"""Capture-gated inspection: the model must not run on an unusable capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from competition.agent.inspection import (
    ADVISORY_FLAGS,
    DEFAULT_BLOCKING_FLAGS,
    InferenceSource,
    InspectionOutcome,
    blocking_flags_present,
    inspect_capture,
)
from competition.agent.trace import TraceTool
from competition.models.adapter import load_condition_model
from competition.vision.degradation import adjust_exposure, gaussian_blur
from competition.vision.fixtures import textured_object
from competition.vision.quality import assess_capture_quality

ARTIFACT_DIR = Path("competition/models/artifacts/mobilenetv3_large_v2exp004")

requires_artifact = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.onnx").is_file(),
    reason="exported ONNX artifact absent; run `python -m competition.models.export`",
)


@pytest.fixture(name="model")
def _model():
    return load_condition_model(ARTIFACT_DIR)


# --- gate policy --------------------------------------------------------------


def test_blocking_and_advisory_flag_sets_are_disjoint():
    assert not (DEFAULT_BLOCKING_FLAGS & ADVISORY_FLAGS)


def test_low_contrast_is_advisory_not_blocking():
    """Documents a deliberate policy choice, so a silent change is caught."""
    assert "LOW_CONTRAST" in ADVISORY_FLAGS
    assert "LOW_CONTRAST" not in DEFAULT_BLOCKING_FLAGS


def test_information_destroying_flags_block():
    for flag in ("BLUR_RISK", "UNDEREXPOSED", "OVEREXPOSED",
                 "SHADOW_CLIPPING", "HIGHLIGHT_CLIPPING", "IMAGE_TOO_SMALL"):
        assert flag in DEFAULT_BLOCKING_FLAGS


def test_blocking_flags_present_reports_only_blocking_flags():
    evidence = assess_capture_quality(gaussian_blur(textured_object(), 8.0))
    blockers = blocking_flags_present(evidence)
    assert "BLUR_RISK" in blockers
    assert all(flag in DEFAULT_BLOCKING_FLAGS for flag in blockers)


# --- required test 9: a rejected capture prevents inference -------------------


@requires_artifact
def test_blurred_capture_blocks_condition_inference(model):
    result = inspect_capture(gaussian_blur(textured_object(), 8.0), model)
    assert result.outcome is InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE
    assert result.condition_evidence is None
    assert not result.inference_ran
    assert result.inference_source is InferenceSource.NONE
    assert result.escalation_required
    assert "BLUR_RISK" in result.blocking_flags


@requires_artifact
def test_crushed_capture_blocks_condition_inference(model):
    result = inspect_capture(adjust_exposure(textured_object(), 0.03), model)
    assert result.outcome is InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE
    assert not result.inference_ran


@requires_artifact
def test_tiny_capture_blocks_condition_inference(model):
    tiny = np.full((8, 8, 3), 128, dtype=np.uint8)
    result = inspect_capture(tiny, model)
    assert result.outcome is InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE
    assert "IMAGE_TOO_SMALL" in result.blocking_flags
    assert not result.inference_ran


@requires_artifact
def test_blocked_inspection_records_the_gate_in_the_trace(model):
    result = inspect_capture(gaussian_blur(textured_object(), 8.0), model)
    assert TraceTool.INSPECTION_GATE.value in result.remediation.trace.tool_sequence
    assert TraceTool.PREDICT_CONDITION.value not in result.remediation.trace.tool_sequence
    gate_step = [
        s for s in result.remediation.trace.steps
        if s.tool is TraceTool.INSPECTION_GATE
    ][0]
    assert gate_step.outputs["inference_executed"] is False


# --- required test 10: an accepted capture permits inference ------------------


@requires_artifact
def test_suitable_capture_permits_inference(model):
    result = inspect_capture(textured_object(), model)
    assert result.outcome is InspectionOutcome.INSPECTED
    assert result.inference_ran
    assert result.condition_evidence is not None
    assert TraceTool.PREDICT_CONDITION.value in result.remediation.trace.tool_sequence


@requires_artifact
def test_gate_precedes_inference_in_the_trace(model):
    result = inspect_capture(textured_object(), model)
    sequence = result.remediation.trace.tool_sequence
    assert sequence.index(TraceTool.INSPECTION_GATE.value) < sequence.index(
        TraceTool.PREDICT_CONDITION.value
    )
    assert sequence[0] == TraceTool.ASSESS_CAPTURE_QUALITY.value


# --- required test 11: the remediated image is the one inferred ---------------


@requires_artifact
def test_accepted_remediation_feeds_the_model_the_remediated_image(model):
    """The image the model saw must be the accepted remediated one, not the input."""
    from competition.agent.remediation import run_capture_remediation
    from competition.models.adapter import array_fingerprint

    # A capture that is remediated and then passes the gate.
    dim = adjust_exposure(textured_object(), 0.5)
    remediation = run_capture_remediation(dim)
    if not remediation.remediation_accepted:
        pytest.skip("policy did not accept remediation for this fixture")

    result = inspect_capture(dim, model)
    if result.outcome is not InspectionOutcome.INSPECTED:
        pytest.skip("gate blocked this fixture; covered by the blocking tests")

    assert result.inference_source is InferenceSource.REMEDIATED
    assert result.condition_evidence.input_sha256 == array_fingerprint(
        result.remediation.canonical_image
    )
    assert result.condition_evidence.input_sha256 != array_fingerprint(dim)


@requires_artifact
def test_unremediated_capture_is_inferred_as_original(model):
    result = inspect_capture(textured_object(), model)
    if result.outcome is not InspectionOutcome.INSPECTED:
        pytest.skip("gate blocked the clean fixture")
    assert result.inference_source is InferenceSource.ORIGINAL

    from competition.models.adapter import array_fingerprint

    assert result.condition_evidence.input_sha256 == array_fingerprint(textured_object())


# --- serialisation and hygiene ------------------------------------------------


@requires_artifact
def test_inspection_result_serialises_without_paths(model):
    for image in (textured_object(), gaussian_blur(textured_object(), 8.0)):
        payload = json.dumps(inspect_capture(image, model).to_dict())
        assert "/home/" not in payload
        assert "/Users/" not in payload
        assert ".onnx" not in payload


@requires_artifact
def test_inspection_result_excludes_image_arrays(model):
    payload = inspect_capture(textured_object(), model).to_dict()
    assert "canonical_image" not in json.dumps(payload)


@requires_artifact
def test_gate_can_be_widened_by_policy(model):
    """The blocking set is policy, not a constant buried in the flow."""
    blurred = gaussian_blur(textured_object(), 8.0)
    assert inspect_capture(blurred, model).outcome is (
        InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE
    )
    permissive = inspect_capture(blurred, model, blocking_flags=frozenset())
    assert permissive.outcome is InspectionOutcome.INSPECTED
    assert permissive.inference_ran


# --- Phase 2b: foreground-restricted gating ----------------------------------


def test_roi_gating_is_off_by_default():
    """Documents a deliberate, evidence-based default so a silent flip is caught.

    ROI measurement is correct; the thresholds are not yet calibrated for it.
    """
    from competition.agent.inspection import DEFAULT_INSPECTION_POLICY

    assert DEFAULT_INSPECTION_POLICY.use_foreground_roi is False
    assert DEFAULT_INSPECTION_POLICY.resegment_after_remediation is True


@requires_artifact
def test_default_gate_scope_is_whole_image(model):
    result = inspect_capture(textured_object(), model)
    assert result.gate_scope == "WHOLE_IMAGE"
    assert result.foreground is None
    assert result.roi_evidence is None


# --- required test 20: background-only clipping stops blocking ---------------


@requires_artifact
def test_background_only_clipping_stops_blocking_under_roi_gating(model):
    """The Phase 2 failure case, end to end.

    A dark subject on a blown-out backdrop is blocked by whole-image
    HIGHLIGHT_CLIPPING. With a valid mask, the clipping is revealed as
    background-only and no longer blocks.
    """
    from competition.agent.inspection import InspectionPolicy
    from competition.vision.fixtures import dark_subject_on_bright_background

    image, _ = dark_subject_on_bright_background()

    whole = inspect_capture(image, model)
    assert "HIGHLIGHT_CLIPPING" in whole.blocking_flags

    roi = inspect_capture(
        image, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
    )
    assert roi.foreground is not None and roi.foreground.valid
    assert roi.gate_scope == "FOREGROUND_MASKED"
    assert "HIGHLIGHT_CLIPPING" not in roi.blocking_flags


# --- required test 19: genuinely bad ROI quality still blocks ----------------


@requires_artifact
def test_genuinely_bad_roi_quality_still_blocks(model):
    """Foreground restriction must not become a way to wave images through."""
    from competition.agent.inspection import InspectionPolicy
    from competition.vision.fixtures import dark_subject_on_bright_background

    image, _ = dark_subject_on_bright_background()
    blurred = gaussian_blur(image, 9.0)

    result = inspect_capture(
        blurred, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
    )
    assert result.outcome is InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE
    assert not result.inference_ran
    assert result.blocking_flags


# --- required test 14: an invalid mask never produces trusted ROI evidence ---


@requires_artifact
def test_invalid_segmentation_does_not_produce_roi_evidence(model):
    """Guards rejected the mask, so the gate must stay on whole-image evidence."""
    from competition.agent.inspection import InspectionPolicy
    from competition.vision.fixtures import multiple_disconnected_regions
    from competition.vision.foreground import isolate_foreground

    image, _ = multiple_disconnected_regions()
    _, foreground = isolate_foreground(image)
    assert not foreground.valid  # premise

    result = inspect_capture(
        image, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
    )
    assert result.foreground is not None
    assert not result.foreground.valid
    assert result.roi_evidence is None
    assert result.gate_scope == "WHOLE_IMAGE"


@requires_artifact
def test_invalid_segmentation_is_recorded_in_the_trace(model):
    from competition.agent.inspection import InspectionPolicy
    from competition.vision.fixtures import multiple_disconnected_regions

    image, _ = multiple_disconnected_regions()
    result = inspect_capture(
        image, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
    )
    steps = [s for s in result.remediation.trace.steps
             if s.tool is TraceTool.ISOLATE_FOREGROUND]
    assert steps
    assert steps[0].outputs["valid"] is False
    assert steps[0].outputs["invalid_reasons"]
    assert steps[0].outputs["gate_scope"] == "WHOLE_IMAGE"


# --- required test 15: the mask describes the image actually gated ----------


@requires_artifact
def test_segmentation_runs_on_the_canonical_image_after_remediation(model):
    """Masks are unstable across tone changes, so the canonical image is segmented.

    The recorded foreground hash must match the canonical image, never the
    original input, whenever remediation was accepted.
    """
    from competition.agent.inspection import InspectionPolicy
    from competition.models.adapter import array_fingerprint

    dim = adjust_exposure(textured_object(), 0.5)
    result = inspect_capture(
        dim, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
    )
    if not result.remediation.remediation_accepted or result.foreground is None:
        pytest.skip("no accepted remediation for this fixture")

    assert result.foreground.input_sha256 == array_fingerprint(
        result.remediation.canonical_image
    )
    assert result.foreground.input_sha256 != array_fingerprint(dim)


@requires_artifact
def test_roi_result_serialises_without_paths(model):
    from competition.agent.inspection import InspectionPolicy
    from competition.vision.fixtures import dark_subject_on_bright_background

    image, _ = dark_subject_on_bright_background()
    payload = json.dumps(
        inspect_capture(
            image, model, inspection_policy=InspectionPolicy(use_foreground_roi=True)
        ).to_dict()
    )
    assert "/home/" not in payload
    assert "/Users/" not in payload
    assert '"gate_scope"' in payload
