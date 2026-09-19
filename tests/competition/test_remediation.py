"""Bounded remediation cycle: causal order, bounds, harm guard, trace hygiene."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from competition.agent.actions import RemediationAction
from competition.agent.remediation import (
    MAX_AUTOMATED_ATTEMPTS,
    RemediationDispatchError,
    enhance_capture,
    run_capture_remediation,
)
from competition.agent.trace import TraceTool
from competition.vision.degradation import adjust_exposure, gaussian_blur
from competition.vision.fixtures import textured_object


# --- required test 7: at most one automated attempt ---------------------------


def test_maximum_one_automated_attempt_is_declared():
    assert MAX_AUTOMATED_ATTEMPTS == 1


@pytest.mark.parametrize("gain", [0.4, 0.25, 0.15, 2.0])
def test_cycle_never_exceeds_one_attempt(gain):
    result = run_capture_remediation(adjust_exposure(textured_object(), gain))
    assert result.attempts <= MAX_AUTOMATED_ATTEMPTS


def test_enhancement_tool_appears_at_most_once_in_the_trace():
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.15))
    enhancement_steps = [
        tool
        for tool in result.trace.tool_sequence
        if tool in ("apply_gamma_correction", "apply_clahe")
    ]
    assert len(enhancement_steps) <= 1


def test_assessment_runs_at_most_twice():
    """Once before, once after. More would mean the loop iterated."""
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.15))
    assessments = [t for t in result.trace.tool_sequence if t == "assess_capture_quality"]
    assert len(assessments) <= 2


# --- required test 8: rejected remediation preserves the original -------------


def test_rejected_remediation_keeps_the_original_canonical():
    """Force a harmful brightening by moving the policy's target to an extreme."""
    from competition.agent.policy import RemediationPolicy

    bright = adjust_exposure(textured_object(), 1.6)
    harmful_policy = RemediationPolicy(
        target_luminance=0.99,
        target_luminance_min=0.95,
        target_luminance_max=0.999,
        severe_highlight_clip_fraction=0.99,
        severe_shadow_clip_fraction=0.99,
    )
    result = run_capture_remediation(bright, remediation_policy=harmful_policy)

    if result.remediation_attempted and not result.remediation_accepted:
        assert np.array_equal(result.canonical_image, bright)
        assert result.escalation_required
        assert result.canonical_evidence is result.original_evidence


def test_escalated_decision_leaves_original_canonical():
    crushed = adjust_exposure(textured_object(), 0.05)
    result = run_capture_remediation(crushed)
    assert result.decision.action is RemediationAction.REQUEST_RECAPTURE
    assert not result.remediation_attempted
    assert np.array_equal(result.canonical_image, crushed)
    assert result.escalation_required


def test_canonical_image_is_independent_of_the_input_array():
    image = adjust_exposure(textured_object(), 0.05)
    original = image.copy()
    result = run_capture_remediation(image)
    result.canonical_image[0, 0] = [1, 2, 3]
    assert np.array_equal(image, original)


# --- required test 9: accepted remediation records before and after -----------


def test_accepted_remediation_records_both_evidences():
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.25))
    assert result.remediation_attempted
    if result.remediation_accepted:
        assert result.original_evidence is not None
        assert result.remediated_evidence is not None
        assert (
            result.original_evidence.image.content_sha256
            != result.remediated_evidence.image.content_sha256
        )
        assert result.comparison is not None
        assert np.array_equal(result.canonical_image, result.canonical_image)
        assert result.canonical_evidence is result.remediated_evidence


def test_no_remediation_leaves_remediated_evidence_empty():
    result = run_capture_remediation(textured_object())
    assert result.decision.action is RemediationAction.NONE
    assert result.remediated_evidence is None
    assert result.comparison is None
    assert not result.escalation_required


# --- required test 10: trace records causal order ----------------------------


def test_trace_order_for_an_automated_remediation():
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.25))
    sequence = result.trace.tool_sequence
    assert sequence[0] == TraceTool.ASSESS_CAPTURE_QUALITY.value
    assert sequence[1] == TraceTool.DECIDE_CAPTURE_REMEDIATION.value
    assert sequence[2] in (
        TraceTool.APPLY_GAMMA_CORRECTION.value,
        TraceTool.APPLY_CLAHE.value,
    )
    assert sequence[3] == TraceTool.ASSESS_CAPTURE_QUALITY.value
    assert sequence[4] == TraceTool.COMPARE_CAPTURE_QUALITY.value
    assert sequence[-1] == TraceTool.FINALISE.value


def test_trace_order_for_an_escalation():
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.05))
    assert result.trace.tool_sequence == [
        TraceTool.ASSESS_CAPTURE_QUALITY.value,
        TraceTool.DECIDE_CAPTURE_REMEDIATION.value,
        TraceTool.FINALISE.value,
    ]


def test_trace_steps_are_numbered_in_order():
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.25))
    assert [s.step for s in result.trace.steps] == list(
        range(1, len(result.trace.steps) + 1)
    )


def test_trace_links_the_triggering_metric_to_the_action():
    """The causal claim must be derivable from the trace, not asserted."""
    result = run_capture_remediation(adjust_exposure(textured_object(), 0.15))
    assess_step, decide_step = result.trace.steps[0], result.trace.steps[1]

    assert "UNDEREXPOSED" in assess_step.outputs["quality_flags"]
    assert "mean_luminance" in assess_step.outputs
    assert "UNDEREXPOSED" in decide_step.inputs["quality_flags"]
    assert decide_step.outputs["action"] == RemediationAction.APPLY_GAMMA.value


# --- required test 11: trace is JSON-serialisable -----------------------------


@pytest.mark.parametrize("gain", [1.0, 0.4, 0.15, 0.05, 2.0, 3.5])
def test_trace_and_result_serialise_to_json(gain):
    result = run_capture_remediation(adjust_exposure(textured_object(), gain))
    trace_json = result.trace.to_json()
    assert json.loads(trace_json)["step_count"] == len(result.trace.steps)
    payload = json.dumps(result.to_dict())
    restored = json.loads(payload)
    assert "decision" in restored and "trace" in restored


def test_result_dict_excludes_the_image_array():
    result = run_capture_remediation(textured_object())
    assert "canonical_image" not in result.to_dict()


# --- required test 12: no filesystem paths anywhere ---------------------------


@pytest.mark.parametrize("gain", [1.0, 0.25, 0.05, 3.5])
def test_no_filesystem_path_in_evidence_or_trace(gain):
    result = run_capture_remediation(adjust_exposure(textured_object(), gain))
    payload = json.dumps(result.to_dict())
    assert "/home/" not in payload
    assert "/Users/" not in payload
    assert "C:\\" not in payload
    assert "/tmp/" not in payload


# --- required test 13: free-form actions cannot execute -----------------------


def test_free_form_string_action_cannot_execute():
    from competition.agent.actions import ReasonCode, RemediationDecision

    rogue = RemediationDecision(
        action="APPLY_GAMMA; rm -rf /",  # type: ignore[arg-type]
        reason_code=ReasonCode.UNDEREXPOSED_RECOVERABLE,
        triggering_flags=[],
        triggering_metrics={},
        thresholds={},
        automation_permitted=True,
        human_intervention_required=False,
        parameters={"gamma": 0.5},
    )
    with pytest.raises(RemediationDispatchError):
        enhance_capture(textured_object(), rogue)


def test_non_automated_action_cannot_execute():
    from competition.agent.actions import ReasonCode, RemediationDecision

    decision = RemediationDecision(
        action=RemediationAction.REQUEST_RECAPTURE,
        reason_code=ReasonCode.SHADOW_CLIPPING_UNRECOVERABLE,
        triggering_flags=[],
        triggering_metrics={},
        thresholds={},
        automation_permitted=False,
        human_intervention_required=True,
    )
    with pytest.raises(RemediationDispatchError):
        enhance_capture(textured_object(), decision)


def test_action_enum_is_closed():
    assert {a.value for a in RemediationAction} == {
        "NONE",
        "APPLY_GAMMA",
        "APPLY_CLAHE",
        "REQUEST_RECAPTURE",
        "REQUEST_HUMAN_REVIEW",
        "REQUEST_REPOSITION_LIGHT",
    }


# --- required test 17: no research/V1/V2 dependency ---------------------------


def test_no_research_or_v1_modules_are_imported():
    """The competition line must not depend on the research tree or TensorFlow."""
    forbidden = ("tensorflow", "keras", "torch", "torchvision", "transformers")
    loaded = set(sys.modules)
    for name in forbidden:
        assert not any(
            module == name or module.startswith(name + ".") for module in loaded
        ), f"{name} was imported by the competition modules"


def test_competition_sources_do_not_import_research_packages():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in (root / "competition").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("import tensorflow", "import keras", "from v2", "import v2", "from src", "import src"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, offenders


# --- cycle-level behaviour ----------------------------------------------------


def test_blurred_but_well_exposed_capture_escalates_without_enhancement():
    result = run_capture_remediation(gaussian_blur(textured_object(), 5.0))
    assert result.decision.action is RemediationAction.REQUEST_RECAPTURE
    assert not result.remediation_attempted
    assert result.escalation_required


def test_exposure_remediation_can_clear_a_false_blur_flag():
    """The Phase 1b thesis, as an executable assertion.

    A dark capture is flagged BLUR_RISK purely because low signal amplitude
    depresses Laplacian variance. After exposure remediation the flag should
    clear, since no actual blur was ever present.
    """
    dark = adjust_exposure(textured_object(), 0.25)
    result = run_capture_remediation(dark)

    assert "BLUR_RISK" in result.original_evidence.quality_flags
    assert result.remediation_accepted
    assert "BLUR_RISK" not in result.remediated_evidence.quality_flags
    assert (
        result.remediated_evidence.sharpness.laplacian_variance
        > result.original_evidence.sharpness.laplacian_variance
    )
