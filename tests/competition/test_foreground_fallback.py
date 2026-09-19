"""Phase 3b: the fallback ladder, its bounds, and what it may not claim."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from competition.agent.decisions import EvidenceMaturity, maturity_of
from competition.agent.orchestrator import (
    OrchestratorPolicy,
    load_locked_policies,
    run_inspection,
)
from competition.evaluation.phase3_agent_scenarios import (
    reference_capture,
    scene_without_a_subject,
)
from competition.vision.foreground import (
    DEFAULT_GUARDS,
    ForegroundMethod,
    evaluate_mask,
    isolate_foreground,
    segment_saturation_otsu,
)
from competition.vision.foreground_fallback import (
    FALLBACK_LADDER,
    GRABCUT_ITERATIONS,
    WORKING_LONG_EDGE,
    FallbackMethod,
    MaskProvenance,
    isolate_foreground_with_fallback,
    segment_chroma_distance,
    segment_grabcut_scaled,
)

ADJUDICATION = (
    Path(__file__).resolve().parents[2]
    / "competition" / "evaluation" / "results" / "phase3b" / "adjudication.json"
)


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return reference_capture()


# --- the primary is not replaced ---------------------------------------------


def test_primary_mask_is_unchanged_when_it_succeeds(reference):
    """The whole of Phase 2c-B and 2d rests on these exact masks."""
    primary_mask, primary_evidence = isolate_foreground(reference)
    mask, evidence, fallback = isolate_foreground_with_fallback(reference)
    assert primary_evidence.valid
    assert np.array_equal(primary_mask, mask)
    assert evidence.mask_sha256 == primary_evidence.mask_sha256
    assert fallback.provenance == MaskProvenance.PRIMARY.value


def test_the_ladder_is_not_consulted_when_the_primary_succeeds(reference):
    _, _, fallback = isolate_foreground_with_fallback(reference)
    assert len(fallback.attempts) == 1
    assert fallback.attempts[0].method == ForegroundMethod.SATURATION_OTSU.value


def test_default_method_is_still_saturation_otsu():
    from competition.vision.foreground import DEFAULT_METHOD

    assert DEFAULT_METHOD is ForegroundMethod.SATURATION_OTSU


# --- the ladder itself --------------------------------------------------------


def test_ladder_order_is_cheapest_first():
    assert FALLBACK_LADDER == (
        FallbackMethod.CHROMA_DISTANCE,
        FallbackMethod.BORDER_LAB_DISTANCE,
        FallbackMethod.GRABCUT_SCALED,
    )


def test_lightness_otsu_was_dropped_from_the_ladder():
    """Measured at median IoU 0.343 against the primary: it finds something else."""
    assert not any("lightness" in m.value for m in FALLBACK_LADDER)


def test_every_ladder_entry_produces_a_mask_of_the_input_shape(reference):
    from competition.vision.foreground_fallback import _FALLBACKS

    for method in FALLBACK_LADDER:
        mask = _FALLBACKS[method](reference)
        assert mask.shape == reference.shape[:2]
        assert mask.dtype == np.uint8


def test_fallback_masks_are_binary(reference):
    from competition.vision.foreground_fallback import _FALLBACKS

    for method in FALLBACK_LADDER:
        assert set(np.unique(_FALLBACKS[method](reference))) <= {0, 255}


def test_grabcut_runs_at_working_resolution_not_full():
    """Phase 2b rejected GrabCut at 649 ms; the downscale is what makes it usable."""
    assert WORKING_LONG_EDGE == 480
    assert GRABCUT_ITERATIONS == 1


def test_grabcut_scaled_is_faster_than_full_resolution_grabcut():
    import time

    from competition.vision.foreground import segment_grabcut_rect

    large = cv2.resize(reference_capture(), (1600, 1200))
    start = time.perf_counter()
    segment_grabcut_scaled(large)
    scaled_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    segment_grabcut_rect(large)
    full_ms = (time.perf_counter() - start) * 1000
    assert scaled_ms < full_ms


def test_chroma_distance_ignores_lightness():
    """A subject differing only in lightness must not be found by a chroma test."""
    image = np.full((240, 240, 3), 120, np.uint8)
    cv2.circle(image, (120, 120), 60, (170, 170, 170), -1)   # grey on grey
    mask = segment_chroma_distance(image)
    coloured = np.full((240, 240, 3), 120, np.uint8)
    cv2.circle(coloured, (120, 120), 60, (40, 160, 220), -1)  # orange on grey
    assert np.count_nonzero(segment_chroma_distance(coloured)) > np.count_nonzero(mask)


# --- guards are not relaxed ---------------------------------------------------


def test_guards_are_identical_for_primary_and_fallback(reference):
    """Widening which images yield a mask must not widen what counts as one."""
    from competition.vision.foreground_fallback import _FALLBACKS

    for method in FALLBACK_LADDER:
        mask = _FALLBACKS[method](reference)
        valid, reasons, _ = evaluate_mask(reference, mask, method.value, DEFAULT_GUARDS)
        recomputed = evaluate_mask(reference, mask, method.value, DEFAULT_GUARDS)
        assert (valid, reasons) == (recomputed[0], recomputed[1])


def test_a_mask_failing_the_guards_is_never_accepted():
    """An image with no subject at all yields NONE, not a fabricated mask."""
    blank = np.full((256, 256, 3), 200, np.uint8)
    _, evidence, fallback = isolate_foreground_with_fallback(blank)
    if not evidence.valid:
        assert fallback.provenance == MaskProvenance.NONE.value
        assert fallback.accepted_method == ""


def test_unresolved_returns_the_original_primary_failure():
    blank = np.full((256, 256, 3), 200, np.uint8)
    primary_mask, primary_evidence = isolate_foreground(blank)
    mask, evidence, fallback = isolate_foreground_with_fallback(blank)
    if fallback.provenance == MaskProvenance.NONE.value:
        assert evidence.valid == primary_evidence.valid
        assert np.array_equal(mask, primary_mask)


# --- provenance is recorded, never inferred -----------------------------------


def test_every_attempt_is_recorded_whether_or_not_it_was_used():
    scene = scene_without_a_subject()
    _, _, fallback = isolate_foreground_with_fallback(scene)
    assert len(fallback.attempts) >= 1
    assert fallback.attempts[0].method == ForegroundMethod.SATURATION_OTSU.value


def test_fallback_evidence_serialises():
    _, _, fallback = isolate_foreground_with_fallback(reference_capture())
    payload = json.loads(json.dumps(fallback.to_dict()))
    assert payload["provenance"] in {m.value for m in MaskProvenance}


def test_deterministic_payload_excludes_timing():
    _, _, fallback = isolate_foreground_with_fallback(reference_capture())
    payload = fallback.deterministic_payload()
    assert "processing_ms" not in payload
    assert all("processing_ms" not in a for a in payload["attempts"])


def test_fallback_evidence_carries_no_filesystem_path():
    _, _, fallback = isolate_foreground_with_fallback(reference_capture())
    text = json.dumps(fallback.to_dict())
    for marker in ("/home/", "/Users/", "/tmp/", "C:\\", ".jpg", ".png"):
        assert marker not in text


def test_the_ladder_is_deterministic(reference):
    first = isolate_foreground_with_fallback(reference)[2].deterministic_payload()
    second = isolate_foreground_with_fallback(reference)[2].deterministic_payload()
    assert first == second


# --- maturity: a recovered mask is not a primary mask -------------------------


def test_recovered_masks_are_provisional_not_calibrated():
    assert maturity_of("foreground.recovered") is EvidenceMaturity.PROVISIONAL
    assert maturity_of("foreground.valid") is EvidenceMaturity.CALIBRATED


def test_recovered_evidence_may_gate_but_is_weaker():
    """PROVISIONAL still gates - refusing to would be worse - but it is marked."""
    assert maturity_of("foreground.recovered").may_gate


# --- orchestrator integration -------------------------------------------------


@pytest.fixture(scope="module")
def policy() -> OrchestratorPolicy:
    return load_locked_policies()


def test_the_fallback_is_off_by_default(policy):
    """Every committed Phase 2c-B, 2d and 3 result was produced without it."""
    assert policy.enable_foreground_fallback is False
    assert OrchestratorPolicy().enable_foreground_fallback is False


def test_enabling_the_fallback_is_recorded_in_the_policy_summary(policy):
    enabled = dataclasses.replace(policy, enable_foreground_fallback=True)
    assert enabled.to_dict()["foreground_fallback_enabled"] is True
    assert policy.to_dict()["foreground_fallback_enabled"] is False


def test_disabled_fallback_leaves_the_trace_unchanged(policy):
    result = run_inspection(reference_capture(), None, policy, run_id="fb-off")
    step = next(s for s in result.trace.steps if s.tool_name == "segment_foreground")
    assert step.evidence_summary["mask_provenance"] == MaskProvenance.PRIMARY.value
    assert step.evidence_maturity == EvidenceMaturity.CALIBRATED.value


def test_a_scene_with_no_subject_still_fails_safe_with_the_fallback_on(policy):
    enabled = dataclasses.replace(policy, enable_foreground_fallback=True)
    result = run_inspection(scene_without_a_subject(), None, enabled, run_id="fb-scene")
    assert not result.inference_ran
    assert result.final_state.requires_human


def test_a_recovered_mask_is_traced_as_provisional_with_its_provenance(policy):
    """The property that makes acceptance non-silent."""
    raw = Path("competition/data/licensed_real/raw")
    matches = sorted(raw.glob("WC-9fd7cd40220b0f*")) if raw.is_dir() else []
    if not matches:
        pytest.skip("licensed corpus imagery is not present in this checkout")
    image = cv2.imread(str(matches[0]))
    enabled = dataclasses.replace(policy, enable_foreground_fallback=True)
    result = run_inspection(image, None, enabled, run_id="fb-recovered")
    step = next(s for s in result.trace.steps if s.tool_name == "segment_foreground")
    assert step.evidence_summary["mask_provenance"] == MaskProvenance.FALLBACK.value
    assert step.evidence_maturity == EvidenceMaturity.PROVISIONAL.value
    assert "foreground.recovered" in step.evidence_ids
    assert "RECOVERED" in step.decision_reason


# --- the adjudication artifact ------------------------------------------------


def test_the_adjudication_states_that_it_is_not_ground_truth():
    record = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    text = record["what_this_is"]
    assert "NOT ground truth" in text
    assert "single" in text.lower() and "rater" in text.lower()


def test_the_adjudication_covers_every_recovered_image():
    record = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    assert len(record["verdicts"]) == 24


def test_the_adjudication_records_the_negative_result():
    """18 of 19 scene recoveries were wrong. That must stay in the artifact."""
    record = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    scenes = [v for v in record["verdicts"] if v["track"] == "NATURAL_SCENE"]
    wrong = [v for v in scenes if v["verdict"] == "WRONG"]
    assert len(scenes) == 19
    assert len(wrong) == 18


def test_every_clean_base_recovery_was_usable():
    record = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    clean = [v for v in record["verdicts"] if v["track"] == "CLEAN_BASE"]
    assert len(clean) == 5
    assert all(v["verdict"] in {"CORRECT", "ACCEPTABLE"} for v in clean)


def test_every_verdict_uses_the_declared_vocabulary():
    record = json.loads(ADJUDICATION.read_text(encoding="utf-8"))
    allowed = set(record["verdict_vocabulary"])
    assert all(v["verdict"] in allowed for v in record["verdicts"])


def test_no_food_safety_vocabulary_in_the_adjudication():
    from competition.models.ontology import PROHIBITED_CLAIM_TERMS

    text = ADJUDICATION.read_text(encoding="utf-8").lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in text
