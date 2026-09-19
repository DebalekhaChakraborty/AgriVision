"""Orchestrator behaviour: gating, bounds, re-perception and failing safe."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.artifact_policy import ArtifactPolicy, CaptureArtifactFlag
from competition.agent.decisions import EvidenceMaturity
from competition.agent.orchestrator import (
    RESIDUAL_BLOCKING_FLAGS,
    ExecutionBudget,
    InspectionRunResult,
    OrchestratorPolicy,
    load_locked_policies,
    run_inspection,
)
from competition.agent.render import render_result, render_trace
from competition.agent.state import InspectionState
from competition.agent.tools import ToolName
from competition.evaluation.phase3_agent_scenarios import (
    compress_tone,
    corrupt_input,
    darken,
    defocus,
    glared,
    reference_capture,
    scene_without_a_subject,
)

ROI_TOOLS = (
    ToolName.ASSESS_CAPTURE_QUALITY.value,
    ToolName.ASSESS_LOCAL_HIGHLIGHTS.value,
    ToolName.ASSESS_VISIBILITY.value,
)


@pytest.fixture(scope="module")
def policy() -> OrchestratorPolicy:
    return load_locked_policies()


@pytest.fixture(scope="module")
def model():
    from competition.evaluation.phase3_agent_scenarios import load_model

    loaded, reason = load_model()
    if loaded is None:
        pytest.skip(f"condition-model artifact unavailable: {reason}")
    return loaded


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return reference_capture()


# --- test 1: normal capture reaches inference --------------------------------


def test_normal_capture_runs_the_condition_model(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t1")
    assert result.final_state is InspectionState.COMPLETE
    assert result.inference_ran
    assert result.condition_evidence is not None


def test_normal_capture_needs_no_remediation(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t1b")
    assert not result.remediation_applied
    assert ToolName.APPLY_GAMMA_CORRECTION.value not in result.trace.tool_sequence


# --- test 2: underexposure -> gamma -> re-perception -------------------------


def test_underexposure_applies_gamma_then_re_perceives(reference, model, policy):
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t2")
    tools = result.trace.tool_sequence
    assert result.remediation_action == RemediationAction.APPLY_GAMMA.value
    assert ToolName.APPLY_GAMMA_CORRECTION.value in tools
    # Test 22: segmentation is recomputed after the pixels changed.
    assert tools.count(ToolName.SEGMENT_FOREGROUND.value) == 2


def test_canonical_hash_changes_after_an_accepted_remediation(reference, model, policy):
    """Test 21."""
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t21")
    assert result.remediation_accepted
    assert result.canonical_image_sha256 != result.original_image_sha256


def test_evidence_after_remediation_describes_the_new_image(reference, model, policy):
    """Test 23: stale evidence is not reused."""
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t23")
    quality_steps = [
        step for step in result.trace.steps
        if step.tool_name == ToolName.ASSESS_CAPTURE_QUALITY.value
    ]
    assert len(quality_steps) == 2
    before, after = quality_steps
    assert before.evidence_summary["mean_luminance"] != after.evidence_summary[
        "mean_luminance"
    ]
    # And the second pass is hashed against the remediated image.
    assert after.input_artifact_hashes["canonical_image_sha256"] == (
        result.canonical_image_sha256
    )


# --- test 3: low contrast -> CLAHE -> re-perception --------------------------


def test_moderate_contrast_applies_clahe(reference, model, policy):
    result = run_inspection(compress_tone(reference, 0.26), model, policy, run_id="t3")
    assert result.remediation_action == RemediationAction.APPLY_CLAHE.value
    assert ToolName.APPLY_CLAHE.value in result.trace.tool_sequence
    assert result.trace.tool_sequence.count(ToolName.SEGMENT_FOREGROUND.value) == 2


# --- test 4: severe blur -> recapture, no inference ---------------------------


def test_severe_blur_requests_recapture_and_never_infers(reference, model, policy):
    result = run_inspection(defocus(reference, 21), model, policy, run_id="t4")
    assert result.final_state is InspectionState.REQUEST_RECAPTURE
    assert not result.inference_ran
    assert ToolName.RUN_CONDITION_MODEL.value not in result.trace.tool_sequence
    assert "BLUR_RISK" in result.quality_flags


def test_blur_is_never_sent_for_enhancement(reference, model, policy):
    result = run_inspection(defocus(reference, 21), model, policy, run_id="t4b")
    assert ToolName.APPLY_GAMMA_CORRECTION.value not in result.trace.tool_sequence
    assert ToolName.APPLY_CLAHE.value not in result.trace.tool_sequence


# --- test 5: invalid foreground -> no ROI detectors, no model ----------------


def test_segmentation_failure_runs_no_roi_detector(model, policy):
    result = run_inspection(scene_without_a_subject(), model, policy, run_id="t5")
    for tool in ROI_TOOLS:
        assert tool not in result.trace.tool_sequence
    assert ToolName.RUN_CONDITION_MODEL.value not in result.trace.tool_sequence


def test_segmentation_failure_passes_through_insufficient_evidence(model, policy):
    result = run_inspection(scene_without_a_subject(), model, policy, run_id="t5b")
    assert InspectionState.INSUFFICIENT_VISUAL_EVIDENCE.value in (
        result.trace.state_sequence
    )
    assert result.final_state is InspectionState.REQUEST_RECAPTURE
    assert not result.foreground_valid


def test_segmentation_failure_is_traced_as_coverage_not_occlusion(model, policy):
    """B11: never 'occlusion detected'."""
    result = run_inspection(scene_without_a_subject(), model, policy, run_id="t5c")
    text = json.dumps(result.trace.to_dict(include_timing=False)).lower()
    assert "occlusion detected" not in text
    assert "hand detected" not in text


def test_no_artifact_evidence_is_fabricated_without_a_mask(model, policy):
    result = run_inspection(scene_without_a_subject(), model, policy, run_id="t5d")
    assert result.artifact_flags == []
    assert result.quality_flags == []


def test_foreground_invalid_action_is_configurable(model, policy):
    review = dataclasses.replace(
        policy, foreground_invalid_action=RemediationAction.REQUEST_HUMAN_REVIEW
    )
    result = run_inspection(scene_without_a_subject(), model, review, run_id="t5e")
    assert result.final_state is InspectionState.REQUEST_HUMAN_REVIEW


# --- test 6: destructive clipping -> recapture -------------------------------


def test_severe_darkness_requests_recapture(reference, model, policy):
    result = run_inspection(darken(reference, 0.05), model, policy, run_id="t6")
    assert result.final_state is InspectionState.REQUEST_RECAPTURE
    assert not result.inference_ran


def test_the_residual_gate_matches_the_phase1b_blocking_set():
    from competition.agent.inspection import DEFAULT_BLOCKING_FLAGS

    assert RESIDUAL_BLOCKING_FLAGS == DEFAULT_BLOCKING_FLAGS


# --- tests 7/8/9: bounded execution ------------------------------------------


def test_remediation_happens_at_most_once(reference, model, policy):
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t7")
    assert result.budget["remediation_attempts"] <= 1
    assert result.budget["budget"]["max_remediation_attempts"] == 1


def test_the_condition_model_is_invoked_at_most_once(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t8")
    assert result.budget["condition_model_calls"] <= 1
    assert result.trace.tool_sequence.count(ToolName.RUN_CONDITION_MODEL.value) == 1


def test_segmentation_is_bounded_to_initial_plus_one(reference, model, policy):
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t8b")
    assert result.budget["segmentation_calls"] <= 2


def test_a_tiny_step_budget_fails_safe_rather_than_continuing(reference, model, policy):
    """Test 9."""
    tight = dataclasses.replace(policy, budget=ExecutionBudget(max_steps=2))
    result = run_inspection(reference, model, tight, run_id="t9")
    assert result.final_state is InspectionState.FAILED_SAFE
    assert not result.inference_ran
    assert "BudgetExceeded" in result.failure


def test_every_run_reports_whether_it_stayed_within_budget(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t9b")
    assert result.budget["within_budget"] is True


# --- test 10: evidence changes the next action -------------------------------


def test_changing_only_the_evidence_changes_the_action(reference, model, policy):
    actions = {}
    for name, image in (
        ("reference", reference),
        ("dark", darken(reference, 0.12)),
        ("blurred", defocus(reference, 21)),
        ("flat", compress_tone(reference, 0.26)),
    ):
        result = run_inspection(image, model, policy, run_id=f"t10_{name}")
        actions[name] = next(
            (step.selected_action for step in result.trace.steps if step.selected_action),
            "",
        )
    assert actions["reference"] != actions["dark"]
    assert actions["dark"] == RemediationAction.APPLY_GAMMA.value
    assert actions["blurred"] == RemediationAction.REQUEST_RECAPTURE.value
    assert actions["flat"] == RemediationAction.APPLY_CLAHE.value


# --- test 11: determinism -----------------------------------------------------


def test_identical_input_and_policy_give_an_identical_decision(reference, model, policy):
    first = run_inspection(reference, model, policy, run_id="a")
    second = run_inspection(reference, model, policy, run_id="b")
    assert first.deterministic_payload() == second.deterministic_payload()


def test_the_run_id_is_excluded_from_the_deterministic_payload(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="unique")
    assert "run_id" not in result.deterministic_payload()


# --- tests 12/13/14: the trace ------------------------------------------------


def test_trace_steps_are_numbered_in_execution_order(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t12")
    assert [step.step_id for step in result.trace.steps] == list(
        range(1, len(result.trace.steps) + 1)
    )


def test_each_step_starts_where_the_previous_one_ended(reference, model, policy):
    result = run_inspection(darken(reference, 0.12), model, policy, run_id="t12b")
    for earlier, later in zip(result.trace.steps, result.trace.steps[1:]):
        assert later.state_before == earlier.state_after


def test_trace_serialises_to_json(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t13")
    payload = json.loads(result.trace.to_json())
    assert payload["step_count"] == len(result.trace.steps)


def test_result_serialises_to_json(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t13b")
    assert json.loads(result.to_json())["final_state"] == "COMPLETE"


def test_no_filesystem_path_appears_in_a_trace(reference, model, policy):
    """Test 14."""
    result = run_inspection(reference, model, policy, run_id="t14")
    payload = json.dumps(result.to_dict())
    for marker in ("/home/", "/Users/", "/tmp/", "C:\\", "/var/", ".jpg", ".png"):
        assert marker not in payload, f"{marker!r} leaked into the trace"


def test_no_image_bytes_appear_in_a_trace(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t14b")
    payload = json.dumps(result.to_dict())
    assert len(payload) < 200_000


def test_timing_is_excluded_from_the_deterministic_payload(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t14c")
    payload = result.trace.deterministic_payload()
    assert all("duration_ms" not in step for step in payload["steps"])


# --- test 15/16: maturity in practice ----------------------------------------


def test_glare_alone_does_not_block_and_is_carried_as_advisory(reference, model, policy):
    result = run_inspection(glared(reference), model, policy, run_id="t15")
    assert CaptureArtifactFlag.GLARE_RISK.value in result.artifact_flags
    assert result.final_state is InspectionState.COMPLETE
    assert result.inference_ran
    assert result.advisory_human_action == (
        RemediationAction.REQUEST_REPOSITION_LIGHT.value
    )
    assert result.requested_human_action == ""


def test_glare_never_invokes_an_enhancement(reference, model, policy):
    result = run_inspection(glared(reference), model, policy, run_id="t15b")
    assert ToolName.APPLY_GAMMA_CORRECTION.value not in result.trace.tool_sequence
    assert ToolName.APPLY_CLAHE.value not in result.trace.tool_sequence


def test_the_advisory_step_states_that_it_does_not_block(reference, model, policy):
    result = run_inspection(glared(reference), model, policy, run_id="t15c")
    advisory = [
        step for step in result.trace.steps
        if step.evidence_maturity == EvidenceMaturity.ADVISORY.value
        and step.selected_action
    ]
    assert advisory
    assert advisory[0].evidence_summary.get("blocks_inference") is False


def test_enabling_glare_blocking_is_an_explicit_policy_change(reference, model, policy):
    blocking = dataclasses.replace(
        policy, artifact_policy=ArtifactPolicy(glare_blocks=True)
    )
    result = run_inspection(glared(reference), model, blocking, run_id="t15d")
    assert result.final_state is InspectionState.REQUEST_REPOSITION_LIGHT
    assert not result.inference_ran


def test_visibility_evidence_is_recorded_but_never_gates(reference, model, policy):
    """Test 16."""
    result = run_inspection(reference, model, policy, run_id="t16")
    visibility_steps = [
        step for step in result.trace.steps
        if step.tool_name == ToolName.ASSESS_VISIBILITY.value
    ]
    assert visibility_steps
    step = visibility_steps[0]
    assert step.evidence_maturity == EvidenceMaturity.UNQUALIFIED.value
    assert step.evidence_ids == []
    assert not step.selected_action


# --- test 17: confidence recorded, never acted on -----------------------------


def test_model_confidence_is_recorded_without_triggering_escalation(
    reference, model, policy
):
    result = run_inspection(reference, model, policy, run_id="t17")
    assert result.model_confidence is not None
    assert result.final_state is InspectionState.COMPLETE
    assert result.requested_human_action == ""


def test_the_inference_step_says_confidence_is_not_thresholded(
    reference, model, policy
):
    result = run_inspection(reference, model, policy, run_id="t17b")
    step = next(
        step for step in result.trace.steps
        if step.tool_name == ToolName.RUN_CONDITION_MODEL.value
    )
    assert "RECORDED ONLY" in step.decision_reason
    assert step.evidence_maturity == EvidenceMaturity.UNQUALIFIED.value


# --- test 18: missing model artifact ------------------------------------------


def test_a_missing_model_gates_the_capture_then_refers_to_a_person(reference, policy):
    result = run_inspection(reference, None, policy, run_id="t18")
    assert result.final_state is InspectionState.REQUEST_HUMAN_REVIEW
    assert not result.inference_ran
    assert result.failure == "MODEL_ARTIFACT_ABSENT"


def test_a_missing_model_does_not_skip_the_capture_gate(policy):
    """A blur refusal still happens with no model present."""
    result = run_inspection(defocus(reference_capture(), 21), None, policy, run_id="t18b")
    assert result.final_state is InspectionState.REQUEST_RECAPTURE
    assert "BLUR_RISK" in result.quality_flags


# --- test 19: tool exceptions fail safe ---------------------------------------


def test_invalid_input_fails_safe(policy):
    result = run_inspection(corrupt_input(), None, policy, run_id="t19")
    assert result.final_state is InspectionState.FAILED_SAFE
    assert not result.inference_ran
    assert ToolName.SEGMENT_FOREGROUND.value not in result.trace.tool_sequence


@pytest.mark.parametrize("bad", [
    None,
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((32, 32, 3), dtype=np.float32),
    np.zeros((32, 32, 4), dtype=np.uint8),
])
def test_malformed_inputs_all_fail_safe(bad, policy):
    result = run_inspection(bad, None, policy, run_id="t19b")
    assert result.final_state is InspectionState.FAILED_SAFE
    assert not result.inference_ran


def test_a_model_that_raises_fails_safe(reference, policy):
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("simulated inference fault")

    result = run_inspection(reference, Exploding(), policy, run_id="t19c")
    assert result.final_state is InspectionState.FAILED_SAFE
    assert not result.inference_ran
    assert "simulated inference fault" in result.failure


def test_a_failure_is_traced_rather_than_swallowed(policy):
    result = run_inspection(corrupt_input(), None, policy, run_id="t19d")
    assert result.trace.steps
    assert result.failure


# --- test 20: no condition evidence when blocked ------------------------------


@pytest.mark.parametrize("build", [
    lambda: defocus(reference_capture(), 21),
    lambda: darken(reference_capture(), 0.05),
    lambda: compress_tone(reference_capture(), 0.18),
    scene_without_a_subject,
    corrupt_input,
])
def test_a_blocked_capture_carries_no_condition_evidence(build, model, policy):
    result = run_inspection(build(), model, policy, run_id="t20")
    assert result.condition_evidence is None
    assert result.model_confidence is None
    assert not result.inference_ran


# --- result contract ----------------------------------------------------------


def test_result_records_both_image_hashes(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t-contract")
    assert len(result.original_image_sha256) == 64
    assert len(result.canonical_image_sha256) == 64


def test_result_records_every_policy_fingerprint(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t-fp")
    for key in ("roi_policy_threshold", "artifact_policy", "highlight_policy",
                "visibility_policy", "remediation_policy"):
        assert result.policy_fingerprints[key]


def test_no_food_safety_vocabulary_in_a_result(reference, model, policy):
    from competition.models.ontology import PROHIBITED_CLAIM_TERMS

    result = run_inspection(reference, model, policy, run_id="t-claims")
    payload = json.dumps(result.to_dict()).lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in payload, f"prohibited term {term!r} in the result"


def test_rendered_output_is_readable_text(reference, model, policy):
    result = run_inspection(reference, model, policy, run_id="t-render")
    text = render_result(result)
    assert "Condition model" in text
    assert "Step 1" in text
    assert "/home/" not in text


def test_rendered_output_marks_advisory_evidence(reference, model, policy):
    result = run_inspection(glared(reference), model, policy, run_id="t-render2")
    text = render_result(result)
    assert "ADVISORY" in text
    assert "cannot block" in text


def test_renderer_handles_a_failed_run(policy):
    result = run_inspection(corrupt_input(), None, policy, run_id="t-render3")
    assert "FAILED_SAFE" in render_result(result)


def test_locked_policy_loads_with_the_phase2cb_fingerprint(policy):
    assert policy.roi_policy.lock_fingerprint() == "15b7fb0908d6d622"
    assert policy.roi_policy.focus_metric == "high_frequency_ratio"


def test_locked_policy_keeps_glare_and_visibility_non_blocking(policy):
    assert policy.artifact_policy.glare_blocks is False
    assert policy.artifact_policy.visibility_blocks is False
