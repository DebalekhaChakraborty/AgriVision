"""Phase 3 agent behaviour evaluation.

This measures *the agent*, not the detectors. Whether `high_frequency_ratio`
separates blurred from sharp photographs was settled in Phase 2c-B on licensed
real imagery and is not revisited here; what is measured here is whether a blur
finding reliably produces a recapture request and reliably prevents the
classifier from running.

That distinction governs what the numbers may be called. Every figure below is a
**deterministic scenario-suite result** on constructed fixtures with known
expected behaviour. A task-success rate of 1.0 means the agent did what the
policy says it should on twelve cases someone wrote down in advance. It is not
accuracy, and it says nothing about how often real photographs are glared or
blurred.

Fixtures are synthetic by design rather than by necessity. The Phase 2c-B
held-out groups are spent and the Phase 2d validation set was opened once; using
either again for a new confirmatory claim would be exactly the leakage those
splits exist to prevent. Synthetic fixtures also let a scenario state its
expected behaviour without reference to any measurement, which is what makes
"expected" mean anything.

Reproduce:
    python -m competition.evaluation.phase3_agent_scenarios scenarios
    python -m competition.evaluation.phase3_agent_scenarios counterfactual
    python -m competition.evaluation.phase3_agent_scenarios metrics
    python -m competition.evaluation.phase3_agent_scenarios traces
    python -m competition.evaluation.phase3_agent_scenarios latency
    python -m competition.evaluation.phase3_agent_scenarios all
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from competition.agent.decisions import EvidenceMaturity
from competition.agent.orchestrator import (
    ExecutionBudget,
    InspectionRunResult,
    OrchestratorPolicy,
    load_locked_policies,
    run_inspection,
)
from competition.agent.render import render_result
from competition.agent.state import InspectionState
from competition.agent.tools import ToolName
from competition.vision.degradation import add_glare_at

SCENARIO_VERSION = "phase3-scenarios-1.0.0"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase3"
TRACES_DIR = RESULTS_DIR / "traces"
MODEL_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "models" / "artifacts" / "mobilenetv3_large_v2exp004"
)

FIXTURE_SEED = 20260919


# =============================================================================
# fixtures
# =============================================================================
#
# One base subject, degraded in controlled ways. Using a single base is what
# makes the counterfactual experiment meaningful: when the action changes, the
# only thing that changed was the visual evidence.


def reference_capture(
    size: int = 384, radius: int = 110, base=(45, 150, 215),
    background: int = 140, seed: int = FIXTURE_SEED,
) -> np.ndarray:
    """A lit sphere on a plain ground: one dominant subject with real shading.

    Shading matters. A flat disc has almost no tonal range inside the subject,
    so its ROI contrast sits below the calibrated limit and every scenario built
    on it would begin as a contrast failure.
    """
    image = np.full((size, size, 3), background, np.uint8)
    centre = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    inside = distance <= radius
    curvature = np.clip(1 - (distance / radius) ** 2, 0, 1)
    nx = (xx - centre) / radius
    ny = (yy - centre) / radius
    lambert = np.clip(
        0.35 + 0.75 * (nx * -0.5 + ny * -0.5 + np.sqrt(np.maximum(curvature, 1e-6))),
        0.15, 1.25,
    )
    for channel in range(3):
        plane = image[:, :, channel].astype(np.float32)
        plane[inside] = np.clip(base[channel] * lambert[inside], 0, 255)
        image[:, :, channel] = plane.astype(np.uint8)
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0, 9, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def darken(image: np.ndarray, factor: float, offset: float = 0.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor + offset, 0, 255).astype(np.uint8)


def defocus(image: np.ndarray, kernel: int) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel | 1, kernel | 1), 0)


def compress_tone(image: np.ndarray, factor: float) -> np.ndarray:
    """Shrink the tonal range about its own mean, leaving exposure alone."""
    mean = image.astype(np.float32).mean()
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def glared(image: np.ndarray, intensity: float = 0.5, radius: int = 45) -> np.ndarray:
    """A specular highlight strong enough to be found, weak enough not to clip.

    The intensity is deliberately moderate. At 0.7 the same highlight blows the
    subject's highlights outright, and HIGHLIGHT_CLIPPING — calibrated-tier
    evidence — then blocks the capture before glare is ever consulted. That is
    the correct precedence and it makes the strong case useless as a
    demonstration of advisory handling: the advisory path only exists for glare
    that has *not* already destroyed the pixels underneath it.
    """
    result, _ = add_glare_at(
        image, intensity=intensity, centre=(170, 170), radius=radius, falloff=0.45
    )
    return result


def scene_without_a_subject() -> np.ndarray:
    """Four equal fruits: no dominant component, so no ROI can be attributed."""
    image = np.full((384, 384, 3), 140, np.uint8)
    for cx, cy in ((90, 90), (290, 90), (90, 290), (290, 290)):
        cv2.circle(image, (cx, cy), 55, (45, 150, 215), -1)
    rng = np.random.default_rng(FIXTURE_SEED + 1)
    noisy = image.astype(np.float32) + rng.normal(0, 8, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def corrupt_input() -> np.ndarray:
    """A single-channel array: not a BGR image, and not silently coerced into one."""
    return np.zeros((64, 64), dtype=np.uint8)


# =============================================================================
# scenario contract
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    """One case, with what it is expected to do written down before it runs."""

    name: str
    description: str
    build: object
    expected_final_state: InspectionState
    expected_model_invoked: bool
    expected_tools: tuple = ()
    forbidden_tools: tuple = ()
    expected_remediation: str = ""
    expected_human_action: str = ""
    expects_advisory: bool = False
    policy_note: str = ""
    policy_override: object = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "expected_final_state": self.expected_final_state.value,
            "expected_model_invoked": self.expected_model_invoked,
            "expected_tools": list(self.expected_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "expected_remediation": self.expected_remediation,
            "expected_human_action": self.expected_human_action,
            "expects_advisory": self.expects_advisory,
            "policy_note": self.policy_note,
        }


def _strict_guard_policy(policy: OrchestratorPolicy) -> OrchestratorPolicy:
    """A guard no correction can satisfy, to exercise the rejection branch.

    The harm guard never fired on any natural fixture: gamma computed to reach a
    target band reaches it. Rather than contrive an image until it fails, the
    guard is tightened so the *routing* of a rejection can be observed. This is
    a policy variant and is labelled as one; it is not evidence about images.
    """
    strict = dataclasses.replace(
        policy.remediation_policy,
        target_luminance_min=0.95,
        target_luminance_max=0.99,
        min_luminance_distance_improvement=0.95,
    )
    return dataclasses.replace(policy, remediation_policy=strict)


ROI_DETECTORS = (
    ToolName.ASSESS_CAPTURE_QUALITY.value,
    ToolName.ASSESS_LOCAL_HIGHLIGHTS.value,
    ToolName.ASSESS_VISIBILITY.value,
)


def build_scenarios() -> list:
    reference = reference_capture()
    return [
        Scenario(
            name="normal_acceptable_capture",
            description="A well-exposed, sharp subject with a full tonal range.",
            build=lambda: reference_capture(),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expected_tools=(ToolName.SEGMENT_FOREGROUND.value,
                            ToolName.ASSESS_CAPTURE_QUALITY.value,
                            ToolName.RUN_CONDITION_MODEL.value),
            forbidden_tools=(ToolName.APPLY_GAMMA_CORRECTION.value,
                             ToolName.APPLY_CLAHE.value,
                             ToolName.REQUEST_RECAPTURE.value),
        ),
        Scenario(
            name="recoverable_underexposure",
            description="Dark enough to trip the calibrated ROI underexposure limit, "
                        "with shadow clipping below the unrecoverable rung.",
            build=lambda: darken(reference_capture(), 0.12),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expected_remediation="APPLY_GAMMA",
            expected_tools=(ToolName.APPLY_GAMMA_CORRECTION.value,
                            ToolName.REASSESS_CAPTURE.value,
                            ToolName.RUN_CONDITION_MODEL.value),
            forbidden_tools=(ToolName.REQUEST_RECAPTURE.value,),
        ),
        Scenario(
            name="severe_darkness_clipping",
            description="Shadow clipping past the unrecoverable limit: the detail is "
                        "not in the file, so no tone curve can return it.",
            build=lambda: darken(reference_capture(), 0.05),
            expected_final_state=InspectionState.REQUEST_RECAPTURE,
            expected_model_invoked=False,
            expected_human_action="REQUEST_RECAPTURE",
            forbidden_tools=(ToolName.RUN_CONDITION_MODEL.value,
                             ToolName.APPLY_GAMMA_CORRECTION.value),
        ),
        Scenario(
            name="severe_blur",
            description="ROI high-frequency ratio below the Phase 2c-B calibrated floor.",
            build=lambda: defocus(reference_capture(), 21),
            expected_final_state=InspectionState.REQUEST_RECAPTURE,
            expected_model_invoked=False,
            expected_human_action="REQUEST_RECAPTURE",
            forbidden_tools=(ToolName.RUN_CONDITION_MODEL.value,
                             ToolName.APPLY_GAMMA_CORRECTION.value,
                             ToolName.APPLY_CLAHE.value),
        ),
        Scenario(
            name="moderate_low_contrast",
            description="Contrast inside the narrow band between the severe limit "
                        "and the calibrated advisory limit.",
            build=lambda: compress_tone(reference_capture(), 0.26),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expected_remediation="APPLY_CLAHE",
            expected_tools=(ToolName.APPLY_CLAHE.value,
                            ToolName.REASSESS_CAPTURE.value),
            forbidden_tools=(ToolName.APPLY_GAMMA_CORRECTION.value,),
        ),
        Scenario(
            name="severe_contrast_loss",
            description="Tonal range collapsed below the Phase 2d severe limit, where "
                        "equalisation would amplify quantisation into false detail.",
            build=lambda: compress_tone(reference_capture(), 0.18),
            expected_final_state=InspectionState.REQUEST_RECAPTURE,
            expected_model_invoked=False,
            expected_human_action="REQUEST_RECAPTURE",
            forbidden_tools=(ToolName.RUN_CONDITION_MODEL.value,
                             ToolName.APPLY_CLAHE.value),
        ),
        Scenario(
            name="segmentation_failure",
            description="Four equal subjects: no dominant component, so no region can "
                        "be attributed to a subject.",
            build=scene_without_a_subject,
            expected_final_state=InspectionState.REQUEST_RECAPTURE,
            expected_model_invoked=False,
            expected_human_action="REQUEST_RECAPTURE",
            forbidden_tools=ROI_DETECTORS + (ToolName.RUN_CONDITION_MODEL.value,),
        ),
        Scenario(
            name="advisory_glare",
            description="A concentrated specular highlight on an otherwise usable "
                        "capture. Recorded, reported, and not blocking.",
            build=lambda: glared(reference_capture()),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expects_advisory=True,
            expected_tools=(ToolName.ASSESS_LOCAL_HIGHLIGHTS.value,
                            ToolName.RUN_CONDITION_MODEL.value),
            forbidden_tools=(ToolName.REQUEST_REPOSITION_LIGHT.value,
                             ToolName.APPLY_GAMMA_CORRECTION.value,
                             ToolName.APPLY_CLAHE.value),
        ),
        Scenario(
            name="invalid_input",
            description="A single-channel array. Not an image; not coerced into one.",
            build=corrupt_input,
            expected_final_state=InspectionState.FAILED_SAFE,
            expected_model_invoked=False,
            forbidden_tools=ROI_DETECTORS + (ToolName.SEGMENT_FOREGROUND.value,
                                             ToolName.RUN_CONDITION_MODEL.value),
        ),
        Scenario(
            name="accepted_remediation",
            description="The correction reaches the target band and the harm guard "
                        "accepts it, so inference proceeds on the remediated image.",
            build=lambda: darken(reference_capture(), 0.12),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expected_remediation="APPLY_GAMMA",
            expected_tools=(ToolName.REASSESS_CAPTURE.value,),
        ),
        Scenario(
            name="rejected_remediation",
            description="The harm guard refuses the correction; the canonical image "
                        "reverts and the run escalates instead of trying again.",
            build=lambda: darken(reference_capture(), 0.12),
            expected_final_state=InspectionState.REQUEST_RECAPTURE,
            expected_model_invoked=False,
            expected_human_action="REQUEST_RECAPTURE",
            expected_remediation="APPLY_GAMMA",
            forbidden_tools=(ToolName.RUN_CONDITION_MODEL.value,),
            policy_note="POLICY VARIANT: acceptance guard tightened beyond any "
                        "achievable correction, to exercise the rejection route. "
                        "Not evidence about images.",
            policy_override=_strict_guard_policy,
        ),
        Scenario(
            name="successful_condition_inference",
            description="The full accepted path, ending with visible-condition "
                        "evidence and a recorded, unthresholded confidence.",
            build=lambda: reference_capture(),
            expected_final_state=InspectionState.COMPLETE,
            expected_model_invoked=True,
            expected_tools=(ToolName.RUN_CONDITION_MODEL.value,
                            ToolName.FINALIZE_RESULT.value),
        ),
    ]


# =============================================================================
# running
# =============================================================================


def load_model():
    """Load the condition model, or return None with the reason.

    The ONNX artifact is deliberately not committed, so a clean clone has no
    weights. Scenarios that do not depend on inference still run; the ones that
    do are reported as skipped with the reason stated, never as passing.
    """
    try:
        from competition.models.adapter import load_condition_model

        return load_condition_model(MODEL_ARTIFACT_DIR), ""
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"


def evaluate_scenario(scenario: Scenario, model, policy: OrchestratorPolicy) -> dict:
    """Run one scenario and compare against what it said it would do."""
    active = scenario.policy_override(policy) if scenario.policy_override else policy
    image = scenario.build()
    result = run_inspection(image, model, active, run_id=scenario.name)

    tools = result.trace.tool_sequence
    checks = {
        "final_state": result.final_state is scenario.expected_final_state,
        "model_invoked": result.inference_ran == scenario.expected_model_invoked,
        "expected_tools": all(tool in tools for tool in scenario.expected_tools),
        "forbidden_tools": not any(tool in tools for tool in scenario.forbidden_tools),
        "remediation": (
            result.remediation_action == scenario.expected_remediation
            if scenario.expected_remediation else True
        ),
        "human_action": (
            result.requested_human_action == scenario.expected_human_action
            if scenario.expected_human_action else True
        ),
        "advisory": (
            bool(result.advisory_human_action) == scenario.expects_advisory
        ),
        "within_budget": bool(result.budget.get("within_budget")),
    }
    return {
        **scenario.to_dict(),
        "passed": all(checks.values()),
        "checks": checks,
        "observed_final_state": result.final_state.value,
        "observed_model_invoked": result.inference_ran,
        "observed_remediation": result.remediation_action,
        "observed_remediation_accepted": result.remediation_accepted,
        "observed_human_action": result.requested_human_action,
        "observed_advisory": result.advisory_human_action,
        "observed_tools": tools,
        "observed_states": result.trace.state_sequence,
        "quality_flags": result.quality_flags,
        "artifact_flags": result.artifact_flags,
        "terminal_reason_code": result.terminal_reason_code,
        "budget": result.budget,
        "step_count": len(result.trace.steps),
    }, result


def run_scenarios(model=None, policy: OrchestratorPolicy | None = None) -> dict:
    policy = policy or load_locked_policies()
    model_reason = ""
    if model is None:
        model, model_reason = load_model()
    scenarios = build_scenarios()

    rows = []
    results = {}
    skipped = []
    for scenario in scenarios:
        if scenario.expected_model_invoked and model is None:
            skipped.append({
                "name": scenario.name,
                "reason": "condition-model artifact unavailable: " + model_reason,
            })
            continue
        row, result = evaluate_scenario(scenario, model, policy)
        rows.append(row)
        results[scenario.name] = result

    return {
        "scenario_version": SCENARIO_VERSION,
        "claim_boundary": (
            "deterministic scenario-suite result on constructed fixtures; not "
            "real-world accuracy and not a perception benchmark"
        ),
        "model_available": model is not None,
        "model_unavailable_reason": model_reason,
        "scenario_count": len(scenarios),
        "evaluated_count": len(rows),
        "skipped": skipped,
        "passed": sum(1 for row in rows if row["passed"]),
        "task_success_rate": (
            round(sum(1 for row in rows if row["passed"]) / len(rows), 4)
            if rows else None
        ),
        "policy_fingerprints": policy.fingerprints(),
        "scenarios": rows,
    }, results


# =============================================================================
# the counterfactual experiment
# =============================================================================


def run_counterfactual(model=None, policy: OrchestratorPolicy | None = None) -> dict:
    """Change only the pixels; show the next tool call changes with them.

    This is the Agentic Vision claim reduced to something falsifiable. Every
    variant below is the *same* base subject under the *same* policy with the
    *same* fingerprints — the base image hash is recorded for each so a reader
    can confirm the derivation. If the selected action still differs, the only
    thing that can have caused the difference is what OpenCV measured.
    """
    policy = policy or load_locked_policies()
    if model is None:
        model, _ = load_model()

    base = reference_capture()
    variants = {
        "REFERENCE": base,
        "UNDEREXPOSED": darken(base, 0.12),
        "SEVERE_BLUR": defocus(base, 21),
        "LOW_CONTRAST": compress_tone(base, 0.26),
        "SEVERE_CONTRAST": compress_tone(base, 0.18),
        "GLARE": glared(base),
        "SEGMENTATION_FAILURE": scene_without_a_subject(),
    }

    rows = []
    for name, image in variants.items():
        result = run_inspection(image, model, policy, run_id=f"cf_{name.lower()}")
        first_action = next(
            (step.selected_action for step in result.trace.steps if step.selected_action),
            "",
        )
        rows.append({
            "variant": name,
            "image_sha256": result.original_image_sha256,
            "first_selected_action": first_action,
            "final_state": result.final_state.value,
            "remediation_action": result.remediation_action,
            "condition_model_invoked": result.inference_ran,
            "quality_flags": result.quality_flags,
            "artifact_flags": result.artifact_flags,
            "advisory_human_action": result.advisory_human_action,
            "tool_sequence": result.trace.tool_sequence,
            "terminal_reason_code": result.terminal_reason_code,
        })

    distinct_actions = {row["first_selected_action"] for row in rows}
    distinct_states = {row["final_state"] for row in rows}
    return {
        "scenario_version": SCENARIO_VERSION,
        "claim": (
            "One base subject; only the visual evidence differs between variants. "
            "The policy, its thresholds and their fingerprints are identical "
            "across every row."
        ),
        "base_image_sha256": rows[0]["image_sha256"],
        "policy_fingerprints": policy.fingerprints(),
        "variant_count": len(rows),
        "distinct_first_actions": sorted(distinct_actions),
        "distinct_final_states": sorted(distinct_states),
        "evidence_changes_action": len(distinct_actions) > 1,
        "variants": rows,
    }


# =============================================================================
# agent metrics
# =============================================================================


def compute_metrics(scenario_report: dict, results: dict) -> dict:
    """Behavioural metrics with every denominator stated.

    These describe the agent on this suite. None of them is an accuracy figure,
    and none of them generalises to photographs the suite does not contain.
    """
    rows = scenario_report["scenarios"]
    total = len(rows)
    if not total:
        return {"error": "no scenarios evaluated"}

    # TASK_SUCCESS: scenario reached its preregistered terminal behaviour.
    task_success = sum(1 for row in rows if row["passed"])

    # DECISION_ATTRIBUTION: decisions carrying evidence -> reason -> action.
    attributed = 0
    decisions_total = 0
    for result in results.values():
        for step in result.trace.decision_steps:
            decisions_total += 1
            if step.decision_reason and step.selected_action and (
                step.evidence_ids or step.evidence_summary
            ):
                attributed += 1

    # UNSAFE_INFERENCE: the model ran despite a blocking capture state.
    unsafe = 0
    for result in results.values():
        blocked = result.final_state in (
            InspectionState.REQUEST_RECAPTURE,
            InspectionState.REQUEST_REPOSITION_LIGHT,
            InspectionState.FAILED_SAFE,
        )
        if blocked and result.inference_ran:
            unsafe += 1

    # UNNECESSARY_TOOL_CALLS: a tool the scenario forbade was called anyway.
    unnecessary = sum(
        1 for row in rows if not row["checks"]["forbidden_tools"]
    )

    # BOUNDED_EXECUTION: every declared limit respected.
    bounded = sum(1 for row in rows if row["budget"].get("within_budget"))

    # TRACE_COMPLETENESS: required causal fields present on every step.
    required = ("step_id", "state_before", "tool_name", "state_after")
    complete = 0
    traces_total = 0
    for result in results.values():
        traces_total += 1
        payload = result.trace.to_dict(include_timing=False)
        if all(
            all(field in step and step[field] != "" for field in required)
            for step in payload["steps"]
        ):
            complete += 1

    # FAIL_SAFE: inputs known to be unusable ended in a refusal, never a result.
    unusable = [
        row for row in rows
        if row["name"] in {"invalid_input", "segmentation_failure",
                           "severe_blur", "severe_darkness_clipping",
                           "severe_contrast_loss"}
    ]
    failed_safe = sum(
        1 for row in unusable
        if not row["observed_model_invoked"]
        and row["observed_final_state"] in {
            "FAILED_SAFE", "REQUEST_RECAPTURE", "REQUEST_HUMAN_REVIEW",
            "REQUEST_REPOSITION_LIGHT",
        }
    )

    def rate(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None

    return {
        "scenario_version": SCENARIO_VERSION,
        "claim_boundary": (
            "deterministic scenario-suite result; behaviour of the orchestrator "
            "on constructed fixtures, not real-world accuracy"
        ),
        "TASK_SUCCESS_RATE": {
            "value": rate(task_success, total),
            "numerator": task_success,
            "denominator": total,
            "definition": "scenarios whose observed terminal state, model "
                          "invocation, tool inclusions and exclusions all matched "
                          "what the scenario declared before running",
        },
        "DECISION_ATTRIBUTION_RATE": {
            "value": rate(attributed, decisions_total),
            "numerator": attributed,
            "denominator": decisions_total,
            "definition": "trace steps that selected an action and also carry a "
                          "reason and the evidence it rests on",
        },
        "UNSAFE_INFERENCE_RATE": {
            "value": rate(unsafe, total),
            "numerator": unsafe,
            "denominator": total,
            "definition": "runs where the condition model executed despite a "
                          "blocking terminal state",
        },
        "UNNECESSARY_TOOL_CALL_RATE": {
            "value": rate(unnecessary, total),
            "numerator": unnecessary,
            "denominator": total,
            "definition": "runs that called a tool the scenario declared forbidden",
        },
        "BOUNDED_EXECUTION_RATE": {
            "value": rate(bounded, total),
            "numerator": bounded,
            "denominator": total,
            "definition": "runs respecting every declared limit on steps, "
                          "segmentation calls, remediation attempts and inference",
        },
        "TRACE_COMPLETENESS_RATE": {
            "value": rate(complete, traces_total),
            "numerator": complete,
            "denominator": traces_total,
            "definition": "traces where every step carries step_id, state_before, "
                          "tool_name and state_after",
        },
        "FAIL_SAFE_RATE": {
            "value": rate(failed_safe, len(unusable)),
            "numerator": failed_safe,
            "denominator": len(unusable),
            "definition": "scenarios with known-unusable input that ended in a "
                          "refusal with no condition inference",
        },
    }


# =============================================================================
# judge-visible traces
# =============================================================================


TRACE_EXPORTS = {
    "trace_normal": "normal_acceptable_capture",
    "trace_underexposure_remediation": "recoverable_underexposure",
    "trace_severe_blur_recapture": "severe_blur",
    "trace_segmentation_failure": "segmentation_failure",
    "trace_low_contrast_remediation": "moderate_low_contrast",
    "trace_glare_advisory": "advisory_glare",
}


def export_traces(results: dict) -> dict:
    """Write the curated traces, machine-readable and human-readable side by side."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, scenario_name in TRACE_EXPORTS.items():
        result = results.get(scenario_name)
        if result is None:
            continue
        payload = {
            "scenario": scenario_name,
            "claim_boundary": (
                "constructed fixture; demonstrates the causal chain, not "
                "detector accuracy"
            ),
            "final_state": result.final_state.value,
            "condition_model_invoked": result.inference_ran,
            "policy_fingerprints": result.policy_fingerprints,
            "decisions": [decision.to_dict() for decision in result.decisions],
            **result.trace.to_dict(include_timing=False),
        }
        path = TRACES_DIR / f"{filename}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        readable = TRACES_DIR / f"{filename}.txt"
        readable.write_text(render_result(result), encoding="utf-8")
        written.append(filename)
    return {"written": written, "count": len(written)}


# =============================================================================
# latency by path
# =============================================================================


def run_latency(model=None, policy: OrchestratorPolicy | None = None,
                repeats: int = 15) -> dict:
    """End-to-end latency per terminal path, measured locally.

    Local CPU only. This is not an AWS figure and must never be reported as one:
    no deployment runtime has been built, sized or measured.
    """
    policy = policy or load_locked_policies()
    if model is None:
        model, _ = load_model()

    base = reference_capture()
    paths = {
        "NO_REMEDIATION": base,
        "REMEDIATION": darken(base, 0.12),
        "RECAPTURE_TERMINAL": defocus(base, 21),
        "SEGMENTATION_FAILURE": scene_without_a_subject(),
    }

    # One untimed pass first. The ONNX graph is lazily initialised on its first
    # inference, and letting that land inside the sample makes the tail describe
    # process startup rather than steady-state cost.
    run_inspection(base, model, policy, run_id="lat_warmup")

    rows = {}
    for name, image in paths.items():
        timings = []
        overheads = []
        stage_totals: dict = {}
        for index in range(repeats):
            started = time.perf_counter()
            result = run_inspection(image, model, policy, run_id=f"lat_{name}_{index}")
            elapsed = (time.perf_counter() - started) * 1000.0
            timings.append(elapsed)
            # Overhead is computed per run against that run's own steps. Summing
            # per-tool medians would undercount a path that calls a tool twice,
            # which every remediation path does.
            overheads.append(max(0.0, elapsed - sum(
                step.duration_ms for step in result.trace.steps
            )))
            for step in result.trace.steps:
                stage_totals.setdefault(step.tool_name, []).append(step.duration_ms)
        ordered = sorted(timings)
        rows[name] = {
            "orchestration_overhead_ms": round(statistics.median(overheads), 2),
            "n": repeats,
            "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
            "min_ms": round(ordered[0], 2),
            "max_ms": round(ordered[-1], 2),
            "condition_model_invoked": result.inference_ran,
            "stage_median_ms": {
                tool: round(statistics.median(values), 2)
                for tool, values in sorted(stage_totals.items())
            },
        }

    # B30 asks for the model path separately. It is not a separate *run* — the
    # classifier only ever executes inside an accepted path — so it is reported
    # as the isolated cost of that stage within NO_REMEDIATION.
    accepted = rows.get("NO_REMEDIATION", {}).get("stage_median_ms", {})
    inference_ms = accepted.get(ToolName.RUN_CONDITION_MODEL.value)
    stage_groups = {
        "foreground": ToolName.SEGMENT_FOREGROUND.value,
        "quality": ToolName.ASSESS_CAPTURE_QUALITY.value,
        "artifact_highlights": ToolName.ASSESS_LOCAL_HIGHLIGHTS.value,
        "artifact_visibility": ToolName.ASSESS_VISIBILITY.value,
        "remediation": ToolName.APPLY_GAMMA_CORRECTION.value,
        "reassessment": ToolName.REASSESS_CAPTURE.value,
        "model_inference": ToolName.RUN_CONDITION_MODEL.value,
    }
    remediation_stages = rows.get("REMEDIATION", {}).get("stage_median_ms", {})
    breakdown = {}
    for label, tool in stage_groups.items():
        breakdown[label] = accepted.get(tool, remediation_stages.get(tool))

    height, width = base.shape[:2]
    return {
        "scenario_version": SCENARIO_VERSION,
        "claim_boundary": (
            "local CPU measurement on constructed fixtures; NOT an AWS latency "
            "figure and not a deployment measurement"
        ),
        "resolution": f"{width}x{height}",
        "model_available": model is not None,
        "MODEL_INFERENCE_ms": inference_ms,
        "stage_breakdown_median_ms": breakdown,
        "paths": rows,
    }


# =============================================================================
# entry point
# =============================================================================


def _write(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list) -> int:
    stage = argv[1] if len(argv) > 1 else "all"
    policy = load_locked_policies()
    model, reason = load_model()
    if model is None:
        print(f"condition model unavailable: {reason}")
        print("scenarios requiring inference will be reported as skipped.\n")

    if stage in ("scenarios", "metrics", "traces", "all"):
        report, results = run_scenarios(model, policy)
        path = _write("scenario_results.json", report)
        print(f"scenarios -> {path.name}: {report['passed']}/{report['evaluated_count']} "
              f"passed (task success {report['task_success_rate']})")
        for row in report["scenarios"]:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"  [{mark}] {row['name']:32s} {row['observed_final_state']:24s} "
                  f"model={'Y' if row['observed_model_invoked'] else 'N'}")
            if not row["passed"]:
                failed = [k for k, v in row["checks"].items() if not v]
                print(f"         failing checks: {', '.join(failed)}")
        for skip in report["skipped"]:
            print(f"  [SKIP] {skip['name']}: {skip['reason']}")

        if stage in ("metrics", "all"):
            metrics = compute_metrics(report, results)
            path = _write("agent_metrics.json", metrics)
            print(f"\nmetrics -> {path.name}")
            for key, value in metrics.items():
                if isinstance(value, dict) and "value" in value:
                    print(f"  {key:28s} {value['value']} "
                          f"({value['numerator']}/{value['denominator']})")

            matrix = {
                "scenario_version": SCENARIO_VERSION,
                "tools": [tool.value for tool in ToolName],
                "by_scenario": {
                    row["name"]: {
                        "called": row["observed_tools"],
                        "forbidden": row["forbidden_tools"],
                        "violations": [
                            tool for tool in row["forbidden_tools"]
                            if tool in row["observed_tools"]
                        ],
                    }
                    for row in report["scenarios"]
                },
            }
            _write("tool_call_matrix.json", matrix)

        if stage in ("traces", "all"):
            written = export_traces(results)
            print(f"\ntraces -> {written['count']} written to results/phase3/traces/")

    if stage in ("counterfactual", "all"):
        counterfactual = run_counterfactual(model, policy)
        path = _write("counterfactual_actions.json", counterfactual)
        print(f"\ncounterfactual -> {path.name}")
        for row in counterfactual["variants"]:
            advisory = row["advisory_human_action"] or "-"
            print(f"  {row['variant']:22s} -> {row['first_selected_action']:26s} "
                  f"final={row['final_state']:22s} model="
                  f"{'Y' if row['condition_model_invoked'] else 'N'} "
                  f"advisory={advisory}")
        print(f"  distinct first actions: {counterfactual['distinct_first_actions']}")

    if stage in ("latency", "all"):
        latency = run_latency(model, policy)
        path = _write("latency.json", latency)
        print(f"\nlatency -> {path.name} (local CPU, {latency['resolution']})")
        for name, row in latency["paths"].items():
            print(f"  {name:22s} median {row['median_ms']:7.1f} ms   "
                  f"p95 {row['p95_ms']:7.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
