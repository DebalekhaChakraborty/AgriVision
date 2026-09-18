"""Bounded perception -> decision -> action -> re-perception cycle.

This is the deterministic control substrate a later orchestrator will drive. It
is not the competition agent: there is no planning, no tool selection by a
model, no multi-step reasoning. It executes exactly one cycle:

    assess -> decide -> (act -> re-assess -> compare) -> finalise

**The loop is bounded to a single automated attempt.** If one remediation cannot
make the capture suitable, the result escalates rather than iterating. An
unbounded enhance-and-recheck loop is both a runaway risk and a way to talk
yourself into a bad image by repeatedly nudging metrics.

The original capture is always retained. A remediated image becomes canonical
only when the comparison explicitly accepts it; otherwise the original stands
and the rejection is recorded, which is what makes a harm rate measurable later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from competition.agent.actions import (
    ComparisonResult,
    ComparisonVerdict,
    RemediationAction,
    RemediationDecision,
)
from competition.agent.policy import (
    DEFAULT_REMEDIATION_POLICY,
    RemediationPolicy,
    compare_capture_quality,
    decide_capture_remediation,
)
from competition.agent.trace import RemediationTrace, TraceTool
from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.enhancement import (
    ClaheParameters,
    EnhancementError,
    GammaParameters,
    apply_clahe,
    apply_gamma_correction,
)
from competition.vision.evidence import PerceptionEvidence
from competition.vision.quality import assess_capture_quality

MAX_AUTOMATED_ATTEMPTS = 1
REMEDIATION_VERSION = "phase1b-remediation-1.0.0"


class RemediationDispatchError(ValueError):
    """Raised when an action outside the permitted enum reaches the dispatcher."""


@dataclass
class RemediationResult:
    """Complete outcome of one bounded remediation cycle.

    `canonical_image` is the image downstream stages should use: the remediated
    one if accepted, otherwise the original. It is a numpy array and therefore
    excluded from `to_dict`; everything else is JSON-serialisable.
    """

    original_evidence: PerceptionEvidence
    decision: RemediationDecision
    trace: RemediationTrace
    canonical_image: np.ndarray
    action_parameters: dict = field(default_factory=dict)
    enhancement_metadata: dict = field(default_factory=dict)
    remediated_evidence: PerceptionEvidence | None = None
    comparison: ComparisonResult | None = None
    remediation_attempted: bool = False
    remediation_accepted: bool = False
    escalation_required: bool = False
    escalation_reason: str = ""
    attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "remediation_version": REMEDIATION_VERSION,
            "original_evidence": self.original_evidence.to_dict(include_timing=False),
            "decision": self.decision.to_dict(),
            "action_parameters": self.action_parameters,
            "enhancement_metadata": {
                key: value
                for key, value in self.enhancement_metadata.items()
                if key != "processing_ms"
            },
            "remediated_evidence": (
                self.remediated_evidence.to_dict(include_timing=False)
                if self.remediated_evidence is not None
                else None
            ),
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "remediation_attempted": self.remediation_attempted,
            "remediation_accepted": self.remediation_accepted,
            "escalation_required": self.escalation_required,
            "escalation_reason": self.escalation_reason,
            "attempts": self.attempts,
            "max_automated_attempts": MAX_AUTOMATED_ATTEMPTS,
            "trace": self.trace.to_dict(),
        }

    @property
    def canonical_evidence(self) -> PerceptionEvidence:
        """Evidence describing whichever image is canonical."""
        if self.remediation_accepted and self.remediated_evidence is not None:
            return self.remediated_evidence
        return self.original_evidence


def _dispatch(
    image: np.ndarray, decision: RemediationDecision
) -> tuple[np.ndarray, dict]:
    """Execute an automated action.

    Dispatch is on the `RemediationAction` enum only. A free-form string can
    never reach an image operation, which keeps the executable surface fixed and
    auditable as higher layers are added above this one.
    """
    if not isinstance(decision.action, RemediationAction):
        raise RemediationDispatchError(
            f"action must be a RemediationAction, got {type(decision.action).__name__}"
        )

    if decision.action is RemediationAction.APPLY_GAMMA:
        return apply_gamma_correction(
            image, GammaParameters(gamma=float(decision.parameters["gamma"]))
        )

    if decision.action is RemediationAction.APPLY_CLAHE:
        return apply_clahe(
            image,
            ClaheParameters(
                clip_limit=float(decision.parameters["clip_limit"]),
                tile_grid_size=int(decision.parameters["tile_grid_size"]),
            ),
        )

    raise RemediationDispatchError(
        f"{decision.action.value} is not an executable enhancement action"
    )


def enhance_capture(
    image: np.ndarray, decision: RemediationDecision
) -> tuple[np.ndarray, dict]:
    """Public entry point for performing a decided enhancement.

    Kept separate from the cycle so an orchestrator can call it on its own.
    """
    if not decision.automation_permitted:
        raise RemediationDispatchError(
            f"{decision.action.value} is not an automated action"
        )
    return _dispatch(image, decision)


def run_capture_remediation(
    image: np.ndarray,
    quality_policy: ThresholdPolicy | None = None,
    remediation_policy: RemediationPolicy | None = None,
) -> RemediationResult:
    """Run one bounded perception-decision-action-re-perception cycle."""
    quality_policy = quality_policy or DEFAULT_POLICY
    remediation_policy = remediation_policy or DEFAULT_REMEDIATION_POLICY

    trace = RemediationTrace()

    # --- step 1: perception ---------------------------------------------------
    original_evidence = assess_capture_quality(image, quality_policy)
    trace.add(
        TraceTool.ASSESS_CAPTURE_QUALITY,
        "Initial capture-quality assessment.",
        inputs={"image_sha256": original_evidence.image.content_sha256},
        outputs={
            "quality_flags": list(original_evidence.quality_flags),
            "mean_luminance": original_evidence.illumination.mean_luminance,
            "contrast_score": original_evidence.illumination.contrast_score,
            "shadow_clip_fraction": original_evidence.illumination.shadow_clip_fraction,
            "highlight_clip_fraction": original_evidence.illumination.highlight_clip_fraction,
            "laplacian_variance": original_evidence.sharpness.laplacian_variance,
        },
    )

    # --- step 2: decision -----------------------------------------------------
    decision = decide_capture_remediation(original_evidence, remediation_policy)
    trace.add(
        TraceTool.DECIDE_CAPTURE_REMEDIATION,
        decision.explanation,
        inputs={
            "quality_flags": list(original_evidence.quality_flags),
            "triggering_metrics": decision.triggering_metrics,
        },
        outputs={
            "action": decision.action.value,
            "reason_code": decision.reason_code.value,
            "parameters": decision.parameters,
            "automation_permitted": decision.automation_permitted,
            "human_intervention_required": decision.human_intervention_required,
            "policy_fingerprint": decision.policy_fingerprint,
        },
    )

    result = RemediationResult(
        original_evidence=original_evidence,
        decision=decision,
        trace=trace,
        canonical_image=image.copy(),
        action_parameters=dict(decision.parameters),
    )

    # --- non-automated outcomes end the cycle here ----------------------------
    if decision.action is RemediationAction.NONE:
        trace.add(
            TraceTool.FINALISE,
            "Capture accepted without remediation.",
            outputs={"canonical": "original", "escalation_required": False},
        )
        return result

    if decision.human_intervention_required:
        result.escalation_required = True
        result.escalation_reason = decision.reason_code.value
        trace.add(
            TraceTool.FINALISE,
            "Escalated without automated remediation; information is not recoverable.",
            outputs={
                "canonical": "original",
                "escalation_required": True,
                "escalation_reason": decision.reason_code.value,
            },
        )
        return result

    # --- step 3: action -------------------------------------------------------
    try:
        enhanced, enhancement_metadata = enhance_capture(image, decision)
    except (EnhancementError, RemediationDispatchError) as error:
        result.escalation_required = True
        result.escalation_reason = "ENHANCEMENT_FAILED"
        trace.add(
            TraceTool.FINALISE,
            f"Enhancement failed: {error}. Original retained.",
            outputs={"canonical": "original", "escalation_required": True},
        )
        return result

    result.remediation_attempted = True
    result.attempts = 1
    result.enhancement_metadata = enhancement_metadata

    tool = (
        TraceTool.APPLY_GAMMA_CORRECTION
        if decision.action is RemediationAction.APPLY_GAMMA
        else TraceTool.APPLY_CLAHE
    )
    trace.add(
        tool,
        f"Applied {enhancement_metadata['operation']} on the L* channel.",
        inputs={"parameters": enhancement_metadata["parameters"]},
        outputs={"colour_space": enhancement_metadata["colour_space"]},
    )

    # --- step 4: re-perception ------------------------------------------------
    remediated_evidence = assess_capture_quality(enhanced, quality_policy)
    result.remediated_evidence = remediated_evidence
    trace.add(
        TraceTool.ASSESS_CAPTURE_QUALITY,
        "Re-assessment of the remediated capture.",
        inputs={"image_sha256": remediated_evidence.image.content_sha256},
        outputs={
            "quality_flags": list(remediated_evidence.quality_flags),
            "mean_luminance": remediated_evidence.illumination.mean_luminance,
            "contrast_score": remediated_evidence.illumination.contrast_score,
            "shadow_clip_fraction": remediated_evidence.illumination.shadow_clip_fraction,
            "highlight_clip_fraction": remediated_evidence.illumination.highlight_clip_fraction,
            "laplacian_variance": remediated_evidence.sharpness.laplacian_variance,
        },
    )

    # --- step 5: comparison ---------------------------------------------------
    comparison = compare_capture_quality(
        original_evidence, remediated_evidence, decision, remediation_policy
    )
    result.comparison = comparison
    trace.add(
        TraceTool.COMPARE_CAPTURE_QUALITY,
        comparison.explanation,
        inputs={"target_metric": comparison.target_metric},
        outputs={
            "verdict": comparison.verdict.value,
            "reason_code": comparison.reason_code.value,
            "target_improved": comparison.target_improved,
            "guardrail_violations": list(comparison.guardrail_violations),
            "deltas": {
                name: delta.to_dict() for name, delta in comparison.deltas.items()
            },
        },
    )

    # --- finalise -------------------------------------------------------------
    if comparison.verdict is ComparisonVerdict.ACCEPT_REMEDIATION:
        result.remediation_accepted = True
        result.canonical_image = enhanced
        # Bounded: no second attempt. If the capture is still flagged after an
        # accepted remediation, a human decides rather than the loop continuing.
        still_flagged = bool(remediated_evidence.quality_flags)
        result.escalation_required = still_flagged
        if still_flagged:
            result.escalation_reason = "RESIDUAL_QUALITY_FLAGS_AFTER_REMEDIATION"
        trace.add(
            TraceTool.FINALISE,
            "Remediation accepted; remediated capture is canonical."
            + (" Residual flags remain, so review is requested." if still_flagged else ""),
            outputs={
                "canonical": "remediated",
                "remediation_accepted": True,
                "escalation_required": still_flagged,
                "residual_flags": list(remediated_evidence.quality_flags),
            },
        )
    else:
        # Harm guard: the original stands.
        result.remediation_accepted = False
        result.escalation_required = True
        result.escalation_reason = comparison.reason_code.value
        trace.add(
            TraceTool.FINALISE,
            "Remediation rejected; original capture retained as canonical.",
            outputs={
                "canonical": "original",
                "remediation_accepted": False,
                "escalation_required": True,
                "escalation_reason": comparison.reason_code.value,
            },
        )

    return result
