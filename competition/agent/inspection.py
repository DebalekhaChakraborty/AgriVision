"""Capture-gated condition inspection.

Wires the condition model behind the Phase 1/1b capture-reliability gate. The
model does **not** run on every image: an unusable capture is refused, not
guessed at. That ordering is the point of the whole pipeline — a classifier that
answers confidently on an image it should have rejected is the failure mode this
project exists to avoid.

    assess capture quality            (Phase 1)
              |
    remediate if policy permits       (Phase 1b, bounded to one attempt)
              |
    is the canonical capture suitable?
       no  -> refuse inference, escalate
       yes -> run condition inference on the canonical image

The canonical image is the remediated one when remediation was accepted, and the
original otherwise. Which one was actually inferred is recorded, because
"the model saw a different image from the one the user uploaded" is exactly the
kind of detail that must not be implicit.

Model-versus-perception conflict reasoning (anomalous regions present but the
model says fresh) is **not** implemented here. That needs surface localisation,
which does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from competition.agent.policy import DEFAULT_REMEDIATION_POLICY, RemediationPolicy
from competition.agent.remediation import RemediationResult, run_capture_remediation
from competition.agent.trace import TraceTool
from competition.models.adapter import ConditionModel, predict_condition
from competition.models.evidence import ConditionModelEvidence
from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.evidence import PerceptionEvidence, QualityFlag

INSPECTION_VERSION = "phase2-inspection-1.0.0"


class InspectionOutcome(str, Enum):
    INSPECTED = "INSPECTED"
    BLOCKED_CAPTURE_UNSUITABLE = "BLOCKED_CAPTURE_UNSUITABLE"
    BLOCKED_INFERENCE_FAILED = "BLOCKED_INFERENCE_FAILED"


class InferenceSource(str, Enum):
    ORIGINAL = "original"
    REMEDIATED = "remediated"
    NONE = "none"


# Flags that block condition inference outright, because each destroys
# information the model needs rather than merely degrading it: insufficient
# resolution, blur, exposure outside the usable band, and clipped detail.
#
# LOW_CONTRAST is deliberately advisory rather than blocking. It degrades the
# signal without removing it, and it is the most common residual flag after tone
# remediation — treating it as blocking would refuse a large fraction of
# successfully remediated captures. It is recorded as a caveat instead.
#
# PROVISIONAL: this split is a judgement call, not a calibrated one.
DEFAULT_BLOCKING_FLAGS: frozenset[str] = frozenset(
    {
        QualityFlag.IMAGE_TOO_SMALL.value,
        QualityFlag.BLUR_RISK.value,
        QualityFlag.UNDEREXPOSED.value,
        QualityFlag.OVEREXPOSED.value,
        QualityFlag.SHADOW_CLIPPING.value,
        QualityFlag.HIGHLIGHT_CLIPPING.value,
    }
)
ADVISORY_FLAGS: frozenset[str] = frozenset({QualityFlag.LOW_CONTRAST.value})


def blocking_flags_present(
    evidence: PerceptionEvidence, blocking: frozenset[str] | None = None
) -> list[str]:
    blocking = blocking if blocking is not None else DEFAULT_BLOCKING_FLAGS
    return [flag for flag in evidence.quality_flags if flag in blocking]


@dataclass
class InspectionResult:
    """Outcome of a gated inspection."""

    remediation: RemediationResult
    outcome: InspectionOutcome
    inference_source: InferenceSource
    condition_evidence: ConditionModelEvidence | None = None
    blocking_flags: list[str] = field(default_factory=list)
    advisory_flags: list[str] = field(default_factory=list)
    escalation_required: bool = False
    escalation_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "inspection_version": INSPECTION_VERSION,
            "outcome": self.outcome.value,
            "inference_source": self.inference_source.value,
            "blocking_flags": list(self.blocking_flags),
            "advisory_flags": list(self.advisory_flags),
            "escalation_required": self.escalation_required,
            "escalation_reason": self.escalation_reason,
            "condition_evidence": (
                self.condition_evidence.to_dict(include_timing=False)
                if self.condition_evidence is not None
                else None
            ),
            "remediation": self.remediation.to_dict(),
        }

    @property
    def inference_ran(self) -> bool:
        return self.condition_evidence is not None


def inspect_capture(
    image: np.ndarray,
    model: ConditionModel,
    quality_policy: ThresholdPolicy | None = None,
    remediation_policy: RemediationPolicy | None = None,
    blocking_flags: frozenset[str] | None = None,
) -> InspectionResult:
    """Assess, remediate if permitted, then infer only if the capture is usable."""
    quality_policy = quality_policy or DEFAULT_POLICY
    remediation_policy = remediation_policy or DEFAULT_REMEDIATION_POLICY

    remediation = run_capture_remediation(image, quality_policy, remediation_policy)
    canonical_evidence = remediation.canonical_evidence
    canonical_image = remediation.canonical_image

    blockers = blocking_flags_present(canonical_evidence, blocking_flags)
    advisory = [flag for flag in canonical_evidence.quality_flags if flag in ADVISORY_FLAGS]

    source = (
        InferenceSource.REMEDIATED
        if remediation.remediation_accepted
        else InferenceSource.ORIGINAL
    )

    if blockers:
        remediation.trace.add(
            TraceTool.INSPECTION_GATE,
            "Capture is not suitable for inspection; condition inference refused.",
            inputs={"canonical_quality_flags": list(canonical_evidence.quality_flags)},
            outputs={
                "outcome": InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE.value,
                "blocking_flags": blockers,
                "inference_executed": False,
            },
        )
        return InspectionResult(
            remediation=remediation,
            outcome=InspectionOutcome.BLOCKED_CAPTURE_UNSUITABLE,
            inference_source=InferenceSource.NONE,
            blocking_flags=blockers,
            advisory_flags=advisory,
            escalation_required=True,
            escalation_reason="CAPTURE_UNSUITABLE_FOR_INSPECTION",
        )

    remediation.trace.add(
        TraceTool.INSPECTION_GATE,
        "Capture accepted for inspection.",
        inputs={"canonical_quality_flags": list(canonical_evidence.quality_flags)},
        outputs={
            "outcome": InspectionOutcome.INSPECTED.value,
            "inference_source": source.value,
            "advisory_flags": advisory,
            "inference_executed": True,
        },
    )

    try:
        condition_evidence = predict_condition(model, canonical_image)
    except Exception as error:  # noqa: BLE001 - any inference failure must fail safe
        remediation.trace.add(
            TraceTool.PREDICT_CONDITION,
            f"Condition inference failed: {type(error).__name__}.",
            outputs={"inference_executed": False},
        )
        return InspectionResult(
            remediation=remediation,
            outcome=InspectionOutcome.BLOCKED_INFERENCE_FAILED,
            inference_source=InferenceSource.NONE,
            advisory_flags=advisory,
            escalation_required=True,
            escalation_reason="CONDITION_INFERENCE_FAILED",
        )

    remediation.trace.add(
        TraceTool.PREDICT_CONDITION,
        "Condition inference on the canonical capture.",
        inputs={
            "inference_source": source.value,
            "input_sha256": condition_evidence.input_sha256,
            "runtime": condition_evidence.runtime,
        },
        outputs={
            "predicted_research_label": condition_evidence.predicted_research_label,
            "fruit_type": condition_evidence.fruit_type.value,
            "visible_condition": condition_evidence.visible_condition.value,
            "confidence": condition_evidence.confidence,
        },
    )

    return InspectionResult(
        remediation=remediation,
        outcome=InspectionOutcome.INSPECTED,
        inference_source=source,
        condition_evidence=condition_evidence,
        advisory_flags=advisory,
        escalation_required=bool(advisory),
        escalation_reason="ADVISORY_QUALITY_FLAGS" if advisory else "",
    )
