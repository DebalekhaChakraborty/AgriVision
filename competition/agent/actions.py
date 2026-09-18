"""Remediation action contract.

Actions are a closed enum, never free-form strings. The dispatcher in
`competition.agent.remediation` refuses anything that is not a `RemediationAction`
member, so a typo or an injected string cannot cause an image operation to run.
This matters more later than it does now: when an orchestrator (and eventually a
language model narrating decisions) sits above this layer, the set of things
that can actually execute must stay fixed and auditable.

A decision is a data record, not a side effect. It names what to do, what
evidence triggered it, which thresholds applied, and whether a human is
required — everything needed to audit or replay the choice later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class RemediationAction(str, Enum):
    """The complete set of permitted responses to a capture-quality assessment."""

    NONE = "NONE"
    APPLY_GAMMA = "APPLY_GAMMA"
    APPLY_CLAHE = "APPLY_CLAHE"
    REQUEST_RECAPTURE = "REQUEST_RECAPTURE"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"

    @property
    def is_automated(self) -> bool:
        """Whether this action is something the system performs itself."""
        return self in (RemediationAction.APPLY_GAMMA, RemediationAction.APPLY_CLAHE)

    @property
    def requires_human(self) -> bool:
        return self in (
            RemediationAction.REQUEST_RECAPTURE,
            RemediationAction.REQUEST_HUMAN_REVIEW,
        )


class ReasonCode(str, Enum):
    """Machine-readable justification for a decision.

    Distinct from the action: several conditions can lead to the same action,
    and later evaluation needs to know which one fired.
    """

    CAPTURE_ACCEPTABLE = "CAPTURE_ACCEPTABLE"
    UNDEREXPOSED_RECOVERABLE = "UNDEREXPOSED_RECOVERABLE"
    OVEREXPOSED_RECOVERABLE = "OVEREXPOSED_RECOVERABLE"
    LOW_CONTRAST_RECOVERABLE = "LOW_CONTRAST_RECOVERABLE"
    SHADOW_CLIPPING_UNRECOVERABLE = "SHADOW_CLIPPING_UNRECOVERABLE"
    HIGHLIGHT_CLIPPING_UNRECOVERABLE = "HIGHLIGHT_CLIPPING_UNRECOVERABLE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    REMEDIATION_INEFFECTIVE = "REMEDIATION_INEFFECTIVE"
    REMEDIATION_HARMFUL = "REMEDIATION_HARMFUL"
    BLUR_NOT_REMEDIABLE = "BLUR_NOT_REMEDIABLE"


class ComparisonVerdict(str, Enum):
    """Outcome of comparing pre- and post-remediation evidence."""

    ACCEPT_REMEDIATION = "ACCEPT_REMEDIATION"
    REJECT_REMEDIATION = "REJECT_REMEDIATION"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class RemediationDecision:
    """A machine-readable decision produced from evidence alone."""

    action: RemediationAction
    reason_code: ReasonCode
    triggering_flags: list[str]
    triggering_metrics: dict[str, float]
    thresholds: dict[str, float]
    automation_permitted: bool
    human_intervention_required: bool
    parameters: dict = field(default_factory=dict)
    policy_version: str = ""
    policy_fingerprint: str = ""
    explanation: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["action"] = self.action.value
        data["reason_code"] = self.reason_code.value
        return data


@dataclass(frozen=True)
class MetricDelta:
    """Change in one metric, with the direction that counts as improvement."""

    metric: str
    before: float
    after: float
    delta: float
    improved: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonResult:
    """Multi-metric before/after judgement.

    Success is never decided from sharpness alone — that would recreate the
    exact exposure/sharpness coupling Phase 1 measured. The targeted metric must
    improve *and* no guardrail metric may materially worsen.
    """

    verdict: ComparisonVerdict
    reason_code: ReasonCode
    target_metric: str
    target_improved: bool
    guardrail_violations: list[str]
    deltas: dict[str, MetricDelta]
    # Which of the two independent acceptance routes succeeded. Recorded so a
    # stored comparison shows *why* the target was considered met.
    reached_target_band: bool = False
    improved_by_margin: bool = False
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reason_code": self.reason_code.value,
            "target_metric": self.target_metric,
            "target_improved": self.target_improved,
            "reached_target_band": self.reached_target_band,
            "improved_by_margin": self.improved_by_margin,
            "guardrail_violations": list(self.guardrail_violations),
            "deltas": {name: delta.to_dict() for name, delta in self.deltas.items()},
            "explanation": self.explanation,
        }

    @property
    def accepted(self) -> bool:
        return self.verdict is ComparisonVerdict.ACCEPT_REMEDIATION
