"""Bounded Agentic Vision orchestrator.

Every earlier phase produced a measurement or a threshold. This one produces
*behaviour*: an explicit loop in which an OpenCV number changes which tool runs
next, and in which the things the system is not allowed to do are refused by
construction rather than by convention.

Four properties are enforced here rather than hoped for.

**Nothing ROI-restricted runs without a valid mask.** Phase 2d measured the cost
of this honestly: only 12 of 29 independent validation photographs produced a
foreground that passed its guards. The response is not to relax the guards. It
is to make segmentation failure a first-class outcome that routes to a person,
because a glare measurement taken on a backdrop is worse than no measurement.

**Experimental evidence cannot become a blocker by accident.** Maturity is
checked when a decision is constructed, so promoting glare from advisory to
gating is an edit to a registry accompanied by the measurement that earns it.

**Remediation happens at most once, and always forces re-perception.** The state
machine has no edge from REASSESSED back to REMEDIATION_SELECTED, so a second
attempt is not merely discouraged, it is unreachable. Evidence computed before a
pixel changed is never reused after it.

**The classifier runs only from ELIGIBLE_FOR_INFERENCE.** A blocked capture
cannot reach it, which is the whole point of the pipeline: a confident answer
about an image that should have been refused is the failure this project exists
to avoid.

No language model participates. Later one may narrate a trace; nothing it says
can select an action, because actions are enum members and tools resolve through
a closed registry.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.artifact_policy import (
    DEFAULT_ARTIFACT_POLICY,
    ArtifactPolicy,
    CaptureArtifactFlag,
    decide_capture_artifacts,
)
from competition.agent.decisions import (
    AgentDecision,
    EvidenceMaturity,
    maturity_of,
    strongest_maturity,
)
from competition.agent.policy import (
    DEFAULT_REMEDIATION_POLICY,
    RemediationPolicy,
    compare_capture_quality,
    decide_capture_remediation,
)
from competition.agent.remediation import RemediationDispatchError, enhance_capture
from competition.agent.state import (
    STATE_MACHINE_VERSION,
    InspectionState,
    StateTransitionError,
    assert_transition,
)
from competition.agent.tools import ToolName
from competition.agent.trace import AgentTrace
from competition.models.adapter import ConditionModel, predict_condition
from competition.vision.blur_metrics import measure_focus
from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.evidence import ImageValidationError, QualityFlag
from competition.vision.foreground import (
    ForegroundError,
    ForegroundEvidence,
    isolate_foreground,
    segment_without_fill,
)
from competition.vision.foreground_fallback import (
    MaskProvenance,
    isolate_foreground_with_fallback,
)
from competition.vision.highlights import (
    DEFAULT_HIGHLIGHT_POLICY,
    LocalHighlightPolicy,
    measure_local_highlights,
)
from competition.vision.quality import (
    assess_capture_quality,
    image_content_sha256,
    validate_image,
)
from competition.vision.roi_policy import RoiQualityPolicy, apply_roi_policy
from competition.vision.visibility import (
    DEFAULT_VISIBILITY_POLICY,
    VisibilityPolicy,
    measure_visibility,
)

# Flags that refuse inference even when no automated correction applies. Taken
# from the Phase 1b gate rather than redefined, so the two layers cannot drift.
RESIDUAL_BLOCKING_FLAGS: frozenset = frozenset({
    QualityFlag.IMAGE_TOO_SMALL.value,
    QualityFlag.BLUR_RISK.value,
    QualityFlag.UNDEREXPOSED.value,
    QualityFlag.OVEREXPOSED.value,
    QualityFlag.SHADOW_CLIPPING.value,
    QualityFlag.HIGHLIGHT_CLIPPING.value,
})

RESIDUAL_FLAG_REASONS: dict = {
    QualityFlag.IMAGE_TOO_SMALL.value: ReasonCode.IMAGE_TOO_SMALL,
    QualityFlag.BLUR_RISK.value: ReasonCode.BLUR_NOT_REMEDIABLE,
    QualityFlag.SHADOW_CLIPPING.value: ReasonCode.SHADOW_CLIPPING_UNRECOVERABLE,
    QualityFlag.HIGHLIGHT_CLIPPING.value: ReasonCode.HIGHLIGHT_CLIPPING_UNRECOVERABLE,
    QualityFlag.UNDEREXPOSED.value: ReasonCode.REMEDIATION_INEFFECTIVE,
    QualityFlag.OVEREXPOSED.value: ReasonCode.REMEDIATION_INEFFECTIVE,
}

RESIDUAL_FLAG_EVIDENCE: dict = {
    QualityFlag.IMAGE_TOO_SMALL.value: "input.dimensions",
    QualityFlag.BLUR_RISK.value: "roi.high_frequency_ratio",
    QualityFlag.SHADOW_CLIPPING.value: "roi.shadow_clip_fraction",
    QualityFlag.HIGHLIGHT_CLIPPING.value: "roi.highlight_clip_fraction",
    QualityFlag.UNDEREXPOSED.value: "roi.mean_luminance.underexposed",
    QualityFlag.OVEREXPOSED.value: "roi.mean_luminance.overexposed",
}

ORCHESTRATOR_VERSION = "phase3-orchestrator-1.0.0"
PIPELINE_VERSION = "agrivision-competition-phase3"

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"
LOCKED_ROI_POLICY_PATH = _RESULTS_DIR / "phase2c_locked_policy.json"
FROZEN_ARTIFACT_POLICY_PATH = _RESULTS_DIR / "phase2d" / "locked_artifact_policy.json"


class OrchestratorError(RuntimeError):
    """Raised for an orchestration fault that is not an ordinary tool failure."""


# =============================================================================
# bounded execution
# =============================================================================


@dataclass(frozen=True)
class ExecutionBudget:
    """Hard limits on what one run may do.

    These are not performance tuning. Each one closes a specific failure mode:
    unbounded remediation (an agent that enhances an image until it likes it),
    repeated inference (a second opinion that is really the same opinion), and
    a step budget that turns any unforeseen cycle into a safe stop instead of a
    hang.
    """

    max_remediation_attempts: int = 1
    max_condition_model_calls: int = 1
    max_segmentation_calls: int = 2  # initial, plus one after remediation
    max_steps: int = 24

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_BUDGET = ExecutionBudget()


class BudgetExceeded(RuntimeError):
    """Raised internally when a limit is hit; always converted to FAILED_SAFE."""


@dataclass
class _BudgetLedger:
    """Live counters for one run."""

    budget: ExecutionBudget
    steps: int = 0
    remediation_attempts: int = 0
    condition_model_calls: int = 0
    segmentation_calls: int = 0

    def spend_step(self) -> None:
        self.steps += 1
        if self.steps > self.budget.max_steps:
            raise BudgetExceeded(
                f"step budget of {self.budget.max_steps} exceeded"
            )

    def spend_segmentation(self) -> None:
        self.segmentation_calls += 1
        if self.segmentation_calls > self.budget.max_segmentation_calls:
            raise BudgetExceeded(
                f"segmentation budget of {self.budget.max_segmentation_calls} exceeded"
            )

    def spend_remediation(self) -> None:
        self.remediation_attempts += 1
        if self.remediation_attempts > self.budget.max_remediation_attempts:
            raise BudgetExceeded(
                f"remediation budget of {self.budget.max_remediation_attempts} exceeded"
            )

    def spend_inference(self) -> None:
        self.condition_model_calls += 1
        if self.condition_model_calls > self.budget.max_condition_model_calls:
            raise BudgetExceeded(
                f"condition-model budget of {self.budget.max_condition_model_calls} exceeded"
            )

    def to_dict(self) -> dict:
        return {
            "budget": self.budget.to_dict(),
            "steps": self.steps,
            "remediation_attempts": self.remediation_attempts,
            "condition_model_calls": self.condition_model_calls,
            "segmentation_calls": self.segmentation_calls,
            "within_budget": (
                self.steps <= self.budget.max_steps
                and self.remediation_attempts <= self.budget.max_remediation_attempts
                and self.condition_model_calls <= self.budget.max_condition_model_calls
                and self.segmentation_calls <= self.budget.max_segmentation_calls
            ),
        }


# =============================================================================
# policy bundle
# =============================================================================


@dataclass(frozen=True)
class OrchestratorPolicy:
    """Every threshold set one run consults, fingerprinted together."""

    roi_policy: RoiQualityPolicy = field(default_factory=RoiQualityPolicy)
    remediation_policy: RemediationPolicy = DEFAULT_REMEDIATION_POLICY
    artifact_policy: ArtifactPolicy = DEFAULT_ARTIFACT_POLICY
    highlight_policy: LocalHighlightPolicy = DEFAULT_HIGHLIGHT_POLICY
    visibility_policy: VisibilityPolicy = DEFAULT_VISIBILITY_POLICY
    quality_policy: ThresholdPolicy = DEFAULT_POLICY
    budget: ExecutionBudget = DEFAULT_BUDGET

    # Where an unusable foreground sends the run. Recapture is the default
    # because the commonest cause is a scene with no single subject, which a
    # person fixes by photographing one item. Review is the alternative for
    # deployments where a second capture is expensive.
    foreground_invalid_action: RemediationAction = RemediationAction.REQUEST_RECAPTURE

    # Phase 3b fallback ladder. OFF by default: every Phase 2c-B, 2d and 3
    # result was produced with the primary alone, and a default that silently
    # changed which mask those numbers describe would invalidate them rather
    # than improve them. Turning it on is a deployment decision whose measured
    # consequences are documented, and the provenance of every recovered mask
    # is recorded in the trace.
    enable_foreground_fallback: bool = False

    def fingerprints(self) -> dict:
        return {
            "roi_policy_threshold": self.roi_policy.threshold_fingerprint(),
            "roi_policy_lock": self.roi_policy.lock_fingerprint(),
            "remediation_policy": self.remediation_policy.fingerprint(),
            "artifact_policy": self.artifact_policy.fingerprint(),
            "highlight_policy": self.highlight_policy.fingerprint(),
            "visibility_policy": self.visibility_policy.fingerprint(),
            "quality_policy": self.quality_policy.fingerprint(),
        }

    def to_dict(self) -> dict:
        return {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "state_machine_version": STATE_MACHINE_VERSION,
            "roi_policy_status": self.roi_policy.status,
            "foreground_fallback_enabled": self.enable_foreground_fallback,
            "foreground_invalid_action": self.foreground_invalid_action.value,
            "budget": self.budget.to_dict(),
            "fingerprints": self.fingerprints(),
        }


def load_locked_policies(budget: ExecutionBudget | None = None) -> OrchestratorPolicy:
    """Assemble the policy bundle from the frozen Phase 2c-B and 2d records.

    Fingerprints are re-derived from the thresholds and compared with what each
    file recorded, so a hand-edited threshold is caught here rather than
    silently changing a decision.
    """
    if not LOCKED_ROI_POLICY_PATH.is_file():
        raise OrchestratorError(
            f"{LOCKED_ROI_POLICY_PATH.name} not found; Phase 2c-B must be run first"
        )
    roi_record = json.loads(LOCKED_ROI_POLICY_PATH.read_text(encoding="utf-8"))
    roi_policy = RoiQualityPolicy.from_dict(roi_record)
    if roi_policy.lock_fingerprint() != roi_record.get("lock_fingerprint"):
        raise OrchestratorError(
            "locked ROI policy fingerprint mismatch: the frozen file was edited"
        )

    artifact_policy = DEFAULT_ARTIFACT_POLICY
    highlight_policy = DEFAULT_HIGHLIGHT_POLICY
    visibility_policy = DEFAULT_VISIBILITY_POLICY
    if FROZEN_ARTIFACT_POLICY_PATH.is_file():
        frozen = json.loads(FROZEN_ARTIFACT_POLICY_PATH.read_text(encoding="utf-8"))
        artifact_policy = ArtifactPolicy(**frozen["artifact_policy"])
        highlight_policy = LocalHighlightPolicy(**{
            **frozen["highlight_policy"],
            "background_sigma_fractions": tuple(
                frozen["highlight_policy"]["background_sigma_fractions"]
            ),
        })
        visibility_policy = VisibilityPolicy(**frozen["visibility_policy"])
        for name, recorded, recomputed in (
            ("artifact", frozen["artifact_policy_fingerprint"],
             artifact_policy.fingerprint()),
            ("highlight", frozen["highlight_policy_fingerprint"],
             highlight_policy.fingerprint()),
            ("visibility", frozen["visibility_policy_fingerprint"],
             visibility_policy.fingerprint()),
        ):
            if recorded != recomputed:
                raise OrchestratorError(
                    f"{name} policy fingerprint mismatch: the frozen file was edited"
                )

    return OrchestratorPolicy(
        roi_policy=roi_policy,
        artifact_policy=artifact_policy,
        highlight_policy=highlight_policy,
        visibility_policy=visibility_policy,
        budget=budget or DEFAULT_BUDGET,
    )


# =============================================================================
# result contract
# =============================================================================


@dataclass
class InspectionRunResult:
    """Everything one run produced, and nothing it did not.

    `condition_evidence` is None whenever the capture was refused. That is not
    an omission to be filled in later by a caller with a fallback classifier: it
    is the record that no inference was performed.
    """

    run_id: str
    final_state: InspectionState
    original_image_sha256: str
    canonical_image_sha256: str
    remediation_applied: bool = False
    remediation_accepted: bool = False
    remediation_action: str = ""
    foreground_valid: bool = False
    foreground_invalid_reasons: list = field(default_factory=list)
    quality_flags: list = field(default_factory=list)
    artifact_flags: list = field(default_factory=list)
    advisory_flags: list = field(default_factory=list)
    requested_human_action: str = ""
    advisory_human_action: str = ""
    terminal_reason_code: str = ""
    condition_evidence: dict | None = None
    model_confidence: float | None = None
    decisions: list = field(default_factory=list)
    trace: AgentTrace = field(default_factory=AgentTrace)
    budget: dict = field(default_factory=dict)
    policy_fingerprints: dict = field(default_factory=dict)
    failure: str = ""
    pipeline_version: str = PIPELINE_VERSION
    orchestrator_version: str = ORCHESTRATOR_VERSION
    processing_ms: float = 0.0

    @property
    def inference_ran(self) -> bool:
        return self.condition_evidence is not None

    @property
    def succeeded(self) -> bool:
        return self.final_state.is_success

    @property
    def human_action_required(self) -> bool:
        return bool(self.requested_human_action)

    def to_dict(self, include_timing: bool = True) -> dict:
        data = {
            "run_id": self.run_id,
            "final_state": self.final_state.value,
            "original_image_sha256": self.original_image_sha256,
            "canonical_image_sha256": self.canonical_image_sha256,
            "remediation_applied": self.remediation_applied,
            "remediation_accepted": self.remediation_accepted,
            "remediation_action": self.remediation_action,
            "foreground_valid": self.foreground_valid,
            "foreground_invalid_reasons": list(self.foreground_invalid_reasons),
            "quality_flags": list(self.quality_flags),
            "artifact_flags": list(self.artifact_flags),
            "advisory_flags": list(self.advisory_flags),
            "requested_human_action": self.requested_human_action,
            "advisory_human_action": self.advisory_human_action,
            "terminal_reason_code": self.terminal_reason_code,
            "condition_evidence": self.condition_evidence,
            "model_confidence": self.model_confidence,
            "inference_ran": self.inference_ran,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "trace": self.trace.to_dict(include_timing=include_timing),
            "budget": self.budget,
            "policy_fingerprints": self.policy_fingerprints,
            "failure": self.failure,
            "pipeline_version": self.pipeline_version,
            "orchestrator_version": self.orchestrator_version,
        }
        if include_timing:
            data["processing_ms"] = self.processing_ms
        return data

    def deterministic_payload(self) -> dict:
        """Timing and run id removed, so identical inputs give identical bytes."""
        payload = self.to_dict(include_timing=False)
        payload.pop("run_id", None)
        payload["trace"] = self.trace.deterministic_payload()
        payload["budget"] = {
            key: value for key, value in (self.budget or {}).items()
        }
        return payload

    def to_json(self, indent: int | None = 2, include_timing: bool = True) -> str:
        return json.dumps(self.to_dict(include_timing), indent=indent, sort_keys=True)


# =============================================================================
# the pure decision function
# =============================================================================


def _flags_of(evidence, calibrated_flags: list) -> list:
    return list(calibrated_flags)


def decide_next_action(
    roi_flags: list,
    roi_evidence,
    artifact_decision,
    foreground_valid: bool,
    policy: OrchestratorPolicy,
    state_before: InspectionState,
    canonical_sha256: str,
    remediation_available: bool = True,
) -> AgentDecision:
    """Map assembled evidence to exactly one action. Pure and deterministic.

    Precedence runs from *information destroyed* to *information degraded*, which
    is the ordering Phase 1b established and Phase 2c-B calibrated. An
    unrecoverable condition is never masked by a recoverable one, so a severely
    clipped image is never sent for gamma correction that would only redistribute
    what survived.

    The tone tier is delegated to `decide_capture_remediation`, unchanged from
    Phase 1b, so the policy that has been under test since then keeps deciding
    exposure. What is new is the tier above it (segmentation validity) and the
    tier below it (the Phase 2d two-tier contrast ladder and advisory artefacts).
    """
    fingerprint = policy.artifact_policy.fingerprint()

    # --- 3. foreground invalid: nothing ROI-restricted may be trusted --------
    if not foreground_valid:
        action = policy.foreground_invalid_action
        return AgentDecision(
            state_before=state_before,
            selected_action=action,
            reason_code=ReasonCode.ARTIFACT_EVIDENCE_UNAVAILABLE,
            state_after=InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
            triggering_evidence_ids=["foreground.valid"],
            evidence_maturity=maturity_of("foreground.valid"),
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=(
                "Foreground isolation did not produce a mask that passed its "
                "validity guards. Every subject-restricted measurement would be "
                "describing an unknown region, so none was taken and the capture "
                "is referred to a person. This is a coverage failure, not a claim "
                "that anything is covering the produce."
            ),
        )

    # --- 1/2/4/5/6. the Phase 1b tone tier, on ROI-calibrated flags ----------
    tone_evidence = SimpleNamespace(
        illumination=roi_evidence.illumination,
        sharpness=roi_evidence.sharpness,
        image=roi_evidence.image,
        quality_flags=list(roi_flags),
        has_flag=lambda flag, flags=frozenset(roi_flags): flag.value in flags,
    )
    tone_decision = decide_capture_remediation(tone_evidence, policy.remediation_policy)

    tone_evidence_ids = {
        ReasonCode.IMAGE_TOO_SMALL: ["input.dimensions"],
        ReasonCode.SHADOW_CLIPPING_UNRECOVERABLE: ["roi.shadow_clip_fraction"],
        ReasonCode.HIGHLIGHT_CLIPPING_UNRECOVERABLE: ["roi.highlight_clip_fraction"],
        ReasonCode.BLUR_NOT_REMEDIABLE: ["roi.high_frequency_ratio"],
        ReasonCode.UNDEREXPOSED_RECOVERABLE: ["roi.mean_luminance.underexposed"],
        ReasonCode.OVEREXPOSED_RECOVERABLE: ["roi.mean_luminance.overexposed"],
        ReasonCode.LOW_CONTRAST_RECOVERABLE: ["roi.contrast_score"],
    }

    if tone_decision.action is RemediationAction.REQUEST_RECAPTURE:
        evidence_ids = tone_evidence_ids.get(
            tone_decision.reason_code, ["roi.high_frequency_ratio"]
        )
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=tone_decision.reason_code,
            state_after=InspectionState.REQUEST_RECAPTURE,
            triggering_evidence_ids=evidence_ids,
            evidence_maturity=strongest_maturity(evidence_ids),
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=tone_decision.explanation,
        )

    if tone_decision.action is RemediationAction.REQUEST_HUMAN_REVIEW:
        evidence_ids = tone_evidence_ids.get(tone_decision.reason_code, [])
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.REQUEST_HUMAN_REVIEW,
            reason_code=tone_decision.reason_code,
            state_after=InspectionState.REQUEST_HUMAN_REVIEW,
            triggering_evidence_ids=evidence_ids,
            evidence_maturity=strongest_maturity(evidence_ids) if evidence_ids
            else EvidenceMaturity.PROVISIONAL,
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=tone_decision.explanation,
        )

    # Exposure correction outranks every tonal-range concern below, because a
    # correctly exposed image is a precondition for judging its contrast.
    if tone_decision.action is RemediationAction.APPLY_GAMMA and remediation_available:
        evidence_ids = tone_evidence_ids.get(
            tone_decision.reason_code, ["roi.mean_luminance.underexposed"]
        )
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.APPLY_GAMMA,
            reason_code=tone_decision.reason_code,
            state_after=InspectionState.REMEDIATION_SELECTED,
            triggering_evidence_ids=evidence_ids,
            evidence_maturity=strongest_maturity(evidence_ids),
            policy_fingerprint=fingerprint,
            automated=True,
            human_action_required=False,
            blocking=False,
            canonical_image_sha256=canonical_sha256,
            explanation=tone_decision.explanation,
        )

    # --- 5b. residual blocking flags, preserved from Phase 1b ----------------
    #
    # The tone tier recaptures only on *severe* clipping, because that is the
    # rung at which enhancement would fabricate detail. Phase 1b additionally
    # refused inference on any blocking flag, severe or not, and that gate is
    # kept: a capture with a sixth of the subject clipped is not something the
    # classifier should be asked about merely because a tone curve cannot fix it.
    residual = [flag for flag in roi_flags if flag in RESIDUAL_BLOCKING_FLAGS]
    if residual:
        evidence_ids = sorted({
            RESIDUAL_FLAG_EVIDENCE[flag] for flag in residual
            if flag in RESIDUAL_FLAG_EVIDENCE
        })
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=RESIDUAL_FLAG_REASONS.get(
                residual[0], ReasonCode.BLUR_NOT_REMEDIABLE
            ),
            state_after=InspectionState.REQUEST_RECAPTURE,
            triggering_evidence_ids=evidence_ids,
            evidence_maturity=strongest_maturity(evidence_ids),
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=(
                f"Capture-quality flags remain after every permitted correction: "
                f"{', '.join(residual)}. These describe information the sensor did "
                f"not record, so condition inference is refused and a new capture "
                f"is requested."
            ),
        )

    # --- 7/8. the Phase 2d contrast ladder ----------------------------------
    flags = list(artifact_decision.flags)

    if CaptureArtifactFlag.SEVERE_LOW_CONTRAST.value in flags:
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=ReasonCode.SEVERE_CONTRAST_LOSS,
            state_after=InspectionState.REQUEST_RECAPTURE,
            triggering_evidence_ids=["artifact.severe_contrast"],
            evidence_maturity=maturity_of("artifact.severe_contrast"),
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=artifact_decision.explanation,
        )

    if CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value in flags and remediation_available:
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.APPLY_CLAHE,
            reason_code=ReasonCode.MODERATE_CONTRAST_LOSS_RECOVERABLE,
            state_after=InspectionState.REMEDIATION_SELECTED,
            triggering_evidence_ids=["artifact.moderate_contrast"],
            evidence_maturity=maturity_of("artifact.moderate_contrast"),
            policy_fingerprint=fingerprint,
            automated=True,
            human_action_required=False,
            blocking=False,
            canonical_image_sha256=canonical_sha256,
            explanation=artifact_decision.explanation,
        )

    # --- 9. glare, advisory unless the policy was explicitly changed ---------
    if CaptureArtifactFlag.GLARE_RISK.value in flags and policy.artifact_policy.glare_blocks:
        return AgentDecision(
            state_before=state_before,
            selected_action=RemediationAction.REQUEST_REPOSITION_LIGHT,
            reason_code=ReasonCode.GLARE_LOCAL_HIGHLIGHT,
            state_after=InspectionState.REQUEST_REPOSITION_LIGHT,
            triggering_evidence_ids=["artifact.glare"],
            evidence_maturity=maturity_of("artifact.glare"),
            # The policy flag is the explicit decision that makes this legal.
            # It is recorded on the decision, so the trace shows a block resting
            # on a detector that did not meet its floor.
            advisory_gating_permitted=True,
            policy_fingerprint=fingerprint,
            automated=False,
            human_action_required=True,
            blocking=True,
            canonical_image_sha256=canonical_sha256,
            explanation=artifact_decision.explanation,
        )

    # --- 10. the capture is usable ------------------------------------------
    # Glare, if present, travels with the run as an advisory human action. It
    # does not block: 0.500 detection on unseen groups with 0.167 false
    # positives is not authority to send a photograph back.
    return AgentDecision(
        state_before=state_before,
        selected_action=RemediationAction.NONE,
        reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
        state_after=InspectionState.ELIGIBLE_FOR_INFERENCE,
        triggering_evidence_ids=["roi.high_frequency_ratio", "roi.contrast_score"],
        evidence_maturity=EvidenceMaturity.CALIBRATED,
        policy_fingerprint=fingerprint,
        automated=False,
        human_action_required=False,
        blocking=False,
        canonical_image_sha256=canonical_sha256,
        explanation=(
            "No calibrated capture-quality failure and no blocking artefact. "
            "The capture is eligible for condition inference."
        ),
    )


# =============================================================================
# the run
# =============================================================================


@dataclass
class _Perception:
    """One complete perception pass over the current canonical image.

    Always produced together and never partially reused: after a pixel changes,
    the whole structure is rebuilt. Carrying the pieces in one object is what
    makes stale-evidence reuse hard to write by accident.
    """

    mask: np.ndarray | None
    foreground: ForegroundEvidence | None
    roi_evidence: object | None
    roi_flags: list
    focus: object | None
    artifact_decision: object | None
    highlights: object | None
    visibility: object | None
    roi_contrast_score: float | None

    @property
    def foreground_valid(self) -> bool:
        return bool(self.foreground is not None and self.foreground.valid)


class _Run:
    """Mutable state of a single inspection. One instance per image."""

    def __init__(
        self,
        image: np.ndarray,
        model: ConditionModel | None,
        policy: OrchestratorPolicy,
        run_id: str,
    ) -> None:
        self.original_image = image
        self.canonical_image = image
        self.model = model
        self.policy = policy
        self.run_id = run_id
        self.state = InspectionState.RECEIVED
        self.trace = AgentTrace(run_id=run_id)
        self.ledger = _BudgetLedger(policy.budget)
        self.decisions: list = []
        self.original_sha = ""
        self.canonical_sha = ""
        self.remediation_applied = False
        self.remediation_accepted = False
        self.remediation_action = ""
        self.advisory_human_action = ""
        self.failure = ""

    # -- state plumbing ------------------------------------------------------

    def step(
        self,
        tool: ToolName,
        state_after: InspectionState,
        started: float,
        **fields,
    ) -> None:
        """Record a tool invocation and move the machine."""
        self.ledger.spend_step()
        # A measurement that leaves the machine where it was is a self-loop, not
        # a transition; the table describes moves between states only.
        if state_after is not self.state:
            assert_transition(self.state, state_after)
        self.trace.add(
            state_before=self.state,
            tool_name=tool,
            state_after=state_after,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            **fields,
        )
        self.state = state_after

    def record(self, decision: AgentDecision) -> None:
        self.decisions.append(decision)

    # -- tools ---------------------------------------------------------------

    def validate_input(self) -> bool:
        started = time.perf_counter()
        try:
            validate_image(self.original_image, self.policy.quality_policy)
        except (ImageValidationError, Exception) as error:  # noqa: BLE001
            self.failure = f"{type(error).__name__}: {error}"
            self.step(
                ToolName.VALIDATE_INPUT,
                InspectionState.FAILED_SAFE,
                started,
                evidence_summary={"valid": False, "error": type(error).__name__},
                evidence_ids=["input.decodable"],
                decision_reason=(
                    "The input could not be read as an image at all. No "
                    "measurement was attempted and no result is claimed."
                ),
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
                evidence_maturity=EvidenceMaturity.CALIBRATED.value,
            )
            return False

        self.original_sha = image_content_sha256(self.original_image)
        self.canonical_sha = self.original_sha
        height, width = self.original_image.shape[:2]
        self.step(
            ToolName.VALIDATE_INPUT,
            InspectionState.INPUT_VALIDATED,
            started,
            evidence_summary={"width": int(width), "height": int(height), "valid": True},
            evidence_ids=["input.decodable", "input.dimensions"],
            output_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            evidence_maturity=EvidenceMaturity.CALIBRATED.value,
        )
        return True

    def perceive(
        self,
        state_after: InspectionState,
        quality_state: InspectionState,
        artifacts_state: InspectionState,
    ) -> _Perception:
        """Segment, then measure — in that order, and never the reverse.

        Returns a perception whose ROI fields are all None when the mask failed
        its guards. Nothing downstream has to remember to check: there is
        nothing to read.
        """
        started = time.perf_counter()
        self.ledger.spend_segmentation()
        mask = None
        foreground = None
        provenance = MaskProvenance.PRIMARY.value
        fallback = None
        try:
            if self.policy.enable_foreground_fallback:
                mask, foreground, fallback = isolate_foreground_with_fallback(
                    self.canonical_image
                )
                provenance = fallback.provenance
            else:
                mask, foreground = isolate_foreground(self.canonical_image)
        except ForegroundError as error:
            self.step(
                ToolName.SEGMENT_FOREGROUND,
                state_after,
                started,
                evidence_summary={"valid": False, "error": str(error)},
                evidence_ids=["foreground.valid"],
                input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
                decision_reason="Segmentation raised; no subject region exists.",
                evidence_maturity=EvidenceMaturity.CALIBRATED.value,
            )
            return _Perception(None, None, None, [], None, None, None, None, None)

        self.step(
            ToolName.SEGMENT_FOREGROUND,
            state_after,
            started,
            evidence_summary={
                "valid": foreground.valid,
                "mask_provenance": provenance,
                "accepted_method": foreground.method,
                "invalid_reasons": list(foreground.invalid_reasons),
                "foreground_fraction": round(foreground.foreground_fraction, 6),
                "component_count": foreground.component_count,
                "method": foreground.method,
            },
            evidence_ids=(
                ["foreground.valid", "foreground.component_count",
                 "foreground.foreground_fraction"]
                + (["foreground.recovered"]
                   if provenance == MaskProvenance.FALLBACK.value else [])
            ),
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            decision_reason=(
                ("Subject isolated by the primary method; ROI measurement permitted."
                 if provenance == MaskProvenance.PRIMARY.value else
                 f"Primary segmentation failed; mask RECOVERED by the "
                 f"{foreground.method} fallback. Adjudicated usable on 5 of 5 "
                 f"single-subject captures and 1 of 19 multi-subject scenes, so "
                 f"this mask is PROVISIONAL, not equivalent to a primary one.")
                if foreground.valid else
                "Mask failed its validity guards; ROI measurement refused."
            ),
            evidence_maturity=(
                EvidenceMaturity.PROVISIONAL.value
                if provenance == MaskProvenance.FALLBACK.value
                else EvidenceMaturity.CALIBRATED.value
            ),
        )

        if not foreground.valid:
            return _Perception(mask, foreground, None, [], None, None, None, None, None)

        return self._measure(mask, foreground, quality_state, artifacts_state)

    def _measure(
        self, mask, foreground,
        quality_state: InspectionState,
        artifacts_state: InspectionState,
    ) -> _Perception:
        """ROI quality and artefact evidence. Only ever called on a valid mask."""
        started = time.perf_counter()
        roi_evidence = assess_capture_quality(
            self.canonical_image, self.policy.quality_policy, mask=mask
        )
        focus = measure_focus(self.canonical_image, mask)
        roi_flags = apply_roi_policy(
            roi_evidence.image, roi_evidence.illumination, focus, self.policy.roi_policy
        )
        self.step(
            ToolName.ASSESS_CAPTURE_QUALITY,
            quality_state,
            started,
            evidence_summary={
                "scope": "FOREGROUND_MASKED",
                "quality_flags": list(roi_flags),
                "high_frequency_ratio": round(focus.high_frequency_ratio, 6),
                "focus_floor": self.policy.roi_policy.focus_floor,
                "mean_luminance": round(roi_evidence.illumination.mean_luminance, 6),
                "contrast_score": round(roi_evidence.illumination.contrast_score, 6),
                "shadow_clip_fraction": round(
                    roi_evidence.illumination.shadow_clip_fraction, 6),
                "highlight_clip_fraction": round(
                    roi_evidence.illumination.highlight_clip_fraction, 6),
            },
            evidence_ids=["roi.high_frequency_ratio", "roi.contrast_score",
                          "roi.mean_luminance.underexposed",
                          "roi.shadow_clip_fraction", "roi.highlight_clip_fraction"],
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            policy_fingerprint=self.policy.roi_policy.threshold_fingerprint(),
            decision_reason=(
                f"Subject-restricted metrics against the Phase 2c-B locked policy "
                f"({self.policy.roi_policy.status})."
            ),
            evidence_maturity=EvidenceMaturity.CALIBRATED.value,
        )

        highlights = visibility = None
        started = time.perf_counter()
        highlights = measure_local_highlights(
            self.canonical_image, mask, self.policy.highlight_policy
        )
        self.step(
            ToolName.ASSESS_LOCAL_HIGHLIGHTS,
            artifacts_state,
            started,
            evidence_summary={
                "glare_flag": bool(highlights.glare_flag),
                "largest_component_fraction": round(
                    highlights.largest_component_fraction, 6),
                "max_local_luminance_excess": round(
                    highlights.max_local_luminance_excess, 3),
            },
            evidence_ids=["artifact.glare"],
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            policy_fingerprint=self.policy.highlight_policy.fingerprint(),
            decision_reason=(
                "Local highlight evidence. ADVISORY: 0.500 detection on unseen "
                "groups with 0.167 false positives, below the 0.70 floor that "
                "would permit it to block."
            ),
            evidence_maturity=EvidenceMaturity.ADVISORY.value,
        )

        started = time.perf_counter()
        try:
            unfilled = segment_without_fill(self.canonical_image)
        except ForegroundError:
            unfilled = None
        visibility = measure_visibility(mask, self.policy.visibility_policy, unfilled)
        self.trace.add(
            state_before=self.state,
            tool_name=ToolName.ASSESS_VISIBILITY,
            state_after=self.state,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            evidence_summary={
                "visibility_sufficient": bool(visibility.visibility_sufficient),
                "insufficiency_reasons": list(visibility.insufficiency_reasons),
                "hull_fill": round(visibility.hull_fill, 6),
                "solidity": round(visibility.solidity, 6),
            },
            evidence_ids=[],
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            policy_fingerprint=self.policy.visibility_policy.fingerprint(),
            decision_reason=(
                "Subject-completeness geometry, RECORDED ONLY. This detector "
                "reached 0.40 in controlled evaluation and never fired; it is "
                "UNQUALIFIED and cannot contribute to any decision. Coverage "
                "failures are reported by the foreground guards instead."
            ),
            evidence_maturity=EvidenceMaturity.UNQUALIFIED.value,
        )
        self.ledger.spend_step()

        artifact_decision = decide_capture_artifacts(
            highlight_evidence=highlights,
            visibility_evidence=visibility,
            roi_contrast_score=roi_evidence.illumination.contrast_score,
            policy=self.policy.artifact_policy,
            segmentation_valid=True,
        )
        return _Perception(
            mask=mask,
            foreground=foreground,
            roi_evidence=roi_evidence,
            roi_flags=roi_flags,
            focus=focus,
            artifact_decision=artifact_decision,
            highlights=highlights,
            visibility=visibility,
            roi_contrast_score=roi_evidence.illumination.contrast_score,
        )

    # -- remediation ---------------------------------------------------------

    def remediate(self, decision: AgentDecision, perception: _Perception) -> bool:
        """Apply one automated correction and establish a new canonical image.

        Returns False if the transform failed, having already routed the run to
        FAILED_SAFE. A failed enhancement is never treated as a no-op that lets
        the original image continue as though nothing was attempted.
        """
        started = time.perf_counter()
        self.ledger.spend_remediation()
        tool = (
            ToolName.APPLY_GAMMA_CORRECTION
            if decision.selected_action is RemediationAction.APPLY_GAMMA
            else ToolName.APPLY_CLAHE
        )
        tone_decision = decide_capture_remediation(
            SimpleNamespace(
                illumination=perception.roi_evidence.illumination,
                sharpness=perception.roi_evidence.sharpness,
                image=perception.roi_evidence.image,
                quality_flags=list(perception.roi_flags),
                has_flag=lambda flag, flags=frozenset(perception.roi_flags):
                    flag.value in flags,
            ),
            self.policy.remediation_policy,
        )
        # The contrast ladder is Phase 2d's, not Phase 1b's, so a CLAHE decision
        # taken on the severe/moderate split carries its own action forward.
        if decision.selected_action is RemediationAction.APPLY_CLAHE:
            from competition.agent.actions import RemediationDecision

            tone_decision = RemediationDecision(
                action=RemediationAction.APPLY_CLAHE,
                reason_code=ReasonCode.MODERATE_CONTRAST_LOSS_RECOVERABLE,
                triggering_flags=[CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value],
                triggering_metrics={
                    "contrast_score": float(perception.roi_contrast_score or 0.0)
                },
                thresholds={
                    "moderate_contrast_limit":
                        self.policy.artifact_policy.moderate_contrast_limit,
                    "severe_contrast_limit":
                        self.policy.artifact_policy.severe_contrast_limit,
                },
                automation_permitted=True,
                human_intervention_required=False,
                parameters={
                    "clip_limit": self.policy.remediation_policy.clahe_clip_limit,
                    "tile_grid_size": self.policy.remediation_policy.clahe_tile_grid_size,
                },
                explanation=decision.explanation,
            )

        try:
            remediated, parameters = enhance_capture(self.canonical_image, tone_decision)
        except (RemediationDispatchError, Exception) as error:  # noqa: BLE001
            self.failure = f"{type(error).__name__}: {error}"
            self.step(
                tool,
                InspectionState.FAILED_SAFE,
                started,
                evidence_summary={"applied": False, "error": type(error).__name__},
                decision_reason=(
                    "The enhancement failed. The original image is not carried "
                    "forward as though it had been corrected."
                ),
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            return False

        previous_sha = self.canonical_sha
        self.canonical_image = remediated
        self.canonical_sha = image_content_sha256(remediated)
        self.remediation_applied = True
        self.remediation_action = decision.selected_action.value
        self.step(
            tool,
            InspectionState.REMEDIATED,
            started,
            evidence_summary={"applied": True, "parameters": parameters},
            input_artifact_hashes={"canonical_image_sha256": previous_sha},
            output_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            decision_reason=(
                "A new canonical image now exists. Every measurement taken "
                "before this point describes a different image and is discarded."
            ),
            selected_action=decision.selected_action.value,
            policy_fingerprint=self.policy.remediation_policy.fingerprint(),
        )
        return True

    def reassess(self, before, after) -> object:
        """Judge whether the correction helped, using the Phase 1b harm guard."""
        started = time.perf_counter()
        from competition.agent.actions import RemediationDecision

        probe = RemediationDecision(
            action=(RemediationAction.APPLY_GAMMA
                    if self.remediation_action == RemediationAction.APPLY_GAMMA.value
                    else RemediationAction.APPLY_CLAHE),
            reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
            triggering_flags=[],
            triggering_metrics={},
            thresholds={},
            automation_permitted=True,
            human_intervention_required=False,
        )
        comparison = compare_capture_quality(
            before, after, probe, self.policy.remediation_policy
        )
        self.trace.add(
            state_before=self.state,
            tool_name=ToolName.REASSESS_CAPTURE,
            state_after=self.state,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            evidence_summary={
                "verdict": comparison.verdict.value,
                "deltas": {name: delta.to_dict()
                           for name, delta in comparison.deltas.items()},
            },
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            decision_reason=comparison.explanation,
            policy_fingerprint=self.policy.remediation_policy.fingerprint(),
            evidence_maturity=EvidenceMaturity.PROVISIONAL.value,
        )
        self.ledger.spend_step()
        return comparison

    # -- inference -----------------------------------------------------------

    def infer(self) -> dict | None:
        """Run the condition model. Reachable only from ELIGIBLE_FOR_INFERENCE."""
        if self.state is not InspectionState.ELIGIBLE_FOR_INFERENCE:
            raise OrchestratorError(
                f"condition inference attempted from {self.state.value}; only "
                "ELIGIBLE_FOR_INFERENCE may reach the model"
            )
        started = time.perf_counter()
        if self.model is None:
            self.failure = "MODEL_ARTIFACT_ABSENT"
            self.step(
                ToolName.RUN_CONDITION_MODEL,
                InspectionState.REQUEST_HUMAN_REVIEW,
                started,
                evidence_summary={"inference_executed": False,
                                  "reason": "MODEL_ARTIFACT_ABSENT"},
                input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
                decision_reason=(
                    "The capture passed the gate but no condition model artifact "
                    "is available in this environment. The capture is referred to "
                    "a person rather than answered without a model."
                ),
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            return None

        self.ledger.spend_inference()
        try:
            evidence = predict_condition(self.model, self.canonical_image)
        except Exception as error:  # noqa: BLE001 - any inference fault fails safe
            self.failure = f"{type(error).__name__}: {error}"
            self.step(
                ToolName.RUN_CONDITION_MODEL,
                InspectionState.FAILED_SAFE,
                started,
                evidence_summary={"inference_executed": False,
                                  "error": type(error).__name__},
                input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
                decision_reason="Condition inference raised; no result is claimed.",
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            return None

        payload = evidence.to_dict(include_timing=False)
        self.step(
            ToolName.RUN_CONDITION_MODEL,
            InspectionState.CONDITION_INFERRED,
            started,
            evidence_summary={
                "inference_executed": True,
                "fruit_type": evidence.fruit_type.value,
                "visible_condition": evidence.visible_condition.value,
                "confidence": round(float(evidence.confidence), 6),
            },
            evidence_ids=[],
            input_artifact_hashes={"canonical_image_sha256": self.canonical_sha},
            decision_reason=(
                "Visible-condition inference on the canonical image. Confidence "
                "is RECORDED ONLY: no confidence threshold has been calibrated, "
                "so no escalation is triggered by it."
            ),
            evidence_maturity=EvidenceMaturity.UNQUALIFIED.value,
        )
        return payload


# The only routes from an action to a terminal state. A closed mapping, so a
# new action cannot silently acquire an undefined ending.
HUMAN_ACTION_ROUTES: dict = {
    RemediationAction.REQUEST_RECAPTURE: (
        ToolName.REQUEST_RECAPTURE, InspectionState.REQUEST_RECAPTURE),
    RemediationAction.REQUEST_REPOSITION_LIGHT: (
        ToolName.REQUEST_REPOSITION_LIGHT, InspectionState.REQUEST_REPOSITION_LIGHT),
    RemediationAction.REQUEST_HUMAN_REVIEW: (
        ToolName.REQUEST_HUMAN_REVIEW, InspectionState.REQUEST_HUMAN_REVIEW),
}


def request_human_action(run: _Run, decision: AgentDecision, started: float) -> None:
    """Terminate the run with a request addressed to a person.

    The run stops here. It does not wait, poll, or pretend the request was
    satisfied; a later capture is a new run with a new id.
    """
    tool, terminal = HUMAN_ACTION_ROUTES[decision.selected_action]
    run.step(
        tool,
        terminal,
        started,
        evidence_summary={"human_action_required": True},
        evidence_ids=list(decision.triggering_evidence_ids),
        input_artifact_hashes={"canonical_image_sha256": run.canonical_sha},
        decision_reason=decision.explanation,
        selected_action=decision.selected_action.value,
        evidence_maturity=decision.evidence_maturity.value,
        policy_fingerprint=decision.policy_fingerprint,
    )


def finalise(run: _Run, perception: _Perception | None, started: float,
              condition: dict | None = None) -> InspectionRunResult:
    """Assemble the result. The only place an InspectionRunResult is built."""
    if run.state is InspectionState.CONDITION_INFERRED:
        run.step(
            ToolName.FINALIZE_RESULT,
            InspectionState.COMPLETE,
            time.perf_counter(),
            evidence_summary={"final_state": InspectionState.COMPLETE.value},
            decision_reason="Inspection complete.",
        )

    artifact_flags: list = []
    advisory_flags: list = []
    if perception is not None and perception.artifact_decision is not None:
        artifact_flags = list(perception.artifact_decision.flags)
        advisory_flags = [
            flag for flag in artifact_flags
            if flag in (CaptureArtifactFlag.GLARE_RISK.value,
                        CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value)
        ]

    terminal_reason = ""
    for decision in reversed(run.decisions):
        terminal_reason = decision.reason_code.value
        break

    return InspectionRunResult(
        run_id=run.run_id,
        final_state=run.state,
        original_image_sha256=run.original_sha,
        canonical_image_sha256=run.canonical_sha,
        remediation_applied=run.remediation_applied,
        remediation_accepted=run.remediation_accepted,
        remediation_action=run.remediation_action,
        foreground_valid=bool(perception and perception.foreground_valid),
        foreground_invalid_reasons=list(
            perception.foreground.invalid_reasons
            if perception is not None and perception.foreground is not None
            else []
        ),
        quality_flags=list(perception.roi_flags) if perception else [],
        artifact_flags=artifact_flags,
        advisory_flags=advisory_flags,
        requested_human_action=(
            run.state.value if run.state.requires_human else ""
        ),
        advisory_human_action=run.advisory_human_action,
        terminal_reason_code=terminal_reason,
        condition_evidence=condition,
        model_confidence=(
            float(condition["confidence"]) if condition and "confidence" in condition
            else None
        ),
        decisions=list(run.decisions),
        trace=run.trace,
        budget=run.ledger.to_dict(),
        policy_fingerprints=run.policy.fingerprints(),
        failure=run.failure,
        processing_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


def run_inspection(
    image: np.ndarray,
    model: ConditionModel | None = None,
    policy: OrchestratorPolicy | None = None,
    run_id: str | None = None,
) -> InspectionRunResult:
    """Run one bounded perception-decision-action loop over a single capture.

    `model` may be None. That is not a degraded mode for testing convenience: a
    clean clone of this repository legitimately has no ONNX artifact, and the
    honest behaviour is to gate the capture exactly as usual and then refer it to
    a person, rather than to answer without a model or to skip the gate because
    the answer would not be used.
    """
    started = time.perf_counter()
    policy = policy or OrchestratorPolicy()
    run = _Run(image, model, policy, run_id or uuid.uuid4().hex[:16])
    perception: _Perception | None = None

    try:
        # --- 1. is this an image at all? -------------------------------------
        if not run.validate_input():
            return finalise(run, None, started)

        # --- 2/3. perceive ---------------------------------------------------
        perception = run.perceive(
            InspectionState.FOREGROUND_ASSESSED,
            InspectionState.QUALITY_ASSESSED,
            InspectionState.ARTIFACTS_ASSESSED,
        )

        # --- 4/5. decide ------------------------------------------------------
        decision = _decide_and_record(run, perception, remediation_available=True)

        # --- 6. act -----------------------------------------------------------
        if decision.state_after is InspectionState.INSUFFICIENT_VISUAL_EVIDENCE:
            return _route_insufficient_evidence(run, perception, decision, started)

        if decision.selected_action in HUMAN_ACTION_ROUTES:
            request_human_action(run, decision, time.perf_counter())
            return finalise(run, perception, started)

        if decision.state_after is InspectionState.REMEDIATION_SELECTED:
            perception = _remediation_excursion(run, perception, decision)
            if run.state.is_terminal:
                return finalise(run, perception, started)
        elif decision.state_after is InspectionState.ELIGIBLE_FOR_INFERENCE:
            run.step(
                ToolName.DECIDE_CAPTURE_ACTION,
                InspectionState.ELIGIBLE_FOR_INFERENCE,
                time.perf_counter(),
                evidence_summary={"quality_flags": list(perception.roi_flags),
                                  "after_remediation": False},
                evidence_ids=list(decision.triggering_evidence_ids),
                decision_reason=decision.explanation,
                selected_action=decision.selected_action.value,
                evidence_maturity=decision.evidence_maturity.value,
            )

        # --- 7. inference ------------------------------------------------------
        if run.state is InspectionState.ELIGIBLE_FOR_INFERENCE:
            _carry_advisories(run, perception)
            condition = run.infer()
            return finalise(run, perception, started, condition)

        return finalise(run, perception, started)

    except BudgetExceeded as error:
        # A budget breach is a safe stop, never a silent continuation.
        run.failure = f"BudgetExceeded: {error}"
        if not run.state.is_terminal:
            run.trace.add(
                state_before=run.state,
                tool_name=ToolName.FINALIZE_RESULT,
                state_after=InspectionState.FAILED_SAFE,
                decision_reason=(
                    f"Execution budget exceeded ({error}). The run stops here "
                    "rather than continuing outside its declared bounds."
                ),
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            run.state = InspectionState.FAILED_SAFE
        return finalise(run, perception, started)

    except (StateTransitionError, OrchestratorError) as error:
        run.failure = f"{type(error).__name__}: {error}"
        if not run.state.is_terminal:
            run.trace.add(
                state_before=run.state,
                tool_name=ToolName.FINALIZE_RESULT,
                state_after=InspectionState.FAILED_SAFE,
                decision_reason=f"Orchestration fault: {error}",
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            run.state = InspectionState.FAILED_SAFE
        return finalise(run, perception, started)

    except Exception as error:  # noqa: BLE001 - an unexpected fault must fail safe
        run.failure = f"{type(error).__name__}: {error}"
        if not run.state.is_terminal:
            run.trace.add(
                state_before=run.state,
                tool_name=ToolName.FINALIZE_RESULT,
                state_after=InspectionState.FAILED_SAFE,
                decision_reason=(
                    f"Unexpected tool fault ({type(error).__name__}); the run "
                    "fails safe rather than continuing on partial evidence."
                ),
                selected_action=RemediationAction.REQUEST_HUMAN_REVIEW.value,
            )
            run.state = InspectionState.FAILED_SAFE
        return finalise(run, perception, started)


def _decide_and_record(
    run: _Run, perception: _Perception, remediation_available: bool
) -> AgentDecision:
    """Call the pure decision function and move the machine to DECISION_MADE."""
    started = time.perf_counter()
    decision = decide_next_action(
        roi_flags=perception.roi_flags,
        roi_evidence=perception.roi_evidence,
        artifact_decision=perception.artifact_decision,
        foreground_valid=perception.foreground_valid,
        policy=run.policy,
        state_before=run.state,
        canonical_sha256=run.canonical_sha,
        remediation_available=remediation_available,
    )
    run.record(decision)
    run.step(
        ToolName.DECIDE_CAPTURE_ACTION,
        InspectionState.DECISION_MADE,
        started,
        evidence_summary={
            "quality_flags": list(perception.roi_flags),
            "artifact_flags": list(
                perception.artifact_decision.flags
                if perception.artifact_decision is not None else []
            ),
            "foreground_valid": perception.foreground_valid,
            "remediation_available": remediation_available,
        },
        evidence_ids=list(decision.triggering_evidence_ids),
        input_artifact_hashes={"canonical_image_sha256": run.canonical_sha},
        decision_reason=decision.explanation,
        selected_action=decision.selected_action.value,
        evidence_maturity=decision.evidence_maturity.value,
        policy_fingerprint=decision.policy_fingerprint,
    )
    return decision


def _route_insufficient_evidence(
    run: _Run, perception: _Perception, decision: AgentDecision, started: float
) -> InspectionRunResult:
    """Enter the waypoint, then route to the human action the policy names."""
    now = time.perf_counter()
    run.step(
        ToolName.DECIDE_CAPTURE_ACTION,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        now,
        evidence_summary={
            "foreground_valid": False,
            "invalid_reasons": list(
                perception.foreground.invalid_reasons
                if perception.foreground is not None else ["SEGMENTATION_FAILED"]
            ),
            "roi_detectors_run": False,
            "condition_model_invoked": False,
        },
        evidence_ids=["foreground.valid"],
        decision_reason=(
            "No subject region passed its validity guards, so no ROI detector "
            "ran and no artefact evidence exists. Recorded as a FOREGROUND / "
            "COVERAGE failure; nothing is claimed about occlusion."
        ),
        selected_action=decision.selected_action.value,
        evidence_maturity=decision.evidence_maturity.value,
    )
    request_human_action(run, decision, time.perf_counter())
    return finalise(run, perception, started)


def _remediation_excursion(
    run: _Run, perception: _Perception, decision: AgentDecision
) -> _Perception:
    """Apply one correction, then perceive the new image from scratch."""
    now = time.perf_counter()
    run.step(
        ToolName.DECIDE_CAPTURE_ACTION,
        InspectionState.REMEDIATION_SELECTED,
        now,
        evidence_summary={"action": decision.selected_action.value},
        evidence_ids=list(decision.triggering_evidence_ids),
        decision_reason=decision.explanation,
        selected_action=decision.selected_action.value,
        evidence_maturity=decision.evidence_maturity.value,
    )
    before_evidence = perception.roi_evidence
    if not run.remediate(decision, perception):
        return perception

    # Re-perception is mandatory. The mask is recomputed on the new pixels, and
    # so is every metric: the state machine has no path that reuses the old ones.
    after = run.perceive(
        InspectionState.RESEGMENTED,
        InspectionState.REASSESSED,
        InspectionState.REASSESSED,
    )
    if after.foreground_valid and after.roi_evidence is not None:
        comparison = run.reassess(before_evidence, after.roi_evidence)
        run.remediation_accepted = bool(comparison.accepted)
        if not comparison.accepted:
            # The enhancement ran and did not earn its place. Revert the
            # canonical image so nothing downstream reads pixels the harm guard
            # rejected, and escalate rather than trying something else.
            run.canonical_image = run.original_image
            run.canonical_sha = run.original_sha
            escalation = AgentDecision(
                state_before=run.state,
                selected_action=RemediationAction.REQUEST_RECAPTURE,
                reason_code=comparison.reason_code,
                state_after=InspectionState.REQUEST_RECAPTURE,
                triggering_evidence_ids=["roi.mean_luminance.underexposed"]
                if run.remediation_action == RemediationAction.APPLY_GAMMA.value
                else ["artifact.moderate_contrast"],
                evidence_maturity=EvidenceMaturity.CALIBRATED,
                policy_fingerprint=run.policy.remediation_policy.fingerprint(),
                automated=False,
                human_action_required=True,
                blocking=True,
                canonical_image_sha256=run.canonical_sha,
                explanation=comparison.explanation,
            )
            run.record(escalation)
            request_human_action(run, escalation, time.perf_counter())
            return after

    second = _decide_after_remediation(run, after)
    if second.state_after is InspectionState.INSUFFICIENT_VISUAL_EVIDENCE:
        run.step(
            ToolName.DECIDE_CAPTURE_ACTION,
            InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
            time.perf_counter(),
            evidence_summary={"foreground_valid": False, "after_remediation": True},
            evidence_ids=["foreground.valid"],
            decision_reason=(
                "The remediated image no longer segments. A correction that "
                "destroys the subject region is not an improvement."
            ),
            selected_action=second.selected_action.value,
            evidence_maturity=second.evidence_maturity.value,
        )
        request_human_action(run, second, time.perf_counter())
        return after

    if second.selected_action in HUMAN_ACTION_ROUTES:
        request_human_action(run, second, time.perf_counter())
        return after

    run.step(
        ToolName.DECIDE_CAPTURE_ACTION,
        InspectionState.ELIGIBLE_FOR_INFERENCE,
        time.perf_counter(),
        evidence_summary={"quality_flags": list(after.roi_flags),
                          "after_remediation": True},
        evidence_ids=list(second.triggering_evidence_ids),
        decision_reason=second.explanation,
        selected_action=second.selected_action.value,
        evidence_maturity=second.evidence_maturity.value,
    )
    return after


def _decide_after_remediation(run: _Run, perception: _Perception) -> AgentDecision:
    """Second and final decision. No further remediation is offered."""
    decision = decide_next_action(
        roi_flags=perception.roi_flags,
        roi_evidence=perception.roi_evidence,
        artifact_decision=perception.artifact_decision,
        foreground_valid=perception.foreground_valid,
        policy=run.policy,
        state_before=run.state,
        canonical_sha256=run.canonical_sha,
        remediation_available=False,
    )
    run.record(decision)
    return decision


def _carry_advisories(run: _Run, perception: _Perception | None) -> None:
    """Attach a non-blocking human suggestion, if the advisory evidence fired.

    This is the whole of glare's authority: it may put a sentence in front of a
    person. It does not stop the inference, and the result explicitly records
    that the condition finding was not blocked on it.
    """
    if perception is None or perception.artifact_decision is None:
        return
    if CaptureArtifactFlag.GLARE_RISK.value in perception.artifact_decision.flags:
        run.advisory_human_action = RemediationAction.REQUEST_REPOSITION_LIGHT.value
        run.trace.add(
            state_before=run.state,
            tool_name=ToolName.DECIDE_CAPTURE_ACTION,
            state_after=run.state,
            evidence_summary={
                "advisory_only": True,
                "blocks_inference": False,
                "largest_component_fraction": round(
                    perception.highlights.largest_component_fraction, 6
                ) if perception.highlights is not None else None,
            },
            evidence_ids=["artifact.glare"],
            decision_reason=(
                "Local highlight evidence is present. It is ADVISORY: the "
                "detector reached 0.500 on unseen groups against a 0.70 floor, "
                "so the capture is NOT blocked on it. The suggestion to "
                "reposition the light accompanies the result for a person to "
                "act on or ignore."
            ),
            selected_action=RemediationAction.REQUEST_REPOSITION_LIGHT.value,
            evidence_maturity=EvidenceMaturity.ADVISORY.value,
        )
        run.ledger.spend_step()
