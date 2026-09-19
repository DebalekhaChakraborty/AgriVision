"""Agent decision contract and evidence maturity.

Two detectors built in Phase 2d did not earn the right to stop an inspection.
Glare reached 0.655 on controlled injections and 0.500 on unseen groups against
a preregistered floor of 0.70; subject visibility reached 0.40 and never fired
at all. Both still produce evidence worth recording. The danger is not that they
exist — it is that six months from now a reader of a trace cannot tell which
numbers were calibrated and which were guesses, and treats them alike.

So maturity is carried in the type system rather than in a comment. Evidence
declares what it is, `may_gate` answers whether it is allowed to stop anything,
and a decision that blocks is *checked* against the maturity of the evidence
that triggered it. Promoting a detector becomes a visible edit to this file
accompanied by the measurement that justifies it, which is exactly how much
friction it should have.

Maturities are taken from the Phase 2c-B locked-policy record, not asserted:
the four thresholds that met the preregistered criteria are CALIBRATED, the
three that were retained at their Phase 1 values are PROVISIONAL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.state import InspectionState

DECISION_CONTRACT_VERSION = "phase3-decision-1.0.0"


class EvidenceMaturity(str, Enum):
    """How much authority a piece of evidence has earned.

    Ordered from most to least. Only the top two may gate.
    """

    # Calibrated on licensed real imagery against preregistered criteria, and
    # met them. Phase 2c-B: separation AUC >= 0.95, exposure sensitivity <= 1.5x,
    # scale sensitivity <= 3x, false accept and false block <= 0.10.
    CALIBRATED = "CALIBRATED"

    # Carried forward from Phase 1 at a whole-image value because calibration
    # was attempted and the criteria were not met. These still gate — refusing
    # to gate at all would be worse — but the trace says they are provisional.
    PROVISIONAL = "PROVISIONAL"

    # Measured, honest, and below the floor that permits blocking. May be
    # reported to a person and may accompany a decision. May not cause one.
    ADVISORY = "ADVISORY"

    # Built and measured and did not work. May appear in a trace as a record of
    # what was looked at. May not influence any decision whatsoever.
    UNQUALIFIED = "UNQUALIFIED"

    @property
    def may_gate(self) -> bool:
        """Whether evidence at this maturity may cause a blocking decision."""
        return self in (EvidenceMaturity.CALIBRATED, EvidenceMaturity.PROVISIONAL)

    @property
    def may_influence(self) -> bool:
        """Whether it may appear among the triggers of a decision at all."""
        return self is not EvidenceMaturity.UNQUALIFIED


# --- the registry -------------------------------------------------------------
#
# Evidence ids are `<scope>.<measurement>`. The scope matters: the same metric
# measured whole-image and ROI-restricted are different evidence with different
# calibration, which is the Phase 2b finding that started all of this.

EVIDENCE_MATURITY: dict[str, EvidenceMaturity] = {
    # Foreground validity guards. Not a threshold sweep but a structural check,
    # and Phase 2c-B evidenced the failure mode directly: the resolution-relative
    # cleanup fix moved mask validity from 42.9% to 85.7% on real photographs.
    "foreground.valid": EvidenceMaturity.CALIBRATED,
    "foreground.component_count": EvidenceMaturity.CALIBRATED,
    "foreground.foreground_fraction": EvidenceMaturity.CALIBRATED,

    # Met the Phase 2c-B criteria. high_frequency_ratio was selected over
    # Laplacian variance despite the latter's AUC of 1.000, because its median
    # moved 88-fold across the exposure ladder and no global threshold can
    # survive that.
    "roi.high_frequency_ratio": EvidenceMaturity.CALIBRATED,
    "roi.contrast_score": EvidenceMaturity.CALIBRATED,
    "roi.mean_luminance.underexposed": EvidenceMaturity.CALIBRATED,

    # Calibration attempted, criteria not met, Phase 1 value retained.
    # Overexposure reached AUC 0.943 against a 0.95 floor; both clipping limits
    # failed on AUC and on achievable false-block rate. Restricted to the
    # subject, highlight clipping is about 22x rarer than whole-image, which is
    # why there was too little signal to calibrate against.
    "roi.mean_luminance.overexposed": EvidenceMaturity.PROVISIONAL,
    "roi.shadow_clip_fraction": EvidenceMaturity.PROVISIONAL,
    "roi.highlight_clip_fraction": EvidenceMaturity.PROVISIONAL,

    # Phase 2d. Severe contrast is the tier that gates; it was calibrated on
    # calibration groups and had zero false positives on the independent
    # validation set.
    "artifact.severe_contrast": EvidenceMaturity.CALIBRATED,
    "artifact.moderate_contrast": EvidenceMaturity.CALIBRATED,

    # 0.655 controlled / 0.500 validation / 0.167 validation false positives,
    # against a 0.70 floor. Advisory.
    "artifact.glare": EvidenceMaturity.ADVISORY,

    # 0.40 detection and never fired once in controlled evaluation: every
    # refusal came from the foreground guards instead. Not a working detector.
    "artifact.visibility": EvidenceMaturity.UNQUALIFIED,

    # No confidence threshold has been calibrated. Phase 2c-B observed a median
    # of 0.416 across six classes on licensed real imagery, which is a
    # domain-shift signal and not a decision rule. Recorded, never acted on.
    "model.confidence": EvidenceMaturity.UNQUALIFIED,

    # Whole-image scope, Phase 1 provisional throughout. Used only when no valid
    # foreground exists, and then only to explain a refusal, never to accept.
    "whole_image.mean_luminance": EvidenceMaturity.PROVISIONAL,
    "whole_image.shadow_clip_fraction": EvidenceMaturity.PROVISIONAL,
    "whole_image.highlight_clip_fraction": EvidenceMaturity.PROVISIONAL,
    "whole_image.laplacian_variance": EvidenceMaturity.PROVISIONAL,

    # Structural facts about the input. Not measurements at all.
    "input.decodable": EvidenceMaturity.CALIBRATED,
    "input.dimensions": EvidenceMaturity.CALIBRATED,
}


class EvidenceMaturityError(RuntimeError):
    """Raised when evidence is used with more authority than it has earned."""


def maturity_of(evidence_id: str) -> EvidenceMaturity:
    """Maturity of one evidence id.

    Unknown ids are UNQUALIFIED rather than an error: a new measurement is
    powerless until someone registers it, which fails in the safe direction.
    """
    return EVIDENCE_MATURITY.get(evidence_id, EvidenceMaturity.UNQUALIFIED)


def weakest_maturity(evidence_ids: list[str]) -> EvidenceMaturity:
    order = [
        EvidenceMaturity.CALIBRATED,
        EvidenceMaturity.PROVISIONAL,
        EvidenceMaturity.ADVISORY,
        EvidenceMaturity.UNQUALIFIED,
    ]
    weakest = EvidenceMaturity.CALIBRATED
    for evidence_id in evidence_ids:
        maturity = maturity_of(evidence_id)
        if order.index(maturity) > order.index(weakest):
            weakest = maturity
    return weakest


def strongest_maturity(evidence_ids: list[str]) -> EvidenceMaturity:
    """The most authoritative evidence in the set.

    This is what a blocking decision is judged against: a block is legitimate
    when *something* trusted supports it, even if advisory evidence is recorded
    alongside. A run may note glare and still refuse for severe blur.
    """
    order = [
        EvidenceMaturity.UNQUALIFIED,
        EvidenceMaturity.ADVISORY,
        EvidenceMaturity.PROVISIONAL,
        EvidenceMaturity.CALIBRATED,
    ]
    strongest = EvidenceMaturity.UNQUALIFIED
    for evidence_id in evidence_ids:
        maturity = maturity_of(evidence_id)
        if order.index(maturity) > order.index(strongest):
            strongest = maturity
    return strongest


@dataclass(frozen=True)
class AgentDecision:
    """One decision: what state, what evidence, what action, what next state.

    Frozen and JSON-serialisable. The action is an enum member, never a string,
    so nothing in a stored decision can be replayed into executing an arbitrary
    tool.
    """

    state_before: InspectionState
    selected_action: RemediationAction
    reason_code: ReasonCode
    state_after: InspectionState
    triggering_evidence_ids: list[str] = field(default_factory=list)
    evidence_maturity: EvidenceMaturity = EvidenceMaturity.CALIBRATED
    policy_fingerprint: str = ""
    automated: bool = False
    human_action_required: bool = False
    blocking: bool = False
    canonical_image_sha256: str = ""
    explanation: str = ""
    # Set only when a policy explicitly enables gating on evidence below the
    # detection floor. It exists so that such a block is *visible* rather than
    # impossible: the guard below stops code from blocking on weak evidence by
    # accident, while an operator who has decided to accept a 0.500-detection
    # signal leaves a record of that decision in every trace it touches.
    # UNQUALIFIED evidence is never admitted this way.
    advisory_gating_permitted: bool = False
    contract_version: str = DECISION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        # A blocking decision must rest on evidence permitted to gate. This is
        # the check that stops an experimental detector becoming a hard blocker
        # by accident: turning glare into a blocker requires changing its
        # registered maturity, which requires the measurement to justify it.
        if self.blocking and self.triggering_evidence_ids:
            supporting = strongest_maturity(self.triggering_evidence_ids)
            permitted = supporting.may_gate or (
                self.advisory_gating_permitted
                and supporting is EvidenceMaturity.ADVISORY
            )
            if not permitted:
                raise EvidenceMaturityError(
                    f"{self.reason_code.value} blocks on {supporting.value} evidence "
                    f"({', '.join(self.triggering_evidence_ids)}); only CALIBRATED or "
                    "PROVISIONAL evidence may gate, unless a policy explicitly sets "
                    "advisory_gating_permitted"
                )
        unqualified = [
            evidence_id for evidence_id in self.triggering_evidence_ids
            if not maturity_of(evidence_id).may_influence
        ]
        if unqualified:
            raise EvidenceMaturityError(
                f"UNQUALIFIED evidence cannot trigger a decision: {', '.join(unqualified)}"
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state_before"] = self.state_before.value
        data["state_after"] = self.state_after.value
        data["selected_action"] = self.selected_action.value
        data["reason_code"] = self.reason_code.value
        data["evidence_maturity"] = self.evidence_maturity.value
        return data

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def maturity_summary() -> dict:
    """Serialisable maturity registry, for documentation and evaluation output."""
    by_maturity: dict[str, list[str]] = {}
    for evidence_id, maturity in sorted(EVIDENCE_MATURITY.items()):
        by_maturity.setdefault(maturity.value, []).append(evidence_id)
    return {
        "contract_version": DECISION_CONTRACT_VERSION,
        "may_gate": sorted(m.value for m in EvidenceMaturity if m.may_gate),
        "may_not_gate": sorted(m.value for m in EvidenceMaturity if not m.may_gate),
        "evidence": by_maturity,
    }
