"""Turn Phase 2d capture-artefact evidence into one bounded, named action.

Three artefacts, three different kinds of problem
-------------------------------------------------
They are handled separately because they are not the same phenomenon and do not
have the same remedy:

* **Glare** is a *local photometric* artefact. A specular highlight returns the
  illuminant rather than the surface, and the surface detail underneath it is
  not attenuated, it is gone. No tone curve recovers it. The only real fix is to
  move the light or the fruit, which is a request to a person.
* **Contrast loss** is a *global tonal* artefact. The tonal range is compressed
  but the ordering of values survives, so a moderate loss genuinely is
  recoverable by local histogram equalisation. A severe loss is not: once the
  range collapses toward a few levels, CLAHE amplifies quantisation and noise
  into something that looks like detail.
* **Visibility insufficiency** is a *coverage* problem. Nothing is wrong with
  the photometry; there is simply not enough subject in the frame to inspect.

Precedence, and why it is in this order
---------------------------------------
Visibility first, then glare, then contrast. If the subject is not fully in
frame, its brightness and tonal range are beside the point - reporting a glare
finding about a fruit that is half outside the picture would be precise about
the wrong thing. Contrast comes last because it is the only one of the three
that can sometimes be fixed automatically.

**Enhancement is never offered for glare.** That is enforced here rather than
left to a caller, because "brighten it and try again" on a blown highlight
produces an image that looks treated and carries no more information than
before, and the pipeline would then infer a condition from it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from competition.agent.actions import ReasonCode, RemediationAction

ARTIFACT_POLICY_VERSION = "phase2d-artifact-policy-1.0.0"


class CaptureArtifactFlag(str, Enum):
    """Closed set of Phase 2d artefact findings.

    Kept separate from `QualityFlag` so the Phase 1/2b/2c evidence vocabulary is
    unchanged and a stored record from an earlier phase stays readable.
    """

    GLARE_RISK = "GLARE_RISK"
    MODERATE_LOW_CONTRAST = "MODERATE_LOW_CONTRAST"
    SEVERE_LOW_CONTRAST = "SEVERE_LOW_CONTRAST"
    SUBJECT_VISIBILITY_INSUFFICIENT = "SUBJECT_VISIBILITY_INSUFFICIENT"
    ARTIFACT_EVIDENCE_UNAVAILABLE = "ARTIFACT_EVIDENCE_UNAVAILABLE"


# Flags that must stop condition inference. `MODERATE_LOW_CONTRAST` is not among
# them: it is the rung Phase 1b judged recoverable, and it routes to enhancement.
BLOCKING_ARTIFACT_FLAGS: frozenset[str] = frozenset({
    CaptureArtifactFlag.GLARE_RISK.value,
    CaptureArtifactFlag.SEVERE_LOW_CONTRAST.value,
    CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value,
    CaptureArtifactFlag.ARTIFACT_EVIDENCE_UNAVAILABLE.value,
})


@dataclass(frozen=True)
class ArtifactPolicy:
    """Thresholds for the artefact decision layer.

    The contrast limits are a two-tier ladder, which is the Phase 2c-B audit's
    conclusion made operational. That audit found the single calibrated limit
    raised `LOW_CONTRAST` on 91% of unusable contrast rungs and blocked none of
    them, because the flag is advisory. Advisory was the right call for *mild*
    loss and the wrong one for *collapse*; one threshold could not express both.
    """

    # Below this, contrast loss is worth acting on at all. This is the value
    # calibrated in Phase 2c-B on licensed real imagery.
    moderate_contrast_limit: float = 0.1314
    # Below this, the tonal range has collapsed and enhancement would fabricate
    # detail rather than reveal it. Calibrated in Phase 2d on calibration groups.
    severe_contrast_limit: float = 0.06

    # Whether a finding may *block*, as opposed to merely being recorded.
    #
    # Phase 2d preregistered a detection floor of 0.70 for enabling a detector.
    # Glare reached 0.655 and visibility 0.40 against honest ground truth, so
    # neither is permitted to gate. The floor is not lowered to let them
    # through: it governed blocking, and a detector below it still produces
    # evidence worth recording in a trace and reporting to a human. What it may
    # not do is send a photograph back on its own authority.
    #
    # Both default to False. Turning one on is a decision with measured
    # consequences, and it should look like one at the call site.
    glare_blocks: bool = False
    visibility_blocks: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_ARTIFACT_POLICY = ArtifactPolicy()


@dataclass(frozen=True)
class ArtifactDecision:
    """One action, one reason code, and the findings that produced them."""

    action: RemediationAction
    reason_code: ReasonCode
    flags: list[str]
    explanation: str
    blocking: bool
    policy_fingerprint: str
    policy_version: str = ARTIFACT_POLICY_VERSION
    evidence_fingerprints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "flags": list(self.flags),
            "explanation": self.explanation,
            "blocking": self.blocking,
            "policy_fingerprint": self.policy_fingerprint,
            "policy_version": self.policy_version,
            "evidence_fingerprints": dict(self.evidence_fingerprints),
        }


def decide_capture_artifacts(
    highlight_evidence=None,
    visibility_evidence=None,
    roi_contrast_score: float | None = None,
    policy: ArtifactPolicy | None = None,
    segmentation_valid: bool = True,
) -> ArtifactDecision:
    """Map artefact evidence to a single bounded action.

    Args:
        highlight_evidence: `LocalHighlightEvidence`, or None if not measured.
        visibility_evidence: `VisibilityEvidence`, or None if not measured.
        roi_contrast_score: foreground-restricted contrast, 0-1.
        policy: threshold policy; defaults to the provisional one.
        segmentation_valid: whether the foreground mask passed its guards.

    Returns:
        An `ArtifactDecision` naming exactly one action from the closed enum.
    """
    policy = policy or DEFAULT_ARTIFACT_POLICY
    fingerprints: dict = {}
    if highlight_evidence is not None:
        fingerprints["highlight_policy"] = highlight_evidence.policy_fingerprint
    if visibility_evidence is not None:
        fingerprints["visibility_policy"] = visibility_evidence.policy_fingerprint

    # Local artefact evidence is only meaningful inside a region we trust. An
    # invalid mask means the "subject" may be background, and a highlight
    # measured on a backdrop says nothing about the fruit.
    if not segmentation_valid or (highlight_evidence is None and visibility_evidence is None):
        return ArtifactDecision(
            action=RemediationAction.REQUEST_HUMAN_REVIEW,
            reason_code=ReasonCode.ARTIFACT_EVIDENCE_UNAVAILABLE,
            flags=[CaptureArtifactFlag.ARTIFACT_EVIDENCE_UNAVAILABLE.value],
            explanation=(
                "Foreground isolation did not produce a mask that passed its "
                "validity guards, so local artefact evidence cannot be attributed "
                "to the subject. Escalating rather than measuring the background."
            ),
            blocking=True,
            policy_fingerprint=policy.fingerprint(),
            evidence_fingerprints=fingerprints,
        )

    flags: list[str] = []
    if visibility_evidence is not None and not visibility_evidence.visibility_sufficient:
        flags.append(CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value)
    if highlight_evidence is not None and highlight_evidence.glare_flag:
        flags.append(CaptureArtifactFlag.GLARE_RISK.value)
    if roi_contrast_score is not None:
        if roi_contrast_score < policy.severe_contrast_limit:
            flags.append(CaptureArtifactFlag.SEVERE_LOW_CONTRAST.value)
        elif roi_contrast_score < policy.moderate_contrast_limit:
            flags.append(CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value)

    # --- precedence: coverage, then photometry, then tone ---------------------
    # A flag that is recorded but not permitted to block falls through to the
    # next candidate, so the finding still appears in `flags` and in the trace.
    if (CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value in flags
            and policy.visibility_blocks):
        reasons = ", ".join(visibility_evidence.insufficiency_reasons)
        return ArtifactDecision(
            action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=ReasonCode.SUBJECT_VISIBILITY_INSUFFICIENT,
            flags=flags,
            explanation=(
                f"The visible subject region is not shaped like a complete subject "
                f"({reasons}). This describes the region, not its cause: no claim "
                f"is made about what, if anything, is covering the produce."
            ),
            blocking=True,
            policy_fingerprint=policy.fingerprint(),
            evidence_fingerprints=fingerprints,
        )

    if CaptureArtifactFlag.GLARE_RISK.value in flags and policy.glare_blocks:
        return ArtifactDecision(
            action=RemediationAction.REQUEST_REPOSITION_LIGHT,
            reason_code=ReasonCode.GLARE_LOCAL_HIGHLIGHT,
            flags=flags,
            explanation=(
                f"A concentrated highlight covers "
                f"{highlight_evidence.largest_component_fraction:.1%} of the subject "
                f"region and sits {highlight_evidence.max_local_luminance_excess:.0f} "
                f"L* above its surroundings. Surface detail under a specular "
                f"highlight is not recoverable by any tone adjustment, so the "
                f"request is to change the lighting, not to enhance the image."
            ),
            blocking=True,
            policy_fingerprint=policy.fingerprint(),
            evidence_fingerprints=fingerprints,
        )

    if CaptureArtifactFlag.SEVERE_LOW_CONTRAST.value in flags:
        return ArtifactDecision(
            action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=ReasonCode.SEVERE_CONTRAST_LOSS,
            flags=flags,
            explanation=(
                f"Foreground contrast is {roi_contrast_score:.3f}, below the severe "
                f"limit of {policy.severe_contrast_limit}. Equalising a range this "
                f"compressed amplifies quantisation into the appearance of detail."
            ),
            blocking=True,
            policy_fingerprint=policy.fingerprint(),
            evidence_fingerprints=fingerprints,
        )

    if CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value in flags:
        return ArtifactDecision(
            action=RemediationAction.APPLY_CLAHE,
            reason_code=ReasonCode.MODERATE_CONTRAST_LOSS_RECOVERABLE,
            flags=flags,
            explanation=(
                f"Foreground contrast is {roi_contrast_score:.3f}, below the "
                f"advisory limit of {policy.moderate_contrast_limit} but above the "
                f"severe limit. Local equalisation is permitted, subject to the "
                f"existing harm guard re-checking the result."
            ),
            blocking=False,
            policy_fingerprint=policy.fingerprint(),
            evidence_fingerprints=fingerprints,
        )

    recorded_only = [
        flag for flag in flags
        if flag != CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value
    ]
    return ArtifactDecision(
        action=RemediationAction.NONE,
        reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
        flags=flags,
        explanation=(
            "No blocking capture artefact was found."
            + (
                f" Recorded but not gating, because the detector did not meet the "
                f"preregistered detection floor: {', '.join(recorded_only)}."
                if recorded_only else ""
            )
        ),
        blocking=False,
        policy_fingerprint=policy.fingerprint(),
        evidence_fingerprints=fingerprints,
    )
