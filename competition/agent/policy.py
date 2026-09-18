"""Deterministic remediation policy.

Two pure functions over evidence, with no image processing of their own:

* `decide_capture_remediation` — what, if anything, to do about a capture.
* `compare_capture_quality` — whether a completed remediation earned its place.

No language model is involved in either. Control decisions are threshold logic
over measured metrics so they are testable, reproducible and auditable; a model
may later *narrate* a decision, but must never be able to change one.

STATUS: every threshold here is PROVISIONAL / DEVELOPMENT ONLY, chosen
conservatively from the Phase 1 synthetic measurements. None is calibrated on
real produce photography.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from competition.agent.actions import (
    ComparisonResult,
    ComparisonVerdict,
    MetricDelta,
    ReasonCode,
    RemediationAction,
    RemediationDecision,
)
from competition.vision.enhancement import (
    ClaheParameters,
    GammaParameters,
    gamma_for_target_luminance,
)
from competition.vision.evidence import PerceptionEvidence, QualityFlag

POLICY_VERSION = "phase1b-remediation-policy-1.0.0"
POLICY_STATUS = "PROVISIONAL_DEVELOPMENT_ONLY"


@dataclass(frozen=True)
class RemediationPolicy:
    """Thresholds governing whether and how to remediate a capture."""

    # --- target exposure band -------------------------------------------------
    target_luminance: float = 0.45
    target_luminance_min: float = 0.25
    target_luminance_max: float = 0.80

    # --- unrecoverable-information gates -------------------------------------
    # Above these, detail is genuinely lost. Enhancement would redistribute what
    # remains and produce a plausible image that has recovered nothing, so the
    # policy refuses and asks for a new capture instead.
    severe_shadow_clip_fraction: float = 0.10
    severe_highlight_clip_fraction: float = 0.10

    # --- gamma bounds ---------------------------------------------------------
    gamma_min: float = 0.25
    gamma_max: float = 3.0

    # --- CLAHE parameters -----------------------------------------------------
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    # --- acceptance guardrails ------------------------------------------------
    # Minimum improvement in the targeted metric for remediation to be worth it
    # WHEN the target band is not reached. Reaching the band is sufficient on its
    # own: see `compare_capture_quality`. Requiring this margin unconditionally
    # was a defect — a capture starting 0.013 from the band could be corrected
    # perfectly and still be rejected, because the margin exceeded the maximum
    # improvement available.
    min_luminance_distance_improvement: float = 0.02
    min_contrast_improvement: float = 0.02
    # Contrast at or above this counts as an acceptable outcome for CLAHE.
    # Should track ThresholdPolicy.low_contrast_limit; kept here so the
    # remediation policy stays self-contained and independently fingerprinted.
    target_contrast_floor: float = 0.20
    # Numeric slack for band membership. Exists so band tests never rely on
    # exact floating-point equality.
    band_tolerance: float = 1e-6
    # A remediation may not increase clipping by more than this.
    max_clipping_increase: float = 0.02
    # Nor reduce contrast by more than this.
    max_contrast_loss: float = 0.02

    @property
    def version(self) -> str:
        return POLICY_VERSION

    @property
    def status(self) -> str:
        return POLICY_STATUS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["policy_version"] = POLICY_VERSION
        data["status"] = POLICY_STATUS
        return data

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_REMEDIATION_POLICY = RemediationPolicy()


def is_within_target_band(
    value: float, lower: float, upper: float, tolerance: float = 1e-6
) -> bool:
    """Whether a value lies inside [lower, upper], within a numeric tolerance.

    A tolerance-based membership test rather than a comparison against an exact
    distance of 0.0: floating-point equality is not a sound way to express
    "landed in the band", and the intent here is a policy statement, not an
    arithmetic coincidence.
    """
    return (lower - tolerance) <= value <= (upper + tolerance)


def luminance_distance(value: float, policy: RemediationPolicy) -> float:
    """Distance of a luminance value from the acceptable band. 0 means inside."""
    if value < policy.target_luminance_min:
        return policy.target_luminance_min - value
    if value > policy.target_luminance_max:
        return value - policy.target_luminance_max
    return 0.0


def _decision(
    action: RemediationAction,
    reason: ReasonCode,
    evidence: PerceptionEvidence,
    policy: RemediationPolicy,
    thresholds: dict,
    metrics: dict,
    parameters: dict | None = None,
    explanation: str = "",
) -> RemediationDecision:
    return RemediationDecision(
        action=action,
        reason_code=reason,
        triggering_flags=list(evidence.quality_flags),
        triggering_metrics=metrics,
        thresholds=thresholds,
        automation_permitted=action.is_automated,
        human_intervention_required=action.requires_human,
        parameters=parameters or {},
        policy_version=policy.version,
        policy_fingerprint=policy.fingerprint(),
        explanation=explanation,
    )


def decide_capture_remediation(
    evidence: PerceptionEvidence, policy: RemediationPolicy | None = None
) -> RemediationDecision:
    """Choose a remediation action from capture-quality evidence alone.

    Deterministic: identical evidence and policy always yield an identical
    decision. Rules are evaluated in order of severity, so an unrecoverable
    condition is never masked by a recoverable one.
    """
    policy = policy or DEFAULT_REMEDIATION_POLICY
    illumination = evidence.illumination
    metrics = {
        "mean_luminance": illumination.mean_luminance,
        "contrast_score": illumination.contrast_score,
        "shadow_clip_fraction": illumination.shadow_clip_fraction,
        "highlight_clip_fraction": illumination.highlight_clip_fraction,
        "laplacian_variance": evidence.sharpness.laplacian_variance,
    }

    # 1. Resolution cannot be added to an image.
    if evidence.has_flag(QualityFlag.IMAGE_TOO_SMALL):
        return _decision(
            RemediationAction.REQUEST_RECAPTURE,
            ReasonCode.IMAGE_TOO_SMALL,
            evidence,
            policy,
            {},
            metrics,
            explanation="Image resolution is below the minimum; no enhancement adds detail.",
        )

    # 2-3. Clipped information is gone. Refuse rather than fabricate.
    if illumination.shadow_clip_fraction > policy.severe_shadow_clip_fraction:
        return _decision(
            RemediationAction.REQUEST_RECAPTURE,
            ReasonCode.SHADOW_CLIPPING_UNRECOVERABLE,
            evidence,
            policy,
            {"severe_shadow_clip_fraction": policy.severe_shadow_clip_fraction},
            metrics,
            explanation=(
                "Shadow detail is clipped beyond recovery. Brightening would "
                "redistribute remaining values without restoring lost detail."
            ),
        )

    if illumination.highlight_clip_fraction > policy.severe_highlight_clip_fraction:
        return _decision(
            RemediationAction.REQUEST_RECAPTURE,
            ReasonCode.HIGHLIGHT_CLIPPING_UNRECOVERABLE,
            evidence,
            policy,
            {"severe_highlight_clip_fraction": policy.severe_highlight_clip_fraction},
            metrics,
            explanation=(
                "Highlight detail is clipped beyond recovery. Darkening would "
                "not restore blown regions."
            ),
        )

    # 4-5. Exposure outside the band, with clipping still recoverable.
    if evidence.has_flag(QualityFlag.UNDEREXPOSED) or evidence.has_flag(
        QualityFlag.OVEREXPOSED
    ):
        underexposed = evidence.has_flag(QualityFlag.UNDEREXPOSED)
        gamma = gamma_for_target_luminance(
            illumination.mean_luminance,
            policy.target_luminance,
            policy.gamma_min,
            policy.gamma_max,
        )
        return _decision(
            RemediationAction.APPLY_GAMMA,
            ReasonCode.UNDEREXPOSED_RECOVERABLE
            if underexposed
            else ReasonCode.OVEREXPOSED_RECOVERABLE,
            evidence,
            policy,
            {
                "target_luminance": policy.target_luminance,
                "target_luminance_min": policy.target_luminance_min,
                "target_luminance_max": policy.target_luminance_max,
            },
            metrics,
            parameters=GammaParameters(gamma=gamma).to_dict(),
            explanation=(
                f"Mean luminance {illumination.mean_luminance:.3f} is outside the "
                f"band [{policy.target_luminance_min}, {policy.target_luminance_max}]; "
                f"applying gamma {gamma:.3f} toward {policy.target_luminance}."
            ),
        )

    # 6. Exposure acceptable but tonal range compressed.
    if evidence.has_flag(QualityFlag.LOW_CONTRAST):
        return _decision(
            RemediationAction.APPLY_CLAHE,
            ReasonCode.LOW_CONTRAST_RECOVERABLE,
            evidence,
            policy,
            {"clahe_clip_limit": policy.clahe_clip_limit},
            metrics,
            parameters=ClaheParameters(
                clip_limit=policy.clahe_clip_limit,
                tile_grid_size=policy.clahe_tile_grid_size,
            ).to_dict(),
            explanation=(
                f"Contrast {illumination.contrast_score:.3f} is low while exposure "
                "is acceptable; CLAHE on L* may expand the usable tonal range."
            ),
        )

    # 7. Blur with healthy illumination is genuine blur, and no tone operation
    #    fixes it. This rule exists because of the Phase 1 finding: only once
    #    exposure is known good can a blur verdict be trusted.
    if evidence.has_flag(QualityFlag.BLUR_RISK):
        return _decision(
            RemediationAction.REQUEST_RECAPTURE,
            ReasonCode.BLUR_NOT_REMEDIABLE,
            evidence,
            policy,
            {},
            metrics,
            explanation=(
                "Illumination is within band, so low sharpness is not an exposure "
                "artefact. Blur cannot be removed by tone adjustment."
            ),
        )

    # 8. Nothing to do.
    return _decision(
        RemediationAction.NONE,
        ReasonCode.CAPTURE_ACCEPTABLE,
        evidence,
        policy,
        {},
        metrics,
        explanation="No capture-quality flag requires remediation.",
    )


def _delta(metric: str, before: float, after: float, improved: bool) -> MetricDelta:
    return MetricDelta(
        metric=metric,
        before=float(before),
        after=float(after),
        delta=float(after - before),
        improved=improved,
    )


def compare_capture_quality(
    before: PerceptionEvidence,
    after: PerceptionEvidence,
    decision: RemediationDecision,
    policy: RemediationPolicy | None = None,
) -> ComparisonResult:
    """Decide whether a completed remediation should be accepted.

    Multi-metric by construction. The targeted problem must improve *and* no
    guardrail metric may materially worsen. Sharpness is recorded for
    interpretation but never decides the verdict: Phase 1 established that
    sharpness moves with exposure, so using it as the success criterion would
    make any brightening look like a win.
    """
    policy = policy or DEFAULT_REMEDIATION_POLICY

    before_illumination = before.illumination
    after_illumination = after.illumination

    before_distance = luminance_distance(before_illumination.mean_luminance, policy)
    after_distance = luminance_distance(after_illumination.mean_luminance, policy)

    # Target success has two independent routes: the remediation reached the
    # acceptable band, or it moved far enough toward it. Reaching the band is
    # sufficient on its own — a capture starting only 0.013 outside it cannot
    # improve by a 0.02 margin no matter how well the correction works, so
    # requiring the margin unconditionally rejected perfect corrections.
    if decision.action is RemediationAction.APPLY_GAMMA:
        target_metric = "luminance_distance"
        reached_band = is_within_target_band(
            after_illumination.mean_luminance,
            policy.target_luminance_min,
            policy.target_luminance_max,
            policy.band_tolerance,
        )
        improved_enough = (
            before_distance - after_distance
        ) >= policy.min_luminance_distance_improvement
        target_improved = reached_band or improved_enough
    elif decision.action is RemediationAction.APPLY_CLAHE:
        target_metric = "contrast_score"
        reached_band = (
            after_illumination.contrast_score
            >= policy.target_contrast_floor - policy.band_tolerance
        )
        improved_enough = (
            after_illumination.contrast_score - before_illumination.contrast_score
        ) >= policy.min_contrast_improvement
        target_improved = reached_band or improved_enough
    else:  # pragma: no cover - comparison is only invoked for automated actions
        target_metric = "none"
        reached_band = False
        improved_enough = False
        target_improved = False

    deltas = {
        "luminance_distance": _delta(
            "luminance_distance",
            before_distance,
            after_distance,
            after_distance < before_distance,
        ),
        "mean_luminance": _delta(
            "mean_luminance",
            before_illumination.mean_luminance,
            after_illumination.mean_luminance,
            after_distance <= before_distance,
        ),
        "contrast_score": _delta(
            "contrast_score",
            before_illumination.contrast_score,
            after_illumination.contrast_score,
            after_illumination.contrast_score >= before_illumination.contrast_score,
        ),
        "shadow_clip_fraction": _delta(
            "shadow_clip_fraction",
            before_illumination.shadow_clip_fraction,
            after_illumination.shadow_clip_fraction,
            after_illumination.shadow_clip_fraction
            <= before_illumination.shadow_clip_fraction,
        ),
        "highlight_clip_fraction": _delta(
            "highlight_clip_fraction",
            before_illumination.highlight_clip_fraction,
            after_illumination.highlight_clip_fraction,
            after_illumination.highlight_clip_fraction
            <= before_illumination.highlight_clip_fraction,
        ),
        # Recorded for interpretation only. Deliberately excluded from both the
        # target test and the guardrails.
        "laplacian_variance": _delta(
            "laplacian_variance",
            before.sharpness.laplacian_variance,
            after.sharpness.laplacian_variance,
            after.sharpness.laplacian_variance >= before.sharpness.laplacian_variance,
        ),
    }

    # Harm guard: independent critical metrics must not materially worsen.
    violations: list[str] = []
    if (
        after_illumination.shadow_clip_fraction
        - before_illumination.shadow_clip_fraction
        > policy.max_clipping_increase
    ):
        violations.append("shadow_clip_fraction increased beyond guardrail")
    if (
        after_illumination.highlight_clip_fraction
        - before_illumination.highlight_clip_fraction
        > policy.max_clipping_increase
    ):
        violations.append("highlight_clip_fraction increased beyond guardrail")
    if (
        before_illumination.contrast_score - after_illumination.contrast_score
        > policy.max_contrast_loss
    ):
        violations.append("contrast_score decreased beyond guardrail")

    if violations:
        return ComparisonResult(
            verdict=ComparisonVerdict.REJECT_REMEDIATION,
            reason_code=ReasonCode.REMEDIATION_HARMFUL,
            target_metric=target_metric,
            target_improved=target_improved,
            reached_target_band=reached_band,
            improved_by_margin=improved_enough,
            guardrail_violations=violations,
            deltas=deltas,
            explanation=(
                "Remediation harmed an independent critical metric; the original "
                "capture is retained as canonical."
            ),
        )

    if not target_improved:
        return ComparisonResult(
            verdict=ComparisonVerdict.REJECT_REMEDIATION,
            reason_code=ReasonCode.REMEDIATION_INEFFECTIVE,
            target_metric=target_metric,
            target_improved=False,
            reached_target_band=reached_band,
            improved_by_margin=improved_enough,
            guardrail_violations=[],
            deltas=deltas,
            explanation=(
                f"Targeted metric {target_metric} did not improve by the required "
                "margin; the original capture is retained as canonical."
            ),
        )

    return ComparisonResult(
        verdict=ComparisonVerdict.ACCEPT_REMEDIATION,
        reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
        target_metric=target_metric,
        target_improved=True,
        reached_target_band=reached_band,
        improved_by_margin=improved_enough,
        guardrail_violations=[],
        deltas=deltas,
        explanation=(
            f"Targeted metric {target_metric} met its criterion ("
            + ("reached the target band" if reached_band else "improved by the required margin")
            + ") with no guardrail violation."
        ),
    )
