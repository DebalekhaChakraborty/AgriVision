"""Remediation policy: decision determinism, safety gates, harm guard."""

from __future__ import annotations

import pytest

from competition.agent.actions import (
    ComparisonVerdict,
    ReasonCode,
    RemediationAction,
)
from competition.agent.policy import (
    DEFAULT_REMEDIATION_POLICY,
    RemediationPolicy,
    compare_capture_quality,
    decide_capture_remediation,
    is_within_target_band,
    luminance_distance,
)
from competition.vision.enhancement import GammaParameters, apply_gamma_correction
from competition.vision.degradation import adjust_exposure, gaussian_blur, reduce_contrast
from competition.vision.fixtures import textured_object, uniform_mid
from competition.vision.quality import assess_capture_quality


def evidence_for(image):
    return assess_capture_quality(image)


# --- required test 5: decision is deterministic -------------------------------


@pytest.mark.parametrize("gain", [1.0, 0.4, 0.15, 0.05, 2.0, 3.5])
def test_decision_is_deterministic_for_identical_evidence(gain):
    evidence = evidence_for(adjust_exposure(textured_object(), gain))
    first = decide_capture_remediation(evidence)
    second = decide_capture_remediation(evidence)
    assert first.to_dict() == second.to_dict()


def test_decision_records_policy_provenance():
    decision = decide_capture_remediation(evidence_for(textured_object()))
    assert decision.policy_version == DEFAULT_REMEDIATION_POLICY.version
    assert len(decision.policy_fingerprint) == 16


def test_policy_fingerprint_changes_with_thresholds():
    a = RemediationPolicy(target_luminance=0.45)
    b = RemediationPolicy(target_luminance=0.50)
    assert a.fingerprint() != b.fingerprint()


# --- required test 15: healthy capture yields NONE ----------------------------


def test_well_exposed_fixture_requires_no_remediation():
    decision = decide_capture_remediation(evidence_for(textured_object()))
    assert decision.action is RemediationAction.NONE
    assert decision.reason_code is ReasonCode.CAPTURE_ACCEPTABLE
    assert not decision.human_intervention_required
    assert not decision.automation_permitted


# --- required test 14: underexposure triggers remediation ---------------------


@pytest.mark.parametrize("gain", [0.4, 0.25, 0.15])
def test_underexposure_triggers_gamma_remediation(gain):
    decision = decide_capture_remediation(
        evidence_for(adjust_exposure(textured_object(), gain))
    )
    assert decision.action is RemediationAction.APPLY_GAMMA
    assert decision.reason_code is ReasonCode.UNDEREXPOSED_RECOVERABLE
    assert decision.automation_permitted
    assert 0 < decision.parameters["gamma"] < 1  # brightening


def test_overexposure_triggers_gamma_darkening():
    decision = decide_capture_remediation(
        evidence_for(adjust_exposure(textured_object(), 2.0))
    )
    assert decision.action is RemediationAction.APPLY_GAMMA
    assert decision.reason_code is ReasonCode.OVEREXPOSED_RECOVERABLE
    assert decision.parameters["gamma"] > 1  # darkening


def test_low_contrast_with_acceptable_exposure_triggers_clahe():
    decision = decide_capture_remediation(
        evidence_for(reduce_contrast(textured_object(), 0.4))
    )
    assert decision.action is RemediationAction.APPLY_CLAHE
    assert decision.reason_code is ReasonCode.LOW_CONTRAST_RECOVERABLE
    assert decision.parameters["clip_limit"] > 0


# --- required test 6: severe clipping escalates, never fakes recovery ---------


def test_severe_shadow_clipping_requests_recapture():
    decision = decide_capture_remediation(
        evidence_for(adjust_exposure(textured_object(), 0.05))
    )
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.SHADOW_CLIPPING_UNRECOVERABLE
    assert decision.human_intervention_required
    assert not decision.automation_permitted


def test_severe_highlight_clipping_requests_recapture():
    decision = decide_capture_remediation(
        evidence_for(adjust_exposure(textured_object(), 3.5))
    )
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.HIGHLIGHT_CLIPPING_UNRECOVERABLE
    assert not decision.automation_permitted


def test_clipping_gate_takes_precedence_over_exposure_remediation():
    """An unrecoverable condition must not be masked by a recoverable one."""
    crushed = evidence_for(adjust_exposure(textured_object(), 0.05))
    assert "UNDEREXPOSED" in crushed.quality_flags  # both conditions present
    decision = decide_capture_remediation(crushed)
    assert decision.action is RemediationAction.REQUEST_RECAPTURE


def test_blur_with_healthy_illumination_is_not_remediable():
    """The Phase 1 finding in policy form: only trust a blur verdict when
    illumination is in band, and then refuse to 'enhance' it away."""
    decision = decide_capture_remediation(evidence_for(gaussian_blur(textured_object(), 5.0)))
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.BLUR_NOT_REMEDIABLE


def test_tiny_image_requests_recapture():
    import numpy as np

    decision = decide_capture_remediation(
        assess_capture_quality(np.full((8, 8, 3), 128, dtype=np.uint8))
    )
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.IMAGE_TOO_SMALL


# --- luminance distance -------------------------------------------------------


def test_luminance_distance_is_zero_inside_the_band():
    policy = DEFAULT_REMEDIATION_POLICY
    assert luminance_distance(0.45, policy) == 0.0
    assert luminance_distance(policy.target_luminance_min, policy) == 0.0
    assert luminance_distance(policy.target_luminance_max, policy) == 0.0


def test_luminance_distance_grows_outside_the_band():
    policy = DEFAULT_REMEDIATION_POLICY
    assert luminance_distance(0.05, policy) > luminance_distance(0.20, policy)
    assert luminance_distance(0.99, policy) > luminance_distance(0.85, policy)


# --- required test 16: harmful enhancement is rejected ------------------------


def test_comparison_rejects_enhancement_that_worsens_clipping():
    """Harm guard: brightening an already-bright capture blows highlights."""
    bright = adjust_exposure(textured_object(), 1.6)
    before = evidence_for(bright)

    # Force an inappropriate brightening rather than the policy's choice.
    from competition.vision.enhancement import GammaParameters, apply_gamma_correction

    harmed, _ = apply_gamma_correction(bright, GammaParameters(gamma=0.25))
    after = evidence_for(harmed)

    decision = decide_capture_remediation(before)
    forced = type(decision)(
        action=RemediationAction.APPLY_GAMMA,
        reason_code=ReasonCode.UNDEREXPOSED_RECOVERABLE,
        triggering_flags=list(before.quality_flags),
        triggering_metrics={},
        thresholds={},
        automation_permitted=True,
        human_intervention_required=False,
        parameters={"gamma": 0.25},
    )

    result = compare_capture_quality(before, after, forced)
    assert result.verdict is ComparisonVerdict.REJECT_REMEDIATION
    assert result.reason_code is ReasonCode.REMEDIATION_HARMFUL
    assert result.guardrail_violations
    assert not result.accepted


def test_comparison_rejects_ineffective_enhancement():
    """Still outside the band and no meaningful movement toward it.

    The image must start outside the target band: a capture already inside it
    satisfies the target by the band route regardless of movement, which is the
    intended behaviour after the acceptance fix.
    """
    image = adjust_exposure(textured_object(), 0.15)  # well below the band
    before = evidence_for(image)
    after = evidence_for(image)  # identical: no movement at all

    decision = type(decide_capture_remediation(before))(
        action=RemediationAction.APPLY_GAMMA,
        reason_code=ReasonCode.UNDEREXPOSED_RECOVERABLE,
        triggering_flags=[],
        triggering_metrics={},
        thresholds={},
        automation_permitted=True,
        human_intervention_required=False,
        parameters={"gamma": 1.0},
    )

    result = compare_capture_quality(before, after, decision)
    assert result.verdict is ComparisonVerdict.REJECT_REMEDIATION
    assert result.reason_code is ReasonCode.REMEDIATION_INEFFECTIVE


def test_comparison_never_decides_on_sharpness_alone():
    """Sharpness is recorded but must not drive the verdict.

    Phase 1 established that sharpness rises with brightness, so if it decided
    acceptance, every brightening would look like a success.
    """
    dark = adjust_exposure(textured_object(), 0.15)
    before = evidence_for(dark)
    decision = decide_capture_remediation(before)

    from competition.agent.remediation import enhance_capture

    enhanced, _ = enhance_capture(dark, decision)
    after = evidence_for(enhanced)

    result = compare_capture_quality(before, after, decision)
    assert "laplacian_variance" in result.deltas
    assert result.target_metric == "luminance_distance"
    assert result.target_metric != "laplacian_variance"


# --- acceptance-fix regression tests -----------------------------------------
#
# A capture starting only marginally outside the target band cannot improve by
# the nominal margin no matter how well it is corrected. Requiring the margin
# unconditionally rejected perfect corrections; these pin the corrected
# semantics: reaching the band is sufficient on its own.


def test_is_within_target_band_uses_tolerance_not_equality():
    assert is_within_target_band(0.25, 0.25, 0.80, 1e-6)
    assert is_within_target_band(0.80, 0.25, 0.80, 1e-6)
    # Just outside by less than the tolerance still counts as inside.
    assert is_within_target_band(0.25 - 5e-7, 0.25, 0.80, 1e-6)
    assert not is_within_target_band(0.24, 0.25, 0.80, 1e-6)
    assert not is_within_target_band(0.81, 0.25, 0.80, 1e-6)


def test_marginally_underexposed_capture_is_just_outside_the_band():
    """Premise check for the regression below: gain 0.5 sits barely too dark."""
    policy = DEFAULT_REMEDIATION_POLICY
    before = evidence_for(adjust_exposure(textured_object(), 0.5))
    distance = luminance_distance(before.illumination.mean_luminance, policy)
    assert 0.0 < distance < policy.min_luminance_distance_improvement


def test_remediation_reaching_the_band_is_accepted_despite_small_improvement():
    """The regression this fix exists for.

    Improvement achievable (~0.013) is smaller than the nominal margin (0.02),
    yet the correction lands inside the band and must be accepted.
    """
    from competition.agent.remediation import run_capture_remediation

    policy = DEFAULT_REMEDIATION_POLICY
    marginal = adjust_exposure(textured_object(), 0.5)
    result = run_capture_remediation(marginal, remediation_policy=policy)

    assert result.remediation_attempted
    assert result.comparison is not None
    before_distance = luminance_distance(
        result.original_evidence.illumination.mean_luminance, policy
    )
    after_distance = luminance_distance(
        result.remediated_evidence.illumination.mean_luminance, policy
    )
    assert before_distance - after_distance < policy.min_luminance_distance_improvement
    assert result.comparison.reached_target_band
    assert not result.comparison.improved_by_margin
    assert result.comparison.verdict is ComparisonVerdict.ACCEPT_REMEDIATION
    assert result.remediation_accepted


def test_small_improvement_still_outside_band_is_rejected():
    """Moving a little but staying outside, by less than the margin, fails."""
    policy = DEFAULT_REMEDIATION_POLICY
    very_dark = adjust_exposure(textured_object(), 0.15)
    before = evidence_for(very_dark)
    # A gamma far too weak to reach the band.
    nudged, _ = apply_gamma_correction(very_dark, GammaParameters(gamma=0.97))
    after = evidence_for(nudged)

    assert luminance_distance(after.illumination.mean_luminance, policy) > 0.0

    decision = decide_capture_remediation(before, policy)
    result = compare_capture_quality(before, after, decision, policy)

    assert not result.reached_target_band
    assert not result.improved_by_margin
    assert result.verdict is ComparisonVerdict.REJECT_REMEDIATION
    assert result.reason_code is ReasonCode.REMEDIATION_INEFFECTIVE


def test_reaching_the_band_but_violating_a_harm_guard_is_rejected():
    """The band route never overrides the harm guard."""
    policy = RemediationPolicy(max_clipping_increase=0.0, max_contrast_loss=0.0)
    dark = adjust_exposure(textured_object(), 0.15)
    before = evidence_for(dark)
    decision = decide_capture_remediation(before, policy)

    # Drive it far past the band so highlights clip badly.
    blown, _ = apply_gamma_correction(dark, GammaParameters(gamma=0.05))
    after = evidence_for(blown)

    result = compare_capture_quality(before, after, decision, policy)
    assert result.verdict is ComparisonVerdict.REJECT_REMEDIATION
    assert result.reason_code is ReasonCode.REMEDIATION_HARMFUL
    assert result.guardrail_violations


def test_clahe_acceptance_also_has_a_band_route():
    """Contrast reaching the acceptable floor is sufficient for CLAHE."""
    policy = DEFAULT_REMEDIATION_POLICY
    low = reduce_contrast(textured_object(), 0.4)
    before = evidence_for(low)
    decision = decide_capture_remediation(before, policy)
    assert decision.action is RemediationAction.APPLY_CLAHE

    from competition.agent.remediation import enhance_capture

    enhanced, _ = enhance_capture(low, decision)
    after = evidence_for(enhanced)
    result = compare_capture_quality(before, after, decision, policy)

    if after.illumination.contrast_score >= policy.target_contrast_floor:
        assert result.reached_target_band
        assert result.verdict is ComparisonVerdict.ACCEPT_REMEDIATION
