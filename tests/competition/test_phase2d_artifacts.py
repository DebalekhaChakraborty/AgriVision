"""Phase 2d: local highlight, visibility and contrast evidence, and their actions."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.artifact_policy import (
    BLOCKING_ARTIFACT_FLAGS,
    ArtifactPolicy,
    CaptureArtifactFlag,
    decide_capture_artifacts,
)
from competition.agent.artifacts import analyse_capture_artifacts
from competition.vision.degradation import add_glare_at, occlude_at
from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import segment_without_fill
from competition.vision.highlights import (
    DEFAULT_HIGHLIGHT_POLICY,
    LocalHighlightPolicy,
    highlight_mask,
    measure_local_highlights,
)
from competition.vision.visibility import (
    DEFAULT_VISIBILITY_POLICY,
    VisibilityPolicy,
    measure_visibility,
)


def textured_subject(size: int = 400, radius: int = 150, colour=(60, 160, 220)):
    """A lit, textured, coloured disc on a neutral ground - a fruit stand-in."""
    rng = np.random.default_rng(7)
    image = np.full((size, size, 3), 140, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    body = np.zeros_like(image)
    cv2.circle(body, (size // 2, size // 2), radius, colour, -1)
    speckle = rng.integers(-25, 25, image.shape, dtype=np.int16)
    body = np.clip(body.astype(np.int16) + speckle, 0, 255).astype(np.uint8)
    image[mask > 0] = body[mask > 0]
    return image, mask


# --- 1-3. evidence contract ---------------------------------------------------


def test_local_highlight_evidence_serialises():
    image, mask = textured_subject()
    payload = json.loads(json.dumps(measure_local_highlights(image, mask).to_dict()))
    assert payload["pipeline_version"].startswith("phase2d")
    assert isinstance(payload["glare_flag"], bool)


def test_visibility_evidence_serialises():
    _, mask = textured_subject()
    payload = json.loads(json.dumps(measure_visibility(mask).to_dict()))
    assert isinstance(payload["visibility_sufficient"], bool)


@pytest.mark.parametrize("builder", ["highlights", "visibility"])
def test_evidence_carries_no_filesystem_path(builder):
    image, mask = textured_subject()
    evidence = (
        measure_local_highlights(image, mask) if builder == "highlights"
        else measure_visibility(mask)
    )
    payload = json.dumps(evidence.to_dict())
    assert "/home/" not in payload and "/Users/" not in payload
    assert not any("path" in field for field in evidence.to_dict())


def test_evidence_is_deterministic_excluding_timing():
    image, mask = textured_subject()
    first = measure_local_highlights(image, mask).deterministic_payload()
    second = measure_local_highlights(image, mask).deterministic_payload()
    assert first == second
    assert "processing_ms" not in first


def test_visibility_evidence_is_deterministic():
    _, mask = textured_subject()
    assert (measure_visibility(mask).deterministic_payload()
            == measure_visibility(mask).deterministic_payload())


# --- 4-7. glare ---------------------------------------------------------------


def test_injected_glare_increases_highlight_evidence():
    image, mask = textured_subject()
    before = measure_local_highlights(image, mask)
    glared, _ = add_glare_at(image, 0.9, (200, 200), 55, falloff=0.35)
    after = measure_local_highlights(glared, mask)
    assert after.largest_component_fraction > before.largest_component_fraction
    assert after.candidate_fraction > before.candidate_fraction


def test_glare_candidates_never_include_background():
    """A highlight on the backdrop is not a highlight on the fruit."""
    image, mask = textured_subject()
    glared, _ = add_glare_at(image, 0.9, (40, 40), 30, falloff=0.35)
    detected = highlight_mask(glared, mask)
    assert np.count_nonzero(detected & (mask == 0)) == 0


def test_a_bright_saturated_fruit_is_not_called_glare():
    """The failure mode this detector must not have."""
    bright_orange, mask = textured_subject(colour=(30, 170, 250))
    evidence = measure_local_highlights(bright_orange, mask)
    assert not evidence.glare_flag, evidence.flag_reasons


def test_a_uniformly_brightened_subject_is_not_called_glare():
    """Global overexposure is an exposure problem, not a local highlight."""
    image, mask = textured_subject()
    brightened = cv2.convertScaleAbs(image, alpha=1.5)
    assert not measure_local_highlights(brightened, mask).glare_flag


def test_glare_response_grows_with_severity():
    image, mask = textured_subject()
    fractions = []
    for intensity in (0.0, 0.4, 0.7, 1.0):
        glared, _ = add_glare_at(image, intensity, (200, 200), 55, falloff=0.35)
        fractions.append(measure_local_highlights(glared, mask).largest_component_fraction)
    assert fractions[-1] > fractions[0]
    assert fractions[-1] >= fractions[1]


def test_multiscale_excess_sees_a_large_highlight():
    """A single background scale cannot see a highlight wider than itself.

    Uses a moderate intensity that brightens without clipping, and puts the
    clipping criterion out of reach in both policies, so the comparison isolates
    the local-excess path. At full intensity the highlight is blown white and
    the clipped-patch rule catches it at any scale, which would make this pass
    without exercising the mechanism it names.
    """
    image, mask = textured_subject()
    glared, _ = add_glare_at(image, 0.30, (200, 200), 120, falloff=0.35)
    single = LocalHighlightPolicy(
        background_sigma_fractions=(0.10,), clipped_luminance=256.0
    )
    multi = LocalHighlightPolicy(clipped_luminance=256.0)
    narrow = measure_local_highlights(glared, mask, single)
    wide = measure_local_highlights(glared, mask, multi)
    assert wide.max_local_luminance_excess > narrow.max_local_luminance_excess
    assert wide.candidate_fraction > narrow.candidate_fraction


def test_highlight_measurement_rejects_a_mismatched_mask():
    image, _ = textured_subject()
    with pytest.raises(ImageValidationError):
        measure_local_highlights(image, np.zeros((10, 10), np.uint8))


# --- 8-9. contrast ------------------------------------------------------------


def test_controlled_contrast_reduction_moves_the_contrast_evidence():
    from competition.vision.config import DEFAULT_POLICY
    from competition.vision.degradation import reduce_contrast
    from competition.vision.quality import measure_illumination

    image, mask = textured_subject()
    before = measure_illumination(image, DEFAULT_POLICY, mask).contrast_score
    after = measure_illumination(reduce_contrast(image, 0.15), DEFAULT_POLICY, mask).contrast_score
    assert after < before / 2


def test_contrast_policy_path_end_to_end():
    """Severe blocks and recaptures; moderate routes to enhancement; intact passes."""
    policy = ArtifactPolicy(moderate_contrast_limit=0.1314, severe_contrast_limit=0.06)
    severe = decide_capture_artifacts(roi_contrast_score=0.03, policy=policy,
                                      visibility_evidence=_sufficient())
    moderate = decide_capture_artifacts(roi_contrast_score=0.10, policy=policy,
                                        visibility_evidence=_sufficient())
    intact = decide_capture_artifacts(roi_contrast_score=0.50, policy=policy,
                                      visibility_evidence=_sufficient())

    assert severe.action is RemediationAction.REQUEST_RECAPTURE
    assert severe.reason_code is ReasonCode.SEVERE_CONTRAST_LOSS
    assert severe.blocking

    assert moderate.action is RemediationAction.APPLY_CLAHE
    assert moderate.reason_code is ReasonCode.MODERATE_CONTRAST_LOSS_RECOVERABLE
    assert not moderate.blocking

    assert intact.action is RemediationAction.NONE


def _sufficient():
    _, mask = textured_subject()
    return measure_visibility(mask)


# --- 10-12. visibility --------------------------------------------------------


def test_controlled_occlusion_moves_visibility_evidence():
    _, mask = textured_subject()
    occluded = mask.copy()
    cv2.rectangle(occluded, (200, 60), (400, 340), 0, -1)
    intact = measure_visibility(mask)
    bitten = measure_visibility(occluded)
    assert bitten.solidity < intact.solidity
    assert bitten.hull_fill < intact.hull_fill


def test_severe_visibility_loss_blocks_inspection():
    _, mask = textured_subject()
    split = mask.copy()
    cv2.rectangle(split, (0, 190), (400, 215), 0, -1)
    evidence = measure_visibility(split)
    assert not evidence.visibility_sufficient
    decision = decide_capture_artifacts(
        visibility_evidence=evidence, policy=ArtifactPolicy(visibility_blocks=True)
    )
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.SUBJECT_VISIBILITY_INSUFFICIENT
    assert decision.blocking


def test_mild_visibility_variation_does_not_block():
    """A lumpy but complete subject must not be sent back."""
    _, mask = textured_subject()
    lumpy = mask.copy()
    for angle in range(0, 360, 45):
        x = int(200 + 140 * np.cos(np.radians(angle)))
        y = int(200 + 140 * np.sin(np.radians(angle)))
        cv2.circle(lumpy, (x, y), 22, 255, -1)
    assert measure_visibility(lumpy).visibility_sufficient


def test_interior_holes_are_measured_on_the_unfilled_mask():
    """Filling the mask erases exactly the evidence an occluder leaves."""
    image, mask = textured_subject()
    covered, _ = occlude_at(image, (200, 200), 90, 90)
    unfilled = segment_without_fill(covered)
    filled_only = measure_visibility(mask)
    with_unfilled = measure_visibility(mask, unfilled_mask=unfilled)
    assert filled_only.internal_hole_fraction == pytest.approx(0.0, abs=1e-6)
    assert with_unfilled.internal_hole_fraction > filled_only.internal_hole_fraction


def test_visibility_rejects_an_unfilled_mask_of_the_wrong_shape():
    _, mask = textured_subject()
    with pytest.raises(ImageValidationError):
        measure_visibility(mask, unfilled_mask=np.zeros((10, 10), np.uint8))


def test_no_semantic_occlusion_claim_appears_in_the_evidence():
    """The vocabulary must not assert a cause it cannot establish."""
    _, mask = textured_subject()
    payload = json.dumps(measure_visibility(mask).to_dict()).lower()
    for forbidden in ("hand", "occluder", "occlusion detected", "covered by"):
        assert forbidden not in payload


# --- 13. untrusted segmentation stops local evidence --------------------------


def test_invalid_segmentation_prevents_trusted_artifact_evidence():
    decision = decide_capture_artifacts(segmentation_valid=False)
    assert decision.reason_code is ReasonCode.ARTIFACT_EVIDENCE_UNAVAILABLE
    assert decision.action is RemediationAction.REQUEST_HUMAN_REVIEW
    assert decision.blocking
    assert CaptureArtifactFlag.ARTIFACT_EVIDENCE_UNAVAILABLE.value in decision.flags


def test_analysis_on_an_unsegmentable_image_escalates_without_measuring():
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    analysis = analyse_capture_artifacts(flat)
    assert analysis.decision.blocking
    assert analysis.highlights is None
    assert analysis.visibility is None


# --- 14-15. actions are closed, and glare is never "enhanced" -----------------


def test_every_decision_uses_the_closed_action_enum():
    _, mask = textured_subject()
    evidence = measure_visibility(mask)
    for contrast in (0.02, 0.10, 0.50, None):
        decision = decide_capture_artifacts(
            visibility_evidence=evidence, roi_contrast_score=contrast
        )
        assert isinstance(decision.action, RemediationAction)
        assert isinstance(decision.reason_code, ReasonCode)
        assert set(decision.flags) <= {flag.value for flag in CaptureArtifactFlag}


def test_glare_never_routes_to_an_enhancement():
    """No tone curve recovers a blown highlight; offering one would fake it."""
    image, mask = textured_subject()
    glared, _ = add_glare_at(image, 1.0, (200, 200), 70, falloff=0.35)
    evidence = measure_local_highlights(glared, mask)
    assert evidence.glare_flag
    decision = decide_capture_artifacts(
        highlight_evidence=evidence, visibility_evidence=measure_visibility(mask),
        roi_contrast_score=0.05, policy=ArtifactPolicy(glare_blocks=True),
    )
    assert decision.action is RemediationAction.REQUEST_REPOSITION_LIGHT
    assert decision.action not in (RemediationAction.APPLY_GAMMA, RemediationAction.APPLY_CLAHE)


def test_reposition_light_is_a_human_action_not_an_automated_one():
    assert RemediationAction.REQUEST_REPOSITION_LIGHT.requires_human
    assert not RemediationAction.REQUEST_REPOSITION_LIGHT.is_automated


def test_blocking_flags_exclude_moderate_contrast():
    assert CaptureArtifactFlag.MODERATE_LOW_CONTRAST.value not in BLOCKING_ARTIFACT_FLAGS
    assert CaptureArtifactFlag.GLARE_RISK.value in BLOCKING_ARTIFACT_FLAGS


def test_visibility_outranks_glare_and_contrast():
    """If the subject is not fully in frame, its photometry is beside the point."""
    image, mask = textured_subject()
    split = mask.copy()
    cv2.rectangle(split, (0, 190), (400, 215), 0, -1)
    glared, _ = add_glare_at(image, 1.0, (200, 200), 70, falloff=0.35)
    decision = decide_capture_artifacts(
        highlight_evidence=measure_local_highlights(glared, mask),
        visibility_evidence=measure_visibility(split),
        roi_contrast_score=0.01,
        policy=ArtifactPolicy(glare_blocks=True, visibility_blocks=True),
    )
    assert decision.reason_code is ReasonCode.SUBJECT_VISIBILITY_INSUFFICIENT


# --- trace --------------------------------------------------------------------


def test_the_analysis_records_a_causal_trace():
    image, _ = textured_subject()
    analysis = analyse_capture_artifacts(image)
    tools = [step["tool"] for step in analysis.trace.to_dict()["steps"]]
    assert tools[0] == "isolate_foreground"
    assert tools[-1] == "decide_capture_artifact"


def test_analysis_payload_is_deterministic_excluding_timing():
    image, _ = textured_subject()
    first = analyse_capture_artifacts(image).deterministic_payload()
    second = analyse_capture_artifacts(image).deterministic_payload()
    for payload in (first, second):
        payload.pop("trace", None)
    assert first == second


def test_policies_fingerprint_their_thresholds():
    assert (LocalHighlightPolicy().fingerprint()
            != LocalHighlightPolicy(local_excess_l=99.0).fingerprint())
    assert (VisibilityPolicy().fingerprint()
            != VisibilityPolicy(min_solidity=0.1).fingerprint())
    assert DEFAULT_HIGHLIGHT_POLICY.fingerprint() == LocalHighlightPolicy().fingerprint()
    assert DEFAULT_VISIBILITY_POLICY.fingerprint() == VisibilityPolicy().fingerprint()


# --- detectors below the preregistered floor record, but do not gate ----------


def test_a_sub_floor_detector_records_its_finding_without_blocking():
    """Glare defaults to evidence-only because it reached 0.655, not 0.70."""
    image, mask = textured_subject()
    glared, _ = add_glare_at(image, 1.0, (200, 200), 70, falloff=0.35)
    evidence = measure_local_highlights(glared, mask)
    assert evidence.glare_flag

    decision = decide_capture_artifacts(
        highlight_evidence=evidence, visibility_evidence=measure_visibility(mask),
        roi_contrast_score=0.50, policy=ArtifactPolicy(),
    )
    assert CaptureArtifactFlag.GLARE_RISK.value in decision.flags
    assert decision.action is RemediationAction.NONE
    assert not decision.blocking
    assert "did not meet the preregistered detection floor" in decision.explanation


def test_visibility_defaults_to_evidence_only():
    _, mask = textured_subject()
    split = mask.copy()
    cv2.rectangle(split, (0, 190), (400, 215), 0, -1)
    decision = decide_capture_artifacts(
        visibility_evidence=measure_visibility(split), policy=ArtifactPolicy()
    )
    assert CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value in decision.flags
    assert decision.action is RemediationAction.NONE


def test_severe_contrast_still_blocks_by_default():
    """Contrast cleared the floor, so it is the one artefact permitted to gate."""
    _, mask = textured_subject()
    decision = decide_capture_artifacts(
        visibility_evidence=measure_visibility(mask), roi_contrast_score=0.02,
        policy=ArtifactPolicy(),
    )
    assert decision.action is RemediationAction.REQUEST_RECAPTURE
    assert decision.reason_code is ReasonCode.SEVERE_CONTRAST_LOSS
    assert decision.blocking


def test_gating_flags_are_off_by_default():
    policy = ArtifactPolicy()
    assert not policy.glare_blocks
    assert not policy.visibility_blocks
