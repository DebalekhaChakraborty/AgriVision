"""Foreground isolation, validity guards, and ROI-restricted quality."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from competition.vision.config import DEFAULT_POLICY
from competition.vision.evidence import ImageValidationError, MeasurementScope, QualityFlag
from competition.vision.fixtures import (
    dark_subject_on_bright_background,
    foreground_fixtures,
    multiple_disconnected_regions,
    small_subject,
    textured_object,
    textured_subject_neutral_background,
)
from competition.vision.foreground import (
    DEFAULT_GUARDS,
    DEFAULT_METHOD,
    ForegroundError,
    ForegroundGuards,
    ForegroundMethod,
    isolate_foreground,
    mask_fingerprint,
    masked_pixels,
)
from competition.vision.quality import MIN_MASKED_PIXELS, assess_capture_quality, measure_sharpness


def iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    p, g = prediction > 0, truth > 0
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 0.0


# --- required test 1-3: evidence contract ------------------------------------


def test_foreground_evidence_serialises():
    image, _ = textured_subject_neutral_background()
    _, evidence = isolate_foreground(image)
    restored = json.loads(json.dumps(evidence.to_dict()))
    assert restored["method"] == DEFAULT_METHOD.value
    assert isinstance(restored["valid"], bool)
    assert 0.0 <= restored["foreground_fraction"] <= 1.0
    assert len(restored["bounding_box"]) == 4


def test_no_filesystem_paths_in_foreground_evidence():
    image, _ = textured_subject_neutral_background()
    payload = json.dumps(isolate_foreground(image)[1].to_dict())
    assert "/home/" not in payload
    assert "/Users/" not in payload
    assert "C:\\" not in payload


def test_foreground_is_deterministic():
    image, _ = textured_subject_neutral_background()
    first_mask, first = isolate_foreground(image)
    second_mask, second = isolate_foreground(image)
    assert np.array_equal(first_mask, second_mask)
    assert first.deterministic_payload() == second.deterministic_payload()
    assert "processing_ms" not in first.deterministic_payload()


def test_mask_hash_changes_with_mask():
    a, _ = isolate_foreground(textured_subject_neutral_background()[0])
    b, _ = isolate_foreground(dark_subject_on_bright_background()[0])
    assert mask_fingerprint(a) != mask_fingerprint(b)


# --- required test 4: invalid input fails safely ------------------------------


@pytest.mark.parametrize("bad", [
    None,
    "not an image",
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((32, 32), dtype=np.uint8),
    np.zeros((32, 32, 3), dtype=np.float32),
])
def test_invalid_image_is_rejected(bad):
    with pytest.raises(ForegroundError):
        isolate_foreground(bad)


def test_unknown_method_is_rejected():
    image, _ = textured_subject_neutral_background()
    with pytest.raises(ForegroundError):
        isolate_foreground(image, method="magic_segmenter")  # type: ignore[arg-type]


# --- required test 5: a known synthetic foreground segments ------------------


@pytest.mark.parametrize("method", list(ForegroundMethod))
@pytest.mark.parametrize("fixture", [
    dark_subject_on_bright_background,
    textured_subject_neutral_background,
])
def test_clear_subject_is_segmented_and_valid(method, fixture):
    image, truth = fixture()
    mask, evidence = isolate_foreground(image, method)
    assert evidence.valid, evidence.invalid_reasons
    assert iou(mask, truth) > 0.85


def test_segmentation_recovers_the_known_area_fraction():
    image, truth = textured_subject_neutral_background()
    _, evidence = isolate_foreground(image)
    expected = float(np.count_nonzero(truth) / truth.size)
    assert abs(evidence.foreground_fraction - expected) < 0.05


# --- required tests 6-9: validity guards --------------------------------------


def test_near_full_frame_mask_is_rejected():
    """A mask covering almost everything has not isolated anything."""
    guards = ForegroundGuards(max_foreground_fraction=0.20)
    image, _ = textured_subject_neutral_background()
    _, evidence = isolate_foreground(image, guards=guards)
    assert not evidence.valid
    assert "FOREGROUND_NEAR_FULL_FRAME" in evidence.invalid_reasons


def test_tiny_mask_is_rejected():
    image, _ = small_subject()
    _, evidence = isolate_foreground(image)
    assert not evidence.valid
    assert "FOREGROUND_TOO_SMALL" in evidence.invalid_reasons


def test_excessive_border_contact_is_rejected():
    guards = ForegroundGuards(max_border_contact_fraction=0.0)
    image, _ = textured_subject_neutral_background()
    _, evidence = isolate_foreground(image, guards=guards)
    if evidence.border_contact_fraction > 0:
        assert "EXCESSIVE_BORDER_CONTACT" in evidence.invalid_reasons


def test_fragmented_mask_is_rejected():
    """Four equal blobs: no dominant component, so the mask is not trusted."""
    image, _ = multiple_disconnected_regions()
    _, evidence = isolate_foreground(image)
    assert not evidence.valid
    assert "NO_DOMINANT_COMPONENT" in evidence.invalid_reasons


def test_component_count_guard_fires():
    guards = ForegroundGuards(max_component_count=0, min_largest_component_dominance=0.0)
    image, _ = textured_subject_neutral_background()
    _, evidence = isolate_foreground(image, guards=guards)
    assert "MASK_FRAGMENTED" in evidence.invalid_reasons


def test_every_invalid_reason_is_reported_not_just_the_first():
    image, _ = small_subject()
    _, evidence = isolate_foreground(image)
    assert len(evidence.invalid_reasons) >= 2


# --- required tests 10-11: masked metrics ignore the background --------------


def test_masked_luminance_ignores_a_bright_background():
    """The Phase 2 failure case, as an assertion."""
    image, truth = dark_subject_on_bright_background()
    whole = assess_capture_quality(image)
    roi = assess_capture_quality(image, mask=truth)
    assert whole.illumination.mean_luminance > roi.illumination.mean_luminance + 0.2


def test_masked_clipping_ignores_background_clipping():
    image, truth = dark_subject_on_bright_background()
    whole = assess_capture_quality(image)
    roi = assess_capture_quality(image, mask=truth)
    assert whole.illumination.highlight_clip_fraction > 0.5
    assert roi.illumination.highlight_clip_fraction < 0.01


def test_masked_shadow_clipping_ignores_a_dark_background():
    from competition.vision.fixtures import bright_subject_on_dark_background

    image, truth = bright_subject_on_dark_background()
    whole = assess_capture_quality(image)
    roi = assess_capture_quality(image, mask=truth)
    assert whole.illumination.shadow_clip_fraction > 0.5
    assert roi.illumination.shadow_clip_fraction < 0.01


# --- required test 12: the mask edge must not create sharpness ---------------


def test_mask_edge_does_not_inflate_sharpness():
    """A binary mask boundary is a step edge; it must not read as detail.

    With no erosion the subject/background boundary is included and the variance
    is inflated roughly threefold. The default erosion removes that.
    """
    image, truth = dark_subject_on_bright_background()
    unerorded = measure_sharpness(image, DEFAULT_POLICY, truth, erosion_px=0)
    eroded = measure_sharpness(image, DEFAULT_POLICY, truth)
    assert eroded.laplacian_variance < unerorded.laplacian_variance * 0.6


def test_sharpness_is_stable_once_the_boundary_is_excluded():
    """Beyond the default erosion, further erosion changes little."""
    image, truth = dark_subject_on_bright_background()
    at_default = measure_sharpness(image, DEFAULT_POLICY, truth).laplacian_variance
    deeper = measure_sharpness(image, DEFAULT_POLICY, truth, erosion_px=11).laplacian_variance
    assert abs(at_default - deeper) / at_default < 0.10


def test_masking_a_uniform_subject_does_not_report_high_sharpness():
    """The failure mode of zeroing the background then differentiating."""
    height = width = 256
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (width // 2, height // 2), 70, 255, -1)
    image[mask > 0] = (40, 40, 40)  # flat, detail-free subject

    result = measure_sharpness(image, DEFAULT_POLICY, mask)
    assert result.laplacian_variance < 1.0


def test_too_few_masked_pixels_raises_rather_than_guessing():
    image, _ = textured_subject_neutral_background()
    tiny = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(tiny, (128, 128), 3, 255, -1)
    with pytest.raises(ImageValidationError):
        assess_capture_quality(image, mask=tiny)


def test_masked_pixels_erodes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)
    assert np.count_nonzero(masked_pixels(mask, 5)) < np.count_nonzero(mask)
    assert np.array_equal(masked_pixels(mask, 0), mask)


# --- required test 13: whole-image metrics remain available ------------------


def test_whole_image_and_roi_evidence_are_distinguishable():
    image, truth = dark_subject_on_bright_background()
    whole = assess_capture_quality(image)
    roi = assess_capture_quality(image, mask=truth)
    assert whole.measurement_scope == MeasurementScope.WHOLE_IMAGE.value
    assert roi.measurement_scope == MeasurementScope.FOREGROUND_MASKED.value
    assert whole.to_dict() != roi.to_dict()


def test_scope_is_present_in_serialised_evidence():
    image, truth = dark_subject_on_bright_background()
    payload = json.loads(assess_capture_quality(image, mask=truth).to_json())
    assert payload["measurement_scope"] == "FOREGROUND_MASKED"


def test_default_assessment_is_still_whole_image():
    evidence = assess_capture_quality(textured_object())
    assert evidence.measurement_scope == MeasurementScope.WHOLE_IMAGE.value


# --- mask validation ----------------------------------------------------------


def test_mismatched_mask_shape_is_rejected():
    image, _ = textured_subject_neutral_background()
    with pytest.raises(ImageValidationError):
        assess_capture_quality(image, mask=np.ones((10, 10), dtype=np.uint8))


def test_three_dimensional_mask_is_rejected():
    image, _ = textured_subject_neutral_background()
    with pytest.raises(ImageValidationError):
        assess_capture_quality(image, mask=np.ones((*image.shape[:2], 3), dtype=np.uint8))


# --- required tests 16-17: nothing mutates the source ------------------------


@pytest.mark.parametrize("method", list(ForegroundMethod))
def test_segmentation_does_not_mutate_the_source(method):
    image, _ = textured_subject_neutral_background()
    before = image.copy()
    isolate_foreground(image, method)
    assert np.array_equal(image, before)


def test_masked_assessment_does_not_mutate_source_or_mask():
    image, truth = dark_subject_on_bright_background()
    image_before, mask_before = image.copy(), truth.copy()
    assess_capture_quality(image, mask=truth)
    assert np.array_equal(image, image_before)
    assert np.array_equal(truth, mask_before)


# --- method coverage ----------------------------------------------------------


def test_all_methods_produce_a_binary_mask_of_the_right_shape():
    image, _ = textured_subject_neutral_background()
    for method in ForegroundMethod:
        mask, _ = isolate_foreground(image, method)
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


def test_default_method_is_a_member_of_the_enum():
    assert DEFAULT_METHOD in set(ForegroundMethod)


def test_guards_are_serialisable():
    assert json.loads(json.dumps(DEFAULT_GUARDS.to_dict()))["sharpness_erosion_px"] > 0


def test_synthetic_fixtures_have_matching_image_and_mask_shapes():
    for name, (image, mask) in foreground_fixtures().items():
        assert mask.shape == image.shape[:2], name
        assert set(np.unique(mask)).issubset({0, 255}), name
