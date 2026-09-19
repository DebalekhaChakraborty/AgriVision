"""Candidate focus metrics and the ROI policy that selects between them."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from competition.vision.blur_metrics import (
    BLUR_METRICS_VERSION,
    MIN_MEASURABLE_PIXELS,
    FocusMetrics,
    high_frequency_ratio,
    measure_focus,
)
from competition.vision.evidence import ImageValidationError
from competition.vision.roi_policy import (
    FOCUS_METRICS,
    PROVISIONAL_ROI_POLICY,
    RoiPolicyError,
    RoiQualityPolicy,
    apply_roi_policy,
    focus_value,
)


@pytest.fixture
def textured() -> np.ndarray:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 1.0)


def test_all_four_metrics_are_reported(textured):
    metrics = measure_focus(textured)
    assert isinstance(metrics, FocusMetrics)
    assert set(metrics.to_dict()) >= set(FOCUS_METRICS)
    assert metrics.metrics_version == BLUR_METRICS_VERSION


@pytest.mark.parametrize("metric", FOCUS_METRICS)
def test_every_metric_falls_when_the_image_is_blurred(textured, metric):
    sharp = getattr(measure_focus(textured), metric)
    blurred = getattr(measure_focus(cv2.GaussianBlur(textured, (0, 0), 4.0)), metric)
    assert blurred < sharp, metric


def test_normalised_gradient_energy_survives_an_exposure_change(textured):
    """The property it exists for: Phase 1 found Laplacian variance scales as g^2."""
    dim = cv2.convertScaleAbs(textured, alpha=0.4)
    sharp = measure_focus(textured)
    darker = measure_focus(dim)

    laplacian_shift = darker.laplacian_variance / sharp.laplacian_variance
    normalised_shift = (
        darker.normalised_gradient_energy / sharp.normalised_gradient_energy
    )
    assert laplacian_shift < 0.5, "expected Laplacian variance to collapse with exposure"
    assert 0.9 < normalised_shift < 1.1, normalised_shift


def test_normalised_gradient_energy_still_falls_for_a_dark_blurred_image(textured):
    """Exposure invariance must not become blur blindness."""
    dim = cv2.convertScaleAbs(textured, alpha=0.4)
    sharp = measure_focus(dim).normalised_gradient_energy
    blurred = measure_focus(cv2.GaussianBlur(dim, (0, 0), 4.0)).normalised_gradient_energy
    assert blurred < sharp


def test_a_mask_restricts_the_measured_population(textured):
    mask = np.zeros(textured.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (160, 160), 90, 255, -1)
    whole = measure_focus(textured)
    roi = measure_focus(textured, mask)
    assert roi.measured_pixels < whole.measured_pixels
    assert roi.measured_pixels > MIN_MEASURABLE_PIXELS


def test_a_mask_that_leaves_too_few_pixels_is_an_error(textured):
    mask = np.zeros(textured.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (160, 160), 4, 255, -1)
    with pytest.raises(ImageValidationError):
        measure_focus(textured, mask)


def test_a_mismatched_mask_is_rejected(textured):
    with pytest.raises(ImageValidationError):
        measure_focus(textured, np.zeros((10, 10), dtype=np.uint8))


def test_measuring_an_empty_image_is_an_error():
    with pytest.raises(ImageValidationError):
        measure_focus(np.zeros((0, 0, 3), dtype=np.uint8))


def test_high_frequency_ratio_is_bounded():
    rng = np.random.default_rng(1)
    noisy = rng.integers(0, 255, (128, 128), dtype=np.uint8)
    flat = np.full((128, 128), 128, dtype=np.uint8)
    assert 0.0 <= high_frequency_ratio(noisy) <= 1.0
    assert 0.0 <= high_frequency_ratio(flat) <= 1.0


def test_measurement_does_not_modify_its_input(textured):
    before = textured.copy()
    measure_focus(textured)
    assert np.array_equal(textured, before)


# --- ROI policy ---------------------------------------------------------------


def test_policy_rejects_an_unknown_focus_metric():
    with pytest.raises(RoiPolicyError):
        RoiQualityPolicy(focus_metric="vibes")


def test_policy_rejects_inverted_exposure_bounds():
    with pytest.raises(RoiPolicyError):
        RoiQualityPolicy(underexposed_mean_luminance=0.9, overexposed_mean_luminance=0.2)


def test_a_policy_with_no_focus_floor_does_not_gate_on_focus():
    policy = RoiQualityPolicy(focus_floor=None)
    assert not policy.gates_on_focus


def test_threshold_fingerprint_ignores_provenance_but_not_thresholds():
    base = RoiQualityPolicy(focus_floor=100.0)
    locked = base.locked({"calibrated_on": "2026-09-18"})
    relaxed = RoiQualityPolicy(focus_floor=10.0)
    assert locked.threshold_fingerprint() == base.threshold_fingerprint()
    assert locked.lock_fingerprint() != base.lock_fingerprint()
    assert relaxed.threshold_fingerprint() != base.threshold_fingerprint()


def test_focus_value_reads_the_named_metric(textured):
    metrics = measure_focus(textured)
    policy = RoiQualityPolicy(focus_metric="tenengrad", focus_floor=1.0)
    assert focus_value(metrics, policy) == metrics.tenengrad


def test_focus_value_errors_when_the_metric_is_absent():
    from types import SimpleNamespace

    with pytest.raises(RoiPolicyError):
        focus_value(SimpleNamespace(laplacian_variance=1.0),
                    RoiQualityPolicy(focus_metric="tenengrad", focus_floor=1.0))


def _metrics(**overrides):
    from types import SimpleNamespace

    values = dict(
        laplacian_variance=500.0, tenengrad=5000.0,
        normalised_gradient_energy=25.0, high_frequency_ratio=0.6,
        mean_luminance=0.5, contrast_score=0.4,
        shadow_clip_fraction=0.0, highlight_clip_fraction=0.0,
        width=1200, height=900,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_a_clean_measurement_raises_no_flags():
    assert apply_roi_policy(_metrics(), _metrics(), _metrics(), PROVISIONAL_ROI_POLICY) == []


def test_focus_below_the_floor_raises_blur_risk():
    flags = apply_roi_policy(
        _metrics(), _metrics(), _metrics(laplacian_variance=5.0), PROVISIONAL_ROI_POLICY
    )
    assert "BLUR_RISK" in flags


def test_no_focus_floor_means_no_blur_flag_however_soft_the_image():
    policy = RoiQualityPolicy(focus_floor=None)
    flags = apply_roi_policy(
        _metrics(), _metrics(), _metrics(laplacian_variance=0.01), policy
    )
    assert "BLUR_RISK" not in flags


@pytest.mark.parametrize("overrides,expected", [
    ({"mean_luminance": 0.05}, "UNDEREXPOSED"),
    ({"mean_luminance": 0.95}, "OVEREXPOSED"),
    ({"contrast_score": 0.02}, "LOW_CONTRAST"),
    ({"shadow_clip_fraction": 0.5}, "SHADOW_CLIPPING"),
    ({"highlight_clip_fraction": 0.5}, "HIGHLIGHT_CLIPPING"),
])
def test_illumination_flags_fire_on_their_own_metric(overrides, expected):
    flags = apply_roi_policy(
        _metrics(), _metrics(**overrides), _metrics(), PROVISIONAL_ROI_POLICY
    )
    assert expected in flags


def test_small_images_are_flagged():
    flags = apply_roi_policy(
        _metrics(width=8, height=8), _metrics(), _metrics(), PROVISIONAL_ROI_POLICY
    )
    assert "IMAGE_TOO_SMALL" in flags


def test_the_provisional_policy_says_it_is_uncalibrated():
    assert "PROVISIONAL" in PROVISIONAL_ROI_POLICY.status
