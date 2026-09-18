"""OpenCV enhancement operations: determinism, safety, colour handling."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from competition.vision.enhancement import (
    ClaheParameters,
    EnhancementError,
    GammaParameters,
    apply_clahe,
    apply_gamma_correction,
    gamma_for_target_luminance,
)
from competition.vision.degradation import adjust_exposure
from competition.vision.fixtures import (
    checkerboard,
    horizontal_gradient,
    textured_object,
    uniform_mid,
)


# --- required test 1: gamma correction is deterministic -----------------------


@pytest.mark.parametrize("gamma", [0.3, 0.5, 1.0, 1.8, 2.5])
def test_gamma_correction_is_deterministic(gamma):
    image = textured_object()
    first, meta_a = apply_gamma_correction(image, GammaParameters(gamma=gamma))
    second, meta_b = apply_gamma_correction(image, GammaParameters(gamma=gamma))
    assert np.array_equal(first, second)
    assert meta_a["parameters"] == meta_b["parameters"]


def test_gamma_metadata_records_exact_parameter():
    _, metadata = apply_gamma_correction(textured_object(), GammaParameters(gamma=0.47))
    assert metadata["parameters"]["gamma"] == 0.47
    assert metadata["operation"] == "gamma_correction"
    assert metadata["opencv_version"] == cv2.__version__
    assert metadata["colour_space"] == "CIELAB L* only"


def test_gamma_below_one_brightens_and_above_one_darkens():
    image = textured_object()
    base = float(cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].mean())
    brighter, _ = apply_gamma_correction(image, GammaParameters(gamma=0.5))
    darker, _ = apply_gamma_correction(image, GammaParameters(gamma=2.0))
    assert float(cv2.cvtColor(brighter, cv2.COLOR_BGR2LAB)[:, :, 0].mean()) > base
    assert float(cv2.cvtColor(darker, cv2.COLOR_BGR2LAB)[:, :, 0].mean()) < base


def test_gamma_of_one_is_near_identity():
    """Exact equality is not expected: BGR->Lab->BGR is lossy at uint8."""
    image = textured_object()
    result, _ = apply_gamma_correction(image, GammaParameters(gamma=1.0))
    assert result.shape == image.shape
    assert np.abs(result.astype(int) - image.astype(int)).max() <= 4


# --- required test 2: CLAHE is deterministic ---------------------------------


@pytest.mark.parametrize("clip,grid", [(2.0, 8), (4.0, 4), (1.5, 16)])
def test_clahe_is_deterministic(clip, grid):
    image = textured_object()
    first, _ = apply_clahe(image, ClaheParameters(clip_limit=clip, tile_grid_size=grid))
    second, _ = apply_clahe(image, ClaheParameters(clip_limit=clip, tile_grid_size=grid))
    assert np.array_equal(first, second)


def test_clahe_metadata_records_parameters():
    _, metadata = apply_clahe(
        textured_object(), ClaheParameters(clip_limit=3.0, tile_grid_size=4)
    )
    assert metadata["parameters"] == {"clip_limit": 3.0, "tile_grid_size": 4}
    assert metadata["operation"] == "clahe"
    assert metadata["opencv_version"] == cv2.__version__
    assert "processing_ms" in metadata


# --- required test 3: source image never mutated ------------------------------


@pytest.mark.parametrize("fixture", [checkerboard, textured_object, horizontal_gradient])
def test_enhancements_never_mutate_the_source(fixture):
    image = fixture()
    original = image.copy()
    apply_gamma_correction(image, GammaParameters(gamma=0.5))
    apply_clahe(image, ClaheParameters())
    assert np.array_equal(image, original)


# --- required test 4: luminance-only round trip yields valid BGR --------------


@pytest.mark.parametrize("fixture", [textured_object, horizontal_gradient, uniform_mid])
def test_enhanced_output_is_valid_bgr(fixture):
    image = fixture()
    for result, _ in (
        apply_gamma_correction(image, GammaParameters(gamma=0.6)),
        apply_clahe(image, ClaheParameters()),
    ):
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert result.ndim == 3 and result.shape[2] == 3
        assert 0 <= int(result.min()) and int(result.max()) <= 255


def test_clahe_preserves_chromaticity_channels():
    """Only L* may change; a and b must be untouched.

    Equalising B, G and R independently would shift colour balance and corrupt
    the colour analysis planned for later phases.
    """
    image = textured_object()
    result, _ = apply_clahe(image, ClaheParameters())

    before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

    # Round-tripping through BGR at uint8 introduces small quantisation error;
    # the point is that a/b are not systematically shifted.
    assert np.abs(after[:, :, 1].astype(int) - before[:, :, 1].astype(int)).mean() < 2.0
    assert np.abs(after[:, :, 2].astype(int) - before[:, :, 2].astype(int)).mean() < 2.0


def test_clahe_increases_contrast_on_a_low_contrast_image():
    dim = adjust_exposure(textured_object(), 0.5)
    result, _ = apply_clahe(dim, ClaheParameters())
    before = float(cv2.cvtColor(dim, cv2.COLOR_BGR2LAB)[:, :, 0].std())
    after = float(cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0].std())
    assert after > before


# --- parameter validation -----------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, 11.0, "x", None])
def test_invalid_gamma_is_rejected(bad):
    with pytest.raises(EnhancementError):
        apply_gamma_correction(textured_object(), GammaParameters(gamma=bad))


@pytest.mark.parametrize("clip,grid", [(0.0, 8), (-1.0, 8), (41.0, 8), (2.0, 0), (2.0, 65)])
def test_invalid_clahe_parameters_are_rejected(clip, grid):
    with pytest.raises(EnhancementError):
        apply_clahe(textured_object(), ClaheParameters(clip_limit=clip, tile_grid_size=grid))


@pytest.mark.parametrize("bad", [
    None,
    "not an image",
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((8, 8), dtype=np.uint8),
    np.zeros((8, 8, 3), dtype=np.float32),
])
def test_invalid_images_are_rejected(bad):
    with pytest.raises(EnhancementError):
        apply_gamma_correction(bad, GammaParameters(gamma=0.5))
    with pytest.raises(EnhancementError):
        apply_clahe(bad, ClaheParameters())


# --- gamma derivation ---------------------------------------------------------


def test_gamma_derivation_matches_the_power_law():
    """gamma = log(target)/log(current), so current**gamma == target."""
    gamma = gamma_for_target_luminance(0.2, 0.45, 0.1, 5.0)
    assert 0.2**gamma == pytest.approx(0.45, rel=1e-6)


def test_gamma_derivation_is_clamped_to_policy_bounds():
    assert gamma_for_target_luminance(0.99, 0.45, 0.25, 3.0) == 3.0
    assert gamma_for_target_luminance(0.001, 0.45, 0.25, 3.0) == 0.25


def test_gamma_derivation_handles_degenerate_luminance():
    """Fully crushed or blown input returns a bound rather than raising."""
    assert gamma_for_target_luminance(0.0, 0.45, 0.25, 3.0) == 0.25
    assert gamma_for_target_luminance(1.0, 0.45, 0.25, 3.0) == 3.0


def test_gamma_derivation_rejects_impossible_target():
    with pytest.raises(EnhancementError):
        gamma_for_target_luminance(0.3, 0.0, 0.25, 3.0)
    with pytest.raises(EnhancementError):
        gamma_for_target_luminance(0.3, 1.0, 0.25, 3.0)
