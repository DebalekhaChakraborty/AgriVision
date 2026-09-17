"""Capture-quality measurement behaviour under controlled degradation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from competition.vision import assess_capture_quality
from competition.vision.config import ThresholdPolicy
from competition.vision.evidence import ImageValidationError, QualityFlag
from competition.vision.degradation import (
    BLUR_SIGMAS,
    OVEREXPOSURE_GAINS,
    UNDEREXPOSURE_GAINS,
    adjust_exposure,
    gaussian_blur,
    reduce_contrast,
)
from competition.vision.fixtures import (
    checkerboard,
    textured_object,
    uniform_bright,
    uniform_dark,
    uniform_mid,
)
from competition.vision.quality import load_image


# --- required test 3: OpenCV major version is 5 -------------------------------


def test_opencv_major_version_is_5():
    major = int(cv2.__version__.split(".")[0])
    assert major == 5, f"expected OpenCV 5, found {cv2.__version__}"


def test_evidence_records_the_running_opencv_version():
    evidence = assess_capture_quality(checkerboard())
    assert evidence.opencv_version == cv2.__version__
    assert evidence.opencv_version.startswith("5.")


# --- required test 4: increasing blur degrades sharpness ----------------------


@pytest.mark.parametrize("fixture", [checkerboard, textured_object])
def test_sharpness_decreases_monotonically_with_blur(fixture):
    """Sharpness must never increase under stronger blur, above the noise floor.

    Monotonicity is asserted only while the variance is above
    `sharpness_noise_floor`. Below that the image retains no detail at all and
    the residual value is dominated by border handling and quantisation rather
    than content — see `test_sharpness_saturates_below_the_noise_floor`, which
    documents that behaviour explicitly rather than hiding it here.
    """
    floor = ThresholdPolicy().sharpness_noise_floor
    image = fixture()
    variances = [
        assess_capture_quality(gaussian_blur(image, sigma)).sharpness.laplacian_variance
        for sigma in BLUR_SIGMAS
    ]

    meaningful = [v for v in variances if v > floor]
    assert len(meaningful) >= 2, f"fixture never rose above the noise floor: {variances}"

    for previous, current in zip(meaningful, meaningful[1:]):
        assert current <= previous + 1e-9, (
            f"sharpness increased under stronger blur: {variances}"
        )
    assert variances[-1] < variances[0], "heaviest blur did not reduce sharpness"


def test_sharpness_saturates_below_the_noise_floor():
    """Documents a measured limitation: below the floor, ordering is meaningless.

    On a low-texture fixture the Laplacian variance bottoms out and then drifts
    slightly upward at larger blur kernels. This test pins the observed
    behaviour so a future change to the metric cannot silently alter it.
    """
    floor = ThresholdPolicy().sharpness_noise_floor
    image = textured_object()
    heavy = [
        assess_capture_quality(gaussian_blur(image, sigma)).sharpness.laplacian_variance
        for sigma in (5.0, 8.0, 12.0)
    ]
    assert all(v < floor for v in heavy), f"expected saturation below {floor}: {heavy}"
    # All effectively "no detail" — the spread is tiny in absolute terms.
    assert max(heavy) - min(heavy) < 1.0


def test_heavy_blur_raises_blur_risk_flag():
    evidence = assess_capture_quality(gaussian_blur(checkerboard(), 8.0))
    assert evidence.has_flag(QualityFlag.BLUR_RISK)


def test_sharp_fixture_does_not_raise_blur_risk():
    evidence = assess_capture_quality(checkerboard())
    assert not evidence.has_flag(QualityFlag.BLUR_RISK)


def test_sharpness_score_is_bounded():
    for sigma in BLUR_SIGMAS:
        score = assess_capture_quality(
            gaussian_blur(checkerboard(), sigma)
        ).sharpness.sharpness_score
        assert 0.0 <= score <= 1.0


# --- required test 5: underexposure moves illumination metrics ----------------


def test_underexposure_lowers_luminance_monotonically():
    image = textured_object()
    means = [
        assess_capture_quality(adjust_exposure(image, gain)).illumination.mean_luminance
        for gain in UNDEREXPOSURE_GAINS
    ]
    for previous, current in zip(means, means[1:]):
        assert current <= previous + 1e-9, f"luminance rose while darkening: {means}"
    assert means[-1] < means[0]


def test_underexposure_raises_underexposed_flag():
    evidence = assess_capture_quality(adjust_exposure(textured_object(), 0.15))
    assert evidence.has_flag(QualityFlag.UNDEREXPOSED)
    assert not evidence.has_flag(QualityFlag.OVEREXPOSED)


def test_extreme_underexposure_increases_shadow_clipping():
    """Shadow clipping requires genuine black crush, not merely a dim capture.

    Gain 0.05, not 0.15: multiplicative underexposure has to be drastic before
    pixels reach the L* shadow floor. See the companion test below.
    """
    image = textured_object()
    base = assess_capture_quality(image).illumination.shadow_clip_fraction
    crushed = assess_capture_quality(
        adjust_exposure(image, 0.05)
    ).illumination.shadow_clip_fraction
    assert base == 0.0
    assert crushed > base


def test_moderate_underexposure_does_not_clip_shadows():
    """Documents the measured separation between UNDEREXPOSED and SHADOW_CLIPPING.

    At gain 0.15 the fixture is clearly too dark and is flagged UNDEREXPOSED,
    but no pixel has reached the shadow floor, so detail is still recoverable
    and SHADOW_CLIPPING must not fire. The two flags mean different things and a
    later enhancement action depends on the distinction: a dim-but-unclipped
    capture can be rescued by CLAHE, a crushed one cannot.
    """
    dim = assess_capture_quality(adjust_exposure(textured_object(), 0.15))
    assert dim.has_flag(QualityFlag.UNDEREXPOSED)
    assert not dim.has_flag(QualityFlag.SHADOW_CLIPPING)
    assert dim.illumination.shadow_clip_fraction == 0.0


def test_uniform_dark_fixture_is_flagged_underexposed():
    assert assess_capture_quality(uniform_dark()).has_flag(QualityFlag.UNDEREXPOSED)


# --- required test 6: overexposure moves illumination metrics -----------------


def test_overexposure_raises_luminance_monotonically():
    image = textured_object()
    means = [
        assess_capture_quality(adjust_exposure(image, gain)).illumination.mean_luminance
        for gain in OVEREXPOSURE_GAINS
    ]
    for previous, current in zip(means, means[1:]):
        assert current >= previous - 1e-9, f"luminance fell while brightening: {means}"
    assert means[-1] > means[0]


def test_overexposure_raises_overexposed_flag():
    evidence = assess_capture_quality(adjust_exposure(textured_object(), 3.5))
    assert evidence.has_flag(QualityFlag.OVEREXPOSED)
    assert not evidence.has_flag(QualityFlag.UNDEREXPOSED)


def test_overexposure_increases_highlight_clipping():
    image = textured_object()
    base = assess_capture_quality(image).illumination.highlight_clip_fraction
    bright = assess_capture_quality(
        adjust_exposure(image, 3.5)
    ).illumination.highlight_clip_fraction
    assert bright > base


def test_uniform_bright_fixture_is_flagged_overexposed():
    assert assess_capture_quality(uniform_bright()).has_flag(QualityFlag.OVEREXPOSED)


# --- contrast -----------------------------------------------------------------


def test_contrast_reduction_lowers_contrast_score():
    image = checkerboard()
    scores = [
        assess_capture_quality(reduce_contrast(image, f)).illumination.contrast_score
        for f in (1.0, 0.6, 0.25, 0.1)
    ]
    for previous, current in zip(scores, scores[1:]):
        assert current <= previous + 1e-9, f"contrast rose while compressing: {scores}"


def test_flat_field_is_flagged_low_contrast():
    assert assess_capture_quality(uniform_mid()).has_flag(QualityFlag.LOW_CONTRAST)


# --- required test 7: invalid input fails safely ------------------------------


@pytest.mark.parametrize(
    "bad,reason",
    [
        (None, "None"),
        ("not an image", "wrong type"),
        (np.zeros((0, 0, 3), dtype=np.uint8), "empty"),
        (np.zeros((16, 16), dtype=np.uint8), "single channel"),
        (np.zeros((16, 16, 4), dtype=np.uint8), "four channels"),
        (np.zeros((16, 16, 3), dtype=np.float32), "wrong dtype"),
    ],
)
def test_invalid_input_raises_image_validation_error(bad, reason):
    with pytest.raises(ImageValidationError):
        assess_capture_quality(bad)


def test_missing_file_raises_rather_than_returning_none(tmp_path):
    """cv2.imread returns None for a missing file; load_image must not pass that on."""
    with pytest.raises(ImageValidationError):
        load_image(tmp_path / "does_not_exist.jpg")


def test_undecodable_file_raises(tmp_path):
    junk = tmp_path / "not_really_an_image.jpg"
    junk.write_bytes(b"this is definitely not a JPEG")
    with pytest.raises(ImageValidationError):
        load_image(junk)


def test_validation_error_message_excludes_directory_path(tmp_path):
    """Error text names the file, never the containing absolute path."""
    missing = tmp_path / "absent.png"
    try:
        load_image(missing)
    except ImageValidationError as error:
        assert "absent.png" in str(error)
        assert str(tmp_path) not in str(error)
    else:  # pragma: no cover
        pytest.fail("expected ImageValidationError")


def test_poor_quality_is_not_an_error():
    """A terrible photograph is valid input, reported through flags."""
    evidence = assess_capture_quality(gaussian_blur(uniform_dark(), 8.0))
    assert evidence.quality_flags  # flagged
    assert evidence.image.width > 0  # but successfully analysed


def test_tiny_image_is_flagged_not_rejected():
    tiny = np.full((8, 8, 3), 128, dtype=np.uint8)
    assert assess_capture_quality(tiny).has_flag(QualityFlag.IMAGE_TOO_SMALL)


# --- input immutability -------------------------------------------------------


def test_assessment_does_not_modify_the_caller_array():
    image = textured_object()
    before = image.copy()
    assess_capture_quality(image)
    assert np.array_equal(image, before)


# --- policy separation --------------------------------------------------------


def test_thresholds_change_flags_without_changing_measurements():
    """Raw metrics are policy-independent; only flags respond to policy."""
    image = gaussian_blur(checkerboard(), 3.0)
    strict = assess_capture_quality(image, ThresholdPolicy(blur_variance_floor=1e6))
    lenient = assess_capture_quality(image, ThresholdPolicy(blur_variance_floor=0.0))

    assert strict.sharpness.laplacian_variance == lenient.sharpness.laplacian_variance
    assert strict.has_flag(QualityFlag.BLUR_RISK)
    assert not lenient.has_flag(QualityFlag.BLUR_RISK)
