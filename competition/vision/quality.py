"""OpenCV 5 capture-quality assessment.

Answers one question: *is this capture visually suitable for reliable automated
inspection, and what measurable visual problems does it have?*

It deliberately answers nothing about the produce itself. Every measurement here
describes the photograph, not the fruit.

Shape of the API
----------------
`assess_capture_quality(image, policy)` takes a decoded BGR array and returns
evidence. It does no file I/O, holds no global state, and never mutates its
input. That is what makes it usable later as an agent tool
(`assess_capture_quality` -> policy decision -> `enhance_capture` ->
`assess_capture_quality` again) without restructuring.

File loading lives in `load_image` and is only used by the CLI.

Measurement definitions
-----------------------
**Sharpness — variance of the Laplacian.** The Laplacian is the sum of second
spatial derivatives; it responds to intensity transitions. A focused image has
many strong edges and therefore a wide spread of Laplacian responses; blurring
attenuates high frequencies and narrows that spread. We report the variance.
Direction: *higher means sharper*.

Known limitations, stated rather than hidden:

* Scale-dependent. The same scene at a different resolution yields a different
  variance, so a single global threshold is a coarse gate rather than a
  calibrated detector.
* Content-dependent. A genuinely smooth subject (a uniform ripe tomato) scores
  lower than a textured one at identical focus. Comparisons are only safe
  *within* an image across a degradation sweep, which is exactly how the Phase 1
  evaluation uses it.
* Cannot distinguish defocus blur from motion blur, and cannot distinguish
  either from a legitimately low-texture subject.
* Noise inflates it, so a noisy out-of-focus image can score misleadingly high.
* **Saturates at a noise floor.** Once blur removes essentially all detail the
  variance bottoms out (near 0.7 on the Phase 1 fixtures) and can then drift
  *upward* slightly with larger kernels, because border handling and uint8
  quantisation dominate what is left. Monotonicity therefore holds only above
  `ThresholdPolicy.sharpness_noise_floor`; below it the metric means "no detail"
  and nothing finer. This was measured, not assumed — see the Phase 1 note.

**Illumination — CIELAB L\\* channel.** L* approximates perceptual lightness far
better than a naive BGR mean, and unlike HSV's V (a plain channel maximum) it
accounts for how the three channels combine. All exposure statistics are taken
on L*.

* `mean_luminance`, `median_luminance` — central tendency, normalised to 0-1.
  Median is reported because a bright specular highlight skews the mean.
* `contrast_score` — the (p95 - p5) percentile spread of L*, normalised.
  Percentiles rather than standard deviation so a few extreme pixels do not
  dominate.
* `shadow_clip_fraction` / `highlight_clip_fraction` — fraction of pixels at or
  beyond the clipping bounds. Clipped pixels carry no recoverable detail, which
  is what distinguishes "dark" from "lost".

Limitations: clipping is measured on L*, so a single saturated colour channel in
an otherwise mid-toned pixel is not counted as clipped. Separately, shadow
clipping is a *severe* condition, not a synonym for "dim": multiplicative
underexposure must be drastic before pixels actually reach the shadow floor
(around gain 0.05 on the Phase 1 fixtures, not 0.15). A merely dim capture is
caught by `UNDEREXPOSED` via mean luminance; `SHADOW_CLIPPING` indicates
genuinely unrecoverable black crush.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.evidence import (
    PIPELINE_VERSION,
    IlluminationMetrics,
    ImageProperties,
    ImageValidationError,
    MeasurementScope,
    PerceptionEvidence,
    QualityFlag,
    SharpnessMetrics,
)
from competition.vision.foreground import DEFAULT_GUARDS, masked_pixels

_L_CHANNEL_MAX = 255.0

# Fewer usable pixels than this and a masked statistic is not worth computing.
MIN_MASKED_PIXELS = 256


def _selected(values: np.ndarray, selection: np.ndarray | None) -> np.ndarray:
    """Flatten either the whole array or just the selected pixels."""
    if selection is None:
        return values.reshape(-1)
    return values[selection > 0]


def _validate_mask(image: np.ndarray, mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise ImageValidationError(f"mask must be an ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ImageValidationError(f"mask must be 2-D, got shape {mask.shape!r}")
    if mask.shape[:2] != image.shape[:2]:
        raise ImageValidationError(
            f"mask shape {mask.shape[:2]} does not match image {image.shape[:2]}"
        )


def _validate(image: np.ndarray, policy: ThresholdPolicy) -> None:
    if image is None:
        raise ImageValidationError("image is None")
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ImageValidationError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError(
            f"expected a 3-channel BGR image, got shape {image.shape!r}"
        )
    if image.dtype != np.uint8:
        raise ImageValidationError(f"expected uint8 image, got dtype {image.dtype}")


def _content_hash(image: np.ndarray) -> str:
    """Identify an image by content, so no filesystem path enters the evidence."""
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def measure_sharpness(
    image: np.ndarray,
    policy: ThresholdPolicy,
    mask: np.ndarray | None = None,
    erosion_px: int | None = None,
) -> SharpnessMetrics:
    """Variance of the Laplacian, optionally restricted to a foreground region.

    The Laplacian is always computed on the **unmodified** grayscale image, and
    the mask only selects which responses are aggregated. Zeroing the background
    and then differentiating would manufacture a step edge at the mask boundary
    and inflate the variance — the more aggressively an image were masked, the
    sharper it would appear. The mask is additionally eroded before aggregation
    so that genuine subject/background edges, which sit just inside the mask,
    do not dominate either.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None:
        erosion = (
            erosion_px if erosion_px is not None else DEFAULT_GUARDS.sharpness_erosion_px
        )
        interior = masked_pixels(mask, erosion)
        values = laplacian[interior > 0]
        if values.size < MIN_MASKED_PIXELS:
            raise ImageValidationError(
                f"only {values.size} pixels remain inside the eroded mask; "
                f"at least {MIN_MASKED_PIXELS} are required"
            )
        variance = float(values.var())
    else:
        variance = float(laplacian.var())
    score = min(1.0, variance / policy.sharpness_reference_variance)
    return SharpnessMetrics(
        laplacian_variance=variance,
        sharpness_score=float(score),
    )


def measure_illumination(
    image: np.ndarray, policy: ThresholdPolicy, mask: np.ndarray | None = None
) -> IlluminationMetrics:
    """Exposure, contrast and clipping statistics on the CIELAB L* channel.

    With a mask, every statistic is computed over the selected pixels only. That
    is the whole point of the foreground work: a blown-out backdrop should not
    count as highlight clipping on the produce.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = _selected(lab[:, :, 0], mask)

    if luminance.size < (MIN_MASKED_PIXELS if mask is not None else 1):
        raise ImageValidationError(
            f"only {luminance.size} pixels selected; at least "
            f"{MIN_MASKED_PIXELS} are required for masked measurement"
        )

    total = float(luminance.size)
    low_p, high_p = np.percentile(
        luminance, [policy.contrast_low_percentile, policy.contrast_high_percentile]
    )

    shadow_clipped = float(np.count_nonzero(luminance <= policy.shadow_clip_value))
    highlight_clipped = float(np.count_nonzero(luminance >= policy.highlight_clip_value))

    return IlluminationMetrics(
        mean_luminance=float(luminance.mean() / _L_CHANNEL_MAX),
        median_luminance=float(np.median(luminance) / _L_CHANNEL_MAX),
        luminance_std=float(luminance.std() / _L_CHANNEL_MAX),
        contrast_score=float((high_p - low_p) / _L_CHANNEL_MAX),
        shadow_clip_fraction=shadow_clipped / total,
        highlight_clip_fraction=highlight_clipped / total,
    )


def derive_quality_flags(
    properties: ImageProperties,
    sharpness: SharpnessMetrics,
    illumination: IlluminationMetrics,
    policy: ThresholdPolicy,
) -> list[str]:
    """Apply threshold policy to raw metrics.

    Kept separate from measurement so recalibrating thresholds never touches a
    measurement function, and so a stored evidence record can in principle be
    re-flagged under a new policy without re-reading the image.
    """
    flags: list[QualityFlag] = []

    if (
        min(properties.width, properties.height) < policy.min_image_dimension
    ):
        flags.append(QualityFlag.IMAGE_TOO_SMALL)

    if sharpness.laplacian_variance < policy.blur_variance_floor:
        flags.append(QualityFlag.BLUR_RISK)

    if illumination.mean_luminance < policy.underexposed_mean_luminance:
        flags.append(QualityFlag.UNDEREXPOSED)
    elif illumination.mean_luminance > policy.overexposed_mean_luminance:
        flags.append(QualityFlag.OVEREXPOSED)

    if illumination.contrast_score < policy.low_contrast_limit:
        flags.append(QualityFlag.LOW_CONTRAST)

    if illumination.shadow_clip_fraction > policy.shadow_clip_fraction_limit:
        flags.append(QualityFlag.SHADOW_CLIPPING)

    if illumination.highlight_clip_fraction > policy.highlight_clip_fraction_limit:
        flags.append(QualityFlag.HIGHLIGHT_CLIPPING)

    return [flag.value for flag in flags]


def assess_capture_quality(
    image: np.ndarray,
    policy: ThresholdPolicy | None = None,
    mask: np.ndarray | None = None,
) -> PerceptionEvidence:
    """Assess whether a capture is suitable for automated inspection.

    One implementation serves both scopes. Passing a mask restricts every metric
    to the selected region and stamps the record `FOREGROUND_MASKED`; passing
    none measures the whole frame and stamps `WHOLE_IMAGE`. There is deliberately
    no second code path, so the two scopes cannot drift apart.

    Args:
        image: decoded BGR uint8 image. Not modified.
        policy: threshold policy; defaults to the provisional development policy.
        mask: optional 2-D uint8 foreground mask, non-zero inside the subject.

    Returns:
        PerceptionEvidence with raw metrics, derived flags and provenance.

    Raises:
        ImageValidationError: input is not an analysable image, the mask does not
            match it, or too few pixels remain to measure. Poor quality is never
            an error — it is reported through flags.
    """
    policy = policy or DEFAULT_POLICY
    started = time.perf_counter()

    _validate(image, policy)
    if mask is not None:
        _validate_mask(image, mask)

    # Defensive copy: guarantees the caller's array cannot be touched, which is
    # what makes this safe to chain as an agent tool.
    working = image.copy()

    height, width, channels = working.shape
    properties = ImageProperties(
        width=int(width),
        height=int(height),
        channels=int(channels),
        content_sha256=_content_hash(working),
    )

    sharpness = measure_sharpness(working, policy, mask)
    illumination = measure_illumination(working, policy, mask)
    flags = derive_quality_flags(properties, sharpness, illumination, policy)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return PerceptionEvidence(
        image=properties,
        sharpness=sharpness,
        illumination=illumination,
        quality_flags=flags,
        opencv_version=cv2.__version__,
        pipeline_version=PIPELINE_VERSION,
        threshold_policy_fingerprint=policy.fingerprint(),
        threshold_policy_status=policy.status,
        measurement_scope=(
            MeasurementScope.FOREGROUND_MASKED.value
            if mask is not None
            else MeasurementScope.WHOLE_IMAGE.value
        ),
        processing_ms=elapsed_ms,
    )


def load_image(path: str | Path) -> np.ndarray:
    """Decode an image from disk. Used by the CLI only.

    Raises ImageValidationError rather than returning None, because OpenCV's
    `imread` silently returns None for a missing or undecodable file and that
    failure mode should not propagate.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageValidationError(f"not a file: {path.name}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(f"could not decode as an image: {path.name}")
    return image


def main(argv: list[str] | None = None) -> int:
    """CLI implementation. Invoked via `python -m competition.vision`."""
    parser = argparse.ArgumentParser(
        prog="python -m competition.vision",
        description="OpenCV 5 capture-quality assessment. Emits JSON evidence.",
    )
    parser.add_argument("image", help="path to an image file")
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="omit processing_ms, producing byte-identical output across runs",
    )
    args = parser.parse_args(argv)

    try:
        image = load_image(args.image)
    except ImageValidationError as error:
        # Message deliberately carries only the file name, never the full path,
        # so CLI output stays free of machine-specific absolute paths.
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2

    evidence = assess_capture_quality(image)
    print(evidence.to_json(include_timing=not args.no_timing))
    return 0


# --- public wrappers for the agent tool layer --------------------------------
#
# Phase 3 needs to validate an image and hash it *without* measuring it: the
# orchestrator checks the input is usable before it spends anything on
# segmentation. These delegate to the functions the module already uses, so the
# agent's notion of "valid" and "which image is this" cannot drift from the
# evidence layer's.


def validate_image(image: np.ndarray, policy: ThresholdPolicy | None = None) -> None:
    """Raise `ImageValidationError` if this array is not a usable BGR image."""
    _validate(image, policy or DEFAULT_POLICY)


def image_content_sha256(image: np.ndarray) -> str:
    """Content hash identical to the one carried in `PerceptionEvidence`."""
    return _content_hash(image)
