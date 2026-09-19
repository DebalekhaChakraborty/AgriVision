"""Local specular-highlight (glare) evidence inside a foreground region.

Why whole-image highlight clipping is not a glare detector
----------------------------------------------------------
Phase 2c-B measured the gap: controlled glare at intensity 0.9 was blocked in
only 10% of cases, and the blocks that did happen came from the patch being
large enough to move a *global* clipped-pixel fraction, not from detecting it.
A specular highlight covering a few percent of a fruit never approaches a
sensible fraction limit, and raising that limit until it does would reject
correctly exposed photographs.

Glare is a **local photometric** artefact. It needs a local measurement.

What separates a highlight from a bright fruit
----------------------------------------------
"Bright pixels" is not the signal. A yellow banana in sunlight is bright; an
orange is bright and vividly coloured. Two properties distinguish a specular
reflection from a brightly coloured surface, and this module requires both:

1. **Local excess luminance.** A highlight is much brighter than its immediate
   surroundings. Subtracting a heavily blurred copy of L\\* leaves the smooth
   illumination gradient near zero and a hotspot standing well above it. A
   uniformly bright banana has a near-zero local excess everywhere.

2. **Desaturation relative to the subject's own chroma.** Specular reflection
   returns the illuminant's colour rather than the surface's, so a highlight
   washes out toward white while the fruit around it keeps its saturation. The
   comparison is against the ROI's *own* median saturation, not an absolute
   number, so a pale fruit is not penalised for being pale.

Both are relative to the region being measured. There is no absolute
white-pixel threshold anywhere in this module.

Scope
-----
Measurement only. Thresholds live in `LocalHighlightPolicy` and the decision
lives in the policy layer, so recalibrating never edits a measurement function.

**Claim boundary.** A detected highlight is a property of the *photograph* -
light reflecting off a surface. It says nothing about the produce, and a bright
region is never evidence of damage, bruising, deterioration or any other
condition.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import DEFAULT_GUARDS, masked_pixels

HIGHLIGHT_VERSION = "phase2d-local-highlights-1.0.0"

MIN_MEASURABLE_PIXELS = 256
_L_MAX = 255.0


@dataclass(frozen=True)
class LocalHighlightPolicy:
    """Thresholds for turning highlight measurements into a flag.

    PROVISIONAL until Phase 2d calibration writes a frozen instance.
    """

    # Sigmas of the blurs used to estimate local background luminance, as
    # fractions of the ROI's equivalent radius. Relative so the measurement does
    # not change meaning with image resolution - the Phase 2b lesson.
    #
    # Several scales, not one, because a single scale can only see highlights
    # smaller than itself: a blur wide enough to leave a small hotspot standing
    # proud will absorb a large one into its own background estimate. Measured
    # on the Phase 2c-B calibration images, a highlight covering 12% of the
    # subject was invisible to a single 0.25-radius estimator - local excess
    # inside it was near zero, with a response only at its rim. The excess is
    # taken as the elementwise maximum across scales.
    background_sigma_fractions: tuple[float, ...] = (0.10, 0.30, 0.90)
    # L* units above the local background before a pixel is a candidate.
    local_excess_l: float = 18.0
    # Candidate saturation must be at most this multiple of the ROI median.
    saturation_ratio: float = 0.75
    # A candidate must also sit in the upper part of the ROI's own L* range.
    luminance_percentile: float = 75.0
    # A severe highlight clips: it is flat white, so it has no local excess at
    # any scale in its interior. Pixels this bright and this desaturated are
    # admitted directly, which is what a blown specular patch actually looks
    # like. Absolute only in L*, where 255 is a hard physical ceiling.
    clipped_luminance: float = 244.0

    # --- flag thresholds ------------------------------------------------------
    # Fraction of the ROI the largest highlight component must occupy.
    min_component_fraction: float = 0.005
    # Total candidate fraction above which the flag fires regardless of shape.
    min_candidate_fraction: float = 0.02
    # Candidates must be concentrated, not scattered texture.
    min_spatial_concentration: float = 0.35
    # Pixels eroded from the ROI before measuring, as elsewhere.
    erosion_px: int = 5

    def to_dict(self) -> dict:
        data = asdict(self)
        data["background_sigma_fractions"] = list(self.background_sigma_fractions)
        return data

    def fingerprint(self) -> str:
        import json

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_HIGHLIGHT_POLICY = LocalHighlightPolicy()


# Above this sigma the Gaussian is computed on a downscaled copy instead. A
# direct blur builds a kernel about six sigma wide, so the widest scale on a
# 1920px photograph would need a ~3000px kernel - minutes per image, and
# pointlessly precise for an estimate of "roughly how bright is this
# neighbourhood". Downscaling by the same factor, blurring at the cap and
# resizing back is the standard equivalence, and it is accurate well within the
# tolerance of a threshold expressed in whole L* units.
_DIRECT_BLUR_SIGMA_CAP = 24.0


def _blur_at_scale(luminance: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= _DIRECT_BLUR_SIGMA_CAP:
        return cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    factor = sigma / _DIRECT_BLUR_SIGMA_CAP
    height, width = luminance.shape[:2]
    small = cv2.resize(
        luminance,
        (max(8, int(round(width / factor))), max(8, int(round(height / factor)))),
        interpolation=cv2.INTER_AREA,
    )
    blurred = cv2.GaussianBlur(
        small, (0, 0), sigmaX=_DIRECT_BLUR_SIGMA_CAP, sigmaY=_DIRECT_BLUR_SIGMA_CAP
    )
    return cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)


def _local_excess(luminance: np.ndarray, equivalent_radius: float,
                  policy: LocalHighlightPolicy) -> np.ndarray:
    """Luminance above the local background, maximised over several scales."""
    excess = None
    for fraction in policy.background_sigma_fractions:
        sigma = max(3.0, equivalent_radius * fraction)
        at_scale = luminance - _blur_at_scale(luminance, sigma)
        excess = at_scale if excess is None else np.maximum(excess, at_scale)
    return excess if excess is not None else np.zeros_like(luminance)


def _candidate_mask(image: np.ndarray, selection: np.ndarray, measured: int,
                    policy: LocalHighlightPolicy) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Shared candidate selection for the evidence and mask entry points."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    luminance = lab[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    equivalent_radius = float(np.sqrt(measured / np.pi))
    excess = _local_excess(luminance, equivalent_radius, policy)

    roi_luminance = luminance[selection]
    roi_saturation = saturation[selection]
    luminance_floor = float(np.percentile(roi_luminance, policy.luminance_percentile))
    median_saturation = float(np.median(roi_saturation))
    saturation_ceiling = max(1.0, median_saturation * policy.saturation_ratio)

    desaturated = saturation <= saturation_ceiling
    locally_bright = (excess >= policy.local_excess_l) & (luminance >= luminance_floor)
    blown = luminance >= policy.clipped_luminance

    candidates = selection & desaturated & (locally_bright | blown)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate_mask = cv2.morphologyEx(
        (candidates.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kernel
    )
    return candidate_mask, excess, median_saturation, float(saturation[candidates].mean()) if np.any(candidates) else 0.0


@dataclass(frozen=True)
class LocalHighlightEvidence:
    """Measured local-highlight geometry and photometry inside a ROI."""

    input_sha256: str
    foreground_mask_sha256: str
    measured_pixels: int
    candidate_fraction: float
    component_count: int
    largest_component_fraction: float
    max_local_luminance_excess: float
    mean_candidate_saturation_ratio: float
    spatial_concentration: float
    largest_component_compactness: float
    glare_flag: bool
    flag_reasons: list[str]
    policy_fingerprint: str
    opencv_version: str
    pipeline_version: str = HIGHLIGHT_VERSION
    processing_ms: float = field(default=0.0)

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        if not include_timing:
            data.pop("processing_ms", None)
        return data

    def deterministic_payload(self) -> dict:
        return self.to_dict(include_timing=False)


def _validate(image: np.ndarray, mask: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageValidationError("image must be a non-empty array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError(f"expected a 3-channel BGR image, got {image.shape!r}")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ImageValidationError("mask must be a 2-D array")
    if mask.shape[:2] != image.shape[:2]:
        raise ImageValidationError(
            f"mask shape {mask.shape[:2]} does not match image {image.shape[:2]}"
        )


def _hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def measure_local_highlights(
    image: np.ndarray,
    mask: np.ndarray,
    policy: LocalHighlightPolicy | None = None,
) -> LocalHighlightEvidence:
    """Find locally bright, locally desaturated regions inside the subject.

    The measurement runs on the unmodified image and the mask only selects which
    pixels are considered, exactly as in `quality.measure_sharpness`: zeroing the
    background and then blurring would drag the local background estimate toward
    zero at the subject's edge and manufacture a rim of false highlights.
    """
    policy = policy or DEFAULT_HIGHLIGHT_POLICY
    _validate(image, mask)
    started = time.perf_counter()

    interior = masked_pixels(mask, policy.erosion_px)
    selection = interior > 0
    measured = int(np.count_nonzero(selection))
    if measured < MIN_MEASURABLE_PIXELS:
        raise ImageValidationError(
            f"only {measured} pixels remain inside the eroded mask; "
            f"at least {MIN_MEASURABLE_PIXELS} are required"
        )

    candidate_mask, local_excess, median_saturation, candidate_saturation = _candidate_mask(
        image, selection, measured, policy
    )

    candidate_count = int(np.count_nonzero(candidate_mask))
    candidate_fraction = candidate_count / measured if measured else 0.0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    # Ignore specks below the same relative floor the foreground guards use.
    significant = areas[areas >= DEFAULT_GUARDS.min_component_area_fraction * measured]
    component_count = int(significant.size)

    if areas.size:
        largest_area = float(areas.max())
        largest_index = int(np.argmax(areas)) + 1
        largest_fraction = largest_area / measured
        concentration = largest_area / float(areas.sum())
        component = (labels == largest_index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            compactness = (
                float(4.0 * np.pi * cv2.contourArea(contour) / (perimeter ** 2))
                if perimeter > 0 else 0.0
            )
        else:
            compactness = 0.0
    else:
        largest_fraction = concentration = compactness = 0.0

    excess_inside = local_excess[selection]
    max_excess = float(excess_inside.max()) if excess_inside.size else 0.0
    saturation_ratio = (
        float(candidate_saturation / median_saturation) if median_saturation > 0 else 1.0
    )

    reasons: list[str] = []
    if largest_fraction >= policy.min_component_fraction and concentration >= policy.min_spatial_concentration:
        reasons.append("CONCENTRATED_HIGHLIGHT_COMPONENT")
    if candidate_fraction >= policy.min_candidate_fraction:
        reasons.append("LARGE_HIGHLIGHT_AREA")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return LocalHighlightEvidence(
        input_sha256=_hash(image),
        foreground_mask_sha256=_hash(mask),
        measured_pixels=measured,
        candidate_fraction=float(candidate_fraction),
        component_count=component_count,
        largest_component_fraction=float(largest_fraction),
        max_local_luminance_excess=max_excess,
        mean_candidate_saturation_ratio=saturation_ratio,
        spatial_concentration=float(concentration),
        largest_component_compactness=compactness,
        glare_flag=bool(reasons),
        flag_reasons=reasons,
        policy_fingerprint=policy.fingerprint(),
        opencv_version=cv2.__version__,
        processing_ms=elapsed_ms,
    )


def highlight_mask(
    image: np.ndarray, mask: np.ndarray, policy: LocalHighlightPolicy | None = None
) -> np.ndarray:
    """The candidate highlight mask itself, for inspection and overlap scoring.

    Always a subset of the eroded foreground: a highlight on the backdrop is not
    a highlight on the fruit, and reporting one as the other is exactly the
    whole-image mistake this module exists to avoid.
    """
    policy = policy or DEFAULT_HIGHLIGHT_POLICY
    _validate(image, mask)

    interior = masked_pixels(mask, policy.erosion_px)
    selection = interior > 0
    measured = int(np.count_nonzero(selection))
    if measured < MIN_MEASURABLE_PIXELS:
        return np.zeros(mask.shape, dtype=np.uint8)
    candidate_mask, _, _, _ = _candidate_mask(image, selection, measured, policy)
    return candidate_mask
