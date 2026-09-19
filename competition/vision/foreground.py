"""Foreground isolation: finding the produce, not judging it.

**Scope boundary, stated first because the distinction matters.** This module
produces a *subject region* — where the produce is in the frame. It does **not**
produce a bruise mask, a rot mask, a lesion mask, or any surface-deterioration
map. Those are anomaly localisation, they do not exist yet, and nothing here may
be described as if they do.

The motivation is measured, not speculative. Phase 2 found the capture gate
rejected 96.7% of real research photographs, with highlight clipping firing on
80% of them, because these are product photographs with bright backgrounds:
whole-image clipping statistics were measuring the backdrop rather than the
fruit. No threshold value fixes that. The metrics have to be restricted to the
subject, which first requires finding it.

Three classical OpenCV methods are implemented and compared. No segmentation
network and no foundation model is used — the point is substantive classical
perception, and a learned segmenter would also reintroduce the dependency
weight Phase 2 worked to remove.

A mask is never trusted on production. `ForegroundEvidence.valid` is an explicit
state derived from geometric sanity checks, and an invalid mask must not be used
to replace whole-image evidence.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

import cv2
import numpy as np

FOREGROUND_VERSION = "phase2c-foreground-1.1.0"

# Morphological cleanup kernel as a fraction of the image's shorter side.
#
# Phase 2b used a fixed 7-pixel kernel, which was correct on the 256x256
# fixtures it was developed against and wrong everywhere else: 7 px is 2.7% of a
# 256 px frame but 0.36% of a 1920 px one, so on full-resolution photographs the
# opening step removed almost nothing and thresholding speckle survived as
# hundreds of tiny components. Measured on the Phase 2c-B corpus, the median
# component count was 19 on plain-background images and 69 on scenes, against a
# fragmentation guard of 12 - so 71% of samples were rejected as "fragmented"
# masks when the real defect was a resolution-dependent constant.
#
# The fraction below is 7/256 exactly, so behaviour at 256 px is unchanged and
# the constant is Phase 2b's own value expressed relatively rather than a new
# number chosen to make the guard pass.
CLEANUP_KERNEL_FRACTION = 7.0 / 256.0


class ForegroundMethod(str, Enum):
    """Closed set of implemented foreground methods."""

    BORDER_LAB_DISTANCE = "border_lab_distance"
    SATURATION_OTSU = "saturation_otsu"
    GRABCUT_RECT = "grabcut_rect"


class ForegroundError(ValueError):
    """Raised for input that cannot be segmented."""


@dataclass(frozen=True)
class ForegroundGuards:
    """Geometric sanity checks a mask must pass to be considered usable.

    These do not measure segmentation *accuracy* — no ground truth exists on
    real photographs. They detect the failure shapes a classical segmenter
    actually produces: grabbing the whole frame, grabbing nothing, grabbing the
    background instead of the subject, or shattering into fragments.

    PROVISIONAL: chosen from the geometry of the research photographs, not
    calibrated on deployment imagery.
    """

    min_foreground_fraction: float = 0.03
    max_foreground_fraction: float = 0.92
    max_border_contact_fraction: float = 0.60
    min_largest_component_dominance: float = 0.65
    max_component_count: int = 12
    min_bounding_box_side: int = 24
    # Components smaller than this fraction of the frame are thresholding
    # speckle, not pieces of a shattered subject, and counting them turns the
    # fragmentation guard into a resolution detector. Relative for the same
    # reason the cleanup kernel is: at 256x256 it is 65 px, at 1920x1440 it is
    # 2,765 px, and in both cases it means "too small to be part of the fruit".
    min_component_area_fraction: float = 0.001
    # Pixels eroded from the mask before sharpness aggregation, to keep the
    # mask boundary itself out of the gradient statistics. See `masked_pixels`.
    sharpness_erosion_px: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_GUARDS = ForegroundGuards()


@dataclass(frozen=True)
class ForegroundEvidence:
    """Geometry of an isolated subject region, plus an explicit validity state.

    The mask itself is not embedded — it is an array, identified here by content
    hash so evidence stays JSON-serialisable and free of filesystem paths.
    """

    input_sha256: str
    mask_sha256: str
    method: str
    valid: bool
    invalid_reasons: list[str]
    foreground_fraction: float
    largest_component_fraction: float
    border_contact_fraction: float
    component_count: int
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    solidity: float
    opencv_version: str
    pipeline_version: str = FOREGROUND_VERSION
    processing_ms: float = field(default=0.0)

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        data["bounding_box"] = list(self.bounding_box)
        if not include_timing:
            data.pop("processing_ms", None)
        return data

    def deterministic_payload(self) -> dict:
        return self.to_dict(include_timing=False)


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ForegroundError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise ForegroundError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ForegroundError(f"expected a 3-channel BGR image, got {image.shape!r}")
    if image.dtype != np.uint8:
        raise ForegroundError(f"expected uint8 image, got dtype {image.dtype}")


def mask_fingerprint(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(mask).tobytes())
    return digest.hexdigest()


def _image_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Shared post-processing
# --------------------------------------------------------------------------

def cleanup_kernel_size(shape: tuple[int, ...]) -> int:
    """Odd morphological kernel scaled to the image, never below 3 px."""
    shorter = min(shape[0], shape[1])
    size = int(round(shorter * CLEANUP_KERNEL_FRACTION)) | 1
    return max(3, size)


def _clean_mask(binary: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """Close gaps, drop speckle, then fill interior holes of the largest region.

    Produce is a solid convex-ish object; a mask of it should not be porous.

    The kernel scales with the image unless one is given explicitly, because a
    fixed pixel count means a different physical amount of cleanup at every
    resolution. See `CLEANUP_KERNEL_FRACTION`.
    """
    if kernel_size is None:
        kernel_size = cleanup_kernel_size(binary.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return opened
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

def segment_border_lab_distance(image: np.ndarray, border_px: int = 12) -> np.ndarray:
    """Model the background from a border ring, keep what differs from it.

    Product photography puts the subject away from the frame edge, so the border
    ring is a cheap and surprisingly robust background sample. Distance is taken
    in CIELAB, where Euclidean distance approximates perceptual difference far
    better than it does in BGR, and the threshold comes from Otsu on the
    distance map rather than from a constant.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    band = max(2, min(border_px, height // 4, width // 4))

    ring = np.concatenate([
        lab[:band, :, :].reshape(-1, 3),
        lab[-band:, :, :].reshape(-1, 3),
        lab[:, :band, :].reshape(-1, 3),
        lab[:, -band:, :].reshape(-1, 3),
    ])
    background = np.median(ring, axis=0)

    distance = np.linalg.norm(lab - background, axis=2)
    scaled = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean_mask(binary)


def segment_saturation_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu on HSV saturation: coloured produce against a neutral backdrop.

    Fails by construction on a saturated background or a desaturated subject,
    which is exactly what the validity guards are for.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean_mask(binary)


def segment_grabcut_rect(image: np.ndarray, inset: float = 0.08, iterations: int = 3) -> np.ndarray:
    """GrabCut seeded with a rectangle inset from the frame edge.

    More expensive than the alternatives and initialised on the same assumption
    (subject central, background at the edges), so it earns its place only if it
    measurably beats them.
    """
    height, width = image.shape[:2]
    margin_x = max(1, int(width * inset))
    margin_y = max(1, int(height * inset))
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))

    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, rect, background_model, foreground_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((height, width), np.uint8)

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _clean_mask(binary)


_METHODS = {
    ForegroundMethod.BORDER_LAB_DISTANCE: segment_border_lab_distance,
    ForegroundMethod.SATURATION_OTSU: segment_saturation_otsu,
    ForegroundMethod.GRABCUT_RECT: segment_grabcut_rect,
}

# Selected default, on measured trade-offs across 270 real photographs and 8
# synthetic fixtures:
#
#   saturation_otsu       validity 0.985   median  8.0 ms   synthetic IoU 0.905
#   border_lab_distance   validity 0.859   median 15.5 ms   synthetic IoU 0.905
#   grabcut_rect          validity 1.000   median  649 ms   synthetic IoU 0.892
#
# GrabCut is rejected on cost: 42x the median latency of the default and a p95
# of 3.3 seconds, for no measured quality advantage.
#
# Caveat recorded rather than hidden: "validity rate" measures how often the
# geometric guards pass, NOT segmentation correctness, and no ground-truth masks
# exist for real photographs. Saturation thresholding also assumes a coloured
# subject against a neutral background; it will fail on a saturated backdrop or
# desaturated produce. `border_lab_distance` models whatever the frame border
# actually contains and is the more general fallback for such scenes.
DEFAULT_METHOD = ForegroundMethod.SATURATION_OTSU


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------

def _border_contact_fraction(mask: np.ndarray) -> float:
    """Proportion of the frame perimeter the mask touches."""
    top, bottom = mask[0, :], mask[-1, :]
    left, right = mask[:, 0], mask[:, -1]
    perimeter = np.concatenate([top, bottom, left, right])
    return float(np.count_nonzero(perimeter) / perimeter.size)


def evaluate_mask(
    image: np.ndarray, mask: np.ndarray, method: str, guards: ForegroundGuards
) -> tuple[bool, list[str], dict]:
    """Apply geometric sanity checks and report every reason for rejection."""
    height, width = mask.shape[:2]
    total = float(height * width)
    foreground = float(np.count_nonzero(mask))
    fraction = foreground / total if total else 0.0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    all_areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    # Speckle is excluded from the *count* but not from the foreground area, so
    # fragmentation still means "the subject broke into pieces" rather than
    # "this image has more pixels for noise to appear in".
    minimum_area = guards.min_component_area_fraction * total
    significant = all_areas[all_areas >= minimum_area]
    component_count = int(significant.size)
    speckle_count = int(all_areas.size - significant.size)
    if all_areas.size:
        areas = all_areas
        largest_area = float(areas.max())
        largest_index = int(np.argmax(areas)) + 1
        largest_fraction = largest_area / foreground if foreground else 0.0
        x = int(stats[largest_index, cv2.CC_STAT_LEFT])
        y = int(stats[largest_index, cv2.CC_STAT_TOP])
        w = int(stats[largest_index, cv2.CC_STAT_WIDTH])
        h = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
        component = (labels == largest_index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
            solidity = float(largest_area / hull_area) if hull_area > 0 else 0.0
        else:
            solidity = 0.0
    else:
        largest_fraction = 0.0
        x = y = w = h = 0
        solidity = 0.0

    border = _border_contact_fraction(mask)

    reasons: list[str] = []
    if foreground == 0:
        reasons.append("MASK_EMPTY")
    if fraction < guards.min_foreground_fraction:
        reasons.append("FOREGROUND_TOO_SMALL")
    if fraction > guards.max_foreground_fraction:
        reasons.append("FOREGROUND_NEAR_FULL_FRAME")
    if border > guards.max_border_contact_fraction:
        reasons.append("EXCESSIVE_BORDER_CONTACT")
    if component_count > guards.max_component_count:
        reasons.append("MASK_FRAGMENTED")
    if component_count and largest_fraction < guards.min_largest_component_dominance:
        reasons.append("NO_DOMINANT_COMPONENT")
    if component_count and min(w, h) < guards.min_bounding_box_side:
        reasons.append("BOUNDING_BOX_TOO_SMALL")

    metrics = {
        "foreground_fraction": round(fraction, 6),
        "largest_component_fraction": round(largest_fraction, 6),
        "border_contact_fraction": round(border, 6),
        "component_count": component_count,
        "speckle_component_count": speckle_count,
        "cleanup_kernel_px": cleanup_kernel_size(mask.shape),
        "bounding_box": (x, y, w, h),
        "solidity": round(solidity, 6),
    }
    return (not reasons), reasons, metrics


def isolate_foreground(
    image: np.ndarray,
    method: ForegroundMethod | None = None,
    guards: ForegroundGuards | None = None,
) -> tuple[np.ndarray, ForegroundEvidence]:
    """Isolate the subject region and report its geometry and validity.

    Returns (mask, evidence). The mask is returned even when invalid, so callers
    can inspect a failure; `evidence.valid` is what decides whether it may be
    used. The source image is never modified.
    """
    method = method or DEFAULT_METHOD
    guards = guards or DEFAULT_GUARDS
    if not isinstance(method, ForegroundMethod):
        raise ForegroundError(f"unknown foreground method {method!r}")

    _validate_image(image)
    started = time.perf_counter()

    raw = _METHODS[method](image.copy())
    mask = _keep_largest_component(raw) if np.count_nonzero(raw) else raw

    # Guards are evaluated on the raw mask so fragmentation is detectable;
    # keeping only the largest component would hide it.
    valid, reasons, metrics = evaluate_mask(image, raw, method.value, guards)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    evidence = ForegroundEvidence(
        input_sha256=_image_fingerprint(image),
        mask_sha256=mask_fingerprint(mask),
        method=method.value,
        valid=valid,
        invalid_reasons=reasons,
        foreground_fraction=metrics["foreground_fraction"],
        largest_component_fraction=metrics["largest_component_fraction"],
        border_contact_fraction=metrics["border_contact_fraction"],
        component_count=metrics["component_count"],
        bounding_box=metrics["bounding_box"],
        solidity=metrics["solidity"],
        opencv_version=cv2.__version__,
        processing_ms=elapsed_ms,
    )
    return mask, evidence


def masked_pixels(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Erode a mask so its own boundary is excluded from statistics.

    Necessary for any gradient-based measurement. A binary mask edge is a step
    discontinuity; if the background is zeroed and a Laplacian taken over the
    result, that artificial edge contributes enormous gradient energy and the
    image appears sharper the more aggressively it was masked. Eroding first and
    aggregating a Laplacian computed on the *original* image avoids inventing
    that energy. See `competition.vision.quality.measure_sharpness`.
    """
    if erosion_px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    return cv2.erode(mask, kernel)
