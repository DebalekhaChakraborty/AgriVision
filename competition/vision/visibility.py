"""Subject-visibility evidence: how much of the subject the frame actually shows.

Naming, because it decides what may be claimed
----------------------------------------------
This module does **not** detect occlusion. Classical single-image OpenCV cannot
establish that a hand, a leaf or a price tag is covering a fruit: there is no
model of what the fruit's complete outline should have been, and a subject that
is genuinely that shape is indistinguishable from one that is partly hidden.

What it can measure is whether the visible subject region is *shaped like a
complete subject*: convex, unbroken, holeless, and inside the frame. When it is
not, something is wrong - the subject is cut off by the frame edge, split by a
foreign region, or bitten into. That is `SUBJECT_VISIBILITY_INSUFFICIENT`, and
it is a statement about the region, not about the cause.

So the evidence is never phrased as "hand detected" or "fruit occluded". A
low-solidity mask is reported as a low-solidity mask, with the interpretation
left to a human who can see the photograph.

Signals
-------
* **Border truncation** - the fraction of the subject's contour lying on the
  frame edge. A fruit running off the side of the frame is not fully visible.
* **Solidity** - contour area over convex-hull area. An occluder cutting into
  the silhouette removes area without shrinking the hull.
* **Hull fill** - visible area over convex-hull area, computed on the filled
  mask, so a bar crossing the middle registers even when it splits the subject.
* **Internal hole fraction** - filled area over raw area. An occluder entirely
  inside the silhouette punches a hole that filling restores.
* **Fragmentation** - a subject split into pieces by something crossing it.
* **Maximum convexity-defect depth**, relative to the equivalent radius: a deep
  bite is different from a slightly irregular outline.

Every ratio is dimensionless or normalised by the subject's own size, so none
of them changes meaning with resolution.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from competition.vision.evidence import ImageValidationError

VISIBILITY_VERSION = "phase2d-visibility-1.0.0"

MIN_SUBJECT_PIXELS = 256


@dataclass(frozen=True)
class VisibilityPolicy:
    """Thresholds separating an incomplete subject from an irregular one.

    PROVISIONAL until Phase 2d calibration writes a frozen instance.
    """

    # Below this, the visible region fills too little of its own convex hull.
    min_hull_fill: float = 0.82
    # Below this, the silhouette has been bitten into rather than merely lumpy.
    min_solidity: float = 0.88
    # Contour fraction on the frame edge above which the subject is cut off.
    max_border_truncation: float = 0.25
    # Holes inside the silhouette, as a fraction of the filled area.
    max_internal_hole_fraction: float = 0.06
    # Deepest convexity defect as a multiple of the equivalent radius.
    max_defect_depth_ratio: float = 0.45
    # A single-subject capture should yield one piece. Two means something
    # crossed it; the guards upstream already reject genuinely multi-subject
    # scenes before this runs.
    max_fragments: int = 1
    # Fragments smaller than this fraction of the subject are not counted.
    min_fragment_fraction: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        import json

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_VISIBILITY_POLICY = VisibilityPolicy()


@dataclass(frozen=True)
class VisibilityEvidence:
    """Geometry of the visible subject region. Never a claim about a cause."""

    foreground_mask_sha256: str
    subject_pixels: int
    hull_fill: float
    solidity: float
    border_truncation_fraction: float
    truncated_sides: int
    internal_hole_fraction: float
    max_defect_depth_ratio: float
    fragment_count: int
    largest_fragment_fraction: float
    visibility_sufficient: bool
    insufficiency_reasons: list[str]
    policy_fingerprint: str
    opencv_version: str
    pipeline_version: str = VISIBILITY_VERSION
    processing_ms: float = field(default=0.0)

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        if not include_timing:
            data.pop("processing_ms", None)
        return data

    def deterministic_payload(self) -> dict:
        return self.to_dict(include_timing=False)


def _hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _border_contour_fraction(mask: np.ndarray, contour_perimeter: float) -> tuple[float, int]:
    """Share of the *subject's own outline* lying on the frame edge.

    Normalised by the subject's perimeter rather than the frame's. Dividing by
    the frame perimeter would make the measure depend on how large the subject
    is: a small fruit running off the edge would score low simply because the
    frame is big, which is the resolution-style confound this project keeps
    running into.
    """
    height, width = mask.shape[:2]
    edges = (mask[0, :], mask[height - 1, :], mask[:, 0], mask[:, width - 1])
    touching = sum(int(np.count_nonzero(edge)) for edge in edges)
    sides = sum(1 for edge in edges if np.count_nonzero(edge))
    if contour_perimeter <= 0:
        return 0.0, sides
    return min(1.0, touching / contour_perimeter), sides


def measure_visibility(
    mask: np.ndarray,
    policy: VisibilityPolicy | None = None,
    unfilled_mask: np.ndarray | None = None,
) -> VisibilityEvidence:
    """Assess whether the visible subject region looks like a complete subject.

    `unfilled_mask` is the same segmentation with interior holes left open (see
    `foreground.segment_without_fill`). Interior evidence is measured on it
    because the filled mask, by construction, cannot show a hole - the fill step
    closes exactly the gap an occluder inside the silhouette creates. Without it
    `internal_hole_fraction` is structurally zero and the field is worthless.
    """
    policy = policy or DEFAULT_VISIBILITY_POLICY
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ImageValidationError("mask must be a 2-D array")
    started = time.perf_counter()

    binary = (mask > 0).astype(np.uint8) * 255
    subject_pixels = int(np.count_nonzero(binary))
    if subject_pixels < MIN_SUBJECT_PIXELS:
        raise ImageValidationError(
            f"only {subject_pixels} subject pixels; at least {MIN_SUBJECT_PIXELS} required"
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, dtype=np.int64)
    significant = areas[areas >= policy.min_fragment_fraction * subject_pixels]
    fragment_count = int(significant.size)
    largest_fragment_fraction = (
        float(areas.max() / subject_pixels) if areas.size else 0.0
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ImageValidationError("mask has no contour to measure")

    # Holes: filling the external contours restores anything fully enclosed.
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    filled_area = float(np.count_nonzero(filled))
    internal_hole_fraction = (
        float((filled_area - subject_pixels) / filled_area) if filled_area else 0.0
    )
    if unfilled_mask is not None:
        if unfilled_mask.shape[:2] != mask.shape[:2]:
            raise ImageValidationError("unfilled mask shape does not match the mask")
        inside = float(np.count_nonzero((unfilled_mask > 0) & (filled > 0)))
        internal_hole_fraction = (
            float((filled_area - inside) / filled_area) if filled_area else 0.0
        )

    # Hull geometry is taken over every significant piece together, so a subject
    # split in two is measured against the outline it would have had whole.
    points = np.vstack([c.reshape(-1, 2) for c in contours])
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    hull_fill = float(filled_area / hull_area) if hull_area > 0 else 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest_contour))
    largest_hull = cv2.convexHull(largest_contour)
    largest_hull_area = float(cv2.contourArea(largest_hull))
    solidity = float(largest_area / largest_hull_area) if largest_hull_area > 0 else 0.0

    equivalent_radius = float(np.sqrt(max(filled_area, 1.0) / np.pi))
    defect_ratio = 0.0
    if len(largest_contour) > 3:
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(largest_contour, hull_indices)
            except cv2.error:
                defects = None
            if defects is not None and len(defects):
                # Depth is fixed-point 1/256 pixels and is the last column.
                # OpenCV 5 returns (N, 4); OpenCV 4 returned (N, 1, 4). Reshaping
                # rather than indexing a fixed rank keeps this working on both.
                depths = np.asarray(defects).reshape(-1, 4)[:, 3]
                deepest = float(depths.max()) / 256.0
                defect_ratio = deepest / equivalent_radius if equivalent_radius else 0.0

    contour_perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    border_fraction, truncated_sides = _border_contour_fraction(binary, contour_perimeter)

    reasons: list[str] = []
    if hull_fill < policy.min_hull_fill:
        reasons.append("SUBJECT_REGION_DOES_NOT_FILL_ITS_OUTLINE")
    if solidity < policy.min_solidity:
        reasons.append("SILHOUETTE_INDENTED")
    if border_fraction > policy.max_border_truncation:
        reasons.append("SUBJECT_TRUNCATED_BY_FRAME")
    if internal_hole_fraction > policy.max_internal_hole_fraction:
        reasons.append("INTERNAL_REGION_EXCLUDED")
    if defect_ratio > policy.max_defect_depth_ratio:
        reasons.append("DEEP_SILHOUETTE_DEFECT")
    if fragment_count > policy.max_fragments:
        reasons.append("SUBJECT_SPLIT_INTO_PIECES")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return VisibilityEvidence(
        foreground_mask_sha256=_hash(mask),
        subject_pixels=subject_pixels,
        hull_fill=hull_fill,
        solidity=solidity,
        border_truncation_fraction=float(border_fraction),
        truncated_sides=int(truncated_sides),
        internal_hole_fraction=float(internal_hole_fraction),
        max_defect_depth_ratio=float(defect_ratio),
        fragment_count=fragment_count,
        largest_fragment_fraction=largest_fragment_fraction,
        visibility_sufficient=not reasons,
        insufficiency_reasons=reasons,
        policy_fingerprint=policy.fingerprint(),
        opencv_version=cv2.__version__,
        processing_ms=elapsed_ms,
    )
