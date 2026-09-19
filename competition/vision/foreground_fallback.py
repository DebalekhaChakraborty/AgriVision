"""Bounded fallback ladder for foreground isolation.

Phase 2d measured the cost of the primary segmenter honestly: 12 of 29
independent validation photographs produced a mask that passed its guards. The
Phase 3 orchestrator handles that safely by refusing to measure a region it
cannot attribute, but 41% assessability is a poor foundation for anything.

This module adds a fallback ladder. It does **not** replace `saturation_otsu`,
which remains the primary and whose code path is untouched — every Phase 2b,
2c-B and 2d result depends on those exact masks, and changing them would
invalidate a calibration rather than improve it.

What the measurements actually showed
-------------------------------------

Diagnosing 65 calibration images split the problem in two, and the split matters
more than the headline number:

    CLEAN_BASE     (single subject)   30/35 valid   =  85.7%
    NATURAL_SCENE  (markets, piles)    9/30 valid   =  30.0%

The dominant failure is one shape: 20 of 26 failures had a foreground fraction
above 0.68 with a single component covering essentially the whole frame. That is
Otsu finding no saturation bimodality because the *entire scene* is colourful.

On CLEAN_BASE — the domain the product actually targets, one fruit photographed
for inspection — the ladder recovers all five failures, and all five masks were
adjudicated correct or acceptable by eye.

On NATURAL_SCENE it "recovers" 19 of 21, and most of those masks are an
arbitrary crop of a fruit pile: compact, dominant, not touching the border, and
therefore indistinguishable from a real subject to any geometric guard. A market
stall does not contain a single subject, so there is no correct mask to find.

Two discriminators were built and both failed
---------------------------------------------

Recorded because a later reader will otherwise try them again.

**Boundary–edge support** — the fraction of the mask outline lying on a strong
image gradient. The idea was that a real object boundary sits on a gradient
ridge while a threshold artifact cut through a pile does not. Measured the wrong
way round: 0.760 median for the arbitrary crops against 0.573 for the correct
clean-base masks. In a dense pile there is texture everywhere, so any cut lands
on an edge.

**Subject multiplicity** — a distance-transform disc ratio, on the theory that
one fruit gives one inscribed disc and a pile gives many. It ranks the clearest
single subjects first (1.09, 1.10, 1.10) and the sprawling market crops last
(3.03, 3.17, 3.59), but it is confounded by shape: a bunch of bananas scores
1.88 and a legitimate three-apple product shot scores worse than a pile. It
measures elongation as much as multiplicity.

Neither is used. No threshold was fitted to make either look better than it is.

What this module therefore claims
---------------------------------

A recovered mask is **not** as trustworthy as a primary mask, and the honest
response is not to hide that but to record it. Every mask carries the method
that produced it and whether it came from the primary or a fallback, so a
recovered mask reaches the orchestrator as PROVISIONAL evidence rather than
CALIBRATED. Acceptance is never silent, which is the property that was asked
for; it is not the same as acceptance always being correct, which no
measurement here supports.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum

import cv2
import numpy as np

from competition.vision.foreground import (
    DEFAULT_GUARDS,
    DEFAULT_METHOD,
    ForegroundError,
    ForegroundEvidence,
    ForegroundGuards,
    ForegroundMethod,
    _clean_mask,
    _image_fingerprint,
    _keep_largest_component,
    _validate_image,
    evaluate_mask,
    isolate_foreground,
    mask_fingerprint,
)

FALLBACK_VERSION = "phase3b-fallback-1.0.0"

# Fallback segmentation runs on a downscaled copy. GrabCut is quadratic-ish in
# pixel count and Phase 2b rejected it outright at 649 ms median; at 480 px it
# is affordable, and its mask is no less precise than the resolution it was
# computed at. Cleanup happens at working resolution too, which costs 2.9 ms
# instead of 31 ms for an IoU of 0.974 against full-resolution cleanup.
#
# The primary path does NOT use this. Its masks must stay byte-identical.
WORKING_LONG_EDGE = 480

# One GrabCut iteration, not three. Measured on the 26 calibration failures:
# one iteration recovered 24, three recovered 22, and cost 909 ms against
# 1263 ms. More refinement made it slightly worse and substantially slower.
GRABCUT_ITERATIONS = 1
GRABCUT_INSET = 0.08


class MaskProvenance(str, Enum):
    """Where a mask came from. Carried into the trace, never inferred."""

    PRIMARY = "PRIMARY"
    FALLBACK = "FALLBACK"
    NONE = "NONE"


class FallbackMethod(str, Enum):
    """Candidates tried, in order, when the primary fails."""

    CHROMA_DISTANCE = "chroma_distance"
    BORDER_LAB_DISTANCE = "border_lab_distance"
    GRABCUT_SCALED = "grabcut_scaled"


# Ordered cheapest-first among those that measurably recover anything.
# `lightness_otsu` was built, measured and dropped: it agreed with the primary
# at a median IoU of 0.343 on images where the primary succeeds, meaning it
# finds something else — bright regions rather than the subject.
FALLBACK_LADDER: tuple = (
    FallbackMethod.CHROMA_DISTANCE,
    FallbackMethod.BORDER_LAB_DISTANCE,
    FallbackMethod.GRABCUT_SCALED,
)


def _downscale(image: np.ndarray, long_edge: int = WORKING_LONG_EDGE):
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    if scale >= 1.0:
        return image, 1.0
    small = cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def _restore(mask: np.ndarray, shape: tuple) -> np.ndarray:
    height, width = shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def segment_chroma_distance(image: np.ndarray, border_px: int = 12) -> np.ndarray:
    """Border-background distance in the a*/b* chroma plane only.

    `border_lab_distance` includes L*, so a subject that differs from its
    background in colour but not in brightness is partly cancelled by the
    lightness term. Dropping L* makes the comparison purely chromatic.

    Of the calibration images where the primary succeeds, this agrees with it at
    a median IoU of 0.821 — the closest of any alternate, which is expected:
    both are asking a colour question, and that similarity is why it is tried
    first.
    """
    working, _ = _downscale(image)
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    band = max(2, min(border_px, height // 4, width // 4))
    ring = np.concatenate([
        lab[:band, :, :].reshape(-1, 3), lab[-band:, :, :].reshape(-1, 3),
        lab[:, :band, :].reshape(-1, 3), lab[:, -band:, :].reshape(-1, 3),
    ])
    background_chroma = np.median(ring, axis=0)[1:]
    distance = np.linalg.norm(lab[:, :, 1:] - background_chroma, axis=2)
    scaled = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _restore(_clean_mask(binary), image.shape)


def segment_border_lab_distance_scaled(image: np.ndarray) -> np.ndarray:
    """The existing Lab-distance method, run at working resolution."""
    from competition.vision.foreground import segment_border_lab_distance

    working, _ = _downscale(image)
    return _restore(segment_border_lab_distance(working), image.shape)


def segment_grabcut_scaled(
    image: np.ndarray,
    long_edge: int = WORKING_LONG_EDGE,
    iterations: int = GRABCUT_ITERATIONS,
    inset: float = GRABCUT_INSET,
) -> np.ndarray:
    """GrabCut seeded with an inset rectangle, at working resolution.

    Phase 2b rejected GrabCut on cost alone — 649 ms median, 3.3 s p95 — while
    recording that it was the only method to pass the guards on every research
    photograph. Downscaling addresses the one objection without touching the
    reason it was attractive.

    It assumes the subject is central and the frame edge is background. That
    assumption is what makes it useful on a single-subject capture and what
    makes its output meaningless on a market stall, where the rectangle simply
    carves a block out of the middle of a pile.
    """
    working, _ = _downscale(image, long_edge)
    height, width = working.shape[:2]
    margin_x = max(1, int(width * inset))
    margin_y = max(1, int(height * inset))
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x),
            max(1, height - 2 * margin_y))

    mask = np.zeros((height, width), np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(working, mask, rect, background_model, foreground_model,
                    iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros(image.shape[:2], np.uint8)

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return _restore(_clean_mask(binary.astype(np.uint8)), image.shape)


_FALLBACKS = {
    FallbackMethod.CHROMA_DISTANCE: segment_chroma_distance,
    FallbackMethod.BORDER_LAB_DISTANCE: segment_border_lab_distance_scaled,
    FallbackMethod.GRABCUT_SCALED: segment_grabcut_scaled,
}


@dataclass(frozen=True)
class FallbackAttempt:
    """One candidate tried, and what it produced. Recorded whether or not used."""

    method: str
    valid: bool
    invalid_reasons: list
    foreground_fraction: float
    processing_ms: float = 0.0

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        if not include_timing:
            data.pop("processing_ms", None)
        return data


@dataclass(frozen=True)
class FallbackEvidence:
    """Which method produced the accepted mask, and what else was tried.

    `provenance` is the field that matters downstream. A FALLBACK mask is not
    interchangeable with a PRIMARY one and the orchestrator must be able to say
    so without re-deriving it.
    """

    provenance: str
    accepted_method: str
    attempts: list = field(default_factory=list)
    ladder_version: str = FALLBACK_VERSION
    processing_ms: float = 0.0

    @property
    def used_fallback(self) -> bool:
        return self.provenance == MaskProvenance.FALLBACK.value

    def to_dict(self, include_timing: bool = True) -> dict:
        data = {
            "provenance": self.provenance,
            "accepted_method": self.accepted_method,
            "ladder_version": self.ladder_version,
            "attempts": [a.to_dict(include_timing) for a in self.attempts],
        }
        if include_timing:
            data["processing_ms"] = self.processing_ms
        return data

    def deterministic_payload(self) -> dict:
        return self.to_dict(include_timing=False)


def isolate_foreground_with_fallback(
    image: np.ndarray,
    method: ForegroundMethod | None = None,
    guards: ForegroundGuards | None = None,
    ladder: tuple | None = None,
) -> tuple[np.ndarray, ForegroundEvidence, FallbackEvidence]:
    """Try the primary; on failure, walk the ladder and take the first valid mask.

    The guards are **not** relaxed for fallbacks. A recovered mask must pass
    exactly the checks the primary would have had to pass, so this widens the
    set of images that yield a mask without widening what counts as a mask.

    That is a real but partial guarantee, and the limit is worth stating at the
    call site: the guards test geometry, not correctness. They cannot tell a
    single apple from a compact block carved out of a heap of apples, because
    the two have the same geometry. What stops that from being *silent* is the
    provenance returned here, not the guards.
    """
    guards = guards or DEFAULT_GUARDS
    started = time.perf_counter()

    mask, evidence = isolate_foreground(image, method or DEFAULT_METHOD, guards)
    attempts = [FallbackAttempt(
        method=evidence.method, valid=evidence.valid,
        invalid_reasons=list(evidence.invalid_reasons),
        foreground_fraction=evidence.foreground_fraction,
        processing_ms=evidence.processing_ms,
    )]

    if evidence.valid:
        return mask, evidence, FallbackEvidence(
            provenance=MaskProvenance.PRIMARY.value,
            accepted_method=evidence.method,
            attempts=attempts,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    for candidate in (ladder if ladder is not None else FALLBACK_LADDER):
        attempt_started = time.perf_counter()
        try:
            raw = _FALLBACKS[candidate](image.copy())
        except cv2.error:
            raw = np.zeros(image.shape[:2], np.uint8)
        attempt_ms = (time.perf_counter() - attempt_started) * 1000.0
        valid, reasons, metrics = evaluate_mask(image, raw, candidate.value, guards)
        attempts.append(FallbackAttempt(
            method=candidate.value, valid=valid, invalid_reasons=list(reasons),
            foreground_fraction=metrics["foreground_fraction"],
            processing_ms=attempt_ms,
        ))
        if not valid:
            continue

        recovered = _keep_largest_component(raw) if np.count_nonzero(raw) else raw
        recovered_evidence = ForegroundEvidence(
            input_sha256=_image_fingerprint(image),
            mask_sha256=mask_fingerprint(recovered),
            method=candidate.value,
            valid=True,
            invalid_reasons=[],
            foreground_fraction=metrics["foreground_fraction"],
            largest_component_fraction=metrics["largest_component_fraction"],
            border_contact_fraction=metrics["border_contact_fraction"],
            component_count=metrics["component_count"],
            bounding_box=metrics["bounding_box"],
            solidity=metrics["solidity"],
            opencv_version=cv2.__version__,
            processing_ms=attempt_ms,
        )
        return recovered, recovered_evidence, FallbackEvidence(
            provenance=MaskProvenance.FALLBACK.value,
            accepted_method=candidate.value,
            attempts=attempts,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    # Nothing worked. The original failure is returned unchanged, so a caller
    # that ignores fallback evidence entirely behaves exactly as before.
    return mask, evidence, FallbackEvidence(
        provenance=MaskProvenance.NONE.value,
        accepted_method="",
        attempts=attempts,
        processing_ms=(time.perf_counter() - started) * 1000.0,
    )
