"""Programmatically generated test images.

Every fixture is synthesised from code. No photograph enters the repository, so
the committed test suite carries no third-party licensing obligation at all.
This matters here specifically: the Kaggle source dataset has an Unknown licence
with no redistribution grant, and FruitVision is CC BY-NC-ND 4.0. Neither may be
committed, and neither is needed to test capture-quality measurement.

Fixtures are deterministic — identical output on every call and every machine —
so they can anchor determinism tests.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SIZE = (256, 256)  # (height, width)


def checkerboard(
    size: tuple[int, int] = DEFAULT_SIZE, square: int = 16
) -> np.ndarray:
    """Edge-rich high-contrast pattern: the sharpness reference fixture.

    Maximal high-frequency content, so Laplacian variance starts high and has
    room to fall across a blur sweep.
    """
    height, width = size
    board = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, square):
        for x in range(0, width, square):
            if ((x // square) + (y // square)) % 2 == 0:
                board[y : y + square, x : x + square] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def uniform(value: int, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Flat field of a single intensity. No edges, no contrast."""
    height, width = size
    return np.full((height, width, 3), int(value), dtype=np.uint8)


def uniform_dark(size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    return uniform(8, size)


def uniform_bright(size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    return uniform(250, size)


def uniform_mid(size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    return uniform(128, size)


def horizontal_gradient(size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Full-range left-to-right ramp: wide tonal spread, very few edges."""
    height, width = size
    ramp = np.linspace(0, 255, width, dtype=np.float64)
    gray = np.tile(ramp, (height, 1)).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def textured_object(size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """A rounded textured object on a plain background.

    Closer in structure to a produce photograph than a checkerboard — a single
    convex subject, mid-tone background, texture on the subject only — while
    remaining entirely synthetic. Useful for checking that metrics behave
    sensibly on something less artificial than a test pattern.
    """
    height, width = size
    image = np.full((height, width, 3), 90, dtype=np.uint8)

    centre = (width // 2, height // 2)
    radius = int(min(height, width) * 0.35)
    cv2.circle(image, centre, radius, (60, 140, 200), thickness=-1)

    # Deterministic speckle texture confined to the object.
    rng = np.random.default_rng(20260917)
    speckle = rng.integers(-28, 29, size=(height, width, 3), dtype=np.int16)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, thickness=-1)
    blended = image.astype(np.int16) + speckle * (mask[:, :, None] > 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def all_fixtures() -> dict[str, np.ndarray]:
    """Named fixture set used by the Phase 1 baseline evaluation."""
    return {
        "checkerboard": checkerboard(),
        "gradient": horizontal_gradient(),
        "textured_object": textured_object(),
        "uniform_mid": uniform_mid(),
    }


# ---------------------------------------------------------------------------
# Foreground fixtures with programmatically known ground-truth masks.
#
# Because the mask is constructed rather than annotated, IoU and Dice are
# genuinely available here. They measure the segmenter against a synthetic
# target and say nothing about its behaviour on real produce photography.
# ---------------------------------------------------------------------------


def _blank(size: tuple[int, int], value: int) -> np.ndarray:
    height, width = size
    return np.full((height, width, 3), int(value), dtype=np.uint8)


def _disc_mask(size: tuple[int, int], centre: tuple[int, int], radius: int) -> np.ndarray:
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, thickness=-1)
    return mask


def _apply_subject(
    background: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int],
    speckle: int = 0, seed: int = 20260918,
) -> np.ndarray:
    image = background.copy()
    image[mask > 0] = colour
    if speckle:
        rng = np.random.default_rng(seed)
        noise = rng.integers(-speckle, speckle + 1, size=image.shape, dtype=np.int16)
        blended = image.astype(np.int16) + noise * (mask[:, :, None] > 0)
        image = np.clip(blended, 0, 255).astype(np.uint8)
    return image


def dark_subject_on_bright_background(size: tuple[int, int] = DEFAULT_SIZE):
    """The Phase 2 failure case: a blown-out backdrop around a darker subject."""
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.3))
    image = _apply_subject(_blank(size, 252), mask, (40, 55, 70), speckle=18)
    return image, mask


def bright_subject_on_dark_background(size: tuple[int, int] = DEFAULT_SIZE):
    """Background value chosen so L* actually reaches the shadow-clip floor.

    BGR 6 maps to L* 5, which sits just *above* the L* <= 4 clip threshold, so a
    brighter-looking choice would not exhibit the phenomenon this fixture exists
    to demonstrate.
    """
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.3))
    image = _apply_subject(_blank(size, 2), mask, (90, 190, 235), speckle=18)
    return image, mask


def textured_subject_neutral_background(size: tuple[int, int] = DEFAULT_SIZE):
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.32))
    image = _apply_subject(_blank(size, 150), mask, (60, 140, 200), speckle=30)
    return image, mask


def subject_touching_one_border(size: tuple[int, int] = DEFAULT_SIZE):
    height, width = size
    radius = int(min(size) * 0.3)
    mask = _disc_mask(size, (radius - 8, height // 2), radius)
    image = _apply_subject(_blank(size, 245), mask, (55, 130, 195), speckle=18)
    return image, mask


def multiple_disconnected_regions(size: tuple[int, int] = DEFAULT_SIZE):
    """Several subject-like blobs: the fragmentation guard should notice."""
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(min(size) * 0.11)
    for cx, cy in ((width // 4, height // 4), (3 * width // 4, height // 4),
                   (width // 4, 3 * height // 4), (3 * width // 4, 3 * height // 4)):
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    image = _apply_subject(_blank(size, 245), mask, (60, 140, 200), speckle=18)
    return image, mask


def small_subject(size: tuple[int, int] = DEFAULT_SIZE):
    """Subject well below the minimum area guard."""
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.05))
    image = _apply_subject(_blank(size, 245), mask, (60, 140, 200), speckle=12)
    return image, mask


def subject_with_distractor(size: tuple[int, int] = DEFAULT_SIZE):
    """A dominant subject plus a smaller clutter object."""
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.3))
    distractor = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(distractor, (8, 8), (8 + int(width * 0.14), 8 + int(height * 0.14)), 255, -1)
    image = _apply_subject(_blank(size, 245), mask, (60, 140, 200), speckle=20)
    image[distractor > 0] = (120, 95, 80)
    return image, mask


def low_contrast_subject(size: tuple[int, int] = DEFAULT_SIZE):
    """Subject barely distinguishable from its background."""
    height, width = size
    mask = _disc_mask(size, (width // 2, height // 2), int(min(size) * 0.3))
    image = _apply_subject(_blank(size, 150), mask, (146, 152, 156), speckle=6)
    return image, mask


def foreground_fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named (image, ground-truth mask) pairs for segmentation evaluation."""
    return {
        "dark_on_bright": dark_subject_on_bright_background(),
        "bright_on_dark": bright_subject_on_dark_background(),
        "textured_neutral": textured_subject_neutral_background(),
        "touching_border": subject_touching_one_border(),
        "multiple_regions": multiple_disconnected_regions(),
        "small_subject": small_subject(),
        "with_distractor": subject_with_distractor(),
        "low_contrast": low_contrast_subject(),
    }
