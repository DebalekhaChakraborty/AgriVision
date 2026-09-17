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
