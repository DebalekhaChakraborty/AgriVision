"""OpenCV 5 capture enhancement operations.

These are the *actions* available to the decision layer. Each is a small, pure,
independently testable operation: it takes an image and explicit parameters,
returns a new image plus metadata, and never mutates its input. Nothing here
decides *whether* to act — that is `competition.agent.policy`.

Both operations work on the CIELAB L* channel and leave chromaticity untouched.
Equalising B, G and R independently shifts colour balance, which would corrupt
the colour analysis planned for later phases; luminance-only processing does
not.

**These operations cannot restore clipped information.** A pixel driven to 0 or
255 has lost its detail irreversibly, and brightening or equalising it only
redistributes what remains. The policy layer is responsible for refusing to
attempt remediation in that case rather than producing a plausible-looking image
that has invented nothing back.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

ENHANCEMENT_VERSION = "phase1b-enhancement-1.0.0"

_LUT_SIZE = 256
_L_CHANNEL_MAX = 255.0


class EnhancementError(ValueError):
    """Raised for invalid enhancement parameters or input."""


@dataclass(frozen=True)
class GammaParameters:
    """Power-law tone adjustment. gamma < 1 brightens, gamma > 1 darkens."""

    gamma: float

    def validate(self) -> None:
        if not isinstance(self.gamma, (int, float)):
            raise EnhancementError(f"gamma must be numeric, got {type(self.gamma).__name__}")
        if not 0.0 < float(self.gamma) <= 10.0:
            raise EnhancementError(f"gamma must be in (0, 10], got {self.gamma}")

    def to_dict(self) -> dict:
        return {"gamma": float(self.gamma)}


@dataclass(frozen=True)
class ClaheParameters:
    """Contrast Limited Adaptive Histogram Equalisation parameters."""

    clip_limit: float = 2.0
    tile_grid_size: int = 8

    def validate(self) -> None:
        if not 0.0 < float(self.clip_limit) <= 40.0:
            raise EnhancementError(f"clip_limit must be in (0, 40], got {self.clip_limit}")
        if not 1 <= int(self.tile_grid_size) <= 64:
            raise EnhancementError(
                f"tile_grid_size must be in [1, 64], got {self.tile_grid_size}"
            )

    def to_dict(self) -> dict:
        return {
            "clip_limit": float(self.clip_limit),
            "tile_grid_size": int(self.tile_grid_size),
        }


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise EnhancementError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise EnhancementError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise EnhancementError(f"expected a 3-channel BGR image, got shape {image.shape!r}")
    if image.dtype != np.uint8:
        raise EnhancementError(f"expected uint8 image, got dtype {image.dtype}")


def gamma_for_target_luminance(
    current: float, target: float, minimum: float, maximum: float
) -> float:
    """Derive the gamma that maps `current` mean luminance toward `target`.

    Not a magic constant. For normalised intensity the power law is
    ``out = in ** gamma``, so requiring ``target = current ** gamma`` gives

        gamma = log(target) / log(current)

    which is then clamped to the policy's permitted range. Degenerate inputs
    (current at 0 or 1, where the logarithm is undefined or the mapping cannot
    help) fall back to the clamp bounds rather than raising: the policy layer
    decides whether such a capture is remediable at all.
    """
    current = float(current)
    target = float(target)

    if not 0.0 < target < 1.0:
        raise EnhancementError(f"target luminance must be in (0, 1), got {target}")
    if current <= 0.0 or current >= 1.0:
        # Fully crushed or fully blown: no exponent recovers detail. Return the
        # bound in the helpful direction and let the policy refuse.
        return minimum if current <= 0.0 else maximum

    gamma = float(np.log(target) / np.log(current))
    return float(min(max(gamma, minimum), maximum))


def apply_gamma_correction(
    image: np.ndarray, parameters: GammaParameters
) -> tuple[np.ndarray, dict]:
    """Apply power-law tone adjustment to the L* channel.

    Returns (enhanced_image, metadata). The source image is not modified.
    """
    _validate_image(image)
    parameters.validate()

    started = time.perf_counter()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)

    # Precomputed LUT: exact, deterministic, and far faster than per-pixel power.
    index = np.arange(_LUT_SIZE, dtype=np.float64) / (_LUT_SIZE - 1)
    table = np.clip(
        np.power(index, float(parameters.gamma)) * (_LUT_SIZE - 1), 0, _LUT_SIZE - 1
    ).astype(np.uint8)
    adjusted = cv2.LUT(luminance, table)

    merged = cv2.merge((adjusted, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    metadata = {
        "operation": "gamma_correction",
        "parameters": parameters.to_dict(),
        "colour_space": "CIELAB L* only",
        "opencv_version": cv2.__version__,
        "enhancement_version": ENHANCEMENT_VERSION,
        "processing_ms": (time.perf_counter() - started) * 1000.0,
    }
    return result, metadata


def apply_clahe(
    image: np.ndarray, parameters: ClaheParameters
) -> tuple[np.ndarray, dict]:
    """Apply CLAHE to the L* channel only.

    Returns (enhanced_image, metadata). The source image is not modified.
    """
    _validate_image(image)
    parameters.validate()

    started = time.perf_counter()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, a_channel, b_channel = cv2.split(lab)

    grid = int(parameters.tile_grid_size)
    clahe = cv2.createCLAHE(
        clipLimit=float(parameters.clip_limit), tileGridSize=(grid, grid)
    )
    equalised = clahe.apply(luminance)

    merged = cv2.merge((equalised, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    metadata = {
        "operation": "clahe",
        "parameters": parameters.to_dict(),
        "colour_space": "CIELAB L* only",
        "opencv_version": cv2.__version__,
        "enhancement_version": ENHANCEMENT_VERSION,
        "processing_ms": (time.perf_counter() - started) * 1000.0,
    }
    return result, metadata
