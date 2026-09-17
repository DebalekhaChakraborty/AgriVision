"""Deterministic controlled degradation.

This is measurement apparatus, not training augmentation. The distinction
matters and drives the design.

Augmentation wants variety and realism, and randomness is a feature. Here we
want the opposite: a *known* degradation of a *known* strength applied to a
known image, so that the perception layer's response can be attributed to that
degradation and nothing else. The chain we are building is:

    known degradation -> measured OpenCV response -> decision/action evaluation

Every transform is therefore parameterised by an explicit level, is reproducible
from `(image, spec)` alone, and never touches the source array.

The parameter sweeps double as ground truth. Because the blur sigma is known, a
sweep tells us whether the sharpness metric is monotone in blur without anyone
labelling an image as "blurry" — and later it tells us whether the agent
*should* have flagged, remediated or escalated, which is the substrate the
evaluation plan is built on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

DEGRADATION_VERSION = "phase1-degradation-1.0.0"

# Explicit swept levels. Named constants rather than inline literals so the
# evaluation, the tests and the documentation all reference the same ladder.
BLUR_SIGMAS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
UNDEREXPOSURE_GAINS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.25, 0.15)
OVEREXPOSURE_GAINS: tuple[float, ...] = (1.0, 1.3, 1.6, 2.0, 2.6, 3.5)
CONTRAST_FACTORS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.25, 0.1)
GLARE_INTENSITIES: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.9)
OCCLUSION_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5)


class DegradationError(ValueError):
    """Raised for an unknown or invalid degradation specification."""


@dataclass(frozen=True)
class DegradationSpec:
    """A single, fully reproducible degradation.

    `seed` only affects transforms with a spatial placement choice (glare,
    occlusion). Deterministic transforms ignore it, but it is always recorded so
    a specification round-trips without special cases.
    """

    kind: str
    level: float
    seed: int = 0
    notes: str = field(default="")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["degradation_version"] = DEGRADATION_VERSION
        return data

    @property
    def is_identity(self) -> bool:
        return self.kind == "none" or (
            self.kind in _IDENTITY_LEVELS and self.level == _IDENTITY_LEVELS[self.kind]
        )


# Level at which each transform is a no-op, used for `is_identity` and for
# asserting sweep ladders start from an undegraded reference.
_IDENTITY_LEVELS: dict[str, float] = {
    "gaussian_blur": 0.0,
    "underexpose": 1.0,
    "overexpose": 1.0,
    "reduce_contrast": 1.0,
    "glare": 0.0,
    "occlude": 0.0,
}


def _kernel_size_for_sigma(sigma: float) -> int:
    """Odd kernel wide enough to contain the Gaussian (~3 sigma each side)."""
    size = int(round(sigma * 6.0)) | 1
    return max(3, size)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Defocus-like blur. sigma == 0 returns an unmodified copy."""
    if sigma < 0:
        raise DegradationError(f"blur sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return image.copy()
    ksize = _kernel_size_for_sigma(sigma)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def adjust_exposure(image: np.ndarray, gain: float) -> np.ndarray:
    """Multiplicative exposure change with saturation.

    gain < 1 underexposes, gain > 1 overexposes. `convertScaleAbs` saturates at
    0 and 255, which is the point: overexposure must genuinely clip so that
    highlight-clipping measurement has something real to detect.
    """
    if gain <= 0:
        raise DegradationError(f"exposure gain must be > 0, got {gain}")
    if gain == 1.0:
        return image.copy()
    return cv2.convertScaleAbs(image, alpha=gain, beta=0.0)


def reduce_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Compress the tonal range toward mid-grey, preserving mean brightness.

    out = (in - 128) * factor + 128, so factor == 1 is identity and factor == 0
    collapses to flat grey.
    """
    if not 0.0 <= factor <= 1.0:
        raise DegradationError(f"contrast factor must be in [0, 1], got {factor}")
    if factor == 1.0:
        return image.copy()
    return cv2.convertScaleAbs(image, alpha=factor, beta=128.0 * (1.0 - factor))


def add_glare(image: np.ndarray, intensity: float, seed: int = 0) -> np.ndarray:
    """Add a soft specular highlight.

    Built as a blurred filled circle used as an additive mask, which produces
    the smooth falloff of a real highlight rather than a hard disc. Position and
    radius are drawn from the seed, so the same seed always yields the same
    glare.
    """
    if not 0.0 <= intensity <= 1.0:
        raise DegradationError(f"glare intensity must be in [0, 1], got {intensity}")
    if intensity == 0.0:
        return image.copy()

    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)

    radius = int(min(height, width) * float(rng.uniform(0.12, 0.22)))
    radius = max(radius, 2)
    centre_x = int(rng.integers(radius, max(radius + 1, width - radius)))
    centre_y = int(rng.integers(radius, max(radius + 1, height - radius)))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, (centre_x, centre_y), radius, 1.0, thickness=-1)
    blur_sigma = max(1.0, radius / 2.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    addition = (mask * intensity * 255.0)[:, :, np.newaxis]
    result = image.astype(np.float32) + addition
    return np.clip(result, 0, 255).astype(np.uint8)


def occlude(image: np.ndarray, area_fraction: float, seed: int = 0) -> np.ndarray:
    """Cover part of the frame with an opaque rectangle.

    Area fraction is of the whole frame. Placement is seeded.
    """
    if not 0.0 <= area_fraction < 1.0:
        raise DegradationError(
            f"occlusion fraction must be in [0, 1), got {area_fraction}"
        )
    if area_fraction == 0.0:
        return image.copy()

    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)

    # Square patch of the requested area, clamped to the frame.
    side = int(round((height * width * area_fraction) ** 0.5))
    side = max(1, min(side, min(height, width)))
    x0 = int(rng.integers(0, max(1, width - side + 1)))
    y0 = int(rng.integers(0, max(1, height - side + 1)))

    result = image.copy()
    cv2.rectangle(result, (x0, y0), (x0 + side - 1, y0 + side - 1), (0, 0, 0), thickness=-1)
    return result


_TRANSFORMS = {
    "none": lambda img, spec: img.copy(),
    "gaussian_blur": lambda img, spec: gaussian_blur(img, spec.level),
    "underexpose": lambda img, spec: adjust_exposure(img, spec.level),
    "overexpose": lambda img, spec: adjust_exposure(img, spec.level),
    "reduce_contrast": lambda img, spec: reduce_contrast(img, spec.level),
    "glare": lambda img, spec: add_glare(img, spec.level, spec.seed),
    "occlude": lambda img, spec: occlude(img, spec.level, spec.seed),
}

SUPPORTED_KINDS: tuple[str, ...] = tuple(_TRANSFORMS)


def apply_degradation(
    image: np.ndarray, spec: DegradationSpec
) -> tuple[np.ndarray, dict]:
    """Apply a degradation, returning the new image and its metadata.

    The source array is never modified; every transform returns a new array.

    Returns:
        (degraded_image, metadata) where metadata records exactly what was done,
        for storage alongside the resulting evidence.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise DegradationError("image must be a non-empty numpy array")
    if spec.kind not in _TRANSFORMS:
        raise DegradationError(
            f"unknown degradation {spec.kind!r}; supported: {', '.join(SUPPORTED_KINDS)}"
        )

    degraded = _TRANSFORMS[spec.kind](image, spec)
    return degraded, spec.to_dict()


def standard_sweep(seed: int = 0) -> list[DegradationSpec]:
    """The Phase 1 measurement ladder.

    Each family begins at its identity level so every curve has an undegraded
    reference point built in.
    """
    specs: list[DegradationSpec] = [DegradationSpec(kind="none", level=0.0, seed=seed)]
    ladders = (
        ("gaussian_blur", BLUR_SIGMAS),
        ("underexpose", UNDEREXPOSURE_GAINS),
        ("overexpose", OVEREXPOSURE_GAINS),
        ("reduce_contrast", CONTRAST_FACTORS),
        ("glare", GLARE_INTENSITIES),
        ("occlude", OCCLUSION_FRACTIONS),
    )
    for kind, levels in ladders:
        for level in levels:
            specs.append(DegradationSpec(kind=kind, level=float(level), seed=seed))
    return specs
