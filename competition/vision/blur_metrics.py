"""Candidate focus metrics, for deciding whether one blur threshold can exist.

Phase 1 measured two properties of variance-of-Laplacian that between them
explain why a single global threshold kept failing:

* it scales with the square of exposure gain, so a dim photograph of a sharp
  subject scores like a bright photograph of a blurred one;
* it scales with resolution and subject scale, so the same fruit photographed
  from further away scores lower without being any less in focus.

Both are properties of the *measurement*, not of the image's usability. A metric
that is invariant to them would let a threshold mean "out of focus" rather than
"dark, small, or out of focus". This module implements three candidates
alongside the incumbent so the question can be settled by measurement.

**Normalised gradient energy** is the one designed against the failure. Under a
multiplicative exposure change `I -> gI`, gradient energy scales as `g^2` and so
does intensity variance, so their ratio is unchanged. It is therefore expected
to be exposure-invariant by construction — which is a prediction this module
makes and the calibration then tests, not a claim made in advance.

Every metric here is a few lines of OpenCV with a closed-form definition. No
learned blur model is used: a neural quality score would be unexplainable to a
judge and impossible to defend against the calibration set it was fitted on.

All four accept the same optional foreground mask and apply the same erosion
discipline as `quality.measure_sharpness`: the operator runs on the unmodified
image and the mask only selects which responses are aggregated, so the mask
boundary never contributes gradient energy of its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import DEFAULT_GUARDS, masked_pixels

BLUR_METRICS_VERSION = "phase2c-blur-metrics-1.0.0"

MIN_MEASURABLE_PIXELS = 256
_EPSILON = 1e-9


@dataclass(frozen=True)
class FocusMetrics:
    """Four focus measurements of one image or region.

    Reported together and never reduced to a single number here. Which one (if
    any) supports a usable threshold is a calibration result, not a design
    assumption.
    """

    laplacian_variance: float
    tenengrad: float
    normalised_gradient_energy: float
    high_frequency_ratio: float
    measured_pixels: int
    metrics_version: str = BLUR_METRICS_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _grayscale(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ImageValidationError("image must be a non-empty array")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ImageValidationError(f"cannot measure focus on shape {image.shape!r}")


def _selection(image: np.ndarray, mask: np.ndarray | None, erosion_px: int | None) -> np.ndarray | None:
    """Boolean selection of pixels whose responses may be aggregated."""
    if mask is None:
        return None
    if mask.shape[:2] != image.shape[:2]:
        raise ImageValidationError(
            f"mask shape {mask.shape[:2]} does not match image {image.shape[:2]}"
        )
    erosion = erosion_px if erosion_px is not None else DEFAULT_GUARDS.sharpness_erosion_px
    interior = masked_pixels(mask, erosion)
    selection = interior > 0
    if int(np.count_nonzero(selection)) < MIN_MEASURABLE_PIXELS:
        raise ImageValidationError(
            f"only {int(np.count_nonzero(selection))} pixels remain inside the eroded "
            f"mask; at least {MIN_MEASURABLE_PIXELS} are required"
        )
    return selection


def high_frequency_ratio(
    gray: np.ndarray, mask: np.ndarray | None = None, cutoff_fraction: float = 0.25
) -> float:
    """Share of spectral energy above a radial cutoff.

    Computed on the mask's bounding box rather than on scattered pixels: a
    Fourier transform needs a contiguous rectangle, and transforming a
    background-zeroed image would put the silhouette's step edge into the
    spectrum as broadband high-frequency energy — the frequency-domain version
    of the mask-edge artefact `masked_pixels` exists to prevent. The box is a
    compromise, and it is recorded as one: this metric sees some background
    whenever the subject is not box-shaped.
    """
    region = gray
    if mask is not None and np.count_nonzero(mask):
        ys, xs = np.nonzero(mask)
        region = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if region.size < MIN_MEASURABLE_PIXELS:
        return 0.0

    windowed = region.astype(np.float32)
    windowed = windowed - float(windowed.mean())
    # Hann window in both axes: without it the rectangle's own edges leak a
    # cross of spurious high-frequency energy across the spectrum.
    rows, columns = windowed.shape
    windowed *= np.outer(np.hanning(rows), np.hanning(columns)).astype(np.float32)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))
    total = float(spectrum.sum())
    if total <= _EPSILON:
        return 0.0

    centre_y, centre_x = rows / 2.0, columns / 2.0
    yy, xx = np.ogrid[:rows, :columns]
    radius = np.sqrt(((yy - centre_y) / max(centre_y, 1)) ** 2
                     + ((xx - centre_x) / max(centre_x, 1)) ** 2)
    return float(spectrum[radius > cutoff_fraction].sum() / total)


def measure_focus(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    erosion_px: int | None = None,
) -> FocusMetrics:
    """All four candidate focus metrics over the same pixel selection."""
    gray = _grayscale(image)
    selection = _selection(gray, mask, erosion_px)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_energy = gradient_x ** 2 + gradient_y ** 2

    if selection is None:
        laplacian_values = laplacian.reshape(-1)
        energy_values = gradient_energy.reshape(-1)
        intensity_values = gray.reshape(-1).astype(np.float64)
    else:
        laplacian_values = laplacian[selection]
        energy_values = gradient_energy[selection]
        intensity_values = gray[selection].astype(np.float64)

    intensity_variance = float(intensity_values.var())
    tenengrad = float(energy_values.mean())

    return FocusMetrics(
        laplacian_variance=float(laplacian_values.var()),
        tenengrad=tenengrad,
        # Dividing by intensity variance removes the multiplicative exposure
        # term that Phase 1 measured. It does not remove *additive* changes, so
        # a glare patch or a raised black level still moves it.
        normalised_gradient_energy=float(tenengrad / (intensity_variance + _EPSILON)),
        high_frequency_ratio=high_frequency_ratio(gray, mask),
        measured_pixels=int(intensity_values.size),
    )
