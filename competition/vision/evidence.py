"""Typed visual evidence produced by the OpenCV perception layer.

Design constraints this module exists to enforce:

* JSON-serialisable, so evidence can cross a process or network boundary and be
  stored as a trace record.
* Deterministic for identical input and configuration. Timing is the one
  inherently non-deterministic field, so it is isolated and excluded from the
  deterministic payload rather than being allowed to contaminate it.
* No machine-specific absolute paths. Inputs are identified by a content hash,
  never by where they happened to live on disk. Research result files in this
  repository already carry stale absolute paths from a previous directory
  layout; that mistake is not repeated here.
* Individually measurable components, never one opaque quality score. Later
  agent branching needs to say *which* measurement caused an action.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

PIPELINE_VERSION = "phase2b-capture-quality-1.1.0"


class MeasurementScope(str, Enum):
    """Region the capture-quality metrics were computed over.

    Recorded on every record so whole-image and foreground-restricted evidence
    can never be silently compared or conflated.
    """

    WHOLE_IMAGE = "WHOLE_IMAGE"
    FOREGROUND_MASKED = "FOREGROUND_MASKED"


class QualityFlag(str, Enum):
    """Descriptive capture-quality flags.

    Flags describe the *capture*, never the produce. Nothing here implies a
    freshness, safety or condition judgement.
    """

    BLUR_RISK = "BLUR_RISK"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    LOW_CONTRAST = "LOW_CONTRAST"
    SHADOW_CLIPPING = "SHADOW_CLIPPING"
    HIGHLIGHT_CLIPPING = "HIGHLIGHT_CLIPPING"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"


@dataclass(frozen=True)
class SharpnessMetrics:
    """Focus/blur measurements. Higher variance means more high-frequency detail."""

    laplacian_variance: float
    sharpness_score: float  # laplacian_variance normalised against policy reference


@dataclass(frozen=True)
class IlluminationMetrics:
    """Exposure and tonal-distribution measurements taken on the CIELAB L* channel."""

    mean_luminance: float  # 0-1
    median_luminance: float  # 0-1
    luminance_std: float  # 0-1
    contrast_score: float  # normalised (p95 - p5) spread of L*
    shadow_clip_fraction: float  # 0-1
    highlight_clip_fraction: float  # 0-1


@dataclass(frozen=True)
class ImageProperties:
    """Basic validity and identity of the analysed image."""

    width: int
    height: int
    channels: int
    content_sha256: str  # identifies the image by content, never by path


@dataclass(frozen=True)
class PerceptionEvidence:
    """Complete capture-quality evidence for a single image.

    This is the contract later phases consume. `assess_capture_quality` is the
    only supported way to construct one.
    """

    image: ImageProperties
    sharpness: SharpnessMetrics
    illumination: IlluminationMetrics
    quality_flags: list[str]
    opencv_version: str
    pipeline_version: str
    threshold_policy_fingerprint: str
    threshold_policy_status: str
    measurement_scope: str = MeasurementScope.WHOLE_IMAGE.value
    processing_ms: float = field(default=0.0)

    # -- serialisation -------------------------------------------------------

    def to_dict(self, include_timing: bool = True) -> dict:
        data = asdict(self)
        if not include_timing:
            data.pop("processing_ms", None)
        return data

    def to_json(self, include_timing: bool = True, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_timing=include_timing), indent=indent, sort_keys=True
        )

    def deterministic_payload(self) -> dict:
        """Everything that must be byte-identical across runs of the same input.

        `processing_ms` is wall-clock and legitimately varies, so it is excluded
        here. Determinism tests compare this, not the full record.
        """
        return self.to_dict(include_timing=False)

    # -- convenience ---------------------------------------------------------

    def has_flag(self, flag: QualityFlag) -> bool:
        return flag.value in self.quality_flags

    @property
    def is_suitable_for_inspection(self) -> bool:
        """True when no capture-quality flag was raised.

        This is a convenience for later policy code, not a verdict about the
        produce. A clean capture can still be inspected incorrectly.
        """
        return not self.quality_flags


class ImageValidationError(ValueError):
    """Raised when input cannot be analysed. Never raised for merely poor quality.

    A dark, blurred, badly exposed photograph is valid input that produces
    evidence with flags set. This error is for input that is not an analysable
    image at all.
    """
