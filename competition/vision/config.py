"""Threshold policy for capture-quality assessment.

Thresholds are deliberately separated from the measurement code in `quality.py`.
Measurement is objective and stable; threshold policy is a judgement call that
will be recalibrated, and later Agentic Vision branching depends on it. Keeping
them apart means a policy change never edits a measurement function, and the
policy in force is recorded in every piece of evidence produced.

STATUS: every threshold below is PROVISIONAL / DEVELOPMENT ONLY.

They are first guesses chosen to be reasonable on synthetic fixtures. None has
been calibrated on real produce photography. Phase 6 calibrates them on a
development split. They must never be tuned on the frozen research test set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

THRESHOLD_STATUS = "PROVISIONAL_DEVELOPMENT_ONLY"


@dataclass(frozen=True)
class ThresholdPolicy:
    """Threshold set consumed by the quality assessor.

    Frozen so a policy cannot be mutated part-way through an assessment, which
    would make the recorded fingerprint a lie.
    """

    # --- sharpness -----------------------------------------------------------
    # Variance of the Laplacian below which a capture is flagged BLUR_RISK.
    # Laplacian variance is scale- and content-dependent (see quality.py
    # limitations), so this is a coarse gate, not a calibrated detector.
    blur_variance_floor: float = 100.0

    # Reference variance mapped to sharpness_score == 1.0. Purely a display
    # normalisation; the raw variance is always reported alongside it.
    sharpness_reference_variance: float = 1000.0

    # Below this variance the Laplacian response is at its numerical floor: the
    # image retains essentially no high-frequency detail, and remaining
    # differences come from border handling and uint8 quantisation rather than
    # image content. Measured on synthetic fixtures during Phase 1, where
    # variance bottomed out near 0.68 and then drifted *up* slightly at larger
    # blur kernels. Comparisons below this floor are not meaningful and
    # monotonicity is not expected to hold there.
    sharpness_noise_floor: float = 2.0

    # --- luminance -----------------------------------------------------------
    # Mean L* (normalised 0-1) outside this band is flagged.
    underexposed_mean_luminance: float = 0.25
    overexposed_mean_luminance: float = 0.80

    # --- clipping ------------------------------------------------------------
    # L* channel values at or beyond these bounds count as clipped.
    shadow_clip_value: int = 4
    highlight_clip_value: int = 251
    # Clipped-pixel fraction above which the corresponding flag is raised.
    shadow_clip_fraction_limit: float = 0.05
    highlight_clip_fraction_limit: float = 0.05

    # --- contrast ------------------------------------------------------------
    # Percentile spread (p95 - p5) of L*, normalised 0-1.
    low_contrast_limit: float = 0.20
    contrast_low_percentile: float = 5.0
    contrast_high_percentile: float = 95.0

    # --- validity ------------------------------------------------------------
    min_image_dimension: int = 32

    @property
    def status(self) -> str:
        return THRESHOLD_STATUS

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = THRESHOLD_STATUS
        return data

    def fingerprint(self) -> str:
        """Stable short hash of the policy, recorded in every evidence record.

        Lets a historical decision be re-derived: if the fingerprint differs,
        the thresholds differed, and the flags are not comparable.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_POLICY = ThresholdPolicy()
