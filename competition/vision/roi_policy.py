"""Threshold policy for foreground-restricted capture quality.

Separate from `config.ThresholdPolicy` on purpose. Phase 2b established that
restricting measurement to the subject changes the measurement domain: ROI
Laplacian variance ran at roughly 5% of the whole-image value, because most of
the gradient energy in a product photograph lives at the subject/background
boundary and in background texture, not on the fruit. A threshold calibrated on
one scope is not merely differently-tuned on the other, it is measuring a
different population. Sharing one dataclass between the two scopes would invite
exactly the substitution that finding warns against, so the two are different
types and a record always says which it carries.

The policy also names the focus metric it uses. Phase 1 found variance of the
Laplacian to be coupled to exposure and to subject scale, so whether it can
support a global threshold at all is an open question that calibration answers.
Naming the metric in the policy means a later switch is a visible change of
fingerprint rather than a silent change of meaning.

Nothing here is calibrated at import time. `PROVISIONAL_ROI_POLICY` carries the
Phase 2b values purely so the module is importable and testable; its status says
so, and the calibrated instance is loaded from disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from competition.vision.evidence import MeasurementScope, QualityFlag

ROI_POLICY_VERSION = "phase2c-roi-policy-1.0.0"

STATUS_PROVISIONAL = "PROVISIONAL_UNCALIBRATED"
STATUS_CALIBRATED = "CALIBRATED_ON_LICENSED_REAL_IMAGERY"
# Used when calibration admitted some thresholds and rejected others, leaving
# the rest at their uncalibrated Phase 1 values. This is the normal outcome and
# it needs its own name: a policy where two of seven thresholds are evidence-
# backed is not a calibrated policy, and calling it one would be the whole-image
# substitution Phase 2b exists to warn against, wearing a better label.
STATUS_PARTIALLY_CALIBRATED = "PARTIALLY_CALIBRATED_SOME_THRESHOLDS_PROVISIONAL"

# Closed set. A policy naming a metric outside it cannot be evaluated.
FOCUS_METRICS: tuple[str, ...] = (
    "laplacian_variance",
    "tenengrad",
    "normalised_gradient_energy",
    "high_frequency_ratio",
)

# Sentinel for "calibration found no defensible threshold for this metric".
# A policy carrying it does not gate on focus at all, which is the honest
# outcome when the measurement cannot separate the classes - far better than a
# number chosen so that the pipeline has one.
NO_DEFENSIBLE_THRESHOLD = None


class RoiPolicyError(ValueError):
    """Raised for a policy that cannot be applied as written."""


@dataclass(frozen=True)
class RoiQualityPolicy:
    """Thresholds applied to foreground-restricted metrics."""

    focus_metric: str = "laplacian_variance"
    focus_floor: float | None = 100.0

    underexposed_mean_luminance: float = 0.25
    overexposed_mean_luminance: float = 0.80
    low_contrast_limit: float = 0.20

    shadow_clip_value: int = 4
    highlight_clip_value: int = 251
    shadow_clip_fraction_limit: float = 0.05
    highlight_clip_fraction_limit: float = 0.05

    min_image_dimension: int = 32

    status: str = STATUS_PROVISIONAL
    measurement_scope: str = MeasurementScope.FOREGROUND_MASKED.value
    policy_version: str = ROI_POLICY_VERSION
    # Filled in when a policy is locked; empty while provisional.
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.focus_metric not in FOCUS_METRICS:
            raise RoiPolicyError(
                f"unknown focus metric {self.focus_metric!r}; "
                f"supported: {', '.join(FOCUS_METRICS)}"
            )
        if self.focus_floor is not None and self.focus_floor < 0:
            raise RoiPolicyError(f"focus floor must be >= 0, got {self.focus_floor}")
        if self.underexposed_mean_luminance >= self.overexposed_mean_luminance:
            raise RoiPolicyError(
                "underexposed bound must sit below the overexposed bound"
            )

    @property
    def gates_on_focus(self) -> bool:
        return self.focus_floor is not None

    def threshold_payload(self) -> dict:
        """Only the numbers that change a decision. Provenance excluded.

        Keeping provenance out means the fingerprint answers exactly one
        question — "would this policy decide differently?" — so two runs of the
        same thresholds compare equal even though they were locked on different
        days.
        """
        data = asdict(self)
        for key in ("provenance", "status"):
            data.pop(key, None)
        return data

    def threshold_fingerprint(self) -> str:
        payload = json.dumps(self.threshold_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def lock_fingerprint(self) -> str:
        """Identifies this policy *and* the corpus and code it was locked against."""
        payload = json.dumps(
            {"thresholds": self.threshold_payload(), "provenance": self.provenance},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def locked(self, provenance: dict, status: str = STATUS_CALIBRATED) -> "RoiQualityPolicy":
        return replace(self, status=status, provenance=dict(provenance))

    @property
    def fully_calibrated(self) -> bool:
        return self.status == STATUS_CALIBRATED

    def to_dict(self) -> dict:
        data = asdict(self)
        data["threshold_fingerprint"] = self.threshold_fingerprint()
        data["lock_fingerprint"] = self.lock_fingerprint()
        data["gates_on_focus"] = self.gates_on_focus
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RoiQualityPolicy":
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


PROVISIONAL_ROI_POLICY = RoiQualityPolicy()


def focus_value(metrics, policy: RoiQualityPolicy) -> float:
    """Read the metric this policy gates on."""
    try:
        return float(getattr(metrics, policy.focus_metric))
    except AttributeError as error:
        raise RoiPolicyError(
            f"focus metrics carry no field {policy.focus_metric!r}"
        ) from error


def apply_roi_policy(properties, illumination, focus, policy: RoiQualityPolicy) -> list[str]:
    """Derive capture-quality flags from foreground-restricted metrics.

    Deliberately mirrors `quality.derive_quality_flags` in shape so the two
    scopes stay comparable, and deliberately does not share its code so that a
    change to one scope's rules cannot silently move the other's.
    """
    flags: list[QualityFlag] = []

    if min(properties.width, properties.height) < policy.min_image_dimension:
        flags.append(QualityFlag.IMAGE_TOO_SMALL)

    if policy.gates_on_focus and focus_value(focus, policy) < policy.focus_floor:
        flags.append(QualityFlag.BLUR_RISK)

    if illumination.mean_luminance < policy.underexposed_mean_luminance:
        flags.append(QualityFlag.UNDEREXPOSED)
    elif illumination.mean_luminance > policy.overexposed_mean_luminance:
        flags.append(QualityFlag.OVEREXPOSED)

    if illumination.contrast_score < policy.low_contrast_limit:
        flags.append(QualityFlag.LOW_CONTRAST)

    if illumination.shadow_clip_fraction > policy.shadow_clip_fraction_limit:
        flags.append(QualityFlag.SHADOW_CLIPPING)

    if illumination.highlight_clip_fraction > policy.highlight_clip_fraction_limit:
        flags.append(QualityFlag.HIGHLIGHT_CLIPPING)

    return [flag.value for flag in flags]
