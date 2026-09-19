"""Controlled counterfactual variants of a single uploaded capture.

The central demonstration of the project. Classification is not the claim; the
claim is that **the same subject produces a different next action when the
visual evidence changes**, and the only honest way to show that is to hold the
subject constant and vary nothing but the pixels.

Variants are derived in memory from the image the user just submitted, which
avoids a licensing problem entirely: there is no bundled demonstration
photograph to attribute, and nothing is persisted. The transformations are the
Phase 1 degradation primitives, unchanged.

Two things this is not, stated here because the UI must repeat them:

- It is **not** a benchmark. These are controlled transformations chosen to land
  either side of calibrated thresholds, not samples of how often real captures
  are dark or blurred.
- It is **not** evidence about real glare or real occluders. A synthesised
  highlight is a characterised perturbation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from competition.agent.orchestrator import (
    OrchestratorPolicy,
    run_inspection,
)
from competition.agent.state import InspectionState
from competition.vision.degradation import (
    adjust_exposure,
    gaussian_blur,
    reduce_contrast,
)

COUNTERFACTUAL_VERSION = "phase5-counterfactual-1.0.0"

CONTROLLED_DEMONSTRATION_NOTICE = (
    "Controlled demonstration. These variants are generated from the image you "
    "uploaded to show how the agent's next action changes when the visual "
    "evidence changes. They are not a real-world accuracy benchmark and do not "
    "represent how often real captures look this way."
)


@dataclass(frozen=True)
class Variant:
    """One controlled transformation, with what it is meant to exercise."""

    key: str
    label: str
    description: str
    transform: object
    expected_finding: str


def _underexpose(image: np.ndarray) -> np.ndarray:
    return adjust_exposure(image, 0.12)


def _blur(image: np.ndarray) -> np.ndarray:
    # Sigma large enough to push the ROI high-frequency ratio below the
    # calibrated floor of 0.3182 on a typical capture.
    return gaussian_blur(image, 6.0)


def _roi_contrast(image: np.ndarray) -> float | None:
    """Subject-restricted contrast, measured directly and cheaply.

    Cheaper than a full inspection: segmentation plus one illumination pass, with
    no artefact detectors and no model. Used only to aim a demonstration
    variant, never to make a decision.
    """
    from competition.vision.foreground import ForegroundError, isolate_foreground
    from competition.vision.quality import measure_illumination
    from competition.vision.config import DEFAULT_POLICY

    try:
        mask, foreground = isolate_foreground(image)
    except ForegroundError:
        return None
    if not foreground.valid:
        return None
    try:
        return float(measure_illumination(image, DEFAULT_POLICY, mask).contrast_score)
    except Exception:  # noqa: BLE001
        return None


def _flatten_to_recoverable(image: np.ndarray, measured_contrast: float | None) -> np.ndarray:
    """Compress the tonal range to land in the *recoverable* contrast band.

    The band between the severe limit (0.12) and the calibrated advisory limit
    (0.1314) is narrow, and it is the band the CLAHE route exists for. A fixed
    factor overshoots it on most images - 0.22 lands in severe territory on a
    typical capture - so the factor is derived from the contrast the reference
    run actually measured on this image.

    This is deliberate and is what a controlled demonstration means: the variant
    is constructed to sit at a specific evidence level so the enhancement route
    can be seen. It is not a claim that real captures cluster there. When the
    reference contrast is unknown (no valid foreground), a fixed fallback is
    used and the variant may land in the severe band instead - which is still an
    honest outcome, just a different one.
    """
    target = 0.125  # mid-band: above severe 0.12, below advisory 0.1314
    if not measured_contrast or measured_contrast <= target:
        return reduce_contrast(image, 0.30)

    # `reduce_contrast` compresses toward the whole-image mean, so the resulting
    # ROI contrast is not a linear function of the factor. One measured
    # correction lands it reliably; a first guess from the ratio alone
    # undershoots by roughly half on a typical capture.
    factor = max(0.05, min(0.95, target / measured_contrast))
    candidate = reduce_contrast(image, factor)
    achieved = _roi_contrast(candidate)
    if achieved and achieved > 0:
        corrected = max(0.05, min(0.95, factor * (target / achieved)))
        candidate = reduce_contrast(image, corrected)
    return candidate


VARIANTS: tuple = (
    Variant("REFERENCE", "Reference",
            "Your image, unchanged.",
            lambda image: image,
            "no blocking finding"),
    Variant("UNDEREXPOSED", "Underexposed",
            "Exposure reduced until the subject falls below the calibrated "
            "underexposure limit.",
            _underexpose,
            "ROI mean luminance below the calibrated floor"),
    Variant("SEVERE_BLUR", "Severe blur",
            "Defocused until subject detail falls below the calibrated "
            "high-frequency floor.",
            _blur,
            "ROI high-frequency ratio below 0.3182"),
    Variant("LOW_CONTRAST", "Recoverable contrast loss",
            "Tonal range compressed to sit in the band the enhancement route "
            "exists for.",
            None,  # derived from the reference measurement; see the runner
            "ROI contrast in the recoverable band below 0.1314"),
)


@dataclass
class VariantOutcome:
    """What the bounded agent did with one variant."""

    key: str
    label: str
    description: str
    expected_finding: str
    final_status: str
    first_action: str
    requested_human_action: str
    condition_model_invoked: bool
    remediation_applied: bool
    remediation_action: str
    quality_flags: list = field(default_factory=list)
    artifact_flags: list = field(default_factory=list)
    headline_evidence: dict = field(default_factory=dict)
    terminal_reason_code: str = ""
    state_path: list = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "description": self.description,
            "expected_finding": self.expected_finding,
            "final_status": self.final_status, "first_action": self.first_action,
            "requested_human_action": self.requested_human_action,
            "condition_model_invoked": self.condition_model_invoked,
            "remediation_applied": self.remediation_applied,
            "remediation_action": self.remediation_action,
            "quality_flags": list(self.quality_flags),
            "artifact_flags": list(self.artifact_flags),
            "headline_evidence": dict(self.headline_evidence),
            "terminal_reason_code": self.terminal_reason_code,
            "state_path": list(self.state_path),
            "duration_ms": round(self.duration_ms, 1),
        }


def _headline_evidence(result) -> dict:
    """The one or two numbers that explain this variant's outcome."""
    for step in result.trace.steps:
        if step.tool_name == "assess_capture_quality":
            summary = step.evidence_summary
            return {
                "high_frequency_ratio": summary.get("high_frequency_ratio"),
                "focus_floor": summary.get("focus_floor"),
                "mean_luminance": summary.get("mean_luminance"),
                "contrast_score": summary.get("contrast_score"),
            }
    for step in result.trace.steps:
        if step.tool_name == "segment_foreground":
            return {
                "foreground_valid": step.evidence_summary.get("valid"),
                "invalid_reasons": step.evidence_summary.get("invalid_reasons", []),
            }
    return {}


def run_counterfactual(
    image: np.ndarray, model, policy: OrchestratorPolicy, run_id: str
) -> dict:
    """Run the bounded agent over every controlled variant of one image.

    Each variant is a full inspection through the same orchestrator and the same
    frozen policy. Nothing is simulated and no outcome is precomputed: if the
    agent behaved differently tomorrow, this page would show that.
    """
    outcomes = []
    reference_contrast: float | None = None
    for index, variant in enumerate(VARIANTS):
        if variant.transform is None:
            derived = _flatten_to_recoverable(image, reference_contrast)
        else:
            derived = variant.transform(image)
        started = time.perf_counter()
        result = run_inspection(derived, model, policy, run_id=f"{run_id}v{index}")
        elapsed = (time.perf_counter() - started) * 1000.0

        first_action = next(
            (s.selected_action for s in result.trace.steps if s.selected_action), "NONE"
        )
        if variant.key == "REFERENCE":
            # Read the subject's own contrast once, so the contrast variant can
            # be aimed at the recoverable band rather than guessing a factor.
            reference_contrast = _headline_evidence(result).get("contrast_score")

        outcomes.append(VariantOutcome(
            key=variant.key, label=variant.label, description=variant.description,
            expected_finding=variant.expected_finding,
            final_status=result.final_state.value,
            first_action=first_action,
            requested_human_action=result.requested_human_action,
            condition_model_invoked=result.inference_ran,
            remediation_applied=result.remediation_applied,
            remediation_action=result.remediation_action,
            quality_flags=list(result.quality_flags),
            artifact_flags=list(result.artifact_flags),
            headline_evidence=_headline_evidence(result),
            terminal_reason_code=result.terminal_reason_code,
            state_path=result.trace.state_sequence,
            duration_ms=elapsed,
        ))

    distinct_actions = sorted({o.first_action for o in outcomes})
    return {
        "counterfactual_version": COUNTERFACTUAL_VERSION,
        "notice": CONTROLLED_DEMONSTRATION_NOTICE,
        "source": "derived in memory from the uploaded image; not persisted",
        "variant_count": len(outcomes),
        "distinct_first_actions": distinct_actions,
        "evidence_changes_action": len(distinct_actions) > 1,
        "policy_fingerprints": (
            policy.fingerprints() if policy is not None else {}
        ),
        "variants": [o.to_dict() for o in outcomes],
    }
