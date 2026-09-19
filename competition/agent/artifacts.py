"""Compose the Phase 2d artefact tools into one traced analysis.

This is the causal chain the Agentic Vision criterion asks for, in its smallest
honest form: a measurement is taken, a named finding follows from it, and a
bounded action follows from the finding. Each step records what it saw, so the
action can be attributed to a specific OpenCV measurement rather than asserted.

    isolate_foreground
        -> measure_local_highlights / measure_visibility / ROI contrast
        -> GLARE_RISK | SUBJECT_VISIBILITY_INSUFFICIENT | SEVERE_LOW_CONTRAST
        -> REQUEST_REPOSITION_LIGHT | REQUEST_RECAPTURE | APPLY_CLAHE

The interactive loop - dispatching the action, re-capturing, re-assessing - is
Phase 3. What exists here is the evidence and the decision, which is what Phase 3
will need in order to act on something other than a bare flag.

Nothing runs on an untrusted mask. If foreground isolation fails its validity
guards, no local evidence is produced at all and the analysis escalates, because
a highlight measured on a backdrop is not a fact about the fruit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from competition.agent.artifact_policy import (
    ArtifactDecision,
    ArtifactPolicy,
    decide_capture_artifacts,
)
from competition.agent.trace import RemediationTrace, TraceTool
from competition.vision.config import DEFAULT_POLICY
from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import (
    ForegroundError,
    ForegroundEvidence,
    isolate_foreground,
)
from competition.vision.highlights import (
    LocalHighlightEvidence,
    LocalHighlightPolicy,
    measure_local_highlights,
)
from competition.vision.quality import measure_illumination
from competition.vision.visibility import (
    VisibilityEvidence,
    VisibilityPolicy,
    measure_visibility,
)

ARTIFACT_ANALYSIS_VERSION = "phase2d-artifact-analysis-1.0.0"


@dataclass
class ArtifactAnalysis:
    """Everything one artefact pass produced, including why."""

    decision: ArtifactDecision
    foreground: ForegroundEvidence | None = None
    highlights: LocalHighlightEvidence | None = None
    visibility: VisibilityEvidence | None = None
    roi_contrast_score: float | None = None
    trace: RemediationTrace = field(default_factory=RemediationTrace)
    analysis_version: str = ARTIFACT_ANALYSIS_VERSION
    processing_ms: float = 0.0

    @property
    def blocking(self) -> bool:
        return self.decision.blocking

    def to_dict(self, include_timing: bool = True) -> dict:
        data = {
            "analysis_version": self.analysis_version,
            "decision": self.decision.to_dict(),
            "foreground": (
                self.foreground.to_dict(include_timing=include_timing)
                if self.foreground else None
            ),
            "highlights": (
                self.highlights.to_dict(include_timing=include_timing)
                if self.highlights else None
            ),
            "visibility": (
                self.visibility.to_dict(include_timing=include_timing)
                if self.visibility else None
            ),
            "roi_contrast_score": self.roi_contrast_score,
            "trace": self.trace.to_dict(),
        }
        if include_timing:
            data["processing_ms"] = self.processing_ms
        return data

    def deterministic_payload(self) -> dict:
        return self.to_dict(include_timing=False)


def analyse_capture_artifacts(
    image: np.ndarray,
    policy: ArtifactPolicy | None = None,
    highlight_policy: LocalHighlightPolicy | None = None,
    visibility_policy: VisibilityPolicy | None = None,
    mask: np.ndarray | None = None,
    foreground: ForegroundEvidence | None = None,
) -> ArtifactAnalysis:
    """Measure local capture artefacts and decide one bounded action.

    A caller that has already segmented the image may pass `mask` and
    `foreground` to avoid segmenting twice; the analysis is identical either way.
    """
    started = time.perf_counter()
    trace = RemediationTrace()

    if mask is None or foreground is None:
        try:
            mask, foreground = isolate_foreground(image)
        except ForegroundError as error:
            trace.add(TraceTool.ISOLATE_FOREGROUND, f"Segmentation failed: {error}",
                      inputs={}, outputs={"valid": False})
            decision = decide_capture_artifacts(policy=policy, segmentation_valid=False)
            trace.add(TraceTool.DECIDE_CAPTURE_ARTIFACT, decision.explanation,
                      inputs={"segmentation_valid": False},
                      outputs={"action": decision.action.value,
                               "reason_code": decision.reason_code.value})
            return ArtifactAnalysis(
                decision=decision, trace=trace,
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )

    trace.add(
        TraceTool.ISOLATE_FOREGROUND,
        "Isolate the subject region before any local measurement.",
        inputs={"input_sha256": foreground.input_sha256},
        outputs={"valid": foreground.valid,
                 "invalid_reasons": list(foreground.invalid_reasons),
                 "foreground_fraction": foreground.foreground_fraction},
    )

    if not foreground.valid:
        decision = decide_capture_artifacts(policy=policy, segmentation_valid=False)
        trace.add(TraceTool.DECIDE_CAPTURE_ARTIFACT, decision.explanation,
                  inputs={"segmentation_valid": False,
                          "invalid_reasons": list(foreground.invalid_reasons)},
                  outputs={"action": decision.action.value,
                           "reason_code": decision.reason_code.value})
        return ArtifactAnalysis(
            decision=decision, foreground=foreground, trace=trace,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    highlights = visibility = None
    contrast_score = None

    try:
        highlights = measure_local_highlights(image, mask, highlight_policy)
        trace.add(
            TraceTool.MEASURE_LOCAL_HIGHLIGHTS,
            "Look for locally bright, locally desaturated regions on the subject.",
            inputs={"measured_pixels": highlights.measured_pixels},
            outputs={"glare_flag": highlights.glare_flag,
                     "largest_component_fraction": highlights.largest_component_fraction,
                     "max_local_luminance_excess": highlights.max_local_luminance_excess,
                     "spatial_concentration": highlights.spatial_concentration},
        )
    except ImageValidationError as error:
        trace.add(TraceTool.MEASURE_LOCAL_HIGHLIGHTS, f"Not measurable: {error}",
                  inputs={}, outputs={"measured": False})

    try:
        visibility = measure_visibility(mask, visibility_policy)
        trace.add(
            TraceTool.MEASURE_VISIBILITY,
            "Check whether the visible region is shaped like a complete subject.",
            inputs={"subject_pixels": visibility.subject_pixels},
            outputs={"visibility_sufficient": visibility.visibility_sufficient,
                     "hull_fill": visibility.hull_fill,
                     "solidity": visibility.solidity,
                     "border_truncation_fraction": visibility.border_truncation_fraction,
                     "reasons": list(visibility.insufficiency_reasons)},
        )
    except ImageValidationError as error:
        trace.add(TraceTool.MEASURE_VISIBILITY, f"Not measurable: {error}",
                  inputs={}, outputs={"measured": False})

    try:
        illumination = measure_illumination(image, DEFAULT_POLICY, mask)
        contrast_score = float(illumination.contrast_score)
        trace.add(
            TraceTool.ASSESS_CAPTURE_QUALITY,
            "Measure foreground-restricted tonal range.",
            inputs={}, outputs={"roi_contrast_score": contrast_score},
        )
    except ImageValidationError as error:
        trace.add(TraceTool.ASSESS_CAPTURE_QUALITY, f"Not measurable: {error}",
                  inputs={}, outputs={"measured": False})

    decision = decide_capture_artifacts(
        highlight_evidence=highlights,
        visibility_evidence=visibility,
        roi_contrast_score=contrast_score,
        policy=policy,
        segmentation_valid=True,
    )
    trace.add(
        TraceTool.DECIDE_CAPTURE_ARTIFACT,
        decision.explanation,
        inputs={"flags": list(decision.flags)},
        outputs={"action": decision.action.value,
                 "reason_code": decision.reason_code.value,
                 "blocking": decision.blocking},
    )

    return ArtifactAnalysis(
        decision=decision,
        foreground=foreground,
        highlights=highlights,
        visibility=visibility,
        roi_contrast_score=contrast_score,
        trace=trace,
        processing_ms=(time.perf_counter() - started) * 1000.0,
    )
