"""Declared tools the orchestrator may invoke.

The registry exists to make one property checkable: *the set of things this
agent can do is fixed and known*. Every capability is declared with its
contracts, its failure modes, whether it changes the canonical image, and
whether it ends in a request to a person.

Nothing here reimplements vision. Each entry names a function that already
exists and was tested in an earlier phase; the registry adds the metadata an
orchestrator needs to reason about a call before making it, and the closed
vocabulary that stops a string from becoming an execution.

`resolve_tool` refuses anything that is not a `ToolName` member. That is the
guarantee that survives contact with a language model later: a narrator may
describe a trace, and no sentence it produces can name a tool into existence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

TOOL_REGISTRY_VERSION = "phase3-tools-1.0.0"


class ToolName(str, Enum):
    """The complete set of invocable tools."""

    VALIDATE_INPUT = "validate_input"
    SEGMENT_FOREGROUND = "segment_foreground"
    ASSESS_CAPTURE_QUALITY = "assess_capture_quality"
    ASSESS_LOCAL_HIGHLIGHTS = "assess_local_highlights"
    ASSESS_CONTRAST = "assess_contrast"
    ASSESS_VISIBILITY = "assess_visibility"
    DECIDE_CAPTURE_ACTION = "decide_capture_action"
    APPLY_GAMMA_CORRECTION = "apply_gamma_correction"
    APPLY_CLAHE = "apply_clahe"
    REASSESS_CAPTURE = "reassess_capture"
    RUN_CONDITION_MODEL = "run_condition_model"
    REQUEST_RECAPTURE = "request_recapture"
    REQUEST_REPOSITION_LIGHT = "request_reposition_light"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    FINALIZE_RESULT = "finalize_result"


class ToolKind(str, Enum):
    """What a tool does to the run."""

    MEASUREMENT = "MEASUREMENT"        # produces evidence, changes nothing
    DECISION = "DECISION"              # maps evidence to an action
    TRANSFORM = "TRANSFORM"            # produces a new canonical image
    INFERENCE = "INFERENCE"            # runs the condition model
    HUMAN_REQUEST = "HUMAN_REQUEST"    # terminates with a request to a person
    TERMINAL = "TERMINAL"              # assembles the result


@dataclass(frozen=True)
class ToolSpec:
    """Everything the orchestrator needs to know before calling a tool."""

    name: ToolName
    kind: ToolKind
    summary: str
    implementation: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    failure_states: tuple[str, ...]
    mutates_canonical_image: bool = False
    requires_human_action: bool = False
    requires_valid_foreground: bool = False
    requires_model_artifact: bool = False
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["name"] = self.name.value
        data["kind"] = self.kind.value
        return data


TOOL_REGISTRY: dict[ToolName, ToolSpec] = {
    ToolName.VALIDATE_INPUT: ToolSpec(
        name=ToolName.VALIDATE_INPUT,
        kind=ToolKind.MEASUREMENT,
        summary="Decode the array and check it is a usable image at all.",
        implementation="competition.vision.quality.validate_image",
        inputs=("image",),
        outputs=("width", "height", "channels", "content_sha256"),
        failure_states=("ImageValidationError", "NOT_AN_ARRAY", "EMPTY_IMAGE"),
        evidence_ids=("input.decodable", "input.dimensions"),
    ),
    ToolName.SEGMENT_FOREGROUND: ToolSpec(
        name=ToolName.SEGMENT_FOREGROUND,
        kind=ToolKind.MEASUREMENT,
        summary="Isolate the subject. Everything ROI-restricted depends on this.",
        implementation="competition.vision.foreground.isolate_foreground",
        inputs=("image", "method", "guards"),
        outputs=("mask", "foreground_evidence"),
        failure_states=("ForegroundError", "MASK_EMPTY", "MASK_FRAGMENTED",
                        "SUBJECT_TOO_SMALL", "BORDER_DOMINATED"),
        evidence_ids=("foreground.valid", "foreground.component_count",
                      "foreground.foreground_fraction"),
    ),
    ToolName.ASSESS_CAPTURE_QUALITY: ToolSpec(
        name=ToolName.ASSESS_CAPTURE_QUALITY,
        kind=ToolKind.MEASUREMENT,
        summary="Focus, exposure, contrast and clipping restricted to the subject.",
        implementation="competition.vision.roi_policy.apply_roi_policy",
        inputs=("image", "mask", "roi_policy"),
        outputs=("quality_flags", "high_frequency_ratio", "mean_luminance",
                 "contrast_score", "shadow_clip_fraction", "highlight_clip_fraction"),
        failure_states=("ImageValidationError", "ROI_TOO_SMALL"),
        requires_valid_foreground=True,
        evidence_ids=("roi.high_frequency_ratio", "roi.contrast_score",
                      "roi.mean_luminance.underexposed",
                      "roi.mean_luminance.overexposed",
                      "roi.shadow_clip_fraction", "roi.highlight_clip_fraction"),
    ),
    ToolName.ASSESS_LOCAL_HIGHLIGHTS: ToolSpec(
        name=ToolName.ASSESS_LOCAL_HIGHLIGHTS,
        kind=ToolKind.MEASUREMENT,
        summary="Multi-scale local excess luminance with desaturation, inside the ROI.",
        implementation="competition.vision.highlights.measure_local_highlights",
        inputs=("image", "mask", "highlight_policy"),
        outputs=("highlight_area_fraction", "largest_highlight_fraction",
                 "highlight_component_count"),
        failure_states=("ImageValidationError", "ROI_TOO_SMALL"),
        requires_valid_foreground=True,
        evidence_ids=("artifact.glare",),
    ),
    ToolName.ASSESS_CONTRAST: ToolSpec(
        name=ToolName.ASSESS_CONTRAST,
        kind=ToolKind.MEASUREMENT,
        summary="Two-tier tonal-range check on the ROI contrast score.",
        implementation="competition.agent.artifact_policy.decide_capture_artifacts",
        inputs=("roi_contrast_score", "artifact_policy"),
        outputs=("severe", "moderate"),
        failure_states=("CONTRAST_UNAVAILABLE",),
        requires_valid_foreground=True,
        evidence_ids=("artifact.severe_contrast", "artifact.moderate_contrast"),
    ),
    ToolName.ASSESS_VISIBILITY: ToolSpec(
        name=ToolName.ASSESS_VISIBILITY,
        kind=ToolKind.MEASUREMENT,
        summary="Subject completeness geometry. Recorded only; may not drive a decision.",
        implementation="competition.vision.visibility.measure_visibility",
        inputs=("mask", "visibility_policy", "unfilled_mask"),
        outputs=("hull_fill", "solidity", "border_truncation",
                 "internal_hole_fraction", "sufficient"),
        failure_states=("ImageValidationError", "NO_CONTOUR"),
        requires_valid_foreground=True,
        evidence_ids=("artifact.visibility",),
    ),
    ToolName.DECIDE_CAPTURE_ACTION: ToolSpec(
        name=ToolName.DECIDE_CAPTURE_ACTION,
        kind=ToolKind.DECISION,
        summary="Map the assembled evidence to exactly one action from the closed enum.",
        implementation="competition.agent.orchestrator.decide_next_action",
        inputs=("quality_flags", "artifact_flags", "foreground_valid", "maturities"),
        outputs=("action", "reason_code", "state_after"),
        failure_states=("EvidenceMaturityError",),
    ),
    ToolName.APPLY_GAMMA_CORRECTION: ToolSpec(
        name=ToolName.APPLY_GAMMA_CORRECTION,
        kind=ToolKind.TRANSFORM,
        summary="Tone correction toward the target luminance band.",
        implementation="competition.agent.remediation.enhance_capture",
        inputs=("image", "decision"),
        outputs=("remediated_image", "parameters"),
        failure_states=("RemediationDispatchError", "cv2.error"),
        mutates_canonical_image=True,
    ),
    ToolName.APPLY_CLAHE: ToolSpec(
        name=ToolName.APPLY_CLAHE,
        kind=ToolKind.TRANSFORM,
        summary="Local histogram equalisation for recoverable contrast loss.",
        implementation="competition.agent.remediation.enhance_capture",
        inputs=("image", "decision"),
        outputs=("remediated_image", "parameters"),
        failure_states=("RemediationDispatchError", "cv2.error"),
        mutates_canonical_image=True,
    ),
    ToolName.REASSESS_CAPTURE: ToolSpec(
        name=ToolName.REASSESS_CAPTURE,
        kind=ToolKind.MEASUREMENT,
        summary="Compare pre- and post-remediation evidence and accept or reject.",
        implementation="competition.agent.policy.compare_capture_quality",
        inputs=("evidence_before", "evidence_after", "decision"),
        outputs=("verdict", "deltas"),
        failure_states=("ImageValidationError",),
    ),
    ToolName.RUN_CONDITION_MODEL: ToolSpec(
        name=ToolName.RUN_CONDITION_MODEL,
        kind=ToolKind.INFERENCE,
        summary="Visible-condition inference. Runs only from ELIGIBLE_FOR_INFERENCE.",
        implementation="competition.models.adapter.predict_condition",
        inputs=("model", "canonical_image"),
        outputs=("fruit_type", "visible_condition", "confidence", "input_sha256"),
        failure_states=("ConditionModelError", "MODEL_ARTIFACT_ABSENT", "cv2.error"),
        requires_model_artifact=True,
        evidence_ids=("model.confidence",),
    ),
    ToolName.REQUEST_RECAPTURE: ToolSpec(
        name=ToolName.REQUEST_RECAPTURE,
        kind=ToolKind.HUMAN_REQUEST,
        summary="Ask for a new photograph. The run stops here.",
        implementation="competition.agent.orchestrator.request_human_action",
        inputs=("reason_code",),
        outputs=("requested_human_action",),
        failure_states=(),
        requires_human_action=True,
    ),
    ToolName.REQUEST_REPOSITION_LIGHT: ToolSpec(
        name=ToolName.REQUEST_REPOSITION_LIGHT,
        kind=ToolKind.HUMAN_REQUEST,
        summary="Ask for the lighting geometry to change. No tone curve does this.",
        implementation="competition.agent.orchestrator.request_human_action",
        inputs=("reason_code",),
        outputs=("requested_human_action",),
        failure_states=(),
        requires_human_action=True,
    ),
    ToolName.REQUEST_HUMAN_REVIEW: ToolSpec(
        name=ToolName.REQUEST_HUMAN_REVIEW,
        kind=ToolKind.HUMAN_REQUEST,
        summary="Hand the capture to a person. Used when the system cannot tell.",
        implementation="competition.agent.orchestrator.request_human_action",
        inputs=("reason_code",),
        outputs=("requested_human_action",),
        failure_states=(),
        requires_human_action=True,
    ),
    ToolName.FINALIZE_RESULT: ToolSpec(
        name=ToolName.FINALIZE_RESULT,
        kind=ToolKind.TERMINAL,
        summary="Assemble the inspection result and its fingerprints.",
        implementation="competition.agent.orchestrator.finalise",
        inputs=("run_state",),
        outputs=("InspectionRunResult",),
        failure_states=(),
    ),
}


class ToolRegistryError(KeyError):
    """Raised when something that is not a declared tool is asked for."""


def resolve_tool(name) -> ToolSpec:
    """Look up a tool. Refuses anything outside the closed vocabulary.

    A plain string is accepted only when it is exactly a `ToolName` value, so a
    caller cannot invent `run_condition_model_v2` or smuggle in a shell command.
    """
    if isinstance(name, ToolName):
        return TOOL_REGISTRY[name]
    if isinstance(name, str):
        try:
            return TOOL_REGISTRY[ToolName(name)]
        except ValueError as error:
            raise ToolRegistryError(
                f"{name!r} is not a declared tool; permitted: "
                + ", ".join(sorted(tool.value for tool in ToolName))
            ) from error
    raise ToolRegistryError(
        f"tool name must be a ToolName or its exact string value, got {type(name).__name__}"
    )


def tools_requiring_valid_foreground() -> list[str]:
    return sorted(
        spec.name.value for spec in TOOL_REGISTRY.values()
        if spec.requires_valid_foreground
    )


def tool_registry_summary() -> dict:
    return {
        "tool_registry_version": TOOL_REGISTRY_VERSION,
        "tool_count": len(TOOL_REGISTRY),
        "tools": [spec.to_dict() for spec in TOOL_REGISTRY.values()],
        "requires_valid_foreground": tools_requiring_valid_foreground(),
        "mutates_canonical_image": sorted(
            spec.name.value for spec in TOOL_REGISTRY.values()
            if spec.mutates_canonical_image
        ),
        "requires_human_action": sorted(
            spec.name.value for spec in TOOL_REGISTRY.values()
            if spec.requires_human_action
        ),
    }
