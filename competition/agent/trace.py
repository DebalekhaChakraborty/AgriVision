"""Structured causal trace for a remediation attempt.

The trace is the artifact that makes the causal chain inspectable: which tool
ran, on what evidence, what it decided, and what happened next. It is
structured and JSON-serialisable throughout — never prose — so it can be stored,
queried, replayed and evaluated.

Its eventual purpose is to answer, for the Agentic Vision requirement, the
question *did an OpenCV measurement actually change what the system did?* Each
step records the metric values it acted on, so the answer is derivable rather
than asserted. Phase 1b produces the trace; measuring attribution over a corpus
comes later.

No filesystem path is ever recorded. Images are referenced by the content hash
already carried in the evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

TRACE_VERSION = "phase1b-trace-1.0.0"


class TraceTool(str, Enum):
    """Closed set of tools that may appear in a trace."""

    ASSESS_CAPTURE_QUALITY = "assess_capture_quality"
    DECIDE_CAPTURE_REMEDIATION = "decide_capture_remediation"
    APPLY_GAMMA_CORRECTION = "apply_gamma_correction"
    APPLY_CLAHE = "apply_clahe"
    ISOLATE_FOREGROUND = "isolate_foreground"
    COMPARE_CAPTURE_QUALITY = "compare_capture_quality"
    INSPECTION_GATE = "inspection_gate"
    PREDICT_CONDITION = "predict_condition"
    MEASURE_LOCAL_HIGHLIGHTS = "measure_local_highlights"
    MEASURE_VISIBILITY = "measure_visibility"
    DECIDE_CAPTURE_ARTIFACT = "decide_capture_artifact"
    FINALISE = "finalise"


@dataclass(frozen=True)
class TraceStep:
    """One tool invocation and its outcome."""

    step: int
    tool: TraceTool
    summary: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tool"] = self.tool.value
        return data


@dataclass
class RemediationTrace:
    """Ordered record of a single bounded remediation attempt."""

    steps: list[TraceStep] = field(default_factory=list)
    trace_version: str = TRACE_VERSION

    def add(
        self,
        tool: TraceTool,
        summary: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
    ) -> TraceStep:
        step = TraceStep(
            step=len(self.steps) + 1,
            tool=tool,
            summary=summary,
            inputs=inputs or {},
            outputs=outputs or {},
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> dict:
        return {
            "trace_version": self.trace_version,
            "step_count": len(self.steps),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def tool_sequence(self) -> list[str]:
        """Tool names in execution order — the causal spine of the attempt."""
        return [step.tool.value for step in self.steps]


def build_remediation_trace(steps: list[TraceStep]) -> RemediationTrace:
    """Assemble a trace from pre-built steps, renumbering to execution order."""
    trace = RemediationTrace()
    for step in steps:
        trace.add(step.tool, step.summary, step.inputs, step.outputs)
    return trace


# =============================================================================
# Phase 3 agent trace
# =============================================================================
#
# `RemediationTrace` above records one bounded remediation attempt and stays
# exactly as Phase 1b left it. The agent trace records a whole run: which state
# the machine was in, which tool it called, what that tool measured, what the
# policy concluded, and where it went next.
#
# The extra fields exist to answer one question without interpretation — *did an
# OpenCV measurement change what the system did?* A reader should be able to
# point at a number, then at the decision it caused, then at the tool that
# decision invoked, without knowing anything about the implementation.
#
# Timing is carried but excluded from the deterministic payload, so two runs of
# the same image compare equal while still being measurable. No filesystem path
# and no image bytes ever appear: images are named by content hash only.

AGENT_TRACE_VERSION = "phase3-agent-trace-1.0.0"


@dataclass(frozen=True)
class AgentTraceStep:
    """One tool invocation, with the causal context that makes it auditable."""

    step_id: int
    state_before: str
    tool_name: str
    state_after: str
    evidence_summary: dict = field(default_factory=dict)
    evidence_ids: list = field(default_factory=list)
    input_artifact_hashes: dict = field(default_factory=dict)
    output_artifact_hashes: dict = field(default_factory=dict)
    decision_reason: str = ""
    selected_action: str = ""
    evidence_maturity: str = ""
    policy_fingerprint: str = ""
    duration_ms: float = 0.0
    timestamp: str = ""

    def to_dict(self, include_timing: bool = True) -> dict:
        data = {
            "step_id": self.step_id,
            "state_before": self.state_before,
            "tool_name": self.tool_name,
            "state_after": self.state_after,
            "evidence_summary": dict(self.evidence_summary),
            "evidence_ids": list(self.evidence_ids),
            "input_artifact_hashes": dict(self.input_artifact_hashes),
            "output_artifact_hashes": dict(self.output_artifact_hashes),
            "decision_reason": self.decision_reason,
            "selected_action": self.selected_action,
            "evidence_maturity": self.evidence_maturity,
            "policy_fingerprint": self.policy_fingerprint,
        }
        if include_timing:
            data["duration_ms"] = self.duration_ms
            data["timestamp"] = self.timestamp
        return data


@dataclass
class AgentTrace:
    """Ordered causal record of one inspection run."""

    run_id: str = ""
    steps: list = field(default_factory=list)
    trace_version: str = AGENT_TRACE_VERSION

    def add(
        self,
        state_before,
        tool_name,
        state_after,
        evidence_summary: dict | None = None,
        evidence_ids: list | None = None,
        input_artifact_hashes: dict | None = None,
        output_artifact_hashes: dict | None = None,
        decision_reason: str = "",
        selected_action: str = "",
        evidence_maturity: str = "",
        policy_fingerprint: str = "",
        duration_ms: float = 0.0,
        timestamp: str = "",
    ) -> AgentTraceStep:
        step = AgentTraceStep(
            step_id=len(self.steps) + 1,
            state_before=getattr(state_before, "value", state_before),
            tool_name=getattr(tool_name, "value", tool_name),
            state_after=getattr(state_after, "value", state_after),
            evidence_summary=evidence_summary or {},
            evidence_ids=list(evidence_ids or []),
            input_artifact_hashes=input_artifact_hashes or {},
            output_artifact_hashes=output_artifact_hashes or {},
            decision_reason=decision_reason,
            selected_action=selected_action,
            evidence_maturity=evidence_maturity,
            policy_fingerprint=policy_fingerprint,
            duration_ms=duration_ms,
            timestamp=timestamp,
        )
        self.steps.append(step)
        return step

    def to_dict(self, include_timing: bool = True) -> dict:
        data = {
            "trace_version": self.trace_version,
            "run_id": self.run_id,
            "step_count": len(self.steps),
            "steps": [step.to_dict(include_timing) for step in self.steps],
        }
        if include_timing:
            data["total_duration_ms"] = round(
                sum(step.duration_ms for step in self.steps), 3
            )
        return data

    def deterministic_payload(self) -> dict:
        """Everything except timing, so identical inputs give identical bytes."""
        payload = self.to_dict(include_timing=False)
        payload.pop("run_id", None)
        return payload

    def to_json(self, indent: int | None = 2, include_timing: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_timing), indent=indent, sort_keys=True
        )

    @property
    def tool_sequence(self) -> list:
        return [step.tool_name for step in self.steps]

    @property
    def state_sequence(self) -> list:
        """States in order of entry — the spine a reader follows."""
        if not self.steps:
            return []
        sequence = [self.steps[0].state_before]
        for step in self.steps:
            if step.state_after != sequence[-1]:
                sequence.append(step.state_after)
        return sequence

    def called(self, tool_name) -> bool:
        return getattr(tool_name, "value", tool_name) in self.tool_sequence

    @property
    def decision_steps(self) -> list:
        """Steps that selected an action — the ones attribution is measured on."""
        return [step for step in self.steps if step.selected_action]
