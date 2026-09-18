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
    COMPARE_CAPTURE_QUALITY = "compare_capture_quality"
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
