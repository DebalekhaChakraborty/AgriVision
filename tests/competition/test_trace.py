"""Trace structure: ordering, serialisation, closed tool vocabulary."""

from __future__ import annotations

import json

import pytest

from competition.agent.trace import (
    TRACE_VERSION,
    RemediationTrace,
    TraceStep,
    TraceTool,
    build_remediation_trace,
)


def test_steps_are_numbered_from_one_in_insertion_order():
    trace = RemediationTrace()
    trace.add(TraceTool.ASSESS_CAPTURE_QUALITY, "first")
    trace.add(TraceTool.DECIDE_CAPTURE_REMEDIATION, "second")
    trace.add(TraceTool.FINALISE, "third")
    assert [s.step for s in trace.steps] == [1, 2, 3]
    assert [s.summary for s in trace.steps] == ["first", "second", "third"]


def test_tool_sequence_reflects_execution_order():
    trace = RemediationTrace()
    trace.add(TraceTool.ASSESS_CAPTURE_QUALITY, "a")
    trace.add(TraceTool.APPLY_CLAHE, "b")
    assert trace.tool_sequence == ["assess_capture_quality", "apply_clahe"]


def test_trace_serialises_to_json():
    trace = RemediationTrace()
    trace.add(
        TraceTool.ASSESS_CAPTURE_QUALITY,
        "assessment",
        inputs={"image_sha256": "abc123"},
        outputs={"mean_luminance": 0.22, "quality_flags": ["UNDEREXPOSED"]},
    )
    restored = json.loads(trace.to_json())
    assert restored["trace_version"] == TRACE_VERSION
    assert restored["step_count"] == 1
    step = restored["steps"][0]
    assert step["tool"] == "assess_capture_quality"
    assert step["outputs"]["quality_flags"] == ["UNDEREXPOSED"]


def test_tool_is_serialised_as_a_plain_string():
    trace = RemediationTrace()
    trace.add(TraceTool.FINALISE, "done")
    tool = json.loads(trace.to_json())["steps"][0]["tool"]
    assert isinstance(tool, str)
    assert tool == TraceTool.FINALISE.value


def test_tool_vocabulary_is_closed():
    assert {tool.value for tool in TraceTool} == {
        "assess_capture_quality",
        "decide_capture_remediation",
        "apply_gamma_correction",
        "apply_clahe",
        "compare_capture_quality",
        "inspection_gate",
        "predict_condition",
        "finalise",
    }


def test_build_remediation_trace_renumbers_steps():
    steps = [
        TraceStep(step=99, tool=TraceTool.ASSESS_CAPTURE_QUALITY, summary="a"),
        TraceStep(step=7, tool=TraceTool.FINALISE, summary="b"),
    ]
    trace = build_remediation_trace(steps)
    assert [s.step for s in trace.steps] == [1, 2]


def test_empty_trace_serialises_cleanly():
    restored = json.loads(RemediationTrace().to_json())
    assert restored["step_count"] == 0
    assert restored["steps"] == []


def test_step_defaults_are_independent_between_steps():
    """Mutable default containers must not be shared across steps."""
    trace = RemediationTrace()
    first = trace.add(TraceTool.ASSESS_CAPTURE_QUALITY, "a")
    second = trace.add(TraceTool.FINALISE, "b")
    assert first.inputs is not second.inputs
    assert first.outputs is not second.outputs


@pytest.mark.parametrize("tool", list(TraceTool))
def test_every_tool_can_be_recorded_and_serialised(tool):
    trace = RemediationTrace()
    trace.add(tool, "summary")
    assert json.loads(trace.to_json())["steps"][0]["tool"] == tool.value
