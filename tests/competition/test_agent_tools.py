"""Tool registry: the closed vocabulary and the contracts it declares."""

from __future__ import annotations

import importlib
import json

import pytest

from competition.agent.tools import (
    TOOL_REGISTRY,
    ToolName,
    ToolRegistryError,
    resolve_tool,
    tool_registry_summary,
)
from competition.agent.tools.registry import ToolKind, tools_requiring_valid_foreground


# --- test 24: a closed enum prevents arbitrary tool execution ----------------


def test_every_registry_key_is_a_tool_name():
    for name, spec in TOOL_REGISTRY.items():
        assert isinstance(name, ToolName)
        assert spec.name is name


def test_every_tool_name_is_registered():
    assert set(TOOL_REGISTRY) == set(ToolName)


def test_resolve_accepts_an_enum_member():
    assert resolve_tool(ToolName.SEGMENT_FOREGROUND).name is ToolName.SEGMENT_FOREGROUND


def test_resolve_accepts_an_exact_string_value():
    assert resolve_tool("segment_foreground").name is ToolName.SEGMENT_FOREGROUND


@pytest.mark.parametrize("bad", [
    "run_condition_model_v2",
    "rm -rf /",
    "SEGMENT_FOREGROUND",          # the member name, not its value
    "assess_capture_quality ",     # trailing space
    "",
])
def test_resolve_refuses_anything_outside_the_vocabulary(bad):
    with pytest.raises(ToolRegistryError):
        resolve_tool(bad)


@pytest.mark.parametrize("bad", [None, 42, ["segment_foreground"], {"tool": "x"}])
def test_resolve_refuses_non_string_types(bad):
    with pytest.raises(ToolRegistryError):
        resolve_tool(bad)


# --- contracts ----------------------------------------------------------------


def test_every_tool_declares_its_contracts():
    for spec in TOOL_REGISTRY.values():
        assert spec.summary
        assert spec.implementation
        assert isinstance(spec.kind, ToolKind)
        assert isinstance(spec.inputs, tuple)
        assert isinstance(spec.outputs, tuple)
        assert isinstance(spec.failure_states, tuple)


def test_every_declared_implementation_actually_exists():
    """A registry that names a function nobody wrote is worse than none."""
    for spec in TOOL_REGISTRY.values():
        module_path, _, attribute = spec.implementation.rpartition(".")
        module = importlib.import_module(module_path)
        assert hasattr(module, attribute), f"{spec.implementation} does not exist"


def test_only_transforms_mutate_the_canonical_image():
    for spec in TOOL_REGISTRY.values():
        if spec.mutates_canonical_image:
            assert spec.kind is ToolKind.TRANSFORM


def test_the_two_enhancements_are_the_only_mutating_tools():
    mutating = {
        spec.name for spec in TOOL_REGISTRY.values() if spec.mutates_canonical_image
    }
    assert mutating == {ToolName.APPLY_GAMMA_CORRECTION, ToolName.APPLY_CLAHE}


def test_human_request_tools_require_human_action():
    for spec in TOOL_REGISTRY.values():
        if spec.kind is ToolKind.HUMAN_REQUEST:
            assert spec.requires_human_action


def test_reposition_light_is_a_declared_human_request():
    spec = resolve_tool(ToolName.REQUEST_REPOSITION_LIGHT)
    assert spec.requires_human_action
    assert not spec.mutates_canonical_image


def test_glare_measurement_is_not_an_enhancement():
    """Nothing about a highlight may route to a tool that rewrites pixels."""
    spec = resolve_tool(ToolName.ASSESS_LOCAL_HIGHLIGHTS)
    assert spec.kind is ToolKind.MEASUREMENT
    assert not spec.mutates_canonical_image


def test_roi_dependent_tools_declare_that_they_need_a_valid_foreground():
    needs = set(tools_requiring_valid_foreground())
    assert {
        ToolName.ASSESS_CAPTURE_QUALITY.value,
        ToolName.ASSESS_LOCAL_HIGHLIGHTS.value,
        ToolName.ASSESS_VISIBILITY.value,
        ToolName.ASSESS_CONTRAST.value,
    } <= needs


def test_segmentation_itself_does_not_require_a_valid_foreground():
    assert not resolve_tool(ToolName.SEGMENT_FOREGROUND).requires_valid_foreground


def test_only_the_condition_model_requires_the_onnx_artifact():
    requiring = {
        spec.name for spec in TOOL_REGISTRY.values() if spec.requires_model_artifact
    }
    assert requiring == {ToolName.RUN_CONDITION_MODEL}


def test_registry_summary_serialises():
    payload = json.loads(json.dumps(tool_registry_summary()))
    assert payload["tool_count"] == len(ToolName)
    assert "request_reposition_light" in payload["requires_human_action"]


def test_no_tool_declares_a_filesystem_path():
    payload = json.dumps(tool_registry_summary())
    assert "/home/" not in payload and "/Users/" not in payload
