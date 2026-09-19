"""Scenario suite, counterfactual experiment and agent metrics."""

from __future__ import annotations

import json

import pytest

from competition.agent.orchestrator import load_locked_policies
from competition.agent.state import InspectionState
from competition.agent.tools import ToolName
from competition.evaluation.phase3_agent_scenarios import (
    ROI_DETECTORS,
    TRACE_EXPORTS,
    build_scenarios,
    compute_metrics,
    load_model,
    run_counterfactual,
    run_scenarios,
)


@pytest.fixture(scope="module")
def policy():
    return load_locked_policies()


@pytest.fixture(scope="module")
def model():
    loaded, reason = load_model()
    if loaded is None:
        pytest.skip(f"condition-model artifact unavailable: {reason}")
    return loaded


@pytest.fixture(scope="module")
def suite(model, policy):
    return run_scenarios(model, policy)


# --- the suite itself ---------------------------------------------------------


def test_the_twelve_required_scenarios_exist():
    names = {scenario.name for scenario in build_scenarios()}
    assert names == {
        "normal_acceptable_capture", "recoverable_underexposure",
        "severe_darkness_clipping", "severe_blur", "moderate_low_contrast",
        "severe_contrast_loss", "segmentation_failure", "advisory_glare",
        "invalid_input", "accepted_remediation", "rejected_remediation",
        "successful_condition_inference",
    }


def test_every_scenario_declares_its_expected_behaviour():
    for scenario in build_scenarios():
        assert scenario.description
        assert isinstance(scenario.expected_final_state, InspectionState)
        assert isinstance(scenario.expected_model_invoked, bool)


def test_the_policy_variant_scenario_says_that_it_is_one():
    variant = next(
        scenario for scenario in build_scenarios()
        if scenario.name == "rejected_remediation"
    )
    assert variant.policy_override is not None
    assert "POLICY VARIANT" in variant.policy_note


def test_every_scenario_passes(suite):
    report, _ = suite
    failures = [row["name"] for row in report["scenarios"] if not row["passed"]]
    assert not failures, f"failing scenarios: {failures}"


def test_the_report_states_its_claim_boundary(suite):
    report, _ = suite
    assert "not real-world accuracy" in report["claim_boundary"]
    assert "scenario-suite" in report["claim_boundary"]


def test_the_report_serialises(suite):
    report, _ = suite
    assert json.loads(json.dumps(report))["task_success_rate"] is not None


# --- the specific behavioural guarantees --------------------------------------


def test_no_scenario_runs_the_model_on_a_blocked_capture(suite):
    report, _ = suite
    for row in report["scenarios"]:
        if row["observed_final_state"] in {
            "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "FAILED_SAFE",
        }:
            assert not row["observed_model_invoked"], row["name"]


def test_segmentation_failure_runs_no_roi_detector_in_the_suite(suite):
    report, _ = suite
    row = next(r for r in report["scenarios"] if r["name"] == "segmentation_failure")
    for tool in ROI_DETECTORS:
        assert tool not in row["observed_tools"]


def test_the_advisory_glare_scenario_completes_without_blocking(suite):
    report, _ = suite
    row = next(r for r in report["scenarios"] if r["name"] == "advisory_glare")
    assert row["observed_final_state"] == "COMPLETE"
    assert row["observed_model_invoked"]
    assert row["observed_advisory"] == "REQUEST_REPOSITION_LIGHT"
    assert row["observed_human_action"] == ""


def test_the_rejected_remediation_scenario_escalates_without_inferring(suite):
    report, _ = suite
    row = next(r for r in report["scenarios"] if r["name"] == "rejected_remediation")
    assert row["observed_remediation"] == "APPLY_GAMMA"
    assert row["observed_remediation_accepted"] is False
    assert not row["observed_model_invoked"]


def test_no_scenario_exceeds_its_budget(suite):
    report, _ = suite
    for row in report["scenarios"]:
        assert row["budget"]["within_budget"], row["name"]
        assert row["budget"]["remediation_attempts"] <= 1
        assert row["budget"]["condition_model_calls"] <= 1


# --- metrics ------------------------------------------------------------------


def test_metrics_report_every_denominator(suite):
    report, results = suite
    metrics = compute_metrics(report, results)
    for key, value in metrics.items():
        if isinstance(value, dict) and "value" in value:
            assert value["denominator"] is not None, key
            assert value["definition"], key


def test_unsafe_inference_rate_is_zero(suite):
    report, results = suite
    metrics = compute_metrics(report, results)
    assert metrics["UNSAFE_INFERENCE_RATE"]["value"] == 0.0


def test_bounded_execution_holds_for_every_run(suite):
    report, results = suite
    metrics = compute_metrics(report, results)
    assert metrics["BOUNDED_EXECUTION_RATE"]["value"] == 1.0


def test_metrics_are_labelled_as_a_scenario_result(suite):
    report, results = suite
    metrics = compute_metrics(report, results)
    assert "not real-world accuracy" in metrics["claim_boundary"]


# --- the counterfactual -------------------------------------------------------


@pytest.fixture(scope="module")
def counterfactual(model, policy):
    return run_counterfactual(model, policy)


def test_the_same_subject_produces_different_actions(counterfactual):
    """The Agentic Vision claim, reduced to something falsifiable."""
    assert counterfactual["evidence_changes_action"]
    assert len(counterfactual["distinct_first_actions"]) >= 4


def test_each_variant_maps_to_the_action_its_evidence_implies(counterfactual):
    actions = {
        row["variant"]: row["first_selected_action"]
        for row in counterfactual["variants"]
    }
    assert actions["REFERENCE"] == "NONE"
    assert actions["UNDEREXPOSED"] == "APPLY_GAMMA"
    assert actions["SEVERE_BLUR"] == "REQUEST_RECAPTURE"
    assert actions["LOW_CONTRAST"] == "APPLY_CLAHE"
    assert actions["SEGMENTATION_FAILURE"] == "REQUEST_RECAPTURE"


def test_glare_differs_from_the_reference_only_in_its_advisory(counterfactual):
    rows = {row["variant"]: row for row in counterfactual["variants"]}
    assert rows["GLARE"]["first_selected_action"] == (
        rows["REFERENCE"]["first_selected_action"]
    )
    assert rows["GLARE"]["advisory_human_action"] == "REQUEST_REPOSITION_LIGHT"
    assert rows["REFERENCE"]["advisory_human_action"] == ""


def test_the_policy_is_identical_across_every_variant(counterfactual):
    """Otherwise the experiment would prove nothing about the evidence."""
    assert counterfactual["policy_fingerprints"]
    assert len(counterfactual["variants"]) == counterfactual["variant_count"]


def test_blocked_variants_never_invoke_the_model(counterfactual):
    for row in counterfactual["variants"]:
        if row["final_state"] != "COMPLETE":
            assert not row["condition_model_invoked"], row["variant"]


# --- exported traces ----------------------------------------------------------


def test_every_required_trace_export_is_declared():
    assert set(TRACE_EXPORTS) >= {
        "trace_normal", "trace_underexposure_remediation",
        "trace_severe_blur_recapture", "trace_segmentation_failure",
        "trace_low_contrast_remediation", "trace_glare_advisory",
    }


def test_exported_trace_names_map_to_real_scenarios():
    names = {scenario.name for scenario in build_scenarios()}
    for scenario_name in TRACE_EXPORTS.values():
        assert scenario_name in names
