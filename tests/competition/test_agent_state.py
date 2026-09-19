"""State machine: legality, terminals, and the moves that must be impossible."""

from __future__ import annotations

import json

import pytest

from competition.agent.decisions import (
    EVIDENCE_MATURITY,
    AgentDecision,
    EvidenceMaturity,
    EvidenceMaturityError,
    maturity_of,
    maturity_summary,
    strongest_maturity,
    weakest_maturity,
)
from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.state import (
    HUMAN_ACTION_STATES,
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    InspectionState,
    StateTransitionError,
    assert_transition,
    reachable_terminals,
    state_machine_summary,
)


# --- structure ----------------------------------------------------------------


def test_every_state_has_a_transition_entry():
    for state in InspectionState:
        assert state in LEGAL_TRANSITIONS


def test_terminal_states_have_no_outgoing_transitions():
    for state in TERMINAL_STATES:
        assert LEGAL_TRANSITIONS[state] == frozenset()
        assert state.is_terminal


def test_only_complete_counts_as_success():
    successes = [state for state in InspectionState if state.is_success]
    assert successes == [InspectionState.COMPLETE]


def test_every_state_can_still_reach_a_terminal():
    """No dead ends: a run can always end."""
    for state in InspectionState:
        assert reachable_terminals(state), f"{state.value} cannot terminate"


def test_failed_safe_is_reachable_from_every_non_terminal_state():
    for state in InspectionState:
        if state.is_terminal:
            continue
        assert InspectionState.FAILED_SAFE in reachable_terminals(state)


# --- test 7: a second remediation attempt is unreachable ----------------------


def test_a_second_remediation_excursion_is_structurally_impossible():
    """The bound is enforced by the graph, not by a counter that could be reset."""
    assert InspectionState.REMEDIATION_SELECTED not in LEGAL_TRANSITIONS[
        InspectionState.REASSESSED
    ]
    assert InspectionState.REMEDIATION_SELECTED not in LEGAL_TRANSITIONS[
        InspectionState.RESEGMENTED
    ]
    assert InspectionState.REMEDIATION_SELECTED not in LEGAL_TRANSITIONS[
        InspectionState.ELIGIBLE_FOR_INFERENCE
    ]


def test_remediation_must_be_followed_by_resegmentation():
    """The only way out of REMEDIATED is to look at the new image."""
    assert LEGAL_TRANSITIONS[InspectionState.REMEDIATED] == frozenset({
        InspectionState.RESEGMENTED, InspectionState.FAILED_SAFE,
    })


def test_inference_is_reachable_only_from_the_eligible_state():
    sources = [
        state for state, targets in LEGAL_TRANSITIONS.items()
        if InspectionState.CONDITION_INFERRED in targets
    ]
    assert sources == [InspectionState.ELIGIBLE_FOR_INFERENCE]


def test_no_state_reaches_complete_without_inferring_first():
    sources = [
        state for state, targets in LEGAL_TRANSITIONS.items()
        if InspectionState.COMPLETE in targets
    ]
    assert sources == [InspectionState.CONDITION_INFERRED]


# --- transitions --------------------------------------------------------------


def test_legal_transition_passes():
    assert_transition(InspectionState.RECEIVED, InspectionState.INPUT_VALIDATED)


def test_illegal_transition_raises():
    with pytest.raises(StateTransitionError):
        assert_transition(InspectionState.RECEIVED, InspectionState.COMPLETE)


def test_no_transition_out_of_a_terminal_state():
    with pytest.raises(StateTransitionError) as error:
        assert_transition(InspectionState.COMPLETE, InspectionState.RECEIVED)
    assert "terminal" in str(error.value)


def test_insufficient_evidence_is_a_waypoint_not_an_ending():
    """It says what is known; a separate state says what someone should do."""
    assert not InspectionState.INSUFFICIENT_VISUAL_EVIDENCE.is_terminal
    assert not InspectionState.INSUFFICIENT_VISUAL_EVIDENCE.requires_human
    onward = LEGAL_TRANSITIONS[InspectionState.INSUFFICIENT_VISUAL_EVIDENCE]
    assert InspectionState.REQUEST_RECAPTURE in onward
    assert InspectionState.REQUEST_HUMAN_REVIEW in onward


def test_human_action_states_are_exactly_the_three_requests():
    assert {state.value for state in HUMAN_ACTION_STATES} == {
        "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "REQUEST_HUMAN_REVIEW",
    }


def test_state_machine_summary_serialises():
    payload = json.loads(json.dumps(state_machine_summary()))
    assert payload["states"]
    assert "COMPLETE" in payload["terminal_states"]


# --- evidence maturity --------------------------------------------------------


def test_only_calibrated_and_provisional_may_gate():
    assert EvidenceMaturity.CALIBRATED.may_gate
    assert EvidenceMaturity.PROVISIONAL.may_gate
    assert not EvidenceMaturity.ADVISORY.may_gate
    assert not EvidenceMaturity.UNQUALIFIED.may_gate


def test_glare_is_advisory_and_visibility_is_unqualified():
    """The two Phase 2d detectors that did not meet their floor."""
    assert maturity_of("artifact.glare") is EvidenceMaturity.ADVISORY
    assert maturity_of("artifact.visibility") is EvidenceMaturity.UNQUALIFIED


def test_model_confidence_is_unqualified():
    """No threshold has been calibrated, so it cannot drive anything."""
    assert maturity_of("model.confidence") is EvidenceMaturity.UNQUALIFIED


def test_calibrated_thresholds_match_the_phase2cb_record():
    for evidence_id in ("roi.high_frequency_ratio", "roi.contrast_score",
                        "roi.mean_luminance.underexposed"):
        assert maturity_of(evidence_id) is EvidenceMaturity.CALIBRATED


def test_retained_phase1_thresholds_are_provisional():
    for evidence_id in ("roi.mean_luminance.overexposed", "roi.shadow_clip_fraction",
                        "roi.highlight_clip_fraction"):
        assert maturity_of(evidence_id) is EvidenceMaturity.PROVISIONAL


def test_an_unregistered_measurement_is_powerless():
    """Fails safe: a new metric gates nothing until someone registers it."""
    assert maturity_of("artifact.brand_new_idea") is EvidenceMaturity.UNQUALIFIED


def test_weakest_and_strongest_maturity():
    ids = ["roi.contrast_score", "artifact.glare"]
    assert weakest_maturity(ids) is EvidenceMaturity.ADVISORY
    assert strongest_maturity(ids) is EvidenceMaturity.CALIBRATED


# --- test 15: experimental evidence cannot silently become a hard blocker -----


def test_a_blocking_decision_on_advisory_evidence_is_refused():
    with pytest.raises(EvidenceMaturityError) as error:
        AgentDecision(
            state_before=InspectionState.DECISION_MADE,
            selected_action=RemediationAction.REQUEST_REPOSITION_LIGHT,
            reason_code=ReasonCode.GLARE_LOCAL_HIGHLIGHT,
            state_after=InspectionState.REQUEST_REPOSITION_LIGHT,
            triggering_evidence_ids=["artifact.glare"],
            blocking=True,
        )
    assert "ADVISORY" in str(error.value)


def test_advisory_evidence_may_accompany_a_non_blocking_decision():
    decision = AgentDecision(
        state_before=InspectionState.DECISION_MADE,
        selected_action=RemediationAction.NONE,
        reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
        state_after=InspectionState.ELIGIBLE_FOR_INFERENCE,
        triggering_evidence_ids=["artifact.glare"],
        blocking=False,
    )
    assert decision.selected_action is RemediationAction.NONE


def test_a_block_is_allowed_when_trusted_evidence_also_supports_it():
    """Glare may travel with a refusal that rests on something calibrated."""
    decision = AgentDecision(
        state_before=InspectionState.DECISION_MADE,
        selected_action=RemediationAction.REQUEST_RECAPTURE,
        reason_code=ReasonCode.BLUR_NOT_REMEDIABLE,
        state_after=InspectionState.REQUEST_RECAPTURE,
        triggering_evidence_ids=["roi.high_frequency_ratio", "artifact.glare"],
        blocking=True,
    )
    assert decision.blocking


# --- test 16: the unqualified detector cannot drive a decision at all ---------


def test_unqualified_evidence_cannot_trigger_any_decision():
    with pytest.raises(EvidenceMaturityError) as error:
        AgentDecision(
            state_before=InspectionState.DECISION_MADE,
            selected_action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=ReasonCode.SUBJECT_VISIBILITY_INSUFFICIENT,
            state_after=InspectionState.REQUEST_RECAPTURE,
            triggering_evidence_ids=["artifact.visibility"],
            blocking=False,
        )
    assert "UNQUALIFIED" in str(error.value)


def test_model_confidence_cannot_trigger_an_escalation():
    """Test 17: no uncalibrated confidence rule may exist."""
    with pytest.raises(EvidenceMaturityError):
        AgentDecision(
            state_before=InspectionState.CONDITION_INFERRED,
            selected_action=RemediationAction.REQUEST_HUMAN_REVIEW,
            reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
            state_after=InspectionState.REQUEST_HUMAN_REVIEW,
            triggering_evidence_ids=["model.confidence"],
            blocking=False,
        )


def test_decision_serialises_and_fingerprints():
    decision = AgentDecision(
        state_before=InspectionState.DECISION_MADE,
        selected_action=RemediationAction.NONE,
        reason_code=ReasonCode.CAPTURE_ACCEPTABLE,
        state_after=InspectionState.ELIGIBLE_FOR_INFERENCE,
    )
    payload = json.loads(json.dumps(decision.to_dict()))
    assert payload["selected_action"] == "NONE"
    assert len(decision.fingerprint()) == 16


def test_identical_decisions_fingerprint_identically():
    """Test 11: same evidence and policy, same decision."""
    def build():
        return AgentDecision(
            state_before=InspectionState.DECISION_MADE,
            selected_action=RemediationAction.APPLY_GAMMA,
            reason_code=ReasonCode.UNDEREXPOSED_RECOVERABLE,
            state_after=InspectionState.REMEDIATION_SELECTED,
            triggering_evidence_ids=["roi.mean_luminance.underexposed"],
        )
    assert build().fingerprint() == build().fingerprint()


def test_maturity_summary_lists_every_registered_evidence_id():
    summary = maturity_summary()
    listed = {item for group in summary["evidence"].values() for item in group}
    assert listed == set(EVIDENCE_MATURITY)


# --- the explicit advisory override -------------------------------------------


def test_advisory_gating_requires_an_explicit_acknowledgement():
    """Off by default: weak evidence cannot block by accident."""
    with pytest.raises(EvidenceMaturityError):
        AgentDecision(
            state_before=InspectionState.DECISION_MADE,
            selected_action=RemediationAction.REQUEST_REPOSITION_LIGHT,
            reason_code=ReasonCode.GLARE_LOCAL_HIGHLIGHT,
            state_after=InspectionState.REQUEST_REPOSITION_LIGHT,
            triggering_evidence_ids=["artifact.glare"],
            blocking=True,
        )


def test_an_acknowledged_advisory_block_is_permitted_and_recorded():
    decision = AgentDecision(
        state_before=InspectionState.DECISION_MADE,
        selected_action=RemediationAction.REQUEST_REPOSITION_LIGHT,
        reason_code=ReasonCode.GLARE_LOCAL_HIGHLIGHT,
        state_after=InspectionState.REQUEST_REPOSITION_LIGHT,
        triggering_evidence_ids=["artifact.glare"],
        evidence_maturity=EvidenceMaturity.ADVISORY,
        advisory_gating_permitted=True,
        blocking=True,
    )
    payload = decision.to_dict()
    assert payload["advisory_gating_permitted"] is True
    assert payload["evidence_maturity"] == "ADVISORY"


def test_the_override_never_admits_unqualified_evidence():
    """Visibility stays powerless however the policy is configured."""
    with pytest.raises(EvidenceMaturityError):
        AgentDecision(
            state_before=InspectionState.DECISION_MADE,
            selected_action=RemediationAction.REQUEST_RECAPTURE,
            reason_code=ReasonCode.SUBJECT_VISIBILITY_INSUFFICIENT,
            state_after=InspectionState.REQUEST_RECAPTURE,
            triggering_evidence_ids=["artifact.visibility"],
            advisory_gating_permitted=True,
            blocking=True,
        )
