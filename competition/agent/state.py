"""Explicit inspection state machine.

Phase 1b and Phase 2d made correct decisions, but the *state* of an inspection
was implicit: it lived in whichever combination of local variables happened to
be set inside `inspect_capture`. That is adequate for a fixed pipeline and
inadequate for an agent, because nothing can be asserted about a state that is
never named. "The classifier did not run on a blocked capture" was true by
construction rather than by check.

Here the state is a value. Every transition is declared, every terminal state is
marked, and an illegal move raises instead of quietly producing a run that looks
finished. The cost is a small table; the benefit is that bounded execution,
fail-safe behaviour and "inference never ran" become properties a test can read
off the trace rather than infer from an absence.

Terminal states are of two kinds and the distinction matters. `COMPLETE` is the
only success. The rest are refusals, and three of them name an action a *person*
must take — the system does not pretend a human acted.
"""

from __future__ import annotations

from enum import Enum

STATE_MACHINE_VERSION = "phase3-state-1.0.0"


class InspectionState(str, Enum):
    """Every state an inspection run may occupy."""

    # --- linear progress ------------------------------------------------------
    RECEIVED = "RECEIVED"
    INPUT_VALIDATED = "INPUT_VALIDATED"
    FOREGROUND_ASSESSED = "FOREGROUND_ASSESSED"
    QUALITY_ASSESSED = "QUALITY_ASSESSED"
    ARTIFACTS_ASSESSED = "ARTIFACTS_ASSESSED"
    DECISION_MADE = "DECISION_MADE"

    # --- the single bounded remediation excursion -----------------------------
    REMEDIATION_SELECTED = "REMEDIATION_SELECTED"
    REMEDIATED = "REMEDIATED"
    RESEGMENTED = "RESEGMENTED"
    REASSESSED = "REASSESSED"

    # --- inference ------------------------------------------------------------
    ELIGIBLE_FOR_INFERENCE = "ELIGIBLE_FOR_INFERENCE"
    CONDITION_INFERRED = "CONDITION_INFERRED"

    # --- terminals ------------------------------------------------------------
    COMPLETE = "COMPLETE"
    REQUEST_RECAPTURE = "REQUEST_RECAPTURE"
    REQUEST_REPOSITION_LIGHT = "REQUEST_REPOSITION_LIGHT"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"
    INSUFFICIENT_VISUAL_EVIDENCE = "INSUFFICIENT_VISUAL_EVIDENCE"
    FAILED_SAFE = "FAILED_SAFE"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def is_success(self) -> bool:
        """Only one terminal state means the inspection actually produced a result."""
        return self is InspectionState.COMPLETE

    @property
    def requires_human(self) -> bool:
        return self in HUMAN_ACTION_STATES


TERMINAL_STATES: frozenset[InspectionState] = frozenset({
    InspectionState.COMPLETE,
    InspectionState.REQUEST_RECAPTURE,
    InspectionState.REQUEST_REPOSITION_LIGHT,
    InspectionState.REQUEST_HUMAN_REVIEW,
    InspectionState.FAILED_SAFE,
})

# States that end with a request addressed to a person. The run stops here and
# waits; it does not simulate the person having complied. A future capture
# arrives as a new run with a new id, never as a continuation of this one.
HUMAN_ACTION_STATES: frozenset[InspectionState] = frozenset({
    InspectionState.REQUEST_RECAPTURE,
    InspectionState.REQUEST_REPOSITION_LIGHT,
    InspectionState.REQUEST_HUMAN_REVIEW,
})

# INSUFFICIENT_VISUAL_EVIDENCE is deliberately neither terminal nor a
# human-action state. It says what the system knows — that no measurement it
# trusts could be taken — and says nothing about what anyone should do. The
# policy then routes it onward to a state that does name an action, so the
# reason a capture was refused stays separable from the remedy proposed for it.
# A run therefore never *ends* in it, and `final_state` is always an action.


LEGAL_TRANSITIONS: dict[InspectionState, frozenset[InspectionState]] = {
    InspectionState.RECEIVED: frozenset({
        InspectionState.INPUT_VALIDATED,
        InspectionState.FAILED_SAFE,
        InspectionState.REQUEST_RECAPTURE,
    }),
    InspectionState.INPUT_VALIDATED: frozenset({
        InspectionState.FOREGROUND_ASSESSED,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        InspectionState.FAILED_SAFE,
        InspectionState.REQUEST_RECAPTURE,
    }),
    InspectionState.FOREGROUND_ASSESSED: frozenset({
        InspectionState.QUALITY_ASSESSED,
        # A failed mask is decided on immediately: there is nothing to measure,
        # so the run reaches a decision without passing through the ROI states.
        InspectionState.DECISION_MADE,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.QUALITY_ASSESSED: frozenset({
        InspectionState.ARTIFACTS_ASSESSED,
        InspectionState.DECISION_MADE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.ARTIFACTS_ASSESSED: frozenset({
        InspectionState.DECISION_MADE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.DECISION_MADE: frozenset({
        InspectionState.REMEDIATION_SELECTED,
        InspectionState.ELIGIBLE_FOR_INFERENCE,
        InspectionState.REQUEST_RECAPTURE,
        InspectionState.REQUEST_REPOSITION_LIGHT,
        InspectionState.REQUEST_HUMAN_REVIEW,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.REMEDIATION_SELECTED: frozenset({
        InspectionState.REMEDIATED,
        InspectionState.FAILED_SAFE,
    }),
    # Re-perception is mandatory after a pixel change: the only way out of
    # REMEDIATED is to segment the new canonical image again.
    InspectionState.REMEDIATED: frozenset({
        InspectionState.RESEGMENTED,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.RESEGMENTED: frozenset({
        InspectionState.REASSESSED,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.REASSESSED: frozenset({
        InspectionState.ELIGIBLE_FOR_INFERENCE,
        InspectionState.REQUEST_RECAPTURE,
        InspectionState.REQUEST_REPOSITION_LIGHT,
        InspectionState.REQUEST_HUMAN_REVIEW,
        InspectionState.INSUFFICIENT_VISUAL_EVIDENCE,
        InspectionState.FAILED_SAFE,
    }),
    InspectionState.ELIGIBLE_FOR_INFERENCE: frozenset({
        InspectionState.CONDITION_INFERRED,
        InspectionState.FAILED_SAFE,
        InspectionState.REQUEST_HUMAN_REVIEW,
    }),
    InspectionState.CONDITION_INFERRED: frozenset({
        InspectionState.COMPLETE,
        InspectionState.FAILED_SAFE,
    }),
}
LEGAL_TRANSITIONS[InspectionState.INSUFFICIENT_VISUAL_EVIDENCE] = frozenset({
    InspectionState.REQUEST_RECAPTURE,
    InspectionState.REQUEST_HUMAN_REVIEW,
    InspectionState.FAILED_SAFE,
})
for _terminal in TERMINAL_STATES:
    LEGAL_TRANSITIONS.setdefault(_terminal, frozenset())


class StateTransitionError(RuntimeError):
    """Raised when a run attempts a transition the machine does not permit."""


def assert_transition(before: InspectionState, after: InspectionState) -> None:
    """Refuse an undeclared move.

    A second remediation excursion is impossible here by construction rather
    than by counter: REASSESSED cannot reach REMEDIATION_SELECTED.
    """
    if before.is_terminal:
        raise StateTransitionError(
            f"{before.value} is terminal; no transition to {after.value} is possible"
        )
    if after not in LEGAL_TRANSITIONS.get(before, frozenset()):
        raise StateTransitionError(
            f"illegal transition {before.value} -> {after.value}"
        )


def reachable_terminals(state: InspectionState, seen: set | None = None) -> set:
    """Terminal states still reachable from here. Used to prove dead ends absent."""
    seen = seen if seen is not None else set()
    if state in seen:
        return set()
    seen.add(state)
    if state.is_terminal:
        return {state}
    found: set = set()
    for nxt in LEGAL_TRANSITIONS.get(state, frozenset()):
        found |= reachable_terminals(nxt, seen)
    return found


def state_machine_summary() -> dict:
    """Serialisable description of the machine, for documentation and tests."""
    return {
        "state_machine_version": STATE_MACHINE_VERSION,
        "states": [state.value for state in InspectionState],
        "terminal_states": sorted(state.value for state in TERMINAL_STATES),
        "human_action_states": sorted(state.value for state in HUMAN_ACTION_STATES),
        "transitions": {
            state.value: sorted(target.value for target in targets)
            for state, targets in LEGAL_TRANSITIONS.items()
        },
    }
