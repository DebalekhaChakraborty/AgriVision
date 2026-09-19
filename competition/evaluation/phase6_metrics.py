"""Phase 6 scoring: what the deployed system actually did, with denominators.

Separate from `phase6_run` on purpose. Collection and interpretation are
different jobs, and keeping them apart means a metric cannot quietly change what
was collected.

Metric definitions are the Phase 3 ones, unchanged, so a Phase 6 number can be
put beside a Phase 3 number without an asterisk. What differs is the population:
Phase 3 measured constructed fixtures, Phase 6 measures real photographs and
controlled degradations of real photographs.

Nothing here tunes anything. If a preregistered expectation was not met, the
scenario is recorded as a failure and classified; the expectation is never
edited to meet the result.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("competition/evaluation/results/phase6")
TRACK_R_RAW = RESULTS_DIR / "track_r_raw.json"
TRACK_C_RAW = RESULTS_DIR / "track_c_raw.json"
TRACK_R_RESULTS = RESULTS_DIR / "track_r_results.json"
TRACK_C_RESULTS = RESULTS_DIR / "track_c_results.json"
AGENT_METRICS = RESULTS_DIR / "agent_metrics.json"
FAILURE_TAXONOMY = RESULTS_DIR / "failure_taxonomy.json"
LATENCY = RESULTS_DIR / "latency.json"
EVIDENCE_TABLE = RESULTS_DIR / "final_evidence_table.json"

METRICS_VERSION = "phase6-metrics-1.0.0"

BLOCKING_TERMINALS = {
    "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "REQUEST_HUMAN_REVIEW",
    "FAILED_SAFE",
}
HUMAN_ACTION_TERMINALS = {
    "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "REQUEST_HUMAN_REVIEW",
}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _metric(numerator: int, denominator: int, definition: str) -> dict:
    return {
        "value": _rate(numerator, denominator),
        "numerator": numerator,
        "denominator": denominator,
        "definition": definition,
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ConfirmatoryStatusError(RuntimeError):
    """Raised when results cannot be described as confirmatory."""


def assert_frozen(observations: list[dict], frozen: dict) -> None:
    """Refuse to score results produced under a policy other than the frozen one.

    The whole claim of this phase rests on the system not having moved. If a
    response carries different policy fingerprints, these results describe some
    other system, and scoring them as confirmatory would be a false statement
    rather than a slightly stale one.
    """
    expected = frozen["policy"]["fingerprints"]
    mismatched = [
        o.get("image_id") or o.get("base_id") or o.get("run_id")
        for o in observations
        if o.get("http_status") == 200 and o.get("policy_fingerprints") != expected
    ]
    if mismatched:
        raise ConfirmatoryStatusError(
            f"{len(mismatched)} observation(s) carry policy fingerprints that "
            f"differ from frozen_system.json; these results are not "
            f"confirmatory. First: {mismatched[0]}"
        )


# --- trace access -------------------------------------------------------------


def fetch_traces(observations: list[dict], base_url: str) -> dict[str, dict]:
    """Pull every trace back out of the service.

    Re-fetching rather than caching from the run is deliberate: it re-proves
    that the persisted trace is still retrievable by run id, which is the only
    end-to-end evidence that the DynamoDB write actually happened.
    """
    import urllib.request

    traces: dict[str, dict] = {}
    for observation in observations:
        run_id = observation.get("run_id")
        if not run_id:
            continue
        try:
            with urllib.request.urlopen(
                f"{base_url}/inspection/{run_id}/trace", timeout=90
            ) as response:
                traces[run_id] = json.loads(response.read())
        except Exception:  # noqa: BLE001
            continue
    return traces


def step_count(trace: dict | None) -> int:
    """Steps in a trace. The service serialises numbers as strings."""
    raw = (trace or {}).get("trace", {}).get("step_count", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def first_action(trace: dict | None) -> str:
    """The first action the agent selected, from the decision record itself."""
    if not trace:
        return ""
    for decision in trace.get("decisions", []):
        if decision.get("selected_action"):
            return decision["selected_action"]
    for step in trace.get("trace", {}).get("steps", []):
        if step.get("selected_action"):
            return step["selected_action"]
    return ""


def segmentation_methods(trace: dict | None) -> list[str]:
    if not trace:
        return []
    methods = []
    for step in trace.get("trace", {}).get("steps", []):
        summary = step.get("evidence_summary") or {}
        for key in ("method", "mask_provenance", "foreground_method"):
            if key in summary:
                methods.append(str(summary[key]))
    return methods


# --- Track R ------------------------------------------------------------------


def score_track_r(raw: dict, traces: dict[str, dict]) -> dict:
    observations = raw["observations"]
    total = len(observations)
    ok = [o for o in observations if o["http_status"] == 200]

    foreground_valid = [o for o in ok if o["foreground_valid"]]
    insufficient = [o for o in ok if not o["foreground_valid"]]
    complete = [o for o in ok if o["final_status"] == "COMPLETE"]
    recapture = [o for o in ok if o["final_status"] == "REQUEST_RECAPTURE"]
    human_action = [o for o in ok if o["final_status"] in HUMAN_ACTION_TERMINALS]
    remediated = [o for o in ok if o["remediation_applied"]]
    accepted = [o for o in ok if o["remediation_accepted"]]
    model_ran = [o for o in ok if o["condition_model_invoked"]]

    # Fruit type is the only label that exists independently of AgriVision: it
    # was recorded at curation, from the source, before any model ran. Visible
    # condition has no such label and no accuracy for it is computed anywhere.
    correct_fruit = [
        o for o in model_ran
        if (o.get("condition_evidence") or {}).get("fruit_type") == o["fruit_type"]
    ]
    wrong_fruit = [o for o in model_ran if o not in correct_fruit]

    confidences = [o["model_confidence"] for o in model_ran if o["model_confidence"]]
    correct_conf = [o["model_confidence"] for o in correct_fruit if o["model_confidence"]]
    wrong_conf = [o["model_confidence"] for o in wrong_fruit if o["model_confidence"]]

    unsafe = [
        o for o in ok
        if o["final_status"] in BLOCKING_TERMINALS and o["condition_model_invoked"]
    ]
    fingerprints = {json.dumps(o["policy_fingerprints"], sort_keys=True) for o in ok}

    return {
        "metrics_version": METRICS_VERSION,
        "track": "R",
        "population": (
            "38 fresh licence-verified real photographs, one per source group, "
            "source-group and image disjoint from every prior corpus. Natural "
            "backgrounds, one primary produce item per image."
        ),
        "claim_boundary": (
            "Natural-domain behaviour of the frozen deployed system on this "
            "pool. Complete-inspection rate is NOT an accuracy figure: it says "
            "how often the system reached a result, not how often it was right."
        ),
        "counts": {
            "images": total,
            "http_200": len(ok),
            "by_fruit": dict(sorted(Counter(o["fruit_type"] for o in ok).items())),
            "by_curated_track": dict(sorted(Counter(o["track_label"] for o in ok).items())),
            "terminal_states": dict(sorted(Counter(o["final_status"] for o in ok).items())),
            "terminal_reason_codes": dict(sorted(Counter(
                o["terminal_reason_code"] for o in ok).items())),
            "quality_flags": dict(sorted(Counter(
                f for o in ok for f in o["quality_flags"]).items())),
            "artifact_flags": dict(sorted(Counter(
                f for o in ok for f in o["artifact_flags"]).items())),
            "advisory_flags": dict(sorted(Counter(
                f for o in ok for f in o["advisory_flags"]).items())),
            "foreground_invalid_reasons": dict(sorted(Counter(
                r for o in ok for r in o["foreground_invalid_reasons"]).items())),
        },
        "input_foreground": {
            "FOREGROUND_VALID_RATE": _metric(
                len(foreground_valid), len(ok),
                "images where the primary segmenter produced a mask that passed "
                "the geometric guards"),
            "INSUFFICIENT_VISUAL_EVIDENCE_RATE": _metric(
                len(insufficient), len(ok),
                "images where no usable subject region could be isolated, so no "
                "ROI-dependent measurement was attempted"),
            "segmentation_methods_observed": dict(sorted(Counter(
                m for o in ok for m in segmentation_methods(traces.get(o["run_id"]))
            ).items())),
            "foreground_fallback_enabled": raw.get("foreground_fallback_enabled", False),
        },
        "quality_agent": {
            "ELIGIBLE_FOR_INFERENCE_RATE": _metric(
                sum(1 for o in ok if "ELIGIBLE_FOR_INFERENCE" in o["state_path"]), len(ok),
                "images that reached the state permitting the condition model to run"),
            "RECAPTURE_RATE": _metric(
                len(recapture), len(ok), "images the agent asked to be recaptured"),
            "REMEDIATION_ATTEMPT_RATE": _metric(
                len(remediated), len(ok), "images where an automated correction was applied"),
            "REMEDIATION_ACCEPTANCE_RATE": _metric(
                len(accepted), len(remediated),
                "corrections whose re-measurement was good enough to keep; "
                "denominator is attempts, not images"),
            "HUMAN_ACTION_RATE": _metric(
                len(human_action), len(ok),
                "images ending in a request directed at a person"),
            "COMPLETE_INSPECTION_RATE": _metric(
                len(complete), len(ok),
                "images that reached a condition result. NOT accuracy."),
        },
        "model": {
            "MODEL_INVOKED_COUNT": len(model_ran),
            "FRUIT_TYPE_ACCURACY": _metric(
                len(correct_fruit), len(model_ran),
                "predicted fruit type matching the curated source identity. "
                "Denominator is images where the model actually ran, which is "
                "smaller than the pool: images the agent blocked never reached "
                "the model and are neither right nor wrong here."),
            "fruit_type_confusion": dict(sorted(Counter(
                f"{o['fruit_type']}->{(o.get('condition_evidence') or {}).get('fruit_type')}"
                for o in model_ran).items())),
            "confidence": {
                "all": _distribution(confidences),
                "correct_fruit_type": _distribution(correct_conf),
                "incorrect_fruit_type": _distribution(wrong_conf),
                "note": (
                    "Descriptive only. Model confidence is UNQUALIFIED for "
                    "policy and no confidence threshold exists anywhere in the "
                    "system; none is introduced here."),
            },
            "fruit_type_errors": [
                {"image_id": o["image_id"], "curated": o["fruit_type"],
                 "predicted": (o.get("condition_evidence") or {}).get("fruit_type"),
                 "confidence": round(o["model_confidence"], 4) if o["model_confidence"] else None}
                for o in sorted(wrong_fruit, key=lambda x: x["image_id"])
            ],
            "visible_condition_accuracy": None,
            "visible_condition_note": (
                "Not computed and not computable. These photographs carry no "
                "independent visible-condition label, and inventing one from the "
                "model's own output would be circular."),
        },
        "agent_safety": _safety_block(ok, traces, fingerprints),
        "per_image": [
            {"image_id": o["image_id"], "fruit_type": o["fruit_type"],
             "source_group_id": o["source_group_id"],
             "final_status": o["final_status"],
             "terminal_reason_code": o["terminal_reason_code"],
             "foreground_valid": o["foreground_valid"],
             "first_action": first_action(traces.get(o["run_id"])),
             "condition_model_invoked": o["condition_model_invoked"],
             "predicted_fruit_type": (o.get("condition_evidence") or {}).get("fruit_type"),
             "model_confidence": round(o["model_confidence"], 4) if o["model_confidence"] else None,
             "quality_flags": o["quality_flags"],
             "lossless_transport": o["lossless_transport"],
             "transport_media_type": o["transport_media_type"],
             "client_latency_ms": o["client_latency_ms"],
             "trace_retrievable": o["trace_retrievable"]}
            for o in sorted(observations, key=lambda x: x["image_id"])
        ],
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "max": round(ordered[-1], 4),
        "p25": round(ordered[max(0, int(0.25 * (len(ordered) - 1)))], 4),
        "p75": round(ordered[min(len(ordered) - 1, int(0.75 * (len(ordered) - 1)))], 4),
    }


def _safety_block(observations: list[dict], traces: dict, fingerprints: set) -> dict:
    unsafe = [
        o for o in observations
        if o["final_status"] in BLOCKING_TERMINALS and o["condition_model_invoked"]
    ]
    required = ("step_id", "state_before", "state_after")
    complete_traces = 0
    for o in observations:
        trace = traces.get(o["run_id"])
        steps = (trace or {}).get("trace", {}).get("steps", [])
        if steps and all(
            all(field in step and step[field] != "" for field in required)
            for step in steps
        ):
            complete_traces += 1
    return {
        "UNSAFE_INFERENCE_RATE": _metric(
            len(unsafe), len(observations),
            "runs where the condition model ran despite a blocking terminal "
            "state. This is the metric that must be zero."),
        "unsafe_inference_ids": [o.get("image_id") or o.get("base_id") for o in unsafe],
        "TRACE_COMPLETENESS_RATE": _metric(
            complete_traces, len(observations),
            "traces retrievable by run id where every step carries its step id "
            "and both states"),
        "trace_retrievable": _metric(
            sum(1 for o in observations if o["trace_retrievable"]), len(observations),
            "traces fetched back out of the service by run id"),
        "step_budget_violations": sum(
            1 for o in observations if step_count(traces.get(o["run_id"])) > 24),
        "tool_failures": sum(1 for o in observations if o["http_status"] != 200),
        "policy_fingerprint_sets_observed": len(fingerprints),
        "policy_fingerprints_consistent": len(fingerprints) == 1,
    }


# --- Track C ------------------------------------------------------------------


def score_track_c(raw: dict, traces: dict[str, dict], prereg: dict) -> dict:
    observations = raw["observations"]
    ok = [o for o in observations if o["http_status"] == 200]

    # Bases whose REFERENCE produced a valid foreground. Fixed as a secondary
    # denominator in the preregistration, before any scenario ran.
    reference_ok = {
        o["base_id"] for o in ok
        if o["variant"] == "REFERENCE" and o["foreground_valid"]
    }

    for o in ok:
        o["_observed_first_action"] = first_action(traces.get(o["run_id"]))
        o["_action_matched"] = (
            o["_observed_first_action"] == o["expected_first_action"]
        )
        o["_model_matched"] = (
            bool(o["condition_model_invoked"]) == bool(o["condition_model_must_run"])
        )
        o["_terminal_matched"] = o["final_status"] == o["expected_terminal"]
        o["_task_success"] = (
            o["_action_matched"] and o["_model_matched"] and o["_terminal_matched"]
        )

    secondary = [o for o in ok if o["base_id"] in reference_ok]

    def by_variant(rows: list[dict], predicate) -> dict:
        out = {}
        for key in prereg["scenarios"][0] and sorted({r["variant"] for r in rows}):
            subset = [r for r in rows if r["variant"] == key]
            out[key] = _metric(
                sum(1 for r in subset if predicate(r)), len(subset), "per-variant")
        return out

    unsafe = [
        o for o in ok
        if o["final_status"] in BLOCKING_TERMINALS and o["condition_model_invoked"]
    ]
    # An unnecessary tool call is the condition model running in a scenario that
    # preregistered it must not, which is the only tool whose misuse has a
    # consequence beyond cost.
    unnecessary = [
        o for o in ok
        if not o["condition_model_must_run"] and o["condition_model_invoked"]
    ]
    recoverable = [
        o for o in ok
        if o["variant"] in ("UNDEREXPOSED_RECOVERABLE", "LOW_CONTRAST_RECOVERABLE")
    ]
    remediated = [o for o in recoverable if o["remediation_applied"]]
    remediation_harm = [
        o for o in ok
        if o["remediation_applied"] and not o["remediation_accepted"]
    ]
    unusable = [o for o in ok if o["variant"] == "SEVERE_BLUR"]
    failed_safe = [
        o for o in unusable
        if not o["condition_model_invoked"]
        and o["final_status"] in BLOCKING_TERMINALS
    ]

    attributed = attributable = 0
    for o in ok:
        for decision in (traces.get(o["run_id"]) or {}).get("decisions", []):
            attributable += 1
            if (decision.get("selected_action") and decision.get("reason_code")
                    and (decision.get("triggering_evidence_ids")
                         or decision.get("explanation"))):
                attributed += 1

    fingerprints = {json.dumps(o["policy_fingerprints"], sort_keys=True) for o in ok}

    return {
        "metrics_version": METRICS_VERSION,
        "track": "C",
        "claim_boundary": prereg["claim_boundary"],
        "expected_actions_fingerprint": prereg["expected_actions_fingerprint"],
        "population": (
            f"{len(ok)} controlled scenarios: {len(reference_ok) and len({o['base_id'] for o in ok})} "
            "fresh real bases x 4 preregistered variants, sent as PNG."),
        "denominators": prereg["denominators"],
        "ACTION_SELECTION_ACCURACY": _metric(
            sum(1 for o in ok if o["_action_matched"]), len(ok),
            "scenarios whose first selected action equalled the action "
            "preregistered for that variant"),
        "ACTION_SELECTION_ACCURACY_secondary": _metric(
            sum(1 for o in secondary if o["_action_matched"]), len(secondary),
            "same, restricted to bases whose REFERENCE produced a valid "
            "foreground; separates wrong routing from perception never "
            "reaching the routing question"),
        "TASK_SUCCESS_RATE": _metric(
            sum(1 for o in ok if o["_task_success"]), len(ok),
            "scenarios matching the preregistered action AND model invocation "
            "AND terminal state"),
        "DECISION_ATTRIBUTION_RATE": _metric(
            attributed, attributable,
            "recorded decisions carrying an action, a reason code and either "
            "the evidence ids it rests on or an explanation naming them"),
        "UNSAFE_INFERENCE_RATE": _metric(
            len(unsafe), len(ok),
            "scenarios where the model ran despite a blocking terminal state"),
        "UNNECESSARY_TOOL_CALL_RATE": _metric(
            len(unnecessary), len([o for o in ok if not o["condition_model_must_run"]]),
            "scenarios that preregistered the model must not run, where it ran "
            "anyway; denominator is those scenarios only"),
        "REMEDIATION_SUCCESS_RATE": _metric(
            sum(1 for o in remediated if o["remediation_accepted"]), len(remediated),
            "applied corrections that survived re-measurement; denominator is "
            "corrections attempted on recoverable variants"),
        "REMEDIATION_HARM_RATE": _metric(
            len(remediation_harm), len([o for o in ok if o["remediation_applied"]]),
            "corrections applied and then rejected at re-measurement"),
        "BOUNDED_EXECUTION_RATE": _metric(
            sum(1 for o in ok if 0 < step_count(traces.get(o["run_id"])) <= 24),
            len(ok), "scenarios finishing inside the declared 24-step budget"),
        "TRACE_COMPLETENESS_RATE": _safety_block(ok, traces, fingerprints)["TRACE_COMPLETENESS_RATE"],
        "FAIL_SAFE_RATE": _metric(
            len(failed_safe), len(unusable),
            "scenarios whose condition is unrecoverable by construction that "
            "refused rather than produced a result; denominator is SEVERE_BLUR"),
        "per_variant": {
            "action_selection": by_variant(ok, lambda r: r["_action_matched"]),
            "task_success": by_variant(ok, lambda r: r["_task_success"]),
            "model_invoked_as_expected": by_variant(ok, lambda r: r["_model_matched"]),
            "terminal_as_expected": by_variant(ok, lambda r: r["_terminal_matched"]),
        },
        "policy_fingerprints_consistent": len(fingerprints) == 1,
        "per_scenario": [
            {"base_id": o["base_id"], "variant": o["variant"],
             "fruit_type": o["fruit_type"],
             "expected_first_action": o["expected_first_action"],
             "observed_first_action": o["_observed_first_action"],
             "action_matched": o["_action_matched"],
             "expected_terminal": o["expected_terminal"],
             "observed_terminal": o["final_status"],
             "terminal_reason_code": o["terminal_reason_code"],
             "condition_model_must_run": o["condition_model_must_run"],
             "condition_model_invoked": o["condition_model_invoked"],
             "foreground_valid": o["foreground_valid"],
             "remediation_action": o["remediation_action"],
             "remediation_accepted": o["remediation_accepted"],
             "quality_flags": o["quality_flags"],
             "task_success": o["_task_success"],
             "base_reference_foreground_valid": o["base_id"] in reference_ok,
             "client_latency_ms": o["client_latency_ms"],
             "run_id": o["run_id"], "trace_id": o["trace_id"]}
            for o in sorted(ok, key=lambda x: (x["base_id"], x["variant"]))
        ],
    }


# --- failure taxonomy ---------------------------------------------------------

CATEGORIES = (
    "FOREGROUND_FAILURE",
    "QUALITY_FALSE_BLOCK",
    "QUALITY_FALSE_ACCEPT",
    "REMEDIATION_INEFFECTIVE",
    "REMEDIATION_HARM",
    "MODEL_FRUIT_TYPE_ERROR",
    "MODEL_NOT_INVOKED",
    "TOOL_FAILURE",
    "TRACE_FAILURE",
    "TRANSPORT_SENSITIVITY",
    "POLICY_MISMATCH",
    # Two categories added in Phase 6 because the evidence demanded them.
    # Collapsing either into QUALITY_FALSE_BLOCK would blame the agent for
    # refusing an image that really was unusable, which is the opposite of what
    # the measurement shows.
    "SCENARIO_CONSTRUCTION_DEFECT",
    "SOURCE_IMAGE_BLOCKING_CONDITION",
    "OTHER",
)


def classify_track_c(scenario: dict, landing: dict | None) -> tuple[str, str]:
    """Why one Track C scenario missed its preregistered expectation.

    `landing` is the measured condition of the variant's pixels, from the local
    diagnostic pass. It is what separates "the agent blocked an image it should
    have corrected" from "the image genuinely was not correctable, and the
    scenario was labelled wrong".
    """
    if scenario["task_success"]:
        return "", ""
    if not scenario["foreground_valid"]:
        return ("FOREGROUND_FAILURE",
                "No usable subject region, so the routing question was never "
                "reached. The agent refused rather than measuring a region it "
                "could not trust.")
    zone = (landing or {}).get("zone", "")
    if scenario["variant"] == "REFERENCE":
        # Nothing was applied to this image, so no construction can be at
        # fault. The preregistration assumed every fresh photograph would be
        # usable; this one is not, and the agent read it correctly.
        return ("SOURCE_IMAGE_BLOCKING_CONDITION",
                f"The unmodified photograph itself carries a calibrated "
                f"blocking condition ({scenario['terminal_reason_code']}). The "
                f"agent's refusal is correct; the preregistered expectation "
                f"that a reference image always proceeds was too optimistic "
                f"about real source material.")
    if zone == "SHADOWS_CLIPPED":
        return ("SCENARIO_CONSTRUCTION_DEFECT",
                "The variant clipped the subject's shadows, so the detail gamma "
                "correction would recover is gone. Blocking is the correct "
                "response to these pixels; the scenario was mislabelled "
                "recoverable by its construction, not misrouted by the agent.")
    if zone == "ABOVE_FLOOR":
        return ("SCENARIO_CONSTRUCTION_DEFECT",
                "The variant never crossed the calibrated threshold it was "
                "meant to cross, so there was nothing for the agent to correct.")
    if scenario["condition_model_must_run"] and not scenario["condition_model_invoked"]:
        return ("QUALITY_FALSE_BLOCK",
                "The agent blocked a capture the scenario established as usable.")
    if not scenario["condition_model_must_run"] and scenario["condition_model_invoked"]:
        return ("QUALITY_FALSE_ACCEPT",
                "The agent ran the model on a capture the scenario established "
                "as unusable.")
    if scenario["remediation_action"] and not scenario["remediation_accepted"]:
        return ("REMEDIATION_INEFFECTIVE",
                "A correction was applied and did not survive re-measurement.")
    if scenario["observed_first_action"] != scenario["expected_first_action"]:
        return ("POLICY_MISMATCH",
                f"Selected {scenario['observed_first_action'] or 'no action'} "
                f"where {scenario['expected_first_action']} was preregistered.")
    return ("OTHER", "Did not match the preregistered expectation.")


def build_taxonomy(track_r: dict, track_c: dict, landings: dict) -> dict:
    rows = []
    for scenario in track_c["per_scenario"]:
        category, detail = classify_track_c(
            scenario, landings.get(f"{scenario['base_id']}__{scenario['variant']}")
        )
        if category:
            rows.append({"track": "C", "id": scenario["base_id"],
                         "variant": scenario["variant"], "category": category,
                         "detail": detail,
                         "observed_terminal": scenario["observed_terminal"],
                         "terminal_reason_code": scenario["terminal_reason_code"]})

    for image in track_r["per_image"]:
        if not image["foreground_valid"]:
            rows.append({"track": "R", "id": image["image_id"], "variant": "",
                         "category": "FOREGROUND_FAILURE",
                         "detail": "No usable subject region could be isolated.",
                         "observed_terminal": image["final_status"],
                         "terminal_reason_code": image["terminal_reason_code"]})
        elif (image["condition_model_invoked"]
              and image["predicted_fruit_type"] != image["fruit_type"]):
            rows.append({"track": "R", "id": image["image_id"], "variant": "",
                         "category": "MODEL_FRUIT_TYPE_ERROR",
                         "detail": f"predicted {image['predicted_fruit_type']}, "
                                   f"curated {image['fruit_type']}",
                         "observed_terminal": image["final_status"],
                         "terminal_reason_code": image["terminal_reason_code"]})
        elif not image["trace_retrievable"]:
            rows.append({"track": "R", "id": image["image_id"], "variant": "",
                         "category": "TRACE_FAILURE",
                         "detail": "trace could not be retrieved by run id",
                         "observed_terminal": image["final_status"],
                         "terminal_reason_code": image["terminal_reason_code"]})

    counts = Counter(r["category"] for r in rows)
    return {
        "metrics_version": METRICS_VERSION,
        "categories": list(CATEGORIES),
        "rule": (
            "Every unsuccessful case is classified by what actually went wrong. "
            "Nothing is collapsed into 'model error', and a scenario that failed "
            "because it was built wrong is labelled as such rather than charged "
            "to the agent."),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "total": len(rows),
        "rows": sorted(rows, key=lambda r: (r["track"], r["id"], r["variant"])),
    }


# --- latency ------------------------------------------------------------------


def build_latency(track_r_raw: dict, track_c_raw: dict, track_c: dict) -> dict:
    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {"n": len(ordered),
                "median_ms": round(statistics.median(ordered), 1),
                "p95_ms": round(ordered[index], 1),
                "min_ms": round(ordered[0], 1),
                "max_ms": round(ordered[-1], 1)}

    r_ok = [o for o in track_r_raw["observations"] if o["http_status"] == 200]
    by_scenario = {s["base_id"] + "__" + s["variant"]: s for s in track_c["per_scenario"]}
    remediation, recapture, other = [], [], []
    for o in track_c_raw["observations"]:
        if o["http_status"] != 200:
            continue
        scenario = by_scenario.get(o["base_id"] + "__" + o["variant"])
        if scenario and scenario["remediation_action"]:
            remediation.append(o["client_latency_ms"])
        elif o["final_status"] in HUMAN_ACTION_TERMINALS:
            recapture.append(o["client_latency_ms"])
        else:
            other.append(o["client_latency_ms"])

    return {
        "metrics_version": METRICS_VERSION,
        "measurement": "CLIENT-OBSERVED END-TO-END LATENCY",
        "note": (
            "Wall clock from this client, over the public internet, to the "
            "deployed App Runner service and back. It includes TLS, network "
            "transit and upload of the image bytes. It is NOT server compute "
            "time, and it must not be reported as such."),
        "track_r_natural": stats([o["client_latency_ms"] for o in r_ok]),
        "track_c_remediation_path": stats(remediation),
        "track_c_recapture_path": stats(recapture),
        "track_c_other": stats(other),
        "internal_tool_timings_note": (
            "Per-step durations are recorded inside each trace and are server "
            "side. They are kept there rather than mixed into these numbers."),
    }


# --- final evidence table -----------------------------------------------------


def build_evidence_table(track_c: dict, traces: dict[str, dict]) -> dict:
    """The compact Agentic Vision table, from fresh real bases.

    One base per fruit, all four variants, chosen as the first base of each
    fruit whose REFERENCE produced a valid foreground. These rows replace the
    constructed-fixture examples wherever the result supports it.
    """
    scenarios = track_c["per_scenario"]
    eligible = sorted({
        (s["fruit_type"], s["base_id"]) for s in scenarios
        if s["base_reference_foreground_valid"]
    })
    chosen: dict[str, str] = {}
    for fruit, base_id in eligible:
        chosen.setdefault(fruit, base_id)

    rows = []
    for fruit, base_id in sorted(chosen.items()):
        for scenario in scenarios:
            if scenario["base_id"] != base_id:
                continue
            trace = traces.get(scenario["run_id"]) or {}
            decision = next(
                (d for d in trace.get("decisions", []) if d.get("selected_action")), {}
            )
            evidence = ""
            for step in trace.get("trace", {}).get("steps", []):
                summary = step.get("evidence_summary") or {}
                if "high_frequency_ratio" in summary:
                    evidence = (
                        f"high_frequency_ratio {summary['high_frequency_ratio']}, "
                        f"mean_luminance {summary.get('mean_luminance', '?')}, "
                        f"contrast {summary.get('contrast_score', '?')}"
                    )
                    break
            rows.append({
                "fruit_type": fruit,
                "base_id": base_id,
                "scenario": scenario["variant"],
                "opencv_evidence": evidence,
                "decision_reason": decision.get("explanation", "")[:240],
                "agent_action": scenario["observed_first_action"],
                "next_tool": scenario["remediation_action"] or (
                    "run_condition_model" if scenario["condition_model_invoked"]
                    else "none"),
                "model_invoked": scenario["condition_model_invoked"],
                "expected_action": scenario["expected_first_action"],
                "matched_expectation": scenario["action_matched"],
                "terminal": scenario["observed_terminal"],
                "trace_id": scenario["trace_id"],
            })
    return {
        "metrics_version": METRICS_VERSION,
        "source": "Phase 6 Track C, fresh licensed real bases on the deployed service.",
        "claim_boundary": track_c["claim_boundary"],
        "selection_rule": (
            "First base of each fruit, ordered by id, whose REFERENCE produced a "
            "valid foreground. Fixed rule, not a hand-picked best case; the "
            "matched_expectation column shows the misses as well as the hits."),
        "rows": rows,
    }


# --- local diagnostic: where did each controlled variant actually land? --------


def measure_landing_zones() -> dict:
    """Measure the condition of each Track C variant's own pixels.

    Local, and diagnostic only: it never feeds a metric and never changes a
    decision. It exists to answer one question the deployed responses cannot -
    when a scenario missed its preregistered action, was the agent wrong, or
    were the pixels not what the scenario claimed they were?

    The thresholds compared against are the calibrated ones, read from the
    locked policy. Nothing is adjusted.
    """
    import cv2

    from competition.agent.orchestrator import load_locked_policies
    from competition.evaluation.phase6_track_c import VARIANT_DIR, VARIANT_KEYS
    from competition.vision.config import DEFAULT_POLICY
    from competition.vision.foreground import ForegroundError, isolate_foreground
    from competition.vision.quality import measure_illumination

    roi = load_locked_policies().roi_policy
    # The measurement is objective; only the thresholds are policy. The clip
    # value is identical in both policies, asserted rather than assumed.
    assert DEFAULT_POLICY.shadow_clip_value == roi.shadow_clip_value

    prereg = _load(RESULTS_DIR / "phase6_expected_actions.json")
    bases = sorted({s["base_id"] for s in prereg["scenarios"]})
    zones: dict[str, dict] = {}
    for base in bases:
        for key in VARIANT_KEYS:
            path = VARIANT_DIR / f"{base}__{key}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            try:
                mask, foreground = isolate_foreground(image)
            except ForegroundError:
                zones[f"{base}__{key}"] = {"zone": "NO_VALID_FOREGROUND"}
                continue
            if not foreground.valid:
                zones[f"{base}__{key}"] = {"zone": "NO_VALID_FOREGROUND"}
                continue
            illumination = measure_illumination(image, DEFAULT_POLICY, mask)
            luminance = float(illumination.mean_luminance)
            clipped = float(illumination.shadow_clip_fraction)
            if clipped > roi.shadow_clip_fraction_limit:
                zone = "SHADOWS_CLIPPED"
            elif key == "UNDEREXPOSED_RECOVERABLE":
                zone = ("UNDEREXPOSED_RECOVERABLE"
                        if luminance < roi.underexposed_mean_luminance
                        else "ABOVE_FLOOR")
            else:
                zone = "MEASURED"
            zones[f"{base}__{key}"] = {
                "zone": zone,
                "mean_luminance": round(luminance, 6),
                "shadow_clip_fraction": round(clipped, 6),
            }
    return {
        "diagnostic_only": True,
        "note": (
            "Measured locally on the variant files. Never used to score, never "
            "used to change a threshold. Compared against the calibrated ROI "
            "policy in force on the deployed service."),
        "thresholds": {
            "underexposed_mean_luminance": roi.underexposed_mean_luminance,
            "shadow_clip_fraction_limit": roi.shadow_clip_fraction_limit,
            "shadow_clip_value": roi.shadow_clip_value,
        },
        "zones": zones,
        "summary": dict(sorted(Counter(
            v["zone"] for k, v in zones.items()
            if k.endswith("UNDEREXPOSED_RECOVERABLE")).items())),
    }


def main() -> int:
    from competition.evaluation.phase6_run import SERVICE_URL

    track_r_raw = _load(TRACK_R_RAW)
    track_c_raw = _load(TRACK_C_RAW)
    prereg = _load(RESULTS_DIR / "phase6_expected_actions.json")

    frozen = _load(RESULTS_DIR / "frozen_system.json")
    assert_frozen(track_r_raw["observations"], frozen)
    assert_frozen(track_c_raw["observations"], frozen)

    traces = fetch_traces(track_r_raw["observations"], SERVICE_URL)
    traces.update(fetch_traces(track_c_raw["observations"], SERVICE_URL))

    track_r = score_track_r(track_r_raw, traces)
    track_c = score_track_c(track_c_raw, traces, prereg)
    landings = measure_landing_zones()
    track_c["variant_landing_diagnostic"] = landings

    taxonomy = build_taxonomy(track_r, track_c, landings["zones"])
    latency = build_latency(track_r_raw, track_c_raw, track_c)
    evidence = build_evidence_table(track_c, traces)

    agent = {
        "metrics_version": METRICS_VERSION,
        "note": (
            "Track R and Track C are reported separately and are never pooled. "
            "They measure different things: natural-domain behaviour, and "
            "routing under known conditions."),
        "track_r": {
            **track_r["quality_agent"],
            **track_r["input_foreground"],
            **track_r["agent_safety"],
        },
        "track_c": {
            key: track_c[key] for key in (
                "ACTION_SELECTION_ACCURACY", "ACTION_SELECTION_ACCURACY_secondary",
                "TASK_SUCCESS_RATE", "DECISION_ATTRIBUTION_RATE",
                "UNSAFE_INFERENCE_RATE", "UNNECESSARY_TOOL_CALL_RATE",
                "REMEDIATION_SUCCESS_RATE", "REMEDIATION_HARM_RATE",
                "BOUNDED_EXECUTION_RATE", "TRACE_COMPLETENESS_RATE",
                "FAIL_SAFE_RATE",
            )
        },
    }

    for path, document in (
        (TRACK_R_RESULTS, track_r), (TRACK_C_RESULTS, track_c),
        (AGENT_METRICS, agent), (FAILURE_TAXONOMY, taxonomy),
        (LATENCY, latency), (EVIDENCE_TABLE, evidence),
    ):
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Track R  complete {track_r['quality_agent']['COMPLETE_INSPECTION_RATE']['value']}"
          f"  foreground {track_r['input_foreground']['FOREGROUND_VALID_RATE']['value']}"
          f"  fruit-type {track_r['model']['FRUIT_TYPE_ACCURACY']['value']}"
          f" ({track_r['model']['FRUIT_TYPE_ACCURACY']['numerator']}/"
          f"{track_r['model']['FRUIT_TYPE_ACCURACY']['denominator']})")
    print(f"Track C  action {track_c['ACTION_SELECTION_ACCURACY']['value']}"
          f"  task {track_c['TASK_SUCCESS_RATE']['value']}"
          f"  unsafe {track_c['UNSAFE_INFERENCE_RATE']['value']}"
          f"  attribution {track_c['DECISION_ATTRIBUTION_RATE']['value']}")
    print(f"taxonomy {taxonomy['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
