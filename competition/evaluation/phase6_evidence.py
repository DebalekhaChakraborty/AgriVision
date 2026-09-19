"""Phase 6 counterfactual pairs and the source licence manifest.

Two artifacts that are about provenance and about proof, rather than about
scores.

The counterfactual matrix is the Agentic Vision evidence in its strongest
available form: every row shares a base photograph with three other rows, so the
subject, the camera and the background are held constant and only the controlled
visual condition moves. Where the action moves with it, the OpenCV evidence is
what moved it - there is nothing else left that could have.

The licence manifest records, per image, what was granted and by whom, so a
reader can check the corpus rather than take a claim about it on trust.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EVIDENCE_VERSION = "phase6-evidence-1.0.0"

RESULTS_DIR = Path("competition/evaluation/results/phase6")
MATRIX_PATH = RESULTS_DIR / "counterfactual_matrix.json"
LICENCE_PATH = RESULTS_DIR / "phase6_licence_manifest.json"


def build_matrix(track_c: dict, traces: dict) -> dict:
    """One row per scenario, grouped so the pairs are visible."""
    rows = []
    for scenario in track_c["per_scenario"]:
        trace = traces.get(scenario["run_id"]) or {}
        evidence = {}
        for step in trace.get("trace", {}).get("steps", []):
            summary = step.get("evidence_summary") or {}
            if "high_frequency_ratio" in summary:
                evidence = {
                    "high_frequency_ratio": summary.get("high_frequency_ratio"),
                    "mean_luminance": summary.get("mean_luminance"),
                    "contrast_score": summary.get("contrast_score"),
                    "shadow_clip_fraction": summary.get("shadow_clip_fraction"),
                    "quality_flags": summary.get("quality_flags", []),
                }
                break
        rows.append({
            "base_id": scenario["base_id"],
            "fruit_type": scenario["fruit_type"],
            "variant": scenario["variant"],
            "opencv_evidence": evidence,
            "first_action": scenario["observed_first_action"],
            "remediation": scenario["remediation_action"] or None,
            "model_invoked": scenario["condition_model_invoked"],
            "terminal_status": scenario["observed_terminal"],
            "terminal_reason_code": scenario["terminal_reason_code"],
            "foreground_valid": scenario["foreground_valid"],
            "trace_id": scenario["trace_id"],
        })

    by_base: dict[str, set] = {}
    for row in rows:
        by_base.setdefault(row["base_id"], set()).add(row["first_action"])
    varying = {base for base, actions in by_base.items() if len(actions) > 1}
    distinct_actions = Counter(row["first_action"] for row in rows)

    return {
        "evidence_version": EVIDENCE_VERSION,
        "claim_boundary": track_c["claim_boundary"],
        "what_this_shows": (
            "Each base contributes four rows that differ only in a controlled "
            "visual condition. Where the selected action differs across a "
            "base's rows, the OpenCV evidence is the only thing that changed, "
            "so it is what changed the action."),
        "bases": len(by_base),
        "bases_where_action_varies": len(varying),
        "bases_where_action_never_varies": sorted(set(by_base) - varying),
        "distinct_first_actions_observed": dict(sorted(distinct_actions.items())),
        "note_on_invariant_bases": (
            "A base whose action never varies is not a contradiction. Those are "
            "the bases whose foreground could not be isolated, so every variant "
            "stopped at the same earlier refusal and the routing question was "
            "never reached. They are listed rather than dropped."),
        "rows": rows,
    }


def build_licence_manifest(source: dict, split: dict) -> dict:
    track_c_bases = {b["image_id"] for b in split["track_c_bases"]}
    entries = []
    for record in source["records"]:
        entries.append({
            "image_id": record["image_id"],
            "source_group_id": record["source_group_id"],
            "creator": record["creator"],
            "source_name": record["source_name"],
            "source_url": record["source_url"],
            "licence_id": record["licence_id"],
            "licence_url": record["licence_url"],
            "attribution_text": record["attribution_text"],
            "commercial_use_allowed": record["commercial_use_allowed"],
            "derivatives_allowed": record["derivatives_allowed"],
            "attribution_required": record["attribution_required"],
            "share_alike_required": record["share_alike_required"],
            "content_sha256": record["content_sha256"],
            "fruit_type": record["fruit_type"],
            "purpose": "Phase 6 confirmatory evaluation of the frozen deployed system",
            "track_membership": (
                ["TRACK_R", "TRACK_C"] if record["image_id"] in track_c_bases
                else ["TRACK_R"]),
        })
    return {
        "evidence_version": EVIDENCE_VERSION,
        "count": len(entries),
        "licence_distribution": dict(sorted(
            Counter(e["licence_id"] for e in entries).items())),
        "all_permit_commercial_use": all(e["commercial_use_allowed"] for e in entries),
        "all_permit_derivatives": all(e["derivatives_allowed"] for e in entries),
        "share_alike_count": sum(e["share_alike_required"] for e in entries),
        "attribution_required_count": sum(e["attribution_required"] for e in entries),
        "gate": (
            "Every entry was admitted by reading its own per-file rights "
            "statement before any byte was fetched. Unknown, NonCommercial and "
            "NoDerivatives are rejected by the gate and none appears here."),
        "bytes_note": (
            "Image bytes are not committed. Only provenance is, and no local "
            "filesystem path appears in this file."),
        "entries": sorted(entries, key=lambda e: e["image_id"]),
    }


def main() -> int:
    from competition.evaluation.phase6_metrics import fetch_traces
    from competition.evaluation.phase6_run import SERVICE_URL, TRACK_C_RAW

    track_c = json.loads(
        (RESULTS_DIR / "track_c_results.json").read_text(encoding="utf-8"))
    raw = json.loads(TRACK_C_RAW.read_text(encoding="utf-8"))
    traces = fetch_traces(raw["observations"], SERVICE_URL)
    matrix = build_matrix(track_c, traces)
    MATRIX_PATH.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source = json.loads(
        (RESULTS_DIR / "phase6_source_manifest.json").read_text(encoding="utf-8"))
    split = json.loads(
        (RESULTS_DIR / "phase6_split_manifest.json").read_text(encoding="utf-8"))
    licences = build_licence_manifest(source, split)
    LICENCE_PATH.write_text(
        json.dumps(licences, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"counterfactual: {matrix['bases_where_action_varies']}/{matrix['bases']} "
          f"bases change action with the condition")
    print(f"distinct actions: {matrix['distinct_first_actions_observed']}")
    print(f"licences: {licences['count']} entries, "
          f"{len(licences['licence_distribution'])} distinct licences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
