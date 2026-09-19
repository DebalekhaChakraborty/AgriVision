"""Trace one degradation family from transform to gate outcome, on calibration data.

Written for the Phase 2c-B consistency audit, which asked why controlled
contrast reduction produced no escalations when a contrast threshold had been
calibrated. Three explanations were possible - the metric does not respond, the
threshold is never crossed, or the flag is raised but does not block - and only
a trace through every stage distinguishes them.

Runs on **calibration groups only**. The held-out artifacts are confirmatory and
frozen; re-deriving a mechanism from them would be a second look at data that
has already been spent.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.agent.inspection import ADVISORY_FLAGS, DEFAULT_BLOCKING_FLAGS
from competition.evaluation.build_licensed_corpus import local_path_for
from competition.evaluation.calibrate_roi_policy import RESULTS_DIR, _measure_one, gate_sample
from competition.evaluation.degrade_licensed_corpus import DEGRADATION_LADDER
from competition.evaluation.evaluate_locked_policy import load_locked_policy
from competition.evaluation.split_licensed_corpus import calibration_view
from competition.vision.degradation import DegradationSpec, apply_degradation

AUDIT_PATH = RESULTS_DIR / "phase2c_contrast_audit.json"
AUDIT_VERSION = "phase2c-contrast-path-audit-1.0.0"
AUDIT_FAMILIES = ("reduce_contrast", "glare", "occlude")


def audit(families: tuple[str, ...] = AUDIT_FAMILIES, limit: int | None = None) -> dict:
    policy = load_locked_policy()
    records = [r for r in calibration_view().records if r.track == "CLEAN_BASE"]
    if limit:
        records = records[:limit]
    rungs = [rung for rung in DEGRADATION_LADDER if rung.kind in families]

    rows: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "blocked": 0, "advisory_flagged": 0, "no_mask": 0,
                 "contrast_scores": [], "flags": defaultdict(int)}
    )

    for record in records:
        base = cv2.imread(
            str(local_path_for(record.image_id, record.file_extension)), cv2.IMREAD_COLOR
        )
        if base is None:
            continue
        for rung in rungs:
            image, _ = apply_degradation(
                base, DegradationSpec(rung.kind, rung.level, 20260918)
            )
            measured = _measure_one(image)
            key = (rung.kind, rung.level, rung.expected.value)
            if not measured.get("roi_available"):
                rows[key]["no_mask"] += 1
                continue
            sample = {"width": record.width, "height": record.height,
                      "roi_available": True, "roi": measured["roi"]}
            flags = set(gate_sample(sample, policy, "roi")["flags"])
            entry = rows[key]
            entry["n"] += 1
            entry["blocked"] += bool(flags & set(DEFAULT_BLOCKING_FLAGS))
            entry["advisory_flagged"] += bool(flags & set(ADVISORY_FLAGS))
            entry["contrast_scores"].append(measured["roi"]["contrast_score"])
            for flag in flags:
                entry["flags"][flag] += 1

    stages = []
    for (kind, level, verdict), entry in sorted(rows.items()):
        scores = entry["contrast_scores"]
        stages.append({
            "family": kind,
            "level": level,
            "expected_verdict": verdict,
            "assessable": entry["n"],
            "no_valid_mask": entry["no_mask"],
            "median_roi_contrast": float(np.median(scores)) if scores else None,
            "advisory_flagged": entry["advisory_flagged"],
            "blocked": entry["blocked"],
            "flag_counts": dict(sorted(entry["flags"].items())),
        })

    by_family = {}
    for family in families:
        scored = [s for s in stages if s["family"] == family
                  and s["expected_verdict"] == "UNUSABLE"]
        assessable = sum(s["assessable"] for s in scored)
        blocked = sum(s["blocked"] for s in scored)
        flagged = sum(s["advisory_flagged"] for s in scored)
        by_family[family] = {
            "scored_unusable": assessable,
            "blocked": blocked,
            "block_rate": blocked / assessable if assessable else None,
            "advisory_flagged": flagged,
            "detection_rate_including_advisory": (
                (blocked + flagged) / assessable if assessable else None
            ),
            "false_accepts": assessable - blocked,
        }

    return {
        "audit_version": AUDIT_VERSION,
        "audited_on": date.today().isoformat(),
        "split": "calibration groups only; held-out artifacts are frozen",
        "policy_threshold_fingerprint": policy.threshold_fingerprint(),
        "low_contrast_limit": policy.low_contrast_limit,
        "blocking_flags": sorted(DEFAULT_BLOCKING_FLAGS),
        "advisory_flags": sorted(ADVISORY_FLAGS),
        "base_images": len(records),
        "stages": stages,
        "by_family": by_family,
        "finding": (
            "LOW_CONTRAST is raised by the calibrated threshold but is classified "
            "advisory, not blocking, so it never escalates. Glare and occlusion "
            "have no detector at all and are blocked only incidentally."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.audit_contrast_path",
        description="Trace degradation families through the ROI gate on calibration data.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    document = audit(limit=args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["by_family"], indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
