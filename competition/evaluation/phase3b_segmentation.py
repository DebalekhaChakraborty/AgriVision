"""Phase 3b — bounded foreground-recovery evaluation.

Measures what the fallback ladder recovers, and reports it split by whether the
photograph contains a single subject at all. That split is the finding: the two
halves behave so differently that a single pooled number misdescribes both.

Reproduce:
    python -m competition.evaluation.phase3b_segmentation diagnose
    python -m competition.evaluation.phase3b_segmentation recover
    python -m competition.evaluation.phase3b_segmentation latency
    python -m competition.evaluation.phase3b_segmentation all
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from competition.vision.foreground import (
    DEFAULT_GUARDS,
    ForegroundMethod,
    isolate_foreground,
)
from competition.vision.foreground_fallback import (
    FALLBACK_LADDER,
    FallbackMethod,
    MaskProvenance,
    isolate_foreground_with_fallback,
)

PHASE3B_VERSION = "phase3b-segmentation-1.0.0"

_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = _ROOT / "data" / "licensed_real" / "manifests"
RAW_DIR = _ROOT / "data" / "licensed_real" / "raw"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase3b"

# Development population only. The Phase 2c-B held-out groups were opened once
# and the Phase 2d validation set was opened once; developing a method against
# either would spend evidence that exists precisely to be unspent.
DEVELOPMENT_SPLIT = "calibration"


class Phase3bError(RuntimeError):
    pass


def load_development_records() -> list:
    """Calibration-split records with a readable local file."""
    split_path = MANIFESTS / "corpus_split.json"
    corpus_path = MANIFESTS / "licensed_corpus.json"
    if not split_path.is_file() or not corpus_path.is_file():
        raise Phase3bError(
            "licensed corpus manifests not found; Phase 2c-B must be run first"
        )
    assignment = json.loads(split_path.read_text(encoding="utf-8"))["assignment"]
    records = json.loads(corpus_path.read_text(encoding="utf-8"))["records"]
    wanted = []
    for record in records:
        if assignment.get(record["source_group_id"]) != DEVELOPMENT_SPLIT:
            continue
        matches = sorted(RAW_DIR.glob(f"{record['image_id']}*"))
        if matches:
            wanted.append((record, matches[0]))
    return wanted


def diagnose() -> dict:
    """Characterise where and how the primary fails, split by track."""
    rows = []
    for record, path in load_development_records():
        image = cv2.imread(str(path))
        if image is None:
            continue
        _, evidence = isolate_foreground(image)
        rows.append({
            "image_id": record["image_id"],
            "track": record["track"],
            "fruit_type": record["fruit_type"],
            "valid": evidence.valid,
            "invalid_reasons": list(evidence.invalid_reasons),
            "foreground_fraction": round(evidence.foreground_fraction, 4),
            "largest_component_fraction": round(evidence.largest_component_fraction, 4),
            "border_contact_fraction": round(evidence.border_contact_fraction, 4),
            "component_count": evidence.component_count,
        })

    reasons = Counter()
    for row in rows:
        for reason in row["invalid_reasons"]:
            reasons[reason] += 1

    by_track = {}
    for track in sorted({row["track"] for row in rows}):
        subset = [row for row in rows if row["track"] == track]
        valid = [row for row in subset if row["valid"]]
        by_track[track] = {
            "n": len(subset),
            "valid": len(valid),
            "validity_rate": round(len(valid) / len(subset), 4) if subset else None,
        }

    # The dominant failure shape, quantified rather than asserted.
    near_full = [
        row for row in rows
        if not row["valid"] and row["foreground_fraction"] >= 0.68
        and row["largest_component_fraction"] >= 0.95
    ]

    return {
        "phase3b_version": PHASE3B_VERSION,
        "population": "Phase 2c-B calibration split (development only)",
        "claim_boundary": (
            "validity means the geometric guards passed, NOT that the mask is "
            "correct; no ground-truth masks exist for these photographs"
        ),
        "image_count": len(rows),
        "valid": sum(1 for row in rows if row["valid"]),
        "validity_rate": round(
            sum(1 for row in rows if row["valid"]) / len(rows), 4) if rows else None,
        "by_track": by_track,
        "failure_reasons": dict(reasons.most_common()),
        "dominant_failure_shape": {
            "description": (
                "foreground fraction >= 0.68 with a single component covering "
                "essentially the whole frame: Otsu found no saturation "
                "bimodality because the entire scene is colourful"
            ),
            "count": len(near_full),
            "image_ids": [row["image_id"] for row in near_full],
        },
        "rows": rows,
    }


def recover() -> dict:
    """Run the ladder and report recovery, split by track."""
    rows = []
    for record, path in load_development_records():
        image = cv2.imread(str(path))
        if image is None:
            continue
        started = time.perf_counter()
        _, evidence, fallback = isolate_foreground_with_fallback(image)
        elapsed = (time.perf_counter() - started) * 1000.0
        rows.append({
            "image_id": record["image_id"],
            "track": record["track"],
            "provenance": fallback.provenance,
            "accepted_method": fallback.accepted_method,
            "valid": evidence.valid,
            "foreground_fraction": round(evidence.foreground_fraction, 4),
            "attempts": [a.to_dict(include_timing=False) for a in fallback.attempts],
            "total_ms": round(elapsed, 1),
        })

    def summarise(subset: list) -> dict:
        primary = [r for r in subset if r["provenance"] == MaskProvenance.PRIMARY.value]
        recovered = [r for r in subset if r["provenance"] == MaskProvenance.FALLBACK.value]
        unresolved = [r for r in subset if r["provenance"] == MaskProvenance.NONE.value]
        return {
            "n": len(subset),
            "primary_valid": len(primary),
            "recovered_by_fallback": len(recovered),
            "still_unresolved": len(unresolved),
            "validity_before": round(len(primary) / len(subset), 4) if subset else None,
            "validity_after": round(
                (len(primary) + len(recovered)) / len(subset), 4) if subset else None,
            "recovered_image_ids": [r["image_id"] for r in recovered],
        }

    by_track = {
        track: summarise([r for r in rows if r["track"] == track])
        for track in sorted({r["track"] for r in rows})
    }

    return {
        "phase3b_version": PHASE3B_VERSION,
        "population": "Phase 2c-B calibration split (development only)",
        "claim_boundary": (
            "'recovered' means a fallback mask passed the same unrelaxed "
            "geometric guards. It does NOT mean the mask is correct. On "
            "multi-subject scenes a compact block carved out of a fruit pile "
            "passes every guard, and the guards cannot distinguish it from a "
            "single subject."
        ),
        "guards_relaxed": False,
        "ladder": [m.value for m in FALLBACK_LADDER],
        "overall": summarise(rows),
        "by_track": by_track,
        "accepted_method_counts": dict(Counter(
            r["accepted_method"] for r in rows if r["accepted_method"]).most_common()),
        "rows": rows,
    }


ADJUDICATION_PATH = RESULTS_DIR / "adjudication.json"


def adjudicate() -> dict:
    """Rates computed from the recorded visual adjudication.

    Separated from `recover` on purpose. `recover` answers "did a mask pass the
    guards"; this answers "was the mask right". They are different questions and
    the phase turns on the gap between their answers, so they are never merged
    into one number.
    """
    if not ADJUDICATION_PATH.is_file():
        raise Phase3bError(f"{ADJUDICATION_PATH.name} not found")
    record = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    verdicts = record["verdicts"]

    def rates(subset: list) -> dict:
        counts = Counter(row["verdict"] for row in subset)
        usable = counts["CORRECT"] + counts["ACCEPTABLE"]
        return {
            "n": len(subset),
            "CORRECT": counts["CORRECT"],
            "ACCEPTABLE": counts["ACCEPTABLE"],
            "WRONG": counts["WRONG"],
            "usable": usable,
            "usable_rate": round(usable / len(subset), 4) if subset else None,
            "wrong_rate": round(counts["WRONG"] / len(subset), 4) if subset else None,
        }

    by_track = {
        track: rates([r for r in verdicts if r["track"] == track])
        for track in sorted({r["track"] for r in verdicts})
    }
    return {
        "phase3b_version": PHASE3B_VERSION,
        "evidence_type": "SINGLE_RATER_VISUAL_ADJUDICATION",
        "claim_boundary": record["what_this_is"],
        "overall": rates(verdicts),
        "by_track": by_track,
        "interpretation": (
            "The ladder is reliable exactly where the product contract applies "
            "and unreliable outside it. On single-subject captures every "
            "recovered mask was usable. On multi-subject scenes 18 of 19 were "
            "an arbitrary block carved out of a pile - which passes every "
            "geometric guard, because a block cut from a heap of apples has the "
            "same geometry as one apple."
        ),
        "verdicts": verdicts,
    }


def latency(repeats: int = 3) -> dict:
    """Cost of the ladder, separated by whether it actually fired."""
    primary_ms, fallback_ms = [], []
    for record, path in load_development_records():
        image = cv2.imread(str(path))
        if image is None:
            continue
        for _ in range(repeats):
            started = time.perf_counter()
            _, _, fallback = isolate_foreground_with_fallback(image)
            elapsed = (time.perf_counter() - started) * 1000.0
            (primary_ms if fallback.provenance == MaskProvenance.PRIMARY.value
             else fallback_ms).append(elapsed)

    def stats(values: list) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "median_ms": round(statistics.median(ordered), 1),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
            "max_ms": round(ordered[-1], 1),
        }

    return {
        "phase3b_version": PHASE3B_VERSION,
        "claim_boundary": (
            "local CPU on full-resolution licensed photographs; NOT an AWS figure"
        ),
        "primary_accepted": stats(primary_ms),
        "fallback_invoked": stats(fallback_ms),
        "note": (
            "The ladder costs nothing when the primary succeeds: it is not "
            "consulted. The fallback figure is the price of an image that would "
            "otherwise have been refused outright."
        ),
    }


def _write(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list) -> int:
    stage = argv[1] if len(argv) > 1 else "all"

    if stage in ("diagnose", "all"):
        report = diagnose()
        _write("diagnosis.json", report)
        print(f"diagnosis: {report['valid']}/{report['image_count']} valid "
              f"({report['validity_rate']:.1%})")
        for track, row in report["by_track"].items():
            print(f"   {track:16s} {row['valid']:2d}/{row['n']:2d}  {row['validity_rate']:.1%}")
        print(f"   dominant failure shape: {report['dominant_failure_shape']['count']} images")
        for reason, count in report["failure_reasons"].items():
            print(f"     {reason:32s} {count}")

    if stage in ("recover", "all"):
        report = recover()
        _write("recovery.json", report)
        overall = report["overall"]
        print(f"\nrecovery: {overall['validity_before']:.1%} -> "
              f"{overall['validity_after']:.1%} "
              f"(+{overall['recovered_by_fallback']} recovered, "
              f"{overall['still_unresolved']} unresolved)")
        for track, row in report["by_track"].items():
            print(f"   {track:16s} {row['validity_before']:.1%} -> {row['validity_after']:.1%}  "
                  f"(+{row['recovered_by_fallback']})")
        print(f"   accepted methods: {report['accepted_method_counts']}")

    if stage in ("adjudicate", "all"):
        report = adjudicate()
        _write("adjudication_rates.json", report)
        print(f"\nadjudication ({report['evidence_type']}):")
        for track, row in report["by_track"].items():
            print(f"   {track:16s} n={row['n']:2d}  usable {row['usable']:2d} "
                  f"({row['usable_rate']:.1%})  wrong {row['WRONG']:2d}")

    if stage in ("latency", "all"):
        report = latency()
        _write("latency.json", report)
        print(f"\nlatency (local CPU):")
        for key in ("primary_accepted", "fallback_invoked"):
            row = report[key]
            if row.get("n"):
                print(f"   {key:20s} n={row['n']:4d}  median {row['median_ms']:7.1f} ms  "
                      f"p95 {row['p95_ms']:7.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
