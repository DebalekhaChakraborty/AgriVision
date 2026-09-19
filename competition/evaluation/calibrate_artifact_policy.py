"""Calibrate the Phase 2d artefact detectors on calibration groups only.

Three detectors, three separate decisions. They are calibrated against their own
controlled degradation family and reported separately, because a single
"artefact accuracy" number would hide that one of them works well and another
barely works at all.

Preregistered rule, identical in shape to Phase 2c-B
----------------------------------------------------
For each detector, over a small explicit parameter grid:

1. reject any setting whose false-positive rate on **undegraded** calibration
   images exceeds `MAX_FALSE_POSITIVE_RATE`;
2. among the rest, take the highest detection rate on the severe rung;
3. if that detection rate is below `MIN_DETECTION_RATE`, declare the detector
   **not usable** rather than lowering the bar to make it pass.

The asymmetry is deliberate and is the same one as before. A false positive here
sends a person back to re-photograph a perfectly good picture; a miss passes a
damaged capture to inference. But a detector that blocks good captures is worse
than no detector, because it will be switched off.

Held-out and validation data are untouched. The Phase 2c-B held-out groups are
spent, and the Phase 2d validation pool is frozen and must not inform a
parameter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import date
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from competition.agent.artifact_policy import ArtifactPolicy
from competition.data.licensed_sources import CorpusError
from competition.evaluation.build_licensed_corpus import local_path_for
from competition.evaluation.split_licensed_corpus import calibration_view
from competition.vision.config import DEFAULT_POLICY
from competition.vision.degradation import (
    DegradationSpec,
    add_glare_at,
    apply_degradation,
    occlude_at,
)
from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import ForegroundError, isolate_foreground
from competition.vision.highlights import LocalHighlightPolicy, measure_local_highlights
from competition.vision.quality import measure_illumination
from competition.vision.visibility import VisibilityPolicy, measure_visibility

RESULTS_DIR = Path("competition/evaluation/results/phase2d")
CALIBRATION_REPORT = RESULTS_DIR / "calibration.json"
FROZEN_POLICY_PATH = RESULTS_DIR / "locked_artifact_policy.json"
CALIBRATION_VERSION = "phase2d-artifact-calibration-1.0.0"
SEED = 20260918

# --- preregistered criteria --------------------------------------------------
MAX_FALSE_POSITIVE_RATE = 0.10
MIN_DETECTION_RATE = 0.70

# --- parameter grids, small and explicit -------------------------------------
GLARE_CANDIDATE_GRID = {
    "local_excess_l": (15.0, 25.0, 35.0),
    "saturation_ratio": (0.75, 1.10),
}
GLARE_FLAG_GRID = {
    "min_component_fraction": (0.005, 0.02, 0.05, 0.10),
    "min_candidate_fraction": (0.02, 0.06, 0.15),
    "min_spatial_concentration": (0.35, 0.60),
}
# Widened after measuring real undegraded masks, which sit far from the synthetic
# discs the provisional values were chosen on: median hull fill 0.85, median
# solidity 0.84, median defect ratio 0.63. A grid that stopped at 0.70 could only
# ever produce false positives.
VISIBILITY_GRID = {
    "min_hull_fill": (0.40, 0.50, 0.60, 0.70),
    "min_solidity": (0.40, 0.50, 0.60, 0.70),
    "max_border_truncation": (0.35, 0.50, 0.70),
    "max_defect_depth_ratio": (0.90, 1.20, 1.60),
    "max_internal_hole_fraction": (0.10, 0.20, 0.35),
}
SEVERE_CONTRAST_GRID = tuple(round(0.02 + 0.01 * i, 3) for i in range(11))


def _iter_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def collect_samples(limit: int | None = None) -> dict:
    """Measure every calibration sample once; thresholds are swept afterwards.

    Foreground isolation and the highlight candidate pass are the expensive
    steps, so masks are computed once per image and the highlight evidence once
    per candidate-selection setting. Flag thresholds are pure arithmetic on the
    stored metrics and cost nothing to sweep.
    """
    records = [r for r in calibration_view().records if r.track == "CLEAN_BASE"]
    if limit:
        records = records[:limit]

    # Glare and occlusion are injected **onto the subject** rather than at a
    # seeded point anywhere in the frame. Measured on this corpus, only about a
    # quarter of the Phase 1 glare landed on the fruit at all, so most "glare
    # positive" samples were photographs with no glare on the produce. Scoring a
    # subject-region detector against that labelling measures the placement.
    # Contrast is global, so its Phase 1 transform is used unchanged.
    conditions = [
        ("undegraded", None, "negative"),
        ("glare_severe", ("glare_at", 0.9), "glare_positive"),
        ("glare_moderate", ("glare_at", 0.5), "glare_borderline"),
        ("occlude_severe", ("occlude_at", 0.25), "visibility_positive"),
        ("occlude_moderate", ("occlude_at", 0.10), "visibility_borderline"),
        ("contrast_severe", DegradationSpec("reduce_contrast", 0.08, SEED), "contrast_positive"),
        ("contrast_unusable", DegradationSpec("reduce_contrast", 0.2, SEED), "contrast_positive"),
        ("contrast_moderate", DegradationSpec("reduce_contrast", 0.5, SEED), "contrast_borderline"),
    ]
    candidate_settings = _iter_grid(GLARE_CANDIDATE_GRID)

    samples: list[dict] = []
    unusable_masks = 0

    for record in records:
        base = cv2.imread(
            str(local_path_for(record.image_id, record.file_extension)), cv2.IMREAD_COLOR
        )
        if base is None:
            continue
        try:
            subject_mask, subject_fg = isolate_foreground(base)
        except ForegroundError:
            continue
        if not subject_fg.valid:
            continue
        ys, xs = np.nonzero(subject_mask)
        centre = (int(xs.mean()), int(ys.mean()))
        subject_area = float(len(ys))
        subject_radius = float(np.sqrt(subject_area / np.pi))

        for name, spec, role in conditions:
            if spec is None:
                image = base
            elif isinstance(spec, tuple) and spec[0] == "glare_at":
                image, _ = add_glare_at(
                    base, spec[1], centre, max(4, int(subject_radius * 0.30)), falloff=0.35
                )
            elif isinstance(spec, tuple) and spec[0] == "occlude_at":
                side = max(4, int(np.sqrt(subject_area * spec[1])))
                image, _ = occlude_at(base, centre, side, side)
            else:
                image = apply_degradation(base, spec)[0]
            try:
                mask, foreground = isolate_foreground(image)
            except ForegroundError:
                unusable_masks += 1
                continue
            if not foreground.valid:
                unusable_masks += 1
                continue

            entry = {
                "image_id": record.image_id, "condition": name, "role": role,
                "fruit_type": record.fruit_type, "highlights": {}, "visibility": None,
                "roi_contrast": None,
            }
            for setting in candidate_settings:
                policy = LocalHighlightPolicy(**setting)
                try:
                    evidence = measure_local_highlights(image, mask, policy)
                except ImageValidationError:
                    continue
                key = f"{setting['local_excess_l']}|{setting['saturation_ratio']}"
                entry["highlights"][key] = {
                    "largest_component_fraction": evidence.largest_component_fraction,
                    "candidate_fraction": evidence.candidate_fraction,
                    "spatial_concentration": evidence.spatial_concentration,
                    "max_local_luminance_excess": evidence.max_local_luminance_excess,
                }
            try:
                visibility = measure_visibility(mask)
                entry["visibility"] = {
                    "hull_fill": visibility.hull_fill,
                    "solidity": visibility.solidity,
                    "border_truncation_fraction": visibility.border_truncation_fraction,
                    "internal_hole_fraction": visibility.internal_hole_fraction,
                    "max_defect_depth_ratio": visibility.max_defect_depth_ratio,
                    "fragment_count": visibility.fragment_count,
                }
            except ImageValidationError:
                pass
            try:
                entry["roi_contrast"] = float(
                    measure_illumination(image, DEFAULT_POLICY, mask).contrast_score
                )
            except ImageValidationError:
                pass
            samples.append(entry)

    return {"samples": samples, "unusable_masks": unusable_masks,
            "base_images": len(records),
            "candidate_settings": [
                f"{s['local_excess_l']}|{s['saturation_ratio']}" for s in candidate_settings
            ]}


def _rates(detections: list[bool], positives: list[bool]) -> tuple[float, float, int, int]:
    """(detection rate on positives, false-positive rate on negatives, n+, n-)."""
    pos = [d for d, p in zip(detections, positives) if p]
    neg = [d for d, p in zip(detections, positives) if not p]
    detection = sum(pos) / len(pos) if pos else 0.0
    false_positive = sum(neg) / len(neg) if neg else 0.0
    return detection, false_positive, len(pos), len(neg)


def calibrate_glare(collected: dict) -> dict:
    samples = collected["samples"]
    results = []
    for candidate_key in collected["candidate_settings"]:
        excess, saturation = candidate_key.split("|")
        usable = [s for s in samples if candidate_key in s["highlights"]
                  and s["role"] in ("negative", "glare_positive")]
        if not usable:
            continue
        positives = [s["role"] == "glare_positive" for s in usable]
        for flags in _iter_grid(GLARE_FLAG_GRID):
            detections = []
            for sample in usable:
                metrics = sample["highlights"][candidate_key]
                concentrated = (
                    metrics["largest_component_fraction"] >= flags["min_component_fraction"]
                    and metrics["spatial_concentration"] >= flags["min_spatial_concentration"]
                )
                large = metrics["candidate_fraction"] >= flags["min_candidate_fraction"]
                detections.append(bool(concentrated or large))
            detection, false_positive, n_pos, n_neg = _rates(detections, positives)
            results.append({
                "local_excess_l": float(excess), "saturation_ratio": float(saturation),
                **flags, "detection_rate": detection, "false_positive_rate": false_positive,
                "positives": n_pos, "negatives": n_neg,
                "admissible": false_positive <= MAX_FALSE_POSITIVE_RATE,
            })

    admissible = [r for r in results if r["admissible"]]
    admissible.sort(key=lambda r: (-r["detection_rate"], r["false_positive_rate"]))
    best = admissible[0] if admissible else None
    usable = bool(best and best["detection_rate"] >= MIN_DETECTION_RATE)
    return {
        "grid_size": len(results),
        "admissible_settings": len(admissible),
        "best": best,
        "usable": usable,
        "rejection_reason": None if usable else (
            "no setting held the false-positive rate within the limit"
            if not admissible else
            f"best detection rate {best['detection_rate']:.2f} is below {MIN_DETECTION_RATE}"
        ),
        "top_settings": admissible[:5],
    }


def calibrate_visibility(collected: dict) -> dict:
    samples = [s for s in collected["samples"] if s["visibility"]
               and s["role"] in ("negative", "visibility_positive")]
    positives = [s["role"] == "visibility_positive" for s in samples]
    results = []
    for setting in _iter_grid(VISIBILITY_GRID):
        detections = []
        for sample in samples:
            v = sample["visibility"]
            insufficient = (
                v["hull_fill"] < setting["min_hull_fill"]
                or v["solidity"] < setting["min_solidity"]
                or v["border_truncation_fraction"] > setting["max_border_truncation"]
                or v["max_defect_depth_ratio"] > setting["max_defect_depth_ratio"]
                or v["internal_hole_fraction"] > setting["max_internal_hole_fraction"]
                or v["fragment_count"] > 1
            )
            detections.append(bool(insufficient))
        detection, false_positive, n_pos, n_neg = _rates(detections, positives)
        results.append({**setting, "detection_rate": detection,
                        "false_positive_rate": false_positive,
                        "positives": n_pos, "negatives": n_neg,
                        "admissible": false_positive <= MAX_FALSE_POSITIVE_RATE})

    admissible = [r for r in results if r["admissible"]]
    admissible.sort(key=lambda r: (-r["detection_rate"], r["false_positive_rate"]))
    best = admissible[0] if admissible else None
    usable = bool(best and best["detection_rate"] >= MIN_DETECTION_RATE)
    return {
        "grid_size": len(results), "admissible_settings": len(admissible),
        "best": best, "usable": usable,
        "rejection_reason": None if usable else (
            "no setting held the false-positive rate within the limit"
            if not admissible else
            f"best detection rate {best['detection_rate']:.2f} is below {MIN_DETECTION_RATE}"
        ),
        "top_settings": admissible[:5],
    }


def calibrate_severe_contrast(collected: dict) -> dict:
    samples = [s for s in collected["samples"] if s["roi_contrast"] is not None]
    negatives = [s["roi_contrast"] for s in samples if s["role"] == "negative"]
    positives = [s["roi_contrast"] for s in samples
                 if s["condition"] in ("contrast_severe", "contrast_unusable")]
    borderline = [s["roi_contrast"] for s in samples if s["role"] == "contrast_borderline"]

    results = []
    for limit in SEVERE_CONTRAST_GRID:
        detection = sum(1 for v in positives if v < limit) / len(positives) if positives else 0.0
        false_positive = sum(1 for v in negatives if v < limit) / len(negatives) if negatives else 0.0
        borderline_rate = (
            sum(1 for v in borderline if v < limit) / len(borderline) if borderline else 0.0
        )
        results.append({"severe_contrast_limit": limit, "detection_rate": detection,
                        "false_positive_rate": false_positive,
                        "borderline_block_rate": borderline_rate,
                        "admissible": false_positive <= MAX_FALSE_POSITIVE_RATE})

    admissible = [r for r in results if r["admissible"]]
    admissible.sort(key=lambda r: (-r["detection_rate"], r["borderline_block_rate"]))
    best = admissible[0] if admissible else None
    usable = bool(best and best["detection_rate"] >= MIN_DETECTION_RATE)
    return {
        "median_contrast": {
            "undegraded": float(np.median(negatives)) if negatives else None,
            "severe_rungs": float(np.median(positives)) if positives else None,
            "borderline_rung": float(np.median(borderline)) if borderline else None,
        },
        "sweep": results, "best": best, "usable": usable,
        "rejection_reason": None if usable else "no admissible limit reached the detection floor",
    }


def _code_fingerprint() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    return f"{head}{'+dirty' if dirty else ''}"


def run(limit: int | None = None) -> dict:
    collected = collect_samples(limit)
    if not collected["samples"]:
        raise CorpusError("no measurable calibration samples")

    glare = calibrate_glare(collected)
    visibility = calibrate_visibility(collected)
    contrast = calibrate_severe_contrast(collected)

    # The best *admissible* setting is adopted whether or not it cleared the
    # detection floor: it is the best-characterised configuration either way.
    # Whether it may block is a separate flag, set from `usable`.
    highlight_policy = LocalHighlightPolicy()
    if glare["best"]:
        best = glare["best"]
        highlight_policy = LocalHighlightPolicy(
            local_excess_l=best["local_excess_l"],
            saturation_ratio=best["saturation_ratio"],
            min_component_fraction=best["min_component_fraction"],
            min_candidate_fraction=best["min_candidate_fraction"],
            min_spatial_concentration=best["min_spatial_concentration"],
        )
    visibility_policy = VisibilityPolicy()
    if visibility["best"]:
        best = visibility["best"]
        visibility_policy = replace(
            visibility_policy,
            min_hull_fill=best["min_hull_fill"],
            min_solidity=best["min_solidity"],
            max_border_truncation=best["max_border_truncation"],
            max_defect_depth_ratio=best["max_defect_depth_ratio"],
            max_internal_hole_fraction=best["max_internal_hole_fraction"],
        )
    artifact_policy = ArtifactPolicy(
        severe_contrast_limit=(
            contrast["best"]["severe_contrast_limit"] if contrast["usable"]
            else ArtifactPolicy().severe_contrast_limit
        ),
        glare_blocks=bool(glare["usable"]),
        visibility_blocks=bool(visibility["usable"]),
    )

    return {
        "calibration_version": CALIBRATION_VERSION,
        "calibrated_on": date.today().isoformat(),
        "opencv_version": cv2.__version__,
        "split": "calibration groups only; held-out and validation untouched",
        "criteria": {
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "min_detection_rate": MIN_DETECTION_RATE,
        },
        "population": {
            "base_images": collected["base_images"],
            "measured_samples": len(collected["samples"]),
            "samples_without_a_usable_mask": collected["unusable_masks"],
        },
        "glare": glare,
        "visibility": visibility,
        "severe_contrast": contrast,
        "frozen": {
            "highlight_policy": highlight_policy.to_dict(),
            "highlight_policy_fingerprint": highlight_policy.fingerprint(),
            "visibility_policy": visibility_policy.to_dict(),
            "visibility_policy_fingerprint": visibility_policy.fingerprint(),
            "artifact_policy": artifact_policy.to_dict(),
            "artifact_policy_fingerprint": artifact_policy.fingerprint(),
            "code_fingerprint": _code_fingerprint(),
            "detectors_usable": {
                "glare": glare["usable"],
                "visibility": visibility["usable"],
                "severe_contrast": contrast["usable"],
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.calibrate_artifact_policy",
        description="Calibrate Phase 2d artefact detectors on calibration groups.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    document = run(args.limit)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REPORT.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    FROZEN_POLICY_PATH.write_text(
        json.dumps(document["frozen"], indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "glare": {k: document["glare"][k] for k in ("usable", "rejection_reason")},
        "glare_best": document["glare"]["best"],
        "visibility": {k: document["visibility"][k] for k in ("usable", "rejection_reason")},
        "visibility_best": document["visibility"]["best"],
        "severe_contrast_best": document["severe_contrast"]["best"],
        "medians": document["severe_contrast"]["median_contrast"],
        "population": document["population"],
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
