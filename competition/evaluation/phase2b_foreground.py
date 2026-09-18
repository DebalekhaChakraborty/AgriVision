"""Phase 2b: foreground isolation and ROI-restricted capture quality.

Tests the hypothesis Phase 2 produced: that the capture gate rejects almost all
real produce photographs because whole-image clipping statistics are dominated
by bright backgrounds rather than by the subject.

Artifacts under --output-dir:

    synthetic_segmentation_metrics.json  IoU/Dice against known masks
    foreground_method_comparison.json    method trade-offs on real images
    real_foreground_statistics.json      validity and geometry distributions
    gate_comparison.json                 whole-image gate vs ROI-restricted gate
    mask_stability.json                  mask agreement across remediation
    mask_edge_sharpness.json             the boundary-energy measurement
    latency.json                         segmentation and ROI-quality timings
    plots/*.png

**Data boundary.** Real photographs are read from local research data in place
and never copied, embedded or committed. Outputs carry hashes, geometry,
metrics and aggregates — no image bytes.

Run with:

    .venv-competition/bin/python -m competition.evaluation.phase2b_foreground
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from competition.agent.inspection import DEFAULT_BLOCKING_FLAGS
from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.enhancement import ClaheParameters, GammaParameters, apply_clahe, apply_gamma_correction
from competition.vision.evidence import ImageValidationError
from competition.vision.fixtures import foreground_fixtures
from competition.vision.foreground import (
    DEFAULT_GUARDS,
    DEFAULT_METHOD,
    ForegroundMethod,
    isolate_foreground,
    masked_pixels,
)
from competition.vision.quality import assess_capture_quality, measure_sharpness

DEFAULT_OUTPUT_DIR = Path("competition/evaluation/results/phase2b")
VALIDATION_SPLIT = Path("v2/data/processed/v1_frozen_split/validation")
REAL_SAMPLE = 270
STABILITY_SAMPLE = 60
LATENCY_WARMUP = 5
LATENCY_SAMPLES = 60


def run_context() -> dict:
    return {
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "foreground_method_default": DEFAULT_METHOD.value,
        "guards": DEFAULT_GUARDS.to_dict(),
    }


def iou_dice(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    p, g = prediction > 0, truth > 0
    intersection = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    total = float(p.sum() + g.sum())
    return (
        intersection / union if union else 0.0,
        2 * intersection / total if total else 0.0,
    )


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "p05": round(float(np.percentile(array, 5)), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "mean": round(float(array.mean()), 6),
    }


# --------------------------------------------------------------------------
# B8: synthetic evaluation against known masks
# --------------------------------------------------------------------------

def synthetic_segmentation(methods: list[ForegroundMethod]) -> dict:
    fixtures = foreground_fixtures()
    results: dict = {"note": (
        "IoU and Dice are valid here because the masks are constructed, not "
        "annotated. These scores describe behaviour on synthetic targets and do "
        "NOT represent real produce segmentation quality."
    ), "methods": {}}

    for method in methods:
        per_fixture = {}
        ious, dices = [], []
        valid = 0
        for name, (image, truth) in fixtures.items():
            mask, evidence = isolate_foreground(image, method)
            score_iou, score_dice = iou_dice(mask, truth)
            ious.append(score_iou)
            dices.append(score_dice)
            valid += int(evidence.valid)
            per_fixture[name] = {
                "iou": round(score_iou, 4),
                "dice": round(score_dice, 4),
                "valid": evidence.valid,
                "invalid_reasons": evidence.invalid_reasons,
                "foreground_fraction": evidence.foreground_fraction,
            }
        results["methods"][method.value] = {
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_dice": round(float(np.mean(dices)), 4),
            "valid_count": valid,
            "fixture_count": len(fixtures),
            "per_fixture": per_fixture,
        }
    return results


# --------------------------------------------------------------------------
# B7/B10: mask-edge sharpness, measured
# --------------------------------------------------------------------------

def mask_edge_sharpness() -> dict:
    """Quantify how much a mask boundary inflates the sharpness metric."""
    rows = []
    for name, (image, truth) in foreground_fixtures().items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        whole = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        zeroed = image.copy()
        zeroed[truth == 0] = 0
        naive = float(
            cv2.Laplacian(cv2.cvtColor(zeroed, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )

        by_erosion = {}
        for erosion in (0, 3, 5, 11):
            try:
                by_erosion[str(erosion)] = round(
                    measure_sharpness(image, DEFAULT_POLICY, truth, erosion_px=erosion
                                      ).laplacian_variance, 4)
            except ImageValidationError:
                by_erosion[str(erosion)] = None

        inflation = None
        if by_erosion.get("0") and by_erosion.get("5"):
            inflation = round(by_erosion["0"] / by_erosion["5"], 3)

        rows.append({
            "fixture": name,
            "whole_image": round(whole, 4),
            "naive_background_zeroing": round(naive, 4),
            "masked_by_erosion_px": by_erosion,
            "edge_inflation_factor_erode0_over_erode5": inflation,
        })

    inflations = [r["edge_inflation_factor_erode0_over_erode5"] for r in rows
                  if r["edge_inflation_factor_erode0_over_erode5"]]
    return {
        "method": (
            "Laplacian computed on the unmodified grayscale image; the mask only "
            "selects which responses are aggregated, and is eroded first."
        ),
        "why": (
            "Zeroing the background and then differentiating manufactures a step "
            "edge at the mask boundary. Including the un-eroded boundary inflates "
            "the variance, so a more aggressively masked image would appear sharper."
        ),
        "rows": rows,
        "median_edge_inflation": round(float(np.median(inflations)), 3) if inflations else None,
        "default_erosion_px": DEFAULT_GUARDS.sharpness_erosion_px,
    }


# --------------------------------------------------------------------------
# B9/B10: real-image statistics and the gate comparison
# --------------------------------------------------------------------------

def real_image_analysis(
    paths: list[str], methods: list[ForegroundMethod], policy: ThresholdPolicy
) -> tuple[dict, dict, dict]:
    method_comparison: dict = {}
    statistics_by_method: dict = {}
    gate: dict = {}

    for method in methods:
        valid = 0
        reasons: Counter = Counter()
        fractions, border_contacts, components, solidities = [], [], [], []
        durations = []

        whole_highlight, whole_shadow = [], []
        roi_highlight, roi_shadow = [], []
        whole_flags: Counter = Counter()
        roi_flags: Counter = Counter()
        whole_blocked = roi_blocked = 0
        roi_evaluated = 0
        considered = 0

        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            considered += 1

            started = time.perf_counter()
            mask, foreground = isolate_foreground(image, method)
            durations.append((time.perf_counter() - started) * 1000.0)

            fractions.append(foreground.foreground_fraction)
            border_contacts.append(foreground.border_contact_fraction)
            components.append(foreground.component_count)
            solidities.append(foreground.solidity)

            whole = assess_capture_quality(image, policy)
            for flag in whole.quality_flags:
                whole_flags[flag] += 1
            whole_is_blocked = any(f in DEFAULT_BLOCKING_FLAGS for f in whole.quality_flags)
            whole_blocked += int(whole_is_blocked)

            if not foreground.valid:
                for reason in foreground.invalid_reasons:
                    reasons[reason] += 1
                # Conservative: an invalid mask never replaces whole-image evidence.
                roi_blocked += int(whole_is_blocked)
                continue

            valid += 1
            try:
                roi = assess_capture_quality(image, policy, mask=mask)
            except ImageValidationError:
                reasons["ROI_TOO_SMALL_TO_MEASURE"] += 1
                roi_blocked += int(whole_is_blocked)
                continue

            roi_evaluated += 1
            for flag in roi.quality_flags:
                roi_flags[flag] += 1
            roi_blocked += int(any(f in DEFAULT_BLOCKING_FLAGS for f in roi.quality_flags))

            whole_highlight.append(whole.illumination.highlight_clip_fraction)
            whole_shadow.append(whole.illumination.shadow_clip_fraction)
            roi_highlight.append(roi.illumination.highlight_clip_fraction)
            roi_shadow.append(roi.illumination.shadow_clip_fraction)

        method_comparison[method.value] = {
            "images": considered,
            "validity_rate": round(valid / considered, 4) if considered else None,
            "median_latency_ms": round(float(np.median(durations)), 3) if durations else None,
            "p95_latency_ms": round(float(np.percentile(durations, 95)), 3) if durations else None,
            "invalid_reasons": dict(reasons.most_common()),
        }

        statistics_by_method[method.value] = {
            "foreground_fraction": _percentiles(fractions),
            "border_contact_fraction": _percentiles(border_contacts),
            "component_count": _percentiles([float(c) for c in components]),
            "solidity": _percentiles(solidities),
        }

        gate[method.value] = {
            "images_considered": considered,
            "segmentation_valid": valid,
            "roi_measured": roi_evaluated,
            "whole_image_blocked": whole_blocked,
            "whole_image_blocked_rate": round(whole_blocked / considered, 4) if considered else None,
            "roi_restricted_blocked": roi_blocked,
            "roi_restricted_blocked_rate": round(roi_blocked / considered, 4) if considered else None,
            "whole_image_flag_rates": {
                flag: round(count / considered, 4) for flag, count in whole_flags.most_common()
            },
            "roi_flag_rates_over_measured": {
                flag: round(count / roi_evaluated, 4) for flag, count in roi_flags.most_common()
            } if roi_evaluated else {},
            "clipping": {
                "whole_highlight": _percentiles(whole_highlight),
                "roi_highlight": _percentiles(roi_highlight),
                "whole_shadow": _percentiles(whole_shadow),
                "roi_shadow": _percentiles(roi_shadow),
                "paired_images": len(roi_highlight),
            },
        }

    return method_comparison, statistics_by_method, gate


# --------------------------------------------------------------------------
# B13: mask stability across remediation
# --------------------------------------------------------------------------

def mask_stability(paths: list[str], method: ForegroundMethod) -> dict:
    """Does a mask computed before remediation still describe the image after?"""
    rows = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        before_mask, before = isolate_foreground(image, method)
        if not before.valid:
            continue

        for label, remediated in (
            ("gamma_0.6", apply_gamma_correction(image, GammaParameters(gamma=0.6))[0]),
            ("gamma_1.6", apply_gamma_correction(image, GammaParameters(gamma=1.6))[0]),
            ("clahe", apply_clahe(image, ClaheParameters())[0]),
        ):
            after_mask, after = isolate_foreground(remediated, method)
            score_iou, score_dice = iou_dice(after_mask, before_mask)
            rows.append({
                "remediation": label,
                "iou": round(score_iou, 4),
                "dice": round(score_dice, 4),
                "still_valid": after.valid,
            })

    summary = {}
    for label in ("gamma_0.6", "gamma_1.6", "clahe"):
        subset = [r for r in rows if r["remediation"] == label]
        if not subset:
            continue
        ious = [r["iou"] for r in subset]
        summary[label] = {
            "samples": len(subset),
            "median_iou": round(float(np.median(ious)), 4),
            "p05_iou": round(float(np.percentile(ious, 5)), 4),
            "fraction_below_0_90_iou": round(
                float(np.mean([i < 0.90 for i in ious])), 4),
            "still_valid_rate": round(
                float(np.mean([r["still_valid"] for r in subset])), 4),
        }
    return {
        "question": (
            "Can a mask computed before remediation be reused afterwards, or "
            "must it be recomputed?"
        ),
        "comparison": "IoU/Dice of the post-remediation mask against the pre-remediation mask",
        "by_remediation": summary,
    }


def latency_benchmark(image: np.ndarray, method: ForegroundMethod) -> dict:
    for _ in range(LATENCY_WARMUP):
        isolate_foreground(image, method)

    segmentation, roi_quality, combined = [], [], []
    for _ in range(LATENCY_SAMPLES):
        start = time.perf_counter()
        mask, evidence = isolate_foreground(image, method)
        mid = time.perf_counter()
        try:
            assess_capture_quality(image, DEFAULT_POLICY, mask=mask if evidence.valid else None)
        except ImageValidationError:
            assess_capture_quality(image, DEFAULT_POLICY)
        end = time.perf_counter()
        segmentation.append((mid - start) * 1000.0)
        roi_quality.append((end - mid) * 1000.0)
        combined.append((end - start) * 1000.0)

    def summarise(values: list[float]) -> dict:
        values = sorted(values)
        return {
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(values[int(0.95 * (len(values) - 1))], 3),
            "sample_count": len(values),
        }

    height, width = image.shape[:2]
    return {
        "scope": "LOCAL_CPU_ONLY - not an AWS or deployed-endpoint measurement",
        "resolution": f"{width}x{height}",
        "method": method.value,
        "warmup_discarded": LATENCY_WARMUP,
        "segmentation": summarise(segmentation),
        "roi_quality": summarise(roi_quality),
        "combined_perception": summarise(combined),
        "note": "Condition-model latency is measured separately in Phase 2.",
    }


def write_plots(synthetic: dict, gate: dict, output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    methods = list(synthetic["methods"])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(methods, [synthetic["methods"][m]["mean_iou"] for m in methods])
    axis.set_ylabel("Mean IoU (synthetic fixtures)")
    axis.set_title("Synthetic segmentation quality by method")
    axis.set_ylim(0, 1)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(plots / "synthetic_iou_by_method.png", dpi=120)
    plt.close(figure)
    written.append("synthetic_iou_by_method.png")

    default = DEFAULT_METHOD.value
    if default in gate:
        clipping = gate[default]["clipping"]
        labels = ["highlight", "shadow"]
        whole = [clipping["whole_highlight"].get("median", 0), clipping["whole_shadow"].get("median", 0)]
        roi = [clipping["roi_highlight"].get("median", 0), clipping["roi_shadow"].get("median", 0)]
        x = np.arange(len(labels))
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(x - 0.2, whole, 0.4, label="whole image")
        axis.bar(x + 0.2, roi, 0.4, label="foreground ROI")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Median clipped-pixel fraction")
        axis.set_title(f"Clipping before and after foreground restriction ({default})")
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(plots / "clipping_whole_vs_roi.png", dpi=120)
        plt.close(figure)
        written.append("clipping_whole_vs_roi.png")

        figure, axis = plt.subplots(figsize=(7, 4.5))
        names = list(gate)
        axis.bar(np.arange(len(names)) - 0.2,
                 [gate[m]["whole_image_blocked_rate"] for m in names], 0.4, label="whole image")
        axis.bar(np.arange(len(names)) + 0.2,
                 [gate[m]["roi_restricted_blocked_rate"] for m in names], 0.4, label="ROI restricted")
        axis.set_xticks(np.arange(len(names)), names, fontsize=7)
        axis.set_ylabel("Blocked rate")
        axis.set_title("Gate blocked rate: whole image vs ROI restricted")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(plots / "gate_block_rate_comparison.png", dpi=120)
        plt.close(figure)
        written.append("gate_block_rate_comparison.png")

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.phase2b_foreground",
        description="Phase 2b foreground isolation and ROI-restricted quality.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-grabcut", action="store_true",
                        help="omit the slowest method from real-image analysis")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = run_context()

    methods = [m for m in ForegroundMethod
               if not (args.skip_grabcut and m is ForegroundMethod.GRABCUT_RECT)]

    synthetic = synthetic_segmentation(list(ForegroundMethod))
    (output_dir / "synthetic_segmentation_metrics.json").write_text(
        json.dumps({"run_context": context, **synthetic}, indent=2, sort_keys=True),
        encoding="utf-8")

    edges = mask_edge_sharpness()
    (output_dir / "mask_edge_sharpness.json").write_text(
        json.dumps({"run_context": context, **edges}, indent=2, sort_keys=True),
        encoding="utf-8")

    paths = sorted(glob.glob(str(VALIDATION_SPLIT / "*" / "*")))[:REAL_SAMPLE]
    if not paths:
        print("local research images unavailable; synthetic artifacts written only")
        return 0

    comparison, real_statistics, gate = real_image_analysis(paths, methods, DEFAULT_POLICY)
    (output_dir / "foreground_method_comparison.json").write_text(
        json.dumps({"run_context": context, "methods": comparison}, indent=2, sort_keys=True),
        encoding="utf-8")
    (output_dir / "real_foreground_statistics.json").write_text(
        json.dumps({"run_context": context,
                    "source": "v2/data/processed/v1_frozen_split/validation (local only; no image bytes stored)",
                    "by_method": real_statistics}, indent=2, sort_keys=True),
        encoding="utf-8")
    (output_dir / "gate_comparison.json").write_text(
        json.dumps({"run_context": context,
                    "hypothesis": (
                        "Whole-image clipping statistics are dominated by bright "
                        "backgrounds, not by the produce."),
                    "by_method": gate}, indent=2, sort_keys=True),
        encoding="utf-8")

    stability = mask_stability(paths[:STABILITY_SAMPLE], DEFAULT_METHOD)
    (output_dir / "mask_stability.json").write_text(
        json.dumps({"run_context": context, **stability}, indent=2, sort_keys=True),
        encoding="utf-8")

    sample_image = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    latency = latency_benchmark(sample_image, DEFAULT_METHOD)
    (output_dir / "latency.json").write_text(
        json.dumps({"run_context": context, **latency}, indent=2, sort_keys=True),
        encoding="utf-8")

    plots = [] if args.skip_plots else write_plots(synthetic, gate, output_dir)

    default = DEFAULT_METHOD.value
    print(f"synthetic mean IoU ({default}): {synthetic['methods'][default]['mean_iou']}")
    print(f"median mask-edge inflation:    {edges['median_edge_inflation']}x")
    print(f"real validity rate:            {comparison[default]['validity_rate']}")
    print(f"gate blocked whole -> ROI:     {gate[default]['whole_image_blocked_rate']} -> {gate[default]['roi_restricted_blocked_rate']}")
    highlight = gate[default]["clipping"]
    print(f"highlight clip median:         {highlight['whole_highlight'].get('median')} -> {highlight['roi_highlight'].get('median')}")
    print(f"segmentation median latency:   {latency['segmentation']['median_ms']} ms")
    print(f"plots written:                 {len(plots)}")
    print(f"output dir:                    {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
