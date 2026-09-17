"""Phase 1 baseline: perception response across controlled degradation.

Runs the capture-quality assessor over every fixture and every degradation level
in the standard sweep, and records what was measured. This is a *characterisation*
of the perception layer, not an accuracy evaluation: no ground-truth condition
labels exist at this phase, so nothing here reports accuracy, precision or any
headline percentage.

Outputs (written under --output-dir):

    phase1_sweep.json    full records, including run context (deterministic:
                         byte-identical across runs, so a diff means a real change)
    phase1_sweep.csv     flat table, one row per (fixture, degradation)
    phase1_latency.json  latency summary (the only non-deterministic artifact)
    plots/*.png          metric response curves

Reproduce with:

    .venv-competition/bin/python -m competition.evaluation.phase1_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.degradation import (
    BLUR_SIGMAS,
    DEGRADATION_VERSION,
    OVEREXPOSURE_GAINS,
    UNDEREXPOSURE_GAINS,
    DegradationSpec,
    apply_degradation,
    standard_sweep,
)
from competition.vision.evidence import PIPELINE_VERSION
from competition.vision.fixtures import all_fixtures
from competition.vision.quality import assess_capture_quality

DEFAULT_OUTPUT_DIR = Path("competition/evaluation/results/phase1")
LATENCY_WARMUP_RUNS = 5
LATENCY_SAMPLE_RUNS = 200


def run_context() -> dict:
    """Runtime context for reproducibility.

    Deliberately excludes hostname and any filesystem path: enough to interpret
    and reproduce the numbers, nothing that identifies the machine or leaks a
    local directory layout.
    """
    return {
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "pipeline_version": PIPELINE_VERSION,
        "degradation_version": DEGRADATION_VERSION,
        "threshold_policy_fingerprint": DEFAULT_POLICY.fingerprint(),
        "threshold_policy_status": DEFAULT_POLICY.status,
    }


@dataclass
class SweepRecord:
    fixture: str
    degradation_kind: str
    degradation_level: float
    seed: int
    evidence: dict

    def flat_row(self) -> dict:
        evidence = self.evidence
        return {
            "fixture": self.fixture,
            "degradation_kind": self.degradation_kind,
            "degradation_level": self.degradation_level,
            "seed": self.seed,
            "width": evidence["image"]["width"],
            "height": evidence["image"]["height"],
            "laplacian_variance": evidence["sharpness"]["laplacian_variance"],
            "sharpness_score": evidence["sharpness"]["sharpness_score"],
            "mean_luminance": evidence["illumination"]["mean_luminance"],
            "median_luminance": evidence["illumination"]["median_luminance"],
            "luminance_std": evidence["illumination"]["luminance_std"],
            "contrast_score": evidence["illumination"]["contrast_score"],
            "shadow_clip_fraction": evidence["illumination"]["shadow_clip_fraction"],
            "highlight_clip_fraction": evidence["illumination"][
                "highlight_clip_fraction"
            ],
            "quality_flags": "|".join(evidence["quality_flags"]),
            "opencv_version": evidence["opencv_version"],
            "pipeline_version": evidence["pipeline_version"],
        }


def run_sweep(policy: ThresholdPolicy, seed: int = 0) -> list[SweepRecord]:
    records: list[SweepRecord] = []
    for fixture_name, image in all_fixtures().items():
        for spec in standard_sweep(seed=seed):
            degraded, _ = apply_degradation(image, spec)
            evidence = assess_capture_quality(degraded, policy)
            records.append(
                SweepRecord(
                    fixture=fixture_name,
                    degradation_kind=spec.kind,
                    degradation_level=spec.level,
                    seed=spec.seed,
                    # Timing is deliberately excluded: a single wall-clock
                    # sample per record is statistically meaningless and would
                    # make these artifacts churn on every run, turning reviewable
                    # evidence into noise. Latency is measured properly, with
                    # warm-up and 200 samples, in phase1_latency.json.
                    evidence=evidence.to_dict(include_timing=False),
                )
            )
    return records


def measure_latency(policy: ThresholdPolicy) -> dict:
    """Local per-image processing latency.

    This is a LOCAL CPU baseline only. It is not, and must not be presented as,
    AWS latency: no container, no cold start, no network. Phase 4 measures the
    deployed endpoint separately.

    Warm-up runs are discarded because the first assessment absorbs OpenCV and
    NumPy lazy initialisation and is roughly two orders of magnitude slower than
    steady state.
    """
    image = all_fixtures()["textured_object"]

    for _ in range(LATENCY_WARMUP_RUNS):
        assess_capture_quality(image, policy)

    samples: list[float] = []
    for _ in range(LATENCY_SAMPLE_RUNS):
        started = time.perf_counter()
        assess_capture_quality(image, policy)
        samples.append((time.perf_counter() - started) * 1000.0)

    samples.sort()
    height, width = image.shape[:2]
    return {
        "scope": "LOCAL_CPU_ONLY — not an AWS or deployed-endpoint measurement",
        "image_size": f"{width}x{height}",
        "sample_count": len(samples),
        "warmup_runs_discarded": LATENCY_WARMUP_RUNS,
        "median_ms": round(statistics.median(samples), 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 4),
        "p99_ms": round(samples[int(0.99 * (len(samples) - 1))], 4),
        "min_ms": round(samples[0], 4),
        "max_ms": round(samples[-1], 4),
        "context": run_context(),
    }


def write_plots(records: list[SweepRecord], output_dir: Path) -> list[str]:
    """Metric response curves. Returns the plot filenames written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib unavailable; skipping plots", file=sys.stderr)
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def series(fixture: str, kind: str, field: str):
        rows = [
            r
            for r in records
            if r.fixture == fixture and r.degradation_kind == kind
        ]
        rows.sort(key=lambda r: r.degradation_level)
        xs = [r.degradation_level for r in rows]
        ys = [r.flat_row()[field] for r in rows]
        return xs, ys

    fixtures = sorted({r.fixture for r in records})

    specs = [
        ("blur_vs_sharpness.png", "gaussian_blur", "laplacian_variance",
         "Gaussian blur sigma", "Laplacian variance (log)",
         "Blur strength vs sharpness metric", True),
        ("underexposure_vs_luminance.png", "underexpose", "mean_luminance",
         "Exposure gain (<1 = darker)", "Mean L* (0-1)",
         "Underexposure vs measured luminance", False),
        ("overexposure_vs_luminance.png", "overexpose", "mean_luminance",
         "Exposure gain (>1 = brighter)", "Mean L* (0-1)",
         "Overexposure vs measured luminance", False),
        ("overexposure_vs_highlight_clipping.png", "overexpose",
         "highlight_clip_fraction", "Exposure gain (>1 = brighter)",
         "Highlight-clipped pixel fraction",
         "Overexposure vs highlight clipping", False),
        ("contrast_reduction_vs_contrast.png", "reduce_contrast", "contrast_score",
         "Contrast factor (1 = unchanged)", "Contrast score (p95-p5 of L*)",
         "Contrast reduction vs measured contrast", False),
    ]

    for filename, kind, field, xlabel, ylabel, title, log_y in specs:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for fixture in fixtures:
            xs, ys = series(fixture, kind, field)
            if xs:
                axis.plot(xs, ys, marker="o", label=fixture)
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=120)
        plt.close(figure)
        written.append(filename)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.phase1_baseline",
        description="Phase 1 perception baseline over controlled degradations.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = DEFAULT_POLICY
    records = run_sweep(policy, seed=args.seed)
    latency = measure_latency(policy)

    payload = {
        "run_context": run_context(),
        "threshold_policy": policy.to_dict(),
        "sweep_levels": {
            "gaussian_blur": list(BLUR_SIGMAS),
            "underexpose": list(UNDEREXPOSURE_GAINS),
            "overexpose": list(OVEREXPOSURE_GAINS),
        },
        "record_count": len(records),
        "timing_note": (
            "Per-record timing is excluded so this artifact is reproducible "
            "byte-for-byte. See phase1_latency.json for measured latency."
        ),
        "records": [
            {
                "fixture": r.fixture,
                "degradation": {
                    "kind": r.degradation_kind,
                    "level": r.degradation_level,
                    "seed": r.seed,
                },
                "evidence": r.evidence,
            }
            for r in records
        ],
    }

    (output_dir / "phase1_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "phase1_latency.json").write_text(
        json.dumps(latency, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = [r.flat_row() for r in records]
    with (output_dir / "phase1_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        # Explicit LF: csv.DictWriter defaults to RFC 4180 CRLF, which git
        # reports as trailing whitespace on every row. This file is generated
        # here, so the format is ours to choose.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plots = [] if args.skip_plots else write_plots(records, output_dir)

    print(f"records:        {len(records)}")
    print(f"latency median: {latency['median_ms']} ms  p95: {latency['p95_ms']} ms")
    print(f"plots written:  {len(plots)}")
    print(f"output dir:     {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
