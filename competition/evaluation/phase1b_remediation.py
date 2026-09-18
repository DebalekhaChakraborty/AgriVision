"""Phase 1b evaluation: exposure/sharpness coupling and remediation behaviour.

Two experiments plus engineering metrics.

**Experiment A — exposure/sharpness coupling (B12).** One synthetic source with
fixed true spatial detail, no blur applied, photographed across an exposure
ladder. Because the underlying detail is identical at every level, any variation
in the sharpness metric is measurement artefact rather than signal. The
experiment quantifies that artefact before and after remediation.

**Experiment B — blur x exposure factorial (B13).** Does exposure remediation
help separate *actually blurred* from *appears low-sharpness because it is
dark*? Each blur level has a ground-truth verdict defined at reference exposure;
the experiment measures how often the verdict at other exposures agrees with it,
before and after remediation.

Neither experiment is used to tune policy parameters. They are engineering
evidence on a small synthetic study, not a calibration set.

Reproduce with:

    .venv-competition/bin/python -m competition.evaluation.phase1b_remediation
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2

from competition.agent.actions import ReasonCode, RemediationAction
from competition.agent.policy import DEFAULT_REMEDIATION_POLICY, RemediationPolicy
from competition.agent.remediation import (
    MAX_AUTOMATED_ATTEMPTS,
    REMEDIATION_VERSION,
    run_capture_remediation,
)
from competition.agent.trace import TRACE_VERSION
from competition.evaluation.phase1_baseline import run_context
from competition.vision.config import DEFAULT_POLICY, ThresholdPolicy
from competition.vision.degradation import adjust_exposure, gaussian_blur
from competition.vision.fixtures import textured_object

DEFAULT_OUTPUT_DIR = Path("competition/evaluation/results/phase1b")

# Exposure ladder for Experiment A. Spans clearly dark to clearly bright,
# including levels the policy will refuse as unrecoverable.
EXPOSURE_LADDER: tuple[float, ...] = (0.05, 0.15, 0.25, 0.4, 0.6, 1.0, 1.6, 2.0, 2.6, 3.5)

# Factorial grid for Experiment B.
FACTORIAL_BLUR_SIGMAS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
FACTORIAL_EXPOSURE_GAINS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.6, 2.6)
REFERENCE_GAIN = 1.0


def coefficient_of_variation(values: list[float]) -> float:
    """Standard deviation divided by the mean.

    Chosen over raw range or standard deviation because Laplacian variance has
    arbitrary units whose magnitude depends on image content and on exposure
    itself. A dimensionless ratio answers the question we actually care about —
    how much does this measurement move, relative to its own typical size,
    across captures whose true detail is identical — and stays comparable when
    remediation shifts the mean.
    """
    if not values:
        return float("nan")
    mean = statistics.fmean(values)
    if mean == 0:
        return float("nan")
    return statistics.pstdev(values) / mean


def _evidence_row(evidence) -> dict:
    return {
        "mean_luminance": evidence.illumination.mean_luminance,
        "contrast_score": evidence.illumination.contrast_score,
        "shadow_clip_fraction": evidence.illumination.shadow_clip_fraction,
        "highlight_clip_fraction": evidence.illumination.highlight_clip_fraction,
        "laplacian_variance": evidence.sharpness.laplacian_variance,
        "quality_flags": "|".join(evidence.quality_flags),
        "blur_risk": "BLUR_RISK" in evidence.quality_flags,
    }


def experiment_exposure_coupling(
    quality_policy: ThresholdPolicy, remediation_policy: RemediationPolicy
) -> dict:
    """Experiment A: quantify exposure-induced variation in the sharpness metric."""
    source = textured_object()
    rows: list[dict] = []

    for gain in EXPOSURE_LADDER:
        image = adjust_exposure(source, gain)
        result = run_capture_remediation(image, quality_policy, remediation_policy)

        before = _evidence_row(result.original_evidence)
        canonical = _evidence_row(result.canonical_evidence)

        rows.append(
            {
                "exposure_gain": gain,
                "action": result.decision.action.value,
                "reason_code": result.decision.reason_code.value,
                "remediation_attempted": result.remediation_attempted,
                "remediation_accepted": result.remediation_accepted,
                "escalation_required": result.escalation_required,
                **{f"before_{k}": v for k, v in before.items()},
                **{f"canonical_{k}": v for k, v in canonical.items()},
            }
        )

    accepted = [r for r in rows if r["remediation_accepted"]]
    # Levels the policy handled automatically, plus any that needed nothing.
    automated_or_clean = [
        r
        for r in rows
        if r["remediation_accepted"] or r["action"] == RemediationAction.NONE.value
    ]

    def cv(subset: list[dict], field: str) -> float:
        return coefficient_of_variation([r[field] for r in subset])

    return {
        "description": (
            "One source, fixed true detail, no blur applied, across an exposure "
            "ladder. Variation in the sharpness metric is measurement artefact."
        ),
        "exposure_levels": list(EXPOSURE_LADDER),
        "rows": rows,
        "sharpness_exposure_sensitivity": {
            "measure": "coefficient_of_variation",
            "justification": (
                "Dimensionless, so it stays comparable when remediation shifts "
                "the mean of a quantity whose units are arbitrary."
            ),
            "all_levels_before": cv(rows, "before_laplacian_variance"),
            "all_levels_canonical": cv(rows, "canonical_laplacian_variance"),
            "accepted_subset_before": cv(accepted, "before_laplacian_variance"),
            "accepted_subset_canonical": cv(accepted, "canonical_laplacian_variance"),
            "automated_or_clean_before": cv(
                automated_or_clean, "before_laplacian_variance"
            ),
            "automated_or_clean_canonical": cv(
                automated_or_clean, "canonical_laplacian_variance"
            ),
            "accepted_count": len(accepted),
            "automated_or_clean_count": len(automated_or_clean),
            "total_levels": len(rows),
        },
        "false_blur_flags_before": sum(1 for r in rows if r["before_blur_risk"]),
        "false_blur_flags_canonical": sum(1 for r in rows if r["canonical_blur_risk"]),
    }


def experiment_blur_exposure_factorial(
    quality_policy: ThresholdPolicy, remediation_policy: RemediationPolicy
) -> dict:
    """Experiment B: does remediation separate real blur from darkness?"""
    source = textured_object()

    # Ground truth per blur level: the verdict at reference exposure, where the
    # sharpness metric is not confounded by exposure.
    reference_verdict: dict[float, bool] = {}
    for sigma in FACTORIAL_BLUR_SIGMAS:
        reference = adjust_exposure(gaussian_blur(source, sigma), REFERENCE_GAIN)
        result = run_capture_remediation(reference, quality_policy, remediation_policy)
        reference_verdict[sigma] = "BLUR_RISK" in result.original_evidence.quality_flags

    rows: list[dict] = []
    for sigma in FACTORIAL_BLUR_SIGMAS:
        for gain in FACTORIAL_EXPOSURE_GAINS:
            image = adjust_exposure(gaussian_blur(source, sigma), gain)
            result = run_capture_remediation(image, quality_policy, remediation_policy)

            before = _evidence_row(result.original_evidence)
            canonical = _evidence_row(result.canonical_evidence)
            truth = reference_verdict[sigma]

            rows.append(
                {
                    "blur_sigma": sigma,
                    "exposure_gain": gain,
                    "reference_blur_verdict": truth,
                    "action": result.decision.action.value,
                    "reason_code": result.decision.reason_code.value,
                    "remediation_attempted": result.remediation_attempted,
                    "remediation_accepted": result.remediation_accepted,
                    "escalation_required": result.escalation_required,
                    "before_blur_verdict": before["blur_risk"],
                    "canonical_blur_verdict": canonical["blur_risk"],
                    "before_agrees_with_reference": before["blur_risk"] == truth,
                    "canonical_agrees_with_reference": canonical["blur_risk"] == truth,
                    **{f"before_{k}": v for k, v in before.items() if k != "blur_risk"},
                    **{
                        f"canonical_{k}": v
                        for k, v in canonical.items()
                        if k != "blur_risk"
                    },
                }
            )

    # Exclude the reference column itself: it is true by construction.
    off_reference = [r for r in rows if r["exposure_gain"] != REFERENCE_GAIN]
    before_agree = sum(1 for r in off_reference if r["before_agrees_with_reference"])
    canonical_agree = sum(
        1 for r in off_reference if r["canonical_agrees_with_reference"]
    )

    return {
        "description": (
            "Blur x exposure factorial. Ground truth per blur level is the blur "
            "verdict at reference exposure, where sharpness is unconfounded."
        ),
        "blur_sigmas": list(FACTORIAL_BLUR_SIGMAS),
        "exposure_gains": list(FACTORIAL_EXPOSURE_GAINS),
        "reference_gain": REFERENCE_GAIN,
        "reference_verdicts": {str(k): v for k, v in reference_verdict.items()},
        "rows": rows,
        "blur_verdict_agreement": {
            "note": "Reference-exposure column excluded (true by construction).",
            "off_reference_count": len(off_reference),
            "before_agreements": before_agree,
            "canonical_agreements": canonical_agree,
            "before_agreement_rate": before_agree / len(off_reference)
            if off_reference
            else float("nan"),
            "canonical_agreement_rate": canonical_agree / len(off_reference)
            if off_reference
            else float("nan"),
        },
    }


def compute_metrics(all_rows: list[dict]) -> dict:
    """Phase 1b engineering metrics over every cycle run in this evaluation."""
    total = len(all_rows)
    attempted = [r for r in all_rows if r["remediation_attempted"]]
    accepted = [r for r in attempted if r["remediation_accepted"]]
    rejected = [r for r in attempted if not r["remediation_accepted"]]
    escalated = [r for r in all_rows if r["escalation_required"]]
    harmful = [r for r in all_rows if r["reason_code"] == ReasonCode.REMEDIATION_HARMFUL.value]

    def rate(count: int, denominator: int) -> float:
        return count / denominator if denominator else float("nan")

    return {
        "cycles": total,
        "remediation_attempt_rate": rate(len(attempted), total),
        "remediation_acceptance_rate": rate(len(accepted), len(attempted)),
        "remediation_rejection_rate": rate(len(rejected), len(attempted)),
        "escalation_rate": rate(len(escalated), total),
        "harm_rate": {
            "definition": (
                "Fraction of ATTEMPTED remediations rejected because an "
                "independent critical metric (shadow clipping, highlight "
                "clipping, or contrast) worsened beyond its guardrail. Because "
                "the harm guard rejects these before they become canonical, "
                "realised harm to the output is zero by construction; this "
                "metric measures how often enhancement WOULD have damaged the "
                "capture, not how often it did."
            ),
            "harmful_rejections": len(harmful),
            "attempted": len(attempted),
            "rate": rate(len(harmful), len(attempted)),
            "realised_harm_to_canonical_output": 0.0,
        },
        "action_distribution": {
            action.value: sum(1 for r in all_rows if r["action"] == action.value)
            for action in RemediationAction
        },
        "reason_code_distribution": {
            reason.value: sum(1 for r in all_rows if r["reason_code"] == reason.value)
            for reason in ReasonCode
            if any(r["reason_code"] == reason.value for r in all_rows)
        },
    }


def write_plots(coupling: dict, factorial: dict, output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    rows = coupling["rows"]
    gains = [r["exposure_gain"] for r in rows]

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(gains, [r["before_laplacian_variance"] for r in rows],
              marker="o", label="before remediation")
    axis.plot(gains, [r["canonical_laplacian_variance"] for r in rows],
              marker="s", label="canonical (after remediation)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Exposure gain (true detail identical at every level)")
    axis.set_ylabel("Laplacian variance")
    axis.set_title("Exposure/sharpness coupling before and after remediation")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "exposure_sharpness_coupling.png", dpi=120)
    plt.close(figure)
    written.append("exposure_sharpness_coupling.png")

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(gains, [r["before_mean_luminance"] for r in rows],
              marker="o", label="before remediation")
    axis.plot(gains, [r["canonical_mean_luminance"] for r in rows],
              marker="s", label="canonical")
    axis.axhspan(DEFAULT_REMEDIATION_POLICY.target_luminance_min,
                 DEFAULT_REMEDIATION_POLICY.target_luminance_max,
                 alpha=0.12, label="target band")
    axis.set_xscale("log")
    axis.set_xlabel("Exposure gain")
    axis.set_ylabel("Mean L* (0-1)")
    axis.set_title("Luminance before and after remediation")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots_dir / "luminance_remediation.png", dpi=120)
    plt.close(figure)
    written.append("luminance_remediation.png")

    # Blur-verdict agreement heat map.
    sigmas = factorial["blur_sigmas"]
    fgains = factorial["exposure_gains"]
    grid = [[0 for _ in fgains] for _ in sigmas]
    for row in factorial["rows"]:
        i = sigmas.index(row["blur_sigma"])
        j = fgains.index(row["exposure_gain"])
        grid[i][j] = (1 if row["before_agrees_with_reference"] else 0) + (
            2 if row["canonical_agrees_with_reference"] else 0
        )

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    image = axis.imshow(grid, cmap="viridis", vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(range(len(fgains)), [str(g) for g in fgains])
    axis.set_yticks(range(len(sigmas)), [str(s) for s in sigmas])
    axis.set_xlabel("Exposure gain")
    axis.set_ylabel("Blur sigma")
    axis.set_title("Blur-verdict agreement with reference exposure\n(0 neither, 1 before only, 2 after only, 3 both)")
    figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3])
    figure.tight_layout()
    figure.savefig(plots_dir / "blur_verdict_agreement.png", dpi=120)
    plt.close(figure)
    written.append("blur_verdict_agreement.png")

    return written


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.phase1b_remediation",
        description="Phase 1b remediation experiments and engineering metrics.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_policy = DEFAULT_POLICY
    remediation_policy = DEFAULT_REMEDIATION_POLICY

    coupling = experiment_exposure_coupling(quality_policy, remediation_policy)
    factorial = experiment_blur_exposure_factorial(quality_policy, remediation_policy)
    metrics = compute_metrics(coupling["rows"] + factorial["rows"])

    context = run_context()
    context.update(
        {
            "remediation_version": REMEDIATION_VERSION,
            "trace_version": TRACE_VERSION,
            "remediation_policy_fingerprint": remediation_policy.fingerprint(),
            "remediation_policy_status": remediation_policy.status,
            "max_automated_attempts": MAX_AUTOMATED_ATTEMPTS,
        }
    )

    payload = {
        "run_context": context,
        "quality_policy": quality_policy.to_dict(),
        "remediation_policy": remediation_policy.to_dict(),
        "experiment_a_exposure_coupling": coupling,
        "experiment_b_blur_exposure_factorial": factorial,
        "metrics": metrics,
        "timing_note": (
            "Per-record timing is excluded so these artifacts are reproducible "
            "byte-for-byte."
        ),
    }

    (output_dir / "phase1b_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "phase1b_exposure_coupling.csv", coupling["rows"])
    _write_csv(output_dir / "phase1b_blur_exposure_factorial.csv", factorial["rows"])

    # One full worked trace, as the human-inspectable causal artifact.
    example = run_capture_remediation(
        adjust_exposure(textured_object(), 0.25), quality_policy, remediation_policy
    )
    (output_dir / "phase1b_example_trace.json").write_text(
        json.dumps(example.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    plots = [] if args.skip_plots else write_plots(coupling, factorial, output_dir)

    sensitivity = coupling["sharpness_exposure_sensitivity"]
    print(f"cycles run:            {metrics['cycles']}")
    print(f"attempt rate:          {metrics['remediation_attempt_rate']:.3f}")
    print(f"acceptance rate:       {metrics['remediation_acceptance_rate']:.3f}")
    print(f"escalation rate:       {metrics['escalation_rate']:.3f}")
    print(f"harm rate:             {metrics['harm_rate']['rate']:.3f}")
    print(
        "sharpness CV (automated/clean): "
        f"{sensitivity['automated_or_clean_before']:.3f} -> "
        f"{sensitivity['automated_or_clean_canonical']:.3f}"
    )
    print(
        "blur-verdict agreement: "
        f"{factorial['blur_verdict_agreement']['before_agreement_rate']:.3f} -> "
        f"{factorial['blur_verdict_agreement']['canonical_agreement_rate']:.3f}"
    )
    print(f"plots written:         {len(plots)}")
    print(f"output dir:            {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
