"""Phase 6 report figures.

Five plots, deliberately plain. Every one carries its own n in the title or the
axis label, because the samples here are small - 38 photographs and 48
scenarios - and a chart that hides that invites a reader to over-read it.

No confidence intervals are drawn. At these counts an interval would be wide
enough to be useless and decorative enough to look authoritative, which is the
worst combination.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path("competition/evaluation/results/phase6")
PLOTS_DIR = RESULTS_DIR / "plots"

FIGURES_VERSION = "phase6-figures-1.0.0"


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def write_plots() -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    track_r = _load("track_r_results.json")
    track_c = _load("track_c_results.json")
    taxonomy = _load("failure_taxonomy.json")
    latency = _load("latency.json")

    # 1. Track R outcomes -----------------------------------------------------
    states = track_r["counts"]["terminal_states"]
    n = sum(states.values())
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.bar(list(states), list(states.values()), color="#3f6fa8")
    axis.set_ylabel("images")
    axis.set_title(f"Track R terminal states on fresh real photographs (n={n})")
    axis.grid(True, axis="y", alpha=0.3)
    for index, value in enumerate(states.values()):
        axis.text(index, value + 0.3, str(value), ha="center")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "track_r_outcomes.png", dpi=120)
    plt.close(figure)
    written.append("track_r_outcomes.png")

    # 2. Track C expected vs actual, per variant ------------------------------
    per_variant = track_c["per_variant"]["action_selection"]
    keys = list(per_variant)
    matched = [per_variant[k]["numerator"] for k in keys]
    missed = [per_variant[k]["denominator"] - per_variant[k]["numerator"] for k in keys]
    figure, axis = plt.subplots(figsize=(9, 4.4))
    axis.bar(keys, matched, label="matched preregistered action", color="#3f8f5f")
    axis.bar(keys, missed, bottom=matched, label="did not match", color="#b4553f")
    axis.set_ylabel("scenarios")
    axis.set_title("Track C: first action vs preregistration (12 bases per variant)")
    axis.legend(loc="lower right", fontsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "track_c_expected_vs_actual.png", dpi=120)
    plt.close(figure)
    written.append("track_c_expected_vs_actual.png")

    # 3. Action-selection confusion matrix ------------------------------------
    pairs = Counter(
        (s["expected_first_action"], s["observed_first_action"] or "NO_ACTION")
        for s in track_c["per_scenario"]
    )
    expected = sorted({e for e, _ in pairs})
    observed = sorted({o for _, o in pairs})
    grid = [[pairs.get((e, o), 0) for o in observed] for e in expected]
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    image = axis.imshow(grid, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(observed)), observed, rotation=30, ha="right", fontsize=8)
    axis.set_yticks(range(len(expected)), expected, fontsize=8)
    axis.set_xlabel("observed first action")
    axis.set_ylabel("preregistered action")
    axis.set_title(f"Track C action selection (n={sum(pairs.values())})")
    for row in range(len(expected)):
        for column in range(len(observed)):
            value = grid[row][column]
            if value:
                axis.text(column, row, str(value), ha="center", va="center",
                          color="white" if value > max(max(grid)) / 2 else "black")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "track_c_action_confusion.png", dpi=120)
    plt.close(figure)
    written.append("track_c_action_confusion.png")

    # 4. Deployed latency -----------------------------------------------------
    groups = [("Track R\nnatural", latency["track_r_natural"]),
              ("Track C\nremediation", latency["track_c_remediation_path"]),
              ("Track C\nrecapture", latency["track_c_recapture_path"]),
              ("Track C\nother", latency["track_c_other"])]
    groups = [(label, stats) for label, stats in groups if stats.get("n")]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    positions = range(len(groups))
    axis.bar(positions, [s["median_ms"] for _, s in groups], color="#3f6fa8",
             label="median")
    axis.plot(positions, [s["p95_ms"] for _, s in groups], "o--", color="#b4553f",
              label="p95")
    axis.set_xticks(list(positions),
                    [f"{label}\n(n={stats['n']})" for label, stats in groups],
                    fontsize=8)
    axis.set_ylabel("client-observed ms")
    axis.set_title("Deployed service latency, client-observed (includes network)")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "deployed_latency.png", dpi=120)
    plt.close(figure)
    written.append("deployed_latency.png")

    # 5. Failure taxonomy -----------------------------------------------------
    counts = taxonomy["counts"]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.barh(list(counts)[::-1], list(counts.values())[::-1], color="#7a5ea8")
    axis.set_xlabel(f"cases (total {taxonomy['total']})")
    axis.set_title("Phase 6 failure taxonomy, both tracks")
    axis.grid(True, axis="x", alpha=0.3)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "failure_taxonomy.png", dpi=120)
    plt.close(figure)
    written.append("failure_taxonomy.png")

    return written


def main() -> int:
    written = write_plots()
    if not written:
        print("matplotlib unavailable; no figures written")
        return 0
    for name in written:
        print(f"  {name}")
    print(f"\n{len(written)} figures in {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
