"""Phase 6 addendum: EXPLORATORY, NOT CONFIRMATORY.

The confirmatory Track C result stands as measured: 2/12 on
UNDEREXPOSED_RECOVERABLE. The diagnostic showed why - a fixed exposure gain of
0.12, chosen against a synthetic 256px fixture, drove every real photograph
either past the shadow-clipping limit or not across the floor at all. Not one of
the twelve landed in the band the scenario was named after.

That leaves a question the confirmatory run cannot answer, because it never
tested it: **is the recoverable-underexposure band reachable on a real
photograph at all**, and if it is, does the agent take the gamma route?

This module answers it separately. Nothing here is merged into the Track C
numbers, nothing here is preregistered, and no threshold is touched. It is a
diagnostic addendum, and if it contradicts the confirmatory result the
confirmatory result is the one that counts.

The search is over the *variant construction* only - the gain applied to the
image - never over a policy value.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from competition.vision.degradation import adjust_exposure

EXPLORATORY_VERSION = "phase6-exploratory-1.0.0"

RESULTS_DIR = Path("competition/evaluation/results/phase6")
EXPLORATORY_RESULTS = RESULTS_DIR / "exploratory_underexposure.json"
VARIANT_DIR = Path("competition/data/licensed_real/phase6_variants")

BANNER = (
    "EXPLORATORY DIAGNOSTIC. Not part of the Phase 6 confirmatory evaluation, "
    "not preregistered, and never pooled with the Track C numbers. It exists to "
    "explain a confirmatory failure, not to replace it."
)

# Coarse-to-fine, and deliberately coarse: the question is whether the band is
# reachable at all, not what the optimal gain is to three decimal places.
GAIN_LADDER = tuple(round(0.02 + 0.02 * i, 3) for i in range(30))


def _measure(image: np.ndarray, policy, roi) -> tuple[float, float] | None:
    from competition.vision.foreground import ForegroundError, isolate_foreground
    from competition.vision.quality import measure_illumination

    try:
        mask, foreground = isolate_foreground(image)
    except ForegroundError:
        return None
    if not foreground.valid:
        return None
    illumination = measure_illumination(image, policy, mask)
    return float(illumination.mean_luminance), float(illumination.shadow_clip_fraction)


def search_recoverable_gain(image: np.ndarray, policy, roi) -> dict:
    """Find a gain landing below the underexposure floor without clipping shadows.

    Returns the best candidate found, or a record saying the band was not
    reachable on this photograph - which is itself a result about the
    calibrated threshold, not a failure of the search.
    """
    attempts = []
    best = None
    for gain in GAIN_LADDER:
        measured = _measure(adjust_exposure(image, gain), policy, roi)
        if measured is None:
            attempts.append({"gain": gain, "zone": "NO_VALID_FOREGROUND"})
            continue
        luminance, clipped = measured
        within_clip = clipped <= roi.shadow_clip_fraction_limit
        below_floor = luminance < roi.underexposed_mean_luminance
        zone = ("UNDEREXPOSED_RECOVERABLE" if (within_clip and below_floor)
                else "SHADOWS_CLIPPED" if not within_clip
                else "ABOVE_FLOOR")
        attempts.append({"gain": gain, "zone": zone,
                         "mean_luminance": round(luminance, 6),
                         "shadow_clip_fraction": round(clipped, 6)})
        if zone == "UNDEREXPOSED_RECOVERABLE" and best is None:
            best = attempts[-1]
    return {"best": best, "attempts": attempts,
            "band_reachable": best is not None}


def main() -> int:
    from competition.agent.orchestrator import load_locked_policies
    from competition.evaluation.phase6_run import SERVICE_URL, post_inspection
    from competition.vision.config import DEFAULT_POLICY

    roi = load_locked_policies().roi_policy
    prereg = json.loads(
        (RESULTS_DIR / "phase6_expected_actions.json").read_text(encoding="utf-8"))
    bases = sorted({s["base_id"] for s in prereg["scenarios"]})

    rows = []
    for base in bases:
        reference = cv2.imread(str(VARIANT_DIR / f"{base}__REFERENCE.png"), cv2.IMREAD_COLOR)
        if reference is None:
            continue
        search = search_recoverable_gain(reference, DEFAULT_POLICY, roi)
        row = {"base_id": base, "band_reachable": search["band_reachable"],
               "gain": search["best"]["gain"] if search["best"] else None,
               "mean_luminance": search["best"]["mean_luminance"] if search["best"] else None,
               "shadow_clip_fraction": (
                   search["best"]["shadow_clip_fraction"] if search["best"] else None),
               "zones_seen": sorted({a["zone"] for a in search["attempts"]})}
        if search["band_reachable"]:
            path = VARIANT_DIR / f"{base}__UNDEREXPOSED_EXPLORATORY.png"
            cv2.imwrite(str(path), adjust_exposure(reference, search["best"]["gain"]))
            status, body, ms = post_inspection(path, SERVICE_URL)
            row.update({
                "http_status": status,
                "final_status": body.get("final_status"),
                "remediation_action": body.get("remediation_action", ""),
                "remediation_accepted": body.get("remediation_accepted"),
                "condition_model_invoked": body.get("condition_model_invoked"),
                "terminal_reason_code": body.get("terminal_reason_code", ""),
                "quality_flags": (body.get("quality_summary") or {}).get("quality_flags", []),
                "client_latency_ms": round(ms, 1),
                "run_id": body.get("run_id", ""),
            })
        rows.append(row)
        print(f"  {base:22} reachable={row['band_reachable']!s:5} "
              f"gain={row['gain']} -> {row.get('final_status', '-')} "
              f"{row.get('remediation_action', '')}")

    reachable = [r for r in rows if r["band_reachable"]]
    gamma = [r for r in reachable if r.get("remediation_action") == "APPLY_GAMMA"]
    document = {
        "exploratory_version": EXPLORATORY_VERSION,
        "status": "EXPLORATORY_NOT_CONFIRMATORY",
        "banner": BANNER,
        "question": (
            "Is the calibrated recoverable-underexposure band reachable on a "
            "real photograph by reducing exposure, and if so does the agent "
            "select APPLY_GAMMA?"),
        "method": (
            "For each Track C base, sweep the exposure gain and keep the first "
            "value whose ROI mean luminance falls below the calibrated floor "
            "while shadow clipping stays within the calibrated limit. Where such "
            "a gain exists, send that image to the deployed service. The search "
            "is over the image, never over a threshold."),
        "thresholds_unchanged": {
            "underexposed_mean_luminance": roi.underexposed_mean_luminance,
            "shadow_clip_fraction_limit": roi.shadow_clip_fraction_limit,
        },
        "bases": len(rows),
        "band_reachable_count": len(reachable),
        "apply_gamma_count": len(gamma),
        "rows": rows,
    }
    EXPLORATORY_RESULTS.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nband reachable on {len(reachable)}/{len(rows)} bases; "
          f"APPLY_GAMMA selected on {len(gamma)} of those")
    print(f"written: {EXPLORATORY_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
