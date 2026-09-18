"""Calibrate ROI capture-quality thresholds. **Skeleton — not yet runnable.**

The methodology is fixed here in advance, deliberately, so that thresholds
cannot later be chosen to flatter a result. What is missing is the only thing
that cannot be written ahead of time: the photographs.

**This command fails clearly when the capture set is absent.** It does not
invent thresholds, fall back to the research photographs, or emit placeholder
numbers. Phase 2b established that the research images are the wrong population
for this purpose, and using them as a proxy would repeat the error the whole
capture exercise exists to correct.

Access discipline
-----------------
Calibration reads ``split == calibration`` only. The held-out evaluation items
stay closed until a policy is locked and fingerprinted, after which
``evaluate_locked_quality_policy`` (a later phase) may open them exactly once.

    .venv-competition/bin/python -m competition.evaluation.calibrate_quality_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from competition.data.capture_schema import CaptureSchemaError, QualityTarget, Split
from competition.evaluation.prepare_capture_manifest import DEFAULT_MANIFEST
from competition.evaluation.split_capture_items import DEFAULT_SPLIT_PATH, load_split

CALIBRATION_METHOD_VERSION = "phase2c-calibration-method-1.0.0"

# Minimum calibration captures before a threshold may be proposed at all. Below
# this the exercise is anecdote, not calibration.
MIN_CALIBRATION_CAPTURES = 40


class CalibrationNotReady(RuntimeError):
    """Raised when calibration is attempted without sufficient captured data."""


#: Locked methodology. Fixed before any data exists so it cannot be shaped by it.
CALIBRATION_METHOD: dict = {
    "version": CALIBRATION_METHOD_VERSION,
    "population": (
        "Captures whose item is assigned to the calibration split, measured in "
        "FOREGROUND_MASKED scope where segmentation is valid. Held-out evaluation "
        "items are never read."
    ),
    "ground_truth": (
        "The preregistered QualityTarget from the capture protocol, assigned by "
        "the photographer's intent and human review - never derived from the "
        "metrics being calibrated."
    ),
    "metrics": {
        "roi_sharpness": {
            "failure_condition": "Insufficient subject detail for reliable inspection.",
            "analysis": (
                "Distribution of ROI Laplacian variance by QualityTarget. Sweep "
                "candidate thresholds; report false-accept and false-block counts "
                "at each. Additionally break the distribution down by fruit type, "
                "capture condition, and subject scale (foreground_fraction and "
                "resolution), because ROI sharpness is known to be content-, "
                "exposure-, enhancement- and scale-dependent."
            ),
            "single_threshold_is_not_assumed": True,
            "fruit_specific_thresholds": (
                "Avoided unless strongly justified: fruit identity is not reliably "
                "known before the quality gate runs, so a fruit-specific threshold "
                "would need information the gate does not have."
            ),
            "if_no_separation_exists": (
                "Report that result and redesign the sharpness measurement - for "
                "example normalising by local contrast, or measuring on a fixed "
                "subject-relative scale. Do NOT tune until a number appears to work."
            ),
        },
        "mean_luminance": {
            "failure_condition": "Subject too dark or too bright to inspect reliably.",
            "analysis": "ROI mean/median L* by QualityTarget; select a band, not a point.",
        },
        "contrast_score": {
            "failure_condition": "Subject tonal range too compressed.",
            "analysis": "ROI p95-p5 spread by QualityTarget; likely remains advisory.",
        },
        "shadow_clip_fraction": {
            "failure_condition": "Unrecoverable shadow crush on the subject.",
            "analysis": (
                "ROI clipped fraction by QualityTarget. Phase 2b measured a median "
                "of 0.0000 on research photographs once background was excluded, so "
                "a threshold near zero may be appropriate - to be confirmed, not assumed."
            ),
        },
        "highlight_clip_fraction": {
            "failure_condition": "Unrecoverable blown highlights on the subject.",
            "analysis": (
                "As above. The GLARE condition exists specifically to populate the "
                "upper tail with genuine subject specularity."
            ),
        },
        "foreground_validity": {
            "failure_condition": "Subject not locatable, so no ROI metric is trustworthy.",
            "analysis": (
                "Validity rate by capture condition. CLUTTERED_BACKGROUND, "
                "SMALL_SUBJECT and PARTIAL_OCCLUSION are the informative cells. "
                "Guard thresholds are reviewed here, not tuned to raise the rate."
            ),
        },
    },
    "objective": (
        "Report the confusion matrix against the QualityTarget rubric. Do NOT "
        "optimise a single accuracy number."
    ),
    "error_asymmetry": (
        "FALSE ACCEPT (an objectively unusable capture is permitted) and FALSE "
        "BLOCK (an acceptable capture is rejected) are reported separately at "
        "every candidate threshold. For a safety-oriented gate false accepts "
        "matter more, but no target percentage is set in advance without "
        "evidence - the trade-off curve is shown rather than a chosen point "
        "being justified after the fact."
    ),
    "advisory_vs_hard_gate": (
        "A metric stays advisory when it separates poorly, when its failure mode "
        "is recoverable, or when blocking on it would reject captures a human "
        "would accept. Promotion to a hard gate requires separation evidence."
    ),
    "locking": (
        "On completion: freeze the threshold set, record its fingerprint and the "
        "code revision, and only then open the held-out evaluation items - once. "
        "Thresholds changed after seeing held-out results make that set "
        "exploratory, not confirmatory, permanently."
    ),
}


def load_calibration_captures(manifest_path: Path, split_path: Path) -> list[dict]:
    """Load calibration-split captures only. Refuses to return evaluation items."""
    split_document = load_split(Path(split_path))
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise CalibrationNotReady(
            "No capture manifest exists. The deployment-domain capture set has not "
            "been collected. See docs/competition/PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_fingerprint") != split_document["fingerprint"]:
        raise CalibrationNotReady(
            "Manifest was built against a different split; regenerate it before calibrating."
        )

    captures = [
        record for record in manifest.get("captures", [])
        if record.get("split") == Split.CALIBRATION.value
    ]

    leaked = [
        record["image_id"] for record in captures
        if split_document["assignment"].get(record.get("item_id")) != Split.CALIBRATION.value
    ]
    if leaked:
        raise CalibrationNotReady(
            f"{len(leaked)} record(s) claim the calibration split but their items "
            f"are assigned elsewhere: {leaked[:5]}"
        )
    return captures


def assert_ready(captures: list[dict]) -> None:
    """Refuse to calibrate on too little data, or without both ends of the rubric."""
    if len(captures) < MIN_CALIBRATION_CAPTURES:
        raise CalibrationNotReady(
            f"only {len(captures)} calibration captures; at least "
            f"{MIN_CALIBRATION_CAPTURES} are required before a threshold may be proposed"
        )

    targets = {record.get("quality_target") for record in captures}
    missing = {QualityTarget.ACCEPTABLE.value, QualityTarget.RECAPTURE_REQUIRED.value} - targets
    if missing:
        raise CalibrationNotReady(
            f"calibration set lacks {sorted(missing)} examples; a threshold cannot "
            "separate classes that are not both present"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.calibrate_quality_policy",
        description="Calibrate ROI capture-quality thresholds (requires captured data).",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--show-method", action="store_true",
                        help="print the locked methodology and exit")
    args = parser.parse_args(argv)

    if args.show_method:
        print(json.dumps(CALIBRATION_METHOD, indent=2, sort_keys=True))
        return 0

    try:
        captures = load_calibration_captures(Path(args.manifest), Path(args.split))
        assert_ready(captures)
    except (CalibrationNotReady, CaptureSchemaError) as error:
        print("CALIBRATION NOT READY")
        print(f"  {error}")
        print()
        print("No thresholds were invented, and the research photographs were not")
        print("used as a substitute: Phase 2b established they are the wrong")
        print("population for this purpose.")
        print("Run with --show-method to review the locked methodology.")
        return 2

    # Reaching here means captures exist; the analysis itself is the next phase.
    print(f"calibration captures available: {len(captures)}")
    print("Threshold selection is Phase 2c-B and is not implemented here.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
