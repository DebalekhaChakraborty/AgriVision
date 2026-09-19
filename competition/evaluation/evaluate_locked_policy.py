"""Open the held-out source groups once, against a policy that is already frozen.

The ordering is the whole point. A held-out result is only evidence if the
decision rule was fixed before the data was seen, so this module refuses to run
without a locked policy file and records the policy's threshold fingerprint in a
ledger alongside every run.

Confirmatory versus exploratory
-------------------------------
The first held-out run for a given set of thresholds is CONFIRMATORY. Any later
run under *different* thresholds is EXPLORATORY and is labelled as such in its
own output, permanently: once the held-out numbers have been seen, a threshold
change is informed by them, and no amount of good intention makes the second
look independent again. The ledger makes that automatic rather than a matter of
remembering. It is append-only by construction - a run is added, never edited.

This does not stop anyone re-tuning and re-running. It stops the result being
quietly reported as if it were the first look.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.agent.inspection import ADVISORY_FLAGS, DEFAULT_BLOCKING_FLAGS
from competition.agent.remediation import MAX_AUTOMATED_ATTEMPTS
from competition.data.licensed_sources import CorpusError, CorpusTrack
from competition.evaluation.calibrate_roi_policy import (
    LOCKED_POLICY_PATH,
    RESULTS_DIR,
    clipping_scope_comparison,
    confusion_matrix,
    gate_sample,
    measure_population,
    segmentation_summary,
)
from competition.evaluation.degrade_licensed_corpus import (
    DEFAULT_SEED as DEGRADATION_SEED,
    ExpectedVerdict,
    load_derived,
)
from competition.evaluation.split_licensed_corpus import held_out_view
from competition.agent.policy import DEFAULT_REMEDIATION_POLICY
from competition.vision.enhancement import (
    GammaParameters,
    apply_gamma_correction,
    gamma_for_target_luminance,
)
from competition.vision.evidence import QualityFlag
from competition.vision.roi_policy import RoiQualityPolicy

HELD_OUT_REPORT = RESULTS_DIR / "phase2c_heldout.json"
LEDGER_PATH = RESULTS_DIR / "phase2c_heldout_ledger.json"
EVALUATION_VERSION = "phase2c-heldout-evaluation-1.0.0"

CONFIRMATORY = "CONFIRMATORY"
EXPLORATORY = "EXPLORATORY"

# Flags a single automated tone correction can plausibly clear. Blur, clipping
# and insufficient resolution cannot be enhanced away: the detail is gone, and
# attempting it would manufacture the appearance of a usable capture.
TONE_REMEDIABLE_FLAGS = frozenset({
    QualityFlag.UNDEREXPOSED.value,
    QualityFlag.OVEREXPOSED.value,
})


def load_locked_policy(path: Path = LOCKED_POLICY_PATH) -> RoiQualityPolicy:
    if not path.is_file():
        raise CorpusError(
            f"{path.name} not found; held-out evaluation requires a frozen policy. "
            "Run calibrate_roi_policy first."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    policy = RoiQualityPolicy.from_dict(document)
    recorded = document.get("threshold_fingerprint")
    if recorded and recorded != policy.threshold_fingerprint():
        raise CorpusError(
            f"locked policy fingerprint mismatch: file records {recorded!r}, "
            f"thresholds hash to {policy.threshold_fingerprint()!r}; the file was edited"
        )
    return policy


def read_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def classify_run(policy: RoiQualityPolicy, ledger: list[dict]) -> tuple[str, str]:
    """Decide whether this run may be reported as confirmatory."""
    fingerprint = policy.threshold_fingerprint()
    if not ledger:
        return CONFIRMATORY, "first held-out run; thresholds were frozen beforehand"
    seen = {run["threshold_fingerprint"] for run in ledger}
    if fingerprint in seen:
        return CONFIRMATORY, (
            "re-run of thresholds already evaluated once; the result is identical "
            "evidence, not a second independent look"
        )
    return EXPLORATORY, (
        f"held-out data was already opened under {len(seen)} other threshold "
        "set(s); any change made after seeing those results is informed by them"
    )


def append_ledger(entry: dict, path: Path = LEDGER_PATH) -> None:
    runs = read_ledger(path)
    runs.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": runs, "count": len(runs)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def gate_outcome(sample: dict, policy: RoiQualityPolicy) -> dict:
    """Classify one sample into the agent's action vocabulary.

    Mirrors the Phase 1b action set without rewiring the agent loop, which is
    Phase 3 work. It answers the question Phase 2c-B actually needs: with the
    calibrated ROI thresholds, how often would the system proceed, attempt a
    correction, or hand back to a human?
    """
    result = gate_sample(sample, policy, "roi")
    if not result["assessable"]:
        return {"outcome": "SEGMENTATION_UNAVAILABLE", "eligible": False,
                "remediation_attempted": False, "escalated": True, "flags": []}

    flags = set(result["flags"])
    blocking = flags & set(DEFAULT_BLOCKING_FLAGS)
    if not blocking:
        return {"outcome": "ELIGIBLE", "eligible": True,
                "remediation_attempted": False, "escalated": False,
                "flags": sorted(flags),
                "advisory": sorted(flags & set(ADVISORY_FLAGS))}
    if blocking <= TONE_REMEDIABLE_FLAGS:
        return {"outcome": "REMEDIATION_CANDIDATE", "eligible": False,
                "remediation_attempted": True, "escalated": False,
                "flags": sorted(flags)}
    return {"outcome": "ESCALATED", "eligible": False,
            "remediation_attempted": False, "escalated": True,
            "flags": sorted(flags)}


def attempt_remediation(
    sample: dict, policy: RoiQualityPolicy
) -> dict | None:
    """One bounded tone correction, then re-measure inside the same region.

    Re-measurement uses the mask derived from the *corrected* image, which is
    what a deployed system would do. It is the honest test: if correction moves
    the subject region, that shows up as a changed outcome rather than being
    hidden by reusing the old mask.
    """
    from competition.evaluation.calibrate_roi_policy import _measure_one

    image = _load_sample_image(sample)
    if image is None:
        return None

    # Target and gamma bounds come from the Phase 1b remediation policy rather
    # than being reinvented here, so this measures the remediation the agent
    # would actually perform.
    remediation = DEFAULT_REMEDIATION_POLICY
    current = sample["roi"]["mean_luminance"]
    gamma = gamma_for_target_luminance(
        current, remediation.target_luminance,
        remediation.gamma_min, remediation.gamma_max,
    )
    corrected, _ = apply_gamma_correction(image, GammaParameters(gamma=gamma))

    measured = _measure_one(corrected)
    if not measured.get("roi_available"):
        return {"accepted": False, "reason": "segmentation failed after correction",
                "gamma": gamma, "attempts": MAX_AUTOMATED_ATTEMPTS}

    after = {**sample, **measured}
    outcome = gate_outcome(after, policy)
    return {
        "accepted": outcome["eligible"],
        "gamma": gamma,
        "attempts": MAX_AUTOMATED_ATTEMPTS,
        "flags_before": sorted(gate_sample(sample, policy, "roi")["flags"]),
        "flags_after": outcome["flags"],
        "luminance_before": current,
        "luminance_after": measured["roi"]["mean_luminance"],
    }


def _load_sample_image(sample: dict) -> np.ndarray | None:
    """Fetch a sample's pixels, regenerating it if it is a derived variant.

    Derived bytes are not stored (see `degrade_licensed_corpus.regenerate`), so
    a derived sample is rebuilt from its base, transformation, level and seed.
    Reading from disk would silently find nothing and skip the sample.
    """
    from competition.evaluation.build_licensed_corpus import RAW_DIR, local_path_for
    from competition.evaluation.degrade_licensed_corpus import regenerate

    base_id = sample["base_image_id"]
    base_path = None
    for extension in (".jpg", ".png", ".jpeg"):
        candidate = local_path_for(base_id, extension, RAW_DIR)
        if candidate.is_file():
            base_path = candidate
            break
    if base_path is None:
        return None

    image = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if image is None or sample["kind"] == "base":
        return image
    return regenerate(
        image, sample["transformation"], float(sample["level"]), DEGRADATION_SEED
    )


def condition_model_check(samples: list[dict], policy: RoiQualityPolicy) -> dict:
    """Run the condition model on eligible undegraded real images.

    The model predicts a joint (fruit, condition) research label, and exactly one
    half of that has ground truth here.

    **Fruit type is scored.** Every corpus image was identified by eye during
    curation, and that identification is the reason it is in the corpus at all -
    it is a real label, recorded before any model ran.

    **Visible condition is not scored.** Commons contributors photograph fruit
    without recording whether it was fresh or deteriorating. Assigning a label by
    eye now, or assuming a market photograph shows fresh produce, would turn an
    unlabelled set into a fabricated benchmark. The predicted conditions are
    reported as a distribution and nothing is claimed about their correctness.
    """
    from competition.models.adapter import load_condition_model, predict_condition

    artifact = Path("competition/models/artifacts/mobilenetv3_large_v2exp004")
    if not (artifact / "model.onnx").is_file():
        return {"ran": False, "reason": "no exported condition-model graph available"}

    try:
        model = load_condition_model(artifact)
    except Exception as error:  # noqa: BLE001
        return {"ran": False, "reason": f"model could not be loaded: {error}"}

    eligible, blocked, failures = [], 0, []
    confidences: list[float] = []
    predictions: Counter = Counter()
    fruit_predictions: Counter = Counter()
    fruit_correct: list[bool] = []
    fruit_confusion: Counter = Counter()

    for sample in samples:
        if sample["kind"] != "base":
            continue
        outcome = gate_outcome(sample, policy)
        if not outcome["eligible"]:
            blocked += 1
            continue
        image = _load_sample_image(sample)
        if image is None:
            failures.append({"content_sha256": sample.get("sample_id", "")[:16],
                             "reason": "unreadable"})
            continue
        try:
            evidence = predict_condition(model, image)
        except Exception as error:  # noqa: BLE001
            failures.append({"content_sha256": sample["sample_id"][:16],
                             "reason": type(error).__name__})
            continue
        eligible.append(sample["sample_id"])
        confidences.append(float(evidence.confidence))
        predictions[evidence.visible_condition.value] += 1
        fruit_predictions[evidence.fruit_type.value] += 1
        correct = evidence.fruit_type.value == sample["fruit_type"]
        fruit_correct.append(bool(correct))
        fruit_confusion[(sample["fruit_type"], evidence.fruit_type.value)] += 1

    array = np.asarray(confidences) if confidences else np.asarray([])
    return {
        "ran": True,
        "eligible_sample_count": len(eligible),
        "blocked_sample_count": blocked,
        "failure_count": len(failures),
        "failures": failures,
        "visible_condition_counts": dict(predictions),
        "fruit_type_counts": dict(fruit_predictions),
        "confidence_distribution": (
            {
                "n": int(array.size),
                "median": float(np.median(array)),
                "p10": float(np.percentile(array, 10)),
                "p90": float(np.percentile(array, 90)),
                "min": float(array.min()),
                "max": float(array.max()),
            } if array.size else {"n": 0}
        ),
        "fruit_type_accuracy": (
            sum(fruit_correct) / len(fruit_correct) if fruit_correct else None
        ),
        "fruit_type_correct": sum(fruit_correct),
        "fruit_type_scored": len(fruit_correct),
        "fruit_type_confusion": {
            f"true={true}|predicted={predicted}": count
            for (true, predicted), count in sorted(fruit_confusion.items())
        },
        "fruit_type_label_source": (
            "visual identification during corpus curation, recorded before any "
            "model was run"
        ),
        "visible_condition_accuracy": None,
        "visible_condition_accuracy_note": (
            "not computed: these images carry no ground-truth visible-condition "
            "label, and none was invented for them"
        ),
    }


def run_evaluation(policy: RoiQualityPolicy | None = None) -> dict:
    policy = policy or load_locked_policy()
    ledger = read_ledger()
    status, status_reason = classify_run(policy, ledger)

    view = held_out_view(policy.threshold_fingerprint())
    if not len(view):
        raise CorpusError("the held-out split is empty")
    derived = [entry for entry in load_derived() if entry["split"] == "held_out"]
    samples = measure_population(view.records, derived)

    undegraded = [s for s in samples if s["kind"] == "base"]
    degraded = [s for s in samples if s["kind"] == "derived"]

    outcomes = {s["sample_id"]: gate_outcome(s, policy) for s in samples}
    counts: Counter = Counter(o["outcome"] for o in outcomes.values())

    remediation_results = []
    for sample in samples:
        if outcomes[sample["sample_id"]]["outcome"] != "REMEDIATION_CANDIDATE":
            continue
        result = attempt_remediation(sample, policy)
        if result is not None:
            remediation_results.append(result)

    attempted = len(remediation_results)
    accepted = sum(1 for r in remediation_results if r["accepted"])
    assessable = sum(1 for o in outcomes.values() if o["outcome"] != "SEGMENTATION_UNAVAILABLE")

    by_track: dict[str, dict] = {}
    for track in (CorpusTrack.CLEAN_BASE.value, CorpusTrack.NATURAL_SCENE.value):
        subset = [s for s in undegraded if s["track"] == track]
        if not subset:
            continue
        track_outcomes = Counter(outcomes[s["sample_id"]]["outcome"] for s in subset)
        by_track[track] = {"n": len(subset), "outcomes": dict(track_outcomes)}

    return {
        "evaluation_version": EVALUATION_VERSION,
        "evaluated_on": date.today().isoformat(),
        "status": status,
        "status_reason": status_reason,
        "policy_threshold_fingerprint": policy.threshold_fingerprint(),
        "policy_lock_fingerprint": policy.lock_fingerprint(),
        "policy_gates_on_focus": policy.gates_on_focus,
        "policy_focus_metric": policy.focus_metric if policy.gates_on_focus else None,
        "population": {
            "held_out_images": len(view.records),
            "held_out_groups": len({r.source_group_id for r in view.records}),
            "derived_samples": len(derived),
            "measured_samples": len(samples),
        },
        # Kept apart deliberately: a controlled degradation of a real photograph
        # and an untouched real photograph support different claims.
        "undegraded_real": {
            "n": len(undegraded),
            "confusion": confusion_matrix(undegraded, policy, "roi"),
            "outcomes": dict(Counter(outcomes[s["sample_id"]]["outcome"] for s in undegraded)),
            "by_track": by_track,
        },
        "controlled_degradations": {
            "n": len(degraded),
            "confusion": confusion_matrix(degraded, policy, "roi"),
            "outcomes": dict(Counter(outcomes[s["sample_id"]]["outcome"] for s in degraded)),
            "by_transformation": {
                kind: dict(Counter(
                    outcomes[s["sample_id"]]["outcome"]
                    for s in degraded if s["transformation"] == kind
                ))
                for kind in sorted({s["transformation"] for s in degraded})
            },
        },
        "confusion_matrix_roi": confusion_matrix(samples, policy, "roi"),
        "confusion_matrix_whole_image": confusion_matrix(samples, policy, "whole_image"),
        "gate": {
            "assessable": assessable,
            "eligible": counts.get("ELIGIBLE", 0),
            "eligibility_rate": counts.get("ELIGIBLE", 0) / assessable if assessable else None,
            "remediation_candidates": counts.get("REMEDIATION_CANDIDATE", 0),
            "escalated": counts.get("ESCALATED", 0),
            "segmentation_unavailable": counts.get("SEGMENTATION_UNAVAILABLE", 0),
            "escalation_rate": counts.get("ESCALATED", 0) / assessable if assessable else None,
            "remediation_attempt_rate": attempted / assessable if assessable else None,
            "remediation_attempted": attempted,
            "remediation_accepted": accepted,
            "remediation_acceptance_rate": (accepted / attempted) if attempted else None,
            "max_automated_attempts": MAX_AUTOMATED_ATTEMPTS,
        },
        "segmentation": segmentation_summary(samples),
        "scope_comparison": clipping_scope_comparison(samples),
        "condition_model": condition_model_check(undegraded, policy),
        "claim_boundary": (
            "held-out licensed real photographs and controlled degradations of "
            "them; not phone-camera deployment captures"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.evaluate_locked_policy",
        description="Single held-out evaluation of a frozen ROI policy.",
    )
    parser.parse_args(argv)

    report = run_evaluation()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HELD_OUT_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    append_ledger({
        "evaluated_on": report["evaluated_on"],
        "status": report["status"],
        "threshold_fingerprint": report["policy_threshold_fingerprint"],
        "lock_fingerprint": report["policy_lock_fingerprint"],
        "held_out_images": report["population"]["held_out_images"],
    })
    print(json.dumps({
        "status": report["status"],
        "gate": report["gate"],
        "undegraded_confusion": report["undegraded_real"]["confusion"],
        "degraded_confusion": report["controlled_degradations"]["confusion"],
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
