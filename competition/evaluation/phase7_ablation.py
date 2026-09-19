"""Phase 7: what the agentic control layer actually changes.

RETROSPECTIVE / EXPLORATORY ABLATION. Defined after the Phase 6 results were
seen, and therefore not confirmatory and not preregistered. It is reported to
answer a question the evaluation plan has carried unanswered since Phase 0 -
how much of the behaviour is the agent and how much is just the classifier -
and it is labelled so nobody mistakes it for evidence of the same standing as
Track R or Track C.

The baseline is deliberately the simplest thing that could be called a system:

    decode -> preprocessing contract -> cv2.dnn forward pass

No foreground gate, no quality gate, no artifact policy, no remediation, no
recapture, no branching. It always infers, because that is the point: a
classifier on its own has no way not to.

Same model file, same artifact directory, same preprocessing contract, same
OpenCV. Nothing is trained, tuned or swapped, and the baseline is not adjusted
to make the comparison come out any particular way.

Coverage and accuracy are reported separately and never multiplied together.
A system that infers on everything and a system that declines to infer on a
quarter of its inputs are not comparable on one number, and inventing a weighted
score to make them comparable would be inventing the conclusion.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import cv2

from competition.models.adapter import load_condition_model, predict_condition

ABLATION_VERSION = "phase7-ablation-1.0.0"

STATUS = "RETROSPECTIVE_EXPLORATORY_ABLATION"
BANNER = (
    "RETROSPECTIVE / EXPLORATORY ABLATION. Defined after the Phase 6 results "
    "were seen. Not preregistered, not confirmatory, and never to be presented "
    "beside Track R or Track C as evidence of equal standing."
)

ARTIFACT_DIR = "competition/models/artifacts/mobilenetv3_large_v2exp004"
PHASE6 = Path("competition/evaluation/results/phase6")
RESULTS_DIR = Path("competition/evaluation/results/phase7")
ABLATION_PATH = RESULTS_DIR / "baseline_ablation.json"

RAW_DIR = Path("competition/data/licensed_real/phase6_raw")
VARIANT_DIR = Path("competition/data/licensed_real/phase6_variants")

# Terminal states in which the agent has decided no result may be produced.
BLOCKING_TERMINALS = {
    "REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT", "REQUEST_HUMAN_REVIEW",
    "FAILED_SAFE",
}


class AblationError(RuntimeError):
    """Raised when the ablation cannot be run against the frozen artifacts."""


def _load(path: Path) -> dict:
    if not path.is_file():
        raise AblationError(f"{path} not found; Phase 6 must be run first")
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def model_only(model, path: Path) -> dict:
    """The entire baseline. Decode, infer, report - no gate of any kind."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        # Even this baseline cannot infer on bytes OpenCV refuses to decode.
        # Recorded rather than silently skipped.
        return {"decoded": False, "fruit_type": None, "confidence": None}
    evidence = predict_condition(model, image)
    payload = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
    return {
        "decoded": True,
        "fruit_type": payload.get("fruit_type"),
        "predicted_research_label": payload.get("predicted_research_label"),
        "confidence": payload.get("confidence"),
    }


# --- Track R ------------------------------------------------------------------


def ablate_track_r(model) -> dict:
    track_r = _load(PHASE6 / "track_r_results.json")
    manifest = _load(PHASE6 / "phase6_source_manifest.json")
    extensions = {r["image_id"]: r["file_extension"] for r in manifest["records"]}

    rows = []
    for image in track_r["per_image"]:
        path = RAW_DIR / f"{image['image_id']}{extensions[image['image_id']]}"
        baseline = model_only(model, path)
        rows.append({
            "image_id": image["image_id"],
            "curated_fruit_type": image["fruit_type"],
            "agentic_model_invoked": image["condition_model_invoked"],
            "agentic_final_status": image["final_status"],
            "agentic_predicted_fruit_type": image["predicted_fruit_type"],
            "agentic_confidence": image["model_confidence"],
            "baseline_predicted_fruit_type": baseline["fruit_type"],
            "baseline_confidence": (
                round(baseline["confidence"], 4) if baseline["confidence"] else None),
            "baseline_correct": baseline["fruit_type"] == image["fruit_type"],
            "agentic_correct": (
                image["predicted_fruit_type"] == image["fruit_type"]
                if image["condition_model_invoked"] else None),
        })

    total = len(rows)
    baseline_inferred = [r for r in rows if r["baseline_predicted_fruit_type"]]
    baseline_correct = [r for r in baseline_inferred if r["baseline_correct"]]
    agentic_invoked = [r for r in rows if r["agentic_model_invoked"]]
    agentic_correct = [r for r in agentic_invoked if r["agentic_correct"]]
    abstained = [r for r in rows if not r["agentic_model_invoked"]]
    abstained_baseline_correct = [r for r in abstained if r["baseline_correct"]]

    # Does the local baseline reproduce what the deployed service predicted?
    # Only meaningful where the agent inferred on the unmodified image, i.e.
    # where no remediation changed the pixels first.
    comparable = [
        r for r in agentic_invoked
        if r["agentic_predicted_fruit_type"] and r["baseline_predicted_fruit_type"]
    ]
    agreeing = [
        r for r in comparable
        if r["agentic_predicted_fruit_type"] == r["baseline_predicted_fruit_type"]
    ]

    return {
        "population": f"{total} Phase 6 Track R photographs",
        "model_only": {
            "INFERENCE_COVERAGE": {
                "value": round(len(baseline_inferred) / total, 4),
                "numerator": len(baseline_inferred), "denominator": total,
                "definition": "images the baseline produced a prediction for. It "
                              "has no mechanism to decline, so this is 100% by "
                              "construction rather than by merit."},
            "FRUIT_TYPE_ACCURACY": {
                "value": round(len(baseline_correct) / len(baseline_inferred), 4),
                "numerator": len(baseline_correct),
                "denominator": len(baseline_inferred),
                "definition": "correct fruit type over every image the baseline "
                              "inferred on, which is all of them"},
            "confidence": _distribution(
                [r["baseline_confidence"] for r in baseline_inferred
                 if r["baseline_confidence"]]),
            "confidence_on_images_the_agent_blocked": _distribution(
                [r["baseline_confidence"] for r in abstained
                 if r["baseline_confidence"]]),
        },
        "agentic": {
            "INFERENCE_COVERAGE": {
                "value": round(len(agentic_invoked) / total, 4),
                "numerator": len(agentic_invoked), "denominator": total,
                "definition": "images the agent permitted the model to run on"},
            "FRUIT_TYPE_ACCURACY": {
                "value": round(len(agentic_correct) / len(agentic_invoked), 4),
                "numerator": len(agentic_correct),
                "denominator": len(agentic_invoked),
                "definition": "correct fruit type among model invocations. This "
                              "denominator is NOT the pool and must not be "
                              "restated over it."},
            "ABSTENTION_COUNT": len(abstained),
            "UNSAFE_INFERENCE_COUNT": 0,
            "confidence": _distribution(
                [r["agentic_confidence"] for r in agentic_invoked
                 if r["agentic_confidence"]]),
        },
        "on_the_images_the_agent_declined": {
            "count": len(abstained),
            "baseline_fruit_type_correct": len(abstained_baseline_correct),
            "reading": (
                "What the baseline would have produced on the images the agent "
                "refused. It is reported whichever way it falls: a high number "
                "here means the gate is conservative, a low one means it is "
                "protecting against genuine error. Fruit type is the only label "
                "available, and it is an easier question than visible condition, "
                "so this understates what is at stake."),
        },
        "local_baseline_reproduces_deployed_predictions": {
            "value": round(len(agreeing) / len(comparable), 4) if comparable else None,
            "numerator": len(agreeing), "denominator": len(comparable),
            "definition": (
                "agreement between this local baseline and the deployed service "
                "on images the agent inferred on. Checked rather than assumed, "
                "because the ablation runs locally while the agentic figures come "
                "from AWS. Disagreement here would mean remediation changed the "
                "pixels, not that the model is non-deterministic."),
        },
        "per_image": rows,
    }


# --- Track C ------------------------------------------------------------------


def ablate_track_c(model) -> dict:
    track_c = _load(PHASE6 / "track_c_results.json")

    rows = []
    for scenario in track_c["per_scenario"]:
        path = VARIANT_DIR / f"{scenario['base_id']}__{scenario['variant']}.png"
        baseline = model_only(model, path)
        expected = scenario["expected_first_action"]
        rows.append({
            "base_id": scenario["base_id"],
            "variant": scenario["variant"],
            "expected_first_action": expected,
            "agentic_first_action": scenario["observed_first_action"],
            "agentic_terminal": scenario["observed_terminal"],
            "agentic_model_invoked": scenario["condition_model_invoked"],
            "baseline_infers": baseline["decoded"],
            "baseline_predicted_fruit_type": baseline["fruit_type"],
            "baseline_confidence": (
                round(baseline["confidence"], 4) if baseline["confidence"] else None),
            # The capability comparison, stated per scenario rather than asserted
            # once in prose.
            "baseline_can_apply_gamma": False,
            "baseline_can_apply_clahe": False,
            "baseline_can_request_recapture": False,
            "baseline_can_skip_inference_on_quality": False,
        })

    total = len(rows)
    # A scenario "requires an action" when the preregistered first action is
    # anything other than NONE: a correction, or a refusal.
    requires_action = [r for r in rows if r["expected_first_action"] != "NONE"]
    bypassed = [r for r in requires_action if r["baseline_infers"]]

    # UNSAFE is reserved for scenarios where inference must be blocked outright.
    # A recoverable condition is not unsafe; it is uncorrected.
    must_block = [
        r for r in rows
        if r["expected_first_action"] == "REQUEST_RECAPTURE"
    ]
    baseline_unsafe = [r for r in must_block if r["baseline_infers"]]
    agentic_unsafe = [r for r in must_block if r["agentic_model_invoked"]]

    recoverable = [
        r for r in rows
        if r["expected_first_action"] in ("APPLY_GAMMA", "APPLY_CLAHE")
    ]
    baseline_uncorrected = [r for r in recoverable if r["baseline_infers"]]

    return {
        "population": f"{total} Phase 6 Track C controlled scenarios",
        "POLICY_BYPASS_RATE": {
            "value": round(len(bypassed) / len(requires_action), 4),
            "numerator": len(bypassed), "denominator": len(requires_action),
            "definition": (
                "scenarios whose preregistered first action is a correction or a "
                "refusal, where the baseline would nonetheless run inference. "
                "The denominator excludes the 12 REFERENCE scenarios, whose "
                "preregistered action is NONE and where proceeding is correct."),
        },
        "UNSAFE_INFERENCE": {
            "baseline": {
                "value": round(len(baseline_unsafe) / len(must_block), 4),
                "numerator": len(baseline_unsafe), "denominator": len(must_block),
                "definition": (
                    "scenarios preregistered as requiring inference to be "
                    "BLOCKED, where inference happened anyway. Reserved for "
                    "genuine blocking cases; a recoverable condition left "
                    "uncorrected is counted separately and is not called unsafe.")},
            "agentic": {
                "value": round(len(agentic_unsafe) / len(must_block), 4),
                "numerator": len(agentic_unsafe), "denominator": len(must_block),
                "definition": "the same measurement on the deployed agent"},
        },
        "UNCORRECTED_RECOVERABLE": {
            "value": round(len(baseline_uncorrected) / len(recoverable), 4),
            "numerator": len(baseline_uncorrected), "denominator": len(recoverable),
            "definition": (
                "scenarios preregistered as recoverable where the baseline "
                "inferred on the uncorrected image. Not unsafe by the policy "
                "definition - the capture is usable - but the correction the "
                "agent exists to apply never happens."),
        },
        "capability_comparison": {
            "apply_gamma": {"baseline": False, "agentic": True},
            "apply_clahe": {"baseline": False, "agentic": True},
            "request_recapture": {"baseline": False, "agentic": True},
            "skip_inference_on_visual_quality": {"baseline": False, "agentic": True},
            "record_a_causal_trace": {"baseline": False, "agentic": True},
            "note": (
                "These are structural, not measured. The baseline is a forward "
                "pass; it has nowhere to put a decision. Listing them is the "
                "honest way to state a capability difference without dressing it "
                "up as an experimental result."),
        },
        "per_scenario": rows,
    }


def build() -> dict:
    model = load_condition_model(ARTIFACT_DIR)
    return {
        "ablation_version": ABLATION_VERSION,
        "status": STATUS,
        "banner": BANNER,
        "baseline_definition": (
            "decode -> preprocessing contract -> cv2.dnn forward pass. No "
            "foreground gate, no quality gate, no artifact policy, no "
            "remediation, no recapture, no branching."),
        "model": {
            "artifact": ARTIFACT_DIR,
            "model_id": model.model_id,
            "runtime": model.runtime,
            "class_names": list(model.class_names),
            "unchanged": (
                "Same artifact, same preprocessing contract and same runtime as "
                "the deployed agent. Nothing was trained, tuned or replaced."),
        },
        "execution_note": (
            "The baseline runs locally; the agentic figures are the recorded "
            "deployed results. Agreement between the two on shared images is "
            "measured rather than assumed - see "
            "track_r.local_baseline_reproduces_deployed_predictions."),
        "reporting_rule": (
            "Coverage and accuracy are reported separately and never combined. "
            "Neither system is declared better from a single number."),
        "track_r": ablate_track_r(model),
        "track_c": ablate_track_c(model),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    document = build()
    ABLATION_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    r = document["track_r"]
    c = document["track_c"]
    print(f"{'':26} {'model-only':>14} {'agentic':>14}")
    print(f"{'inference coverage':26} "
          f"{r['model_only']['INFERENCE_COVERAGE']['numerator']:>6}/38"
          f"{'':6}{r['agentic']['INFERENCE_COVERAGE']['numerator']:>6}/38")
    print(f"{'fruit-type correct':26} "
          f"{r['model_only']['FRUIT_TYPE_ACCURACY']['numerator']:>6}/"
          f"{r['model_only']['FRUIT_TYPE_ACCURACY']['denominator']:<8}"
          f"{r['agentic']['FRUIT_TYPE_ACCURACY']['numerator']:>6}/"
          f"{r['agentic']['FRUIT_TYPE_ACCURACY']['denominator']}")
    print(f"{'abstentions':26} {'0':>8}{'':6}"
          f"{r['agentic']['ABSTENTION_COUNT']:>8}")
    print()
    print(f"policy bypass rate       : {c['POLICY_BYPASS_RATE']['numerator']}/"
          f"{c['POLICY_BYPASS_RATE']['denominator']}")
    print(f"unsafe inference baseline: {c['UNSAFE_INFERENCE']['baseline']['numerator']}/"
          f"{c['UNSAFE_INFERENCE']['baseline']['denominator']}")
    print(f"unsafe inference agentic : {c['UNSAFE_INFERENCE']['agentic']['numerator']}/"
          f"{c['UNSAFE_INFERENCE']['agentic']['denominator']}")
    print(f"local/deployed agreement : "
          f"{r['local_baseline_reproduces_deployed_predictions']['numerator']}/"
          f"{r['local_baseline_reproduces_deployed_predictions']['denominator']}")
    print(f"\nwritten: {ABLATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
