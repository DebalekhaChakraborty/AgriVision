"""Phase 2d evaluation: controlled ground truth, then one look at new real data.

Three families, reported separately
-----------------------------------
Glare, contrast and visibility are different problems with different remedies,
and a single "artefact accuracy" would average a detector that works with one
that does not. Each has its own section and its own numbers.

Two populations, never mixed
----------------------------
*Controlled* results come from injections with a known affected region into
calibration images. They quantify the detector against a characterised
perturbation - a smooth additive highlight, a black rectangle - and they are not
evidence about real glare or real occluders.

*Validation* results come from the Phase 2d pool: licensed real photographs from
Commons categories never used in Phase 2c-B, with no contributor shared with the
calibration or held-out groups. Detector parameters are frozen before it is
opened, and the fingerprints are recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.agent.artifact_policy import ArtifactPolicy, CaptureArtifactFlag
from competition.agent.artifacts import analyse_capture_artifacts
from competition.data.licensed_sources import CorpusError
from competition.evaluation.build_licensed_corpus import local_path_for
from competition.evaluation.build_validation_pool import (
    load_validation_set,
    validation_image_path,
)
from competition.evaluation.calibrate_artifact_policy import (
    FROZEN_POLICY_PATH,
    RESULTS_DIR,
)
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
from competition.vision.highlights import (
    LocalHighlightPolicy,
    highlight_mask,
    measure_local_highlights,
)
from competition.vision.quality import measure_illumination
from competition.vision.visibility import VisibilityPolicy

CONTROLLED_RESULTS = RESULTS_DIR / "controlled.json"
VALIDATION_RESULTS = RESULTS_DIR / "validation.json"
LATENCY_RESULTS = RESULTS_DIR / "latency.json"
EVALUATION_VERSION = "phase2d-artifact-evaluation-1.0.0"

GLARE_SEVERITIES = (0.0, 0.3, 0.6, 0.9)
GLARE_RADIUS_FRACTION = 0.30
OCCLUSION_COVERAGES = (0.0, 0.10, 0.25, 0.45)
CONTRAST_FACTORS = (1.0, 0.5, 0.2, 0.08)


def load_frozen_policies() -> tuple[LocalHighlightPolicy, VisibilityPolicy, ArtifactPolicy, dict]:
    if not FROZEN_POLICY_PATH.is_file():
        raise CorpusError(
            f"{FROZEN_POLICY_PATH.name} not found; run calibrate_artifact_policy first"
        )
    frozen = json.loads(FROZEN_POLICY_PATH.read_text(encoding="utf-8"))
    highlight = LocalHighlightPolicy(**{
        **frozen["highlight_policy"],
        "background_sigma_fractions": tuple(
            frozen["highlight_policy"]["background_sigma_fractions"]
        ),
    })
    visibility = VisibilityPolicy(**frozen["visibility_policy"])
    artifact = ArtifactPolicy(**frozen["artifact_policy"])
    for name, recorded, recomputed in (
        ("highlight", frozen["highlight_policy_fingerprint"], highlight.fingerprint()),
        ("visibility", frozen["visibility_policy_fingerprint"], visibility.fingerprint()),
        ("artifact", frozen["artifact_policy_fingerprint"], artifact.fingerprint()),
    ):
        if recorded != recomputed:
            raise CorpusError(
                f"{name} policy fingerprint mismatch: file records {recorded!r}, "
                f"thresholds hash to {recomputed!r}; the frozen file was edited"
            )
    return highlight, visibility, artifact, frozen


def _subject_centre_and_radius(mask: np.ndarray) -> tuple[tuple[int, int], float]:
    ys, xs = np.nonzero(mask)
    area = float(len(ys))
    return (int(xs.mean()), int(ys.mean())), float(np.sqrt(area / np.pi))


def _analyse(image, highlight, visibility, artifact):
    return analyse_capture_artifacts(
        image, policy=artifact, highlight_policy=highlight, visibility_policy=visibility
    )


def controlled_glare(records, highlight, visibility, artifact, limit=None) -> dict:
    """Inject a highlight of known extent onto the subject and score the detector."""
    rows: dict[float, dict] = {s: {"n": 0, "flagged": 0, "no_mask": 0,
                                   "ious": [], "detected_fractions": [],
                                   "injected_fractions": []}
                               for s in GLARE_SEVERITIES}
    for record in (records[:limit] if limit else records):
        base = cv2.imread(str(local_path_for(record.image_id, record.file_extension)),
                          cv2.IMREAD_COLOR)
        if base is None:
            continue
        try:
            mask, foreground = isolate_foreground(base)
        except ForegroundError:
            continue
        if not foreground.valid:
            continue
        centre, radius = _subject_centre_and_radius(mask)
        injection_radius = max(4, int(radius * GLARE_RADIUS_FRACTION))

        for severity in GLARE_SEVERITIES:
            if severity == 0.0:
                image, region = base, np.zeros(mask.shape, np.uint8)
            else:
                image, region = add_glare_at(base, severity, centre, injection_radius,
                                             falloff=0.35)
            try:
                working_mask, working_fg = isolate_foreground(image)
            except ForegroundError:
                rows[severity]["no_mask"] += 1
                continue
            if not working_fg.valid:
                rows[severity]["no_mask"] += 1
                continue
            try:
                evidence = measure_local_highlights(image, working_mask, highlight)
                detected = highlight_mask(image, working_mask, highlight) > 0
            except ImageValidationError:
                rows[severity]["no_mask"] += 1
                continue

            injected = (region > 0) & (working_mask > 0)
            roi = float(np.count_nonzero(working_mask))
            union = float(np.count_nonzero(detected | injected))
            rows[severity]["n"] += 1
            rows[severity]["flagged"] += int(evidence.glare_flag)
            rows[severity]["detected_fractions"].append(
                float(np.count_nonzero(detected)) / roi if roi else 0.0)
            rows[severity]["injected_fractions"].append(
                float(np.count_nonzero(injected)) / roi if roi else 0.0)
            if severity > 0.0:
                rows[severity]["ious"].append(
                    float(np.count_nonzero(detected & injected)) / union if union else 0.0)

    summary = {}
    for severity, row in rows.items():
        summary[f"severity={severity}"] = {
            "assessable": row["n"],
            "no_usable_mask": row["no_mask"],
            "flag_rate": row["flagged"] / row["n"] if row["n"] else None,
            "median_detected_roi_fraction": (
                float(np.median(row["detected_fractions"])) if row["detected_fractions"] else None),
            "median_injected_roi_fraction": (
                float(np.median(row["injected_fractions"])) if row["injected_fractions"] else None),
            "median_iou_with_injected_region": (
                float(np.median(row["ious"])) if row["ious"] else None),
        }
    zero = summary.get("severity=0.0", {})
    severe = summary.get("severity=0.9", {})
    return {
        "injection": {
            "shape": "blurred disc centred on the subject",
            "radius_fraction_of_subject_radius": GLARE_RADIUS_FRACTION,
            "falloff": 0.35,
        },
        "by_severity": summary,
        "false_positive_rate_on_undegraded": zero.get("flag_rate"),
        "detection_rate_at_severity_0_9": severe.get("flag_rate"),
        "claim_boundary": (
            "a smooth additive highlight injected into a real photograph; not "
            "equivalent to a real specular reflection, which has sharper structure "
            "and interacts with surface microgeometry"
        ),
    }


def controlled_visibility(records, highlight, visibility, artifact, limit=None) -> dict:
    """Cover a known fraction of the subject and record what the system does.

    Reports the segmentation-failure path separately from the visibility flag.
    Those are different outcomes with the same practical effect - both escalate -
    and conflating them would credit the visibility detector with refusals the
    foreground guards made.
    """
    rows: dict[float, Counter] = {c: Counter() for c in OCCLUSION_COVERAGES}
    for record in (records[:limit] if limit else records):
        base = cv2.imread(str(local_path_for(record.image_id, record.file_extension)),
                          cv2.IMREAD_COLOR)
        if base is None:
            continue
        try:
            mask, foreground = isolate_foreground(base)
        except ForegroundError:
            continue
        if not foreground.valid:
            continue
        centre, _ = _subject_centre_and_radius(mask)
        subject_area = float(np.count_nonzero(mask))

        for coverage in OCCLUSION_COVERAGES:
            if coverage == 0.0:
                image = base
            else:
                side = max(4, int(np.sqrt(subject_area * coverage)))
                image, _ = occlude_at(base, centre, side, side)
            analysis = _analyse(image, highlight, visibility, artifact)
            reason = analysis.decision.reason_code.value
            rows[coverage][reason] += 1
            rows[coverage]["_total"] += 1
            if analysis.decision.blocking:
                rows[coverage]["_blocked"] += 1

    summary = {}
    for coverage, counter in rows.items():
        total = counter.pop("_total", 0)
        blocked = counter.pop("_blocked", 0)
        summary[f"coverage={coverage}"] = {
            "n": total,
            "blocked": blocked,
            "block_rate": blocked / total if total else None,
            "by_reason_code": dict(sorted(counter.items())),
        }
    return {
        "injection": {"shape": "opaque black square centred on the subject",
                      "coverage_is_of": "subject mask area"},
        "by_coverage": summary,
        "claim_boundary": (
            "an opaque geometric patch, not a hand, leaf or label; real occluders "
            "are textured and coloured and are not represented here"
        ),
    }


def controlled_contrast(records, highlight, visibility, artifact, limit=None) -> dict:
    rows: dict[float, Counter] = {f: Counter() for f in CONTRAST_FACTORS}
    contrasts: dict[float, list] = {f: [] for f in CONTRAST_FACTORS}
    for record in (records[:limit] if limit else records):
        base = cv2.imread(str(local_path_for(record.image_id, record.file_extension)),
                          cv2.IMREAD_COLOR)
        if base is None:
            continue
        for factor in CONTRAST_FACTORS:
            image = base if factor == 1.0 else apply_degradation(
                base, DegradationSpec("reduce_contrast", factor, 20260918))[0]
            analysis = _analyse(image, highlight, visibility, artifact)
            rows[factor][analysis.decision.reason_code.value] += 1
            rows[factor]["_total"] += 1
            if analysis.decision.blocking:
                rows[factor]["_blocked"] += 1
            if analysis.roi_contrast_score is not None:
                contrasts[factor].append(analysis.roi_contrast_score)

    summary = {}
    for factor, counter in rows.items():
        total = counter.pop("_total", 0)
        blocked = counter.pop("_blocked", 0)
        summary[f"factor={factor}"] = {
            "n": total, "blocked": blocked,
            "block_rate": blocked / total if total else None,
            "median_roi_contrast": (
                float(np.median(contrasts[factor])) if contrasts[factor] else None),
            "by_reason_code": dict(sorted(counter.items())),
        }
    return {"by_factor": summary}


def validation_evaluation(highlight, visibility, artifact) -> dict:
    """One pass over the independent pool. Undegraded only, plus injections."""
    records = load_validation_set()
    undegraded: Counter = Counter()
    flags: Counter = Counter()
    segmentation_valid = 0
    injected_glare_flagged = injected_glare_n = 0

    for record in records:
        image = cv2.imread(str(validation_image_path(record)), cv2.IMREAD_COLOR)
        if image is None:
            continue
        analysis = _analyse(image, highlight, visibility, artifact)
        undegraded[analysis.decision.reason_code.value] += 1
        for flag in analysis.decision.flags:
            flags[flag] += 1
        if analysis.foreground is not None and analysis.foreground.valid:
            segmentation_valid += 1
            try:
                mask, _ = isolate_foreground(image)
                centre, radius = _subject_centre_and_radius(mask)
                glared, _ = add_glare_at(
                    image, 0.9, centre, max(4, int(radius * GLARE_RADIUS_FRACTION)),
                    falloff=0.35,
                )
                glared_analysis = _analyse(glared, highlight, visibility, artifact)
                injected_glare_n += 1
                injected_glare_flagged += int(
                    CaptureArtifactFlag.GLARE_RISK.value in glared_analysis.decision.flags
                )
            except (ForegroundError, ImageValidationError):
                pass

    total = sum(undegraded.values())
    clean = [r for r in records if r.track == "CLEAN_BASE"]
    return {
        "population": {
            "images": total,
            "clean_base": len(clean),
            "natural_scene": len(records) - len(clean),
            "source_groups": len({r.source_group_id for r in records}),
            "segmentation_valid": segmentation_valid,
        },
        "undegraded_outcomes": dict(sorted(undegraded.items())),
        "undegraded_flag_counts": dict(sorted(flags.items())),
        "glare_false_positive_rate": (
            flags.get(CaptureArtifactFlag.GLARE_RISK.value, 0) / segmentation_valid
            if segmentation_valid else None),
        "severe_contrast_false_positive_rate": (
            flags.get(CaptureArtifactFlag.SEVERE_LOW_CONTRAST.value, 0) / segmentation_valid
            if segmentation_valid else None),
        "visibility_false_positive_rate": (
            flags.get(CaptureArtifactFlag.SUBJECT_VISIBILITY_INSUFFICIENT.value, 0)
            / segmentation_valid if segmentation_valid else None),
        "injected_glare": {
            "n": injected_glare_n,
            "flagged": injected_glare_flagged,
            "detection_rate": (
                injected_glare_flagged / injected_glare_n if injected_glare_n else None),
            "note": "confirmatory detection on unseen source groups, injected highlight",
        },
    }


def benchmark(records, highlight, visibility, artifact, limit: int = 12) -> dict:
    """Per-stage CPU latency. Local machine only; says nothing about AWS."""
    stages: dict[str, list] = {"foreground": [], "glare": [], "visibility": [],
                               "contrast": [], "combined": []}
    resolutions = []
    for record in records[:limit]:
        image = cv2.imread(str(local_path_for(record.image_id, record.file_extension)),
                           cv2.IMREAD_COLOR)
        if image is None:
            continue
        started = time.perf_counter()
        try:
            mask, foreground = isolate_foreground(image)
        except ForegroundError:
            continue
        stages["foreground"].append((time.perf_counter() - started) * 1000.0)
        if not foreground.valid:
            continue
        resolutions.append((image.shape[1], image.shape[0]))

        started = time.perf_counter()
        try:
            measure_local_highlights(image, mask, highlight)
        except ImageValidationError:
            continue
        stages["glare"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        from competition.vision.visibility import measure_visibility
        measure_visibility(mask, visibility)
        stages["visibility"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        measure_illumination(image, DEFAULT_POLICY, mask)
        stages["contrast"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        _analyse(image, highlight, visibility, artifact)
        stages["combined"].append((time.perf_counter() - started) * 1000.0)

    def summarise(values: list) -> dict:
        if not values:
            return {"n": 0}
        array = np.asarray(values)
        return {"n": int(array.size), "median_ms": float(np.median(array)),
                "p95_ms": float(np.percentile(array, 95))}

    return {
        "environment": "local CPU; not an AWS measurement and not a deployment figure",
        "opencv_version": cv2.__version__,
        "resolutions": {
            "n": len(resolutions),
            "median_width": int(np.median([w for w, _ in resolutions])) if resolutions else None,
            "median_height": int(np.median([h for _, h in resolutions])) if resolutions else None,
        },
        "stages": {name: summarise(values) for name, values in stages.items()},
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.phase2d_artifacts",
        description="Phase 2d controlled and validation evaluation.",
    )
    parser.add_argument("stage", choices=["controlled", "validation", "latency", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    highlight, visibility, artifact, frozen = load_frozen_policies()
    records = [r for r in calibration_view().records if r.track == "CLEAN_BASE"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluated_on": date.today().isoformat(),
        "highlight_policy_fingerprint": highlight.fingerprint(),
        "visibility_policy_fingerprint": visibility.fingerprint(),
        "artifact_policy_fingerprint": artifact.fingerprint(),
        "detectors_usable": frozen["detectors_usable"],
        "code_fingerprint": _code_fingerprint(),
    }
    output = {}

    if args.stage in ("controlled", "all"):
        document = {
            **provenance,
            "population": "calibration groups only",
            "glare": controlled_glare(records, highlight, visibility, artifact, args.limit),
            "visibility": controlled_visibility(records, highlight, visibility, artifact, args.limit),
            "contrast": controlled_contrast(records, highlight, visibility, artifact, args.limit),
        }
        CONTROLLED_RESULTS.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8")
        output["controlled"] = {
            "glare_fp": document["glare"]["false_positive_rate_on_undegraded"],
            "glare_detection_0_9": document["glare"]["detection_rate_at_severity_0_9"],
        }

    if args.stage in ("validation", "all"):
        document = {**provenance,
                    "status": "CONFIRMATORY",
                    "population": "Phase 2d independent validation pool",
                    **validation_evaluation(highlight, visibility, artifact)}
        VALIDATION_RESULTS.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8")
        output["validation"] = {
            k: document[k] for k in
            ("glare_false_positive_rate", "visibility_false_positive_rate",
             "severe_contrast_false_positive_rate")
        }
        output["validation"]["injected_glare"] = document["injected_glare"]

    if args.stage in ("latency", "all"):
        document = {**provenance, **benchmark(records, highlight, visibility, artifact)}
        LATENCY_RESULTS.write_text(
            json.dumps(document, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8")
        output["latency"] = document["stages"]

    print(json.dumps(output, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
