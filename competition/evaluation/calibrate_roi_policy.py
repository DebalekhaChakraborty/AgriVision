"""Calibrate the foreground-restricted capture-quality policy.

Reads calibration source groups only. `CorpusView` enforces that structurally:
this module never receives held-out records, so it cannot inspect them by
accident or by a later careless edit.

The decision rule is preregistered
----------------------------------
Every acceptance criterion in this file is a named constant with a stated
justification, fixed before the corpus was measured. That matters more than
usual here, because the interesting outcome is a *negative* one: Phase 1 and
Phase 2b both suggested variance-of-Laplacian may not support a single global
threshold, and a rule invented after seeing the numbers could be shaped to
rescue it. The rule below can fail, and if every candidate metric fails it, the
policy is locked with no focus gate at all rather than with a number chosen to
make the pipeline look finished.

`MAX_EXPOSURE_SENSITIVITY` encodes the Phase 1 finding directly. A focus metric
whose value on *undegraded* images moves substantially when the exposure changes
cannot carry a global threshold, because the same subject would cross it merely
by being photographed in a darker kitchen. That is a statement about the metric,
not about this corpus, which is why it is allowed to disqualify a metric even if
that metric happens to separate the classes well here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.data.licensed_sources import CorpusError, CorpusTrack, LicensedImageRecord
from competition.evaluation.build_licensed_corpus import (
    CORPUS_MANIFEST,
    RAW_DIR,
    load_corpus,
    local_path_for,
)
from competition.evaluation.degrade_licensed_corpus import (
    BASE_EXPECTED,
    DERIVED_MANIFEST,
    ExpectedVerdict,
    derived_digest,
    load_derived,
    regenerate,
)
from competition.evaluation.split_licensed_corpus import (
    SPLIT_PATH,
    calibration_view,
    load_split,
)
from competition.vision.blur_metrics import measure_focus
from competition.vision.evidence import ImageValidationError
from competition.vision.foreground import ForegroundError, isolate_foreground
from competition.vision.quality import measure_illumination
from competition.vision.roi_policy import (
    FOCUS_METRICS,
    STATUS_CALIBRATED,
    STATUS_PARTIALLY_CALIBRATED,
    RoiQualityPolicy,
)
from competition.vision.config import DEFAULT_POLICY

RESULTS_DIR = Path("competition/evaluation/results")
CALIBRATION_REPORT = RESULTS_DIR / "phase2c_calibration.json"
LOCKED_POLICY_PATH = RESULTS_DIR / "phase2c_locked_policy.json"

CALIBRATION_VERSION = "phase2c-roi-calibration-1.0.0"

# --- preregistered acceptance criteria ---------------------------------------

# Probability that a randomly chosen UNUSABLE sample scores below a randomly
# chosen ACCEPTABLE one. 0.95 is demanding on purpose: a metric that only sorts
# the classes four times in five cannot carry a single global threshold.
MIN_SEPARATION_AUC = 0.95

# A false accept passes an unusable photograph to condition inference, and the
# system then answers confidently about an image it cannot actually read. A
# false block costs one retake. The asymmetry is why the rule constrains false
# accepts first and minimises false blocks second, rather than maximising
# accuracy - which would trade the two off at par.
MAX_FALSE_ACCEPT_RATE = 0.10
MAX_FALSE_BLOCK_RATE = 0.10

# Largest tolerated ratio between the highest and lowest median value a metric
# takes on the *same* undegraded images across the exposure ladder. Encodes the
# Phase 1 exposure-coupling finding as a disqualifying property.
MAX_EXPOSURE_SENSITIVITY = 1.5

# Same idea for subject scale, the Phase 2b confound. Looser, because scale
# genuinely removes detail: some movement is real signal, not measurement drift.
MAX_SCALE_SENSITIVITY = 3.0


@dataclass(frozen=True)
class ThresholdChoice:
    """One threshold candidate and the cost it carries."""

    threshold: float | None
    direction: str  # "floor": block below; "ceiling": block above
    false_accept_rate: float
    false_block_rate: float
    false_accepts: int
    false_blocks: int
    acceptable_n: int
    unusable_n: int
    separation_auc: float
    admissible: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold, "direction": self.direction,
            "false_accept_rate": self.false_accept_rate,
            "false_block_rate": self.false_block_rate,
            "false_accepts": self.false_accepts, "false_blocks": self.false_blocks,
            "acceptable_n": self.acceptable_n, "unusable_n": self.unusable_n,
            "separation_auc": self.separation_auc,
            "admissible": self.admissible, "reasons": list(self.reasons),
        }


def separation_auc(acceptable: list[float], unusable: list[float], direction: str) -> float:
    """Rank statistic: P(unusable is on the blocking side of acceptable).

    Ties count a half, which is the standard convention and matters here because
    saturated metrics produce many identical values.
    """
    if not acceptable or not unusable:
        return 0.0
    wins = 0.0
    for bad in unusable:
        for good in acceptable:
            if direction == "floor":
                wins += 1.0 if bad < good else 0.5 if bad == good else 0.0
            else:
                wins += 1.0 if bad > good else 0.5 if bad == good else 0.0
    return wins / (len(acceptable) * len(unusable))


def calibrate_threshold(
    acceptable: list[float], unusable: list[float], direction: str
) -> tuple[ThresholdChoice, list[dict]]:
    """Sweep every candidate threshold and apply the preregistered rule.

    Candidates are midpoints between adjacent observed values, so the sweep
    covers every decision boundary the data can distinguish without inventing
    precision the samples do not support.
    """
    auc = separation_auc(acceptable, unusable, direction)
    values = sorted(set(acceptable) | set(unusable))
    if len(values) < 2:
        return ThresholdChoice(None, direction, 1.0, 0.0, len(unusable), 0,
                               len(acceptable), len(unusable), auc, False,
                               ["fewer than two distinct values to threshold"]), []

    candidates = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    sweep: list[dict] = []
    best: ThresholdChoice | None = None

    for threshold in candidates:
        if direction == "floor":
            false_accepts = sum(1 for v in unusable if v >= threshold)
            false_blocks = sum(1 for v in acceptable if v < threshold)
        else:
            false_accepts = sum(1 for v in unusable if v <= threshold)
            false_blocks = sum(1 for v in acceptable if v > threshold)

        fa_rate = false_accepts / len(unusable) if unusable else 0.0
        fb_rate = false_blocks / len(acceptable) if acceptable else 0.0
        sweep.append({"threshold": threshold, "false_accept_rate": fa_rate,
                      "false_block_rate": fb_rate, "false_accepts": false_accepts,
                      "false_blocks": false_blocks})

        if fa_rate > MAX_FALSE_ACCEPT_RATE:
            continue
        if best is None or fb_rate < best.false_block_rate:
            best = ThresholdChoice(
                threshold, direction, fa_rate, fb_rate, false_accepts, false_blocks,
                len(acceptable), len(unusable), auc, True, [],
            )

    if best is None:
        return ThresholdChoice(
            None, direction, 1.0, 0.0, len(unusable), 0, len(acceptable),
            len(unusable), auc, False,
            [f"no threshold holds the false-accept rate at or below {MAX_FALSE_ACCEPT_RATE}"],
        ), sweep

    reasons: list[str] = []
    if auc < MIN_SEPARATION_AUC:
        reasons.append(f"separation AUC {auc:.3f} below {MIN_SEPARATION_AUC}")
    if best.false_block_rate > MAX_FALSE_BLOCK_RATE:
        reasons.append(
            f"best achievable false-block rate {best.false_block_rate:.3f} "
            f"exceeds {MAX_FALSE_BLOCK_RATE}"
        )
    if reasons:
        best = ThresholdChoice(**{**best.to_dict(), "admissible": False,
                                  "reasons": reasons})
    return best, sweep


def _measure_one(image: np.ndarray) -> dict:
    """Foreground isolation plus ROI and whole-image metrics for one image."""
    result: dict = {"roi_available": False, "segmentation_valid": False}
    try:
        mask, foreground = isolate_foreground(image)
    except ForegroundError as error:
        result["segmentation_error"] = str(error)
        return result

    result.update({
        "segmentation_valid": foreground.valid,
        "invalid_reasons": foreground.invalid_reasons,
        "foreground_fraction": foreground.foreground_fraction,
        "largest_component_fraction": foreground.largest_component_fraction,
        "border_contact_fraction": foreground.border_contact_fraction,
        "component_count": foreground.component_count,
        "solidity": foreground.solidity,
        "method": foreground.method,
    })

    whole_focus = measure_focus(image)
    whole_illumination = measure_illumination(image, DEFAULT_POLICY)
    result["whole_image"] = {
        **whole_focus.to_dict(),
        "mean_luminance": whole_illumination.mean_luminance,
        "contrast_score": whole_illumination.contrast_score,
        "shadow_clip_fraction": whole_illumination.shadow_clip_fraction,
        "highlight_clip_fraction": whole_illumination.highlight_clip_fraction,
    }

    if foreground.valid:
        try:
            roi_focus = measure_focus(image, mask)
            roi_illumination = measure_illumination(image, DEFAULT_POLICY, mask)
        except (ImageValidationError, ForegroundError) as error:
            result["roi_error"] = str(error)
            return result
        result["roi_available"] = True
        result["roi"] = {
            **roi_focus.to_dict(),
            "mean_luminance": roi_illumination.mean_luminance,
            "contrast_score": roi_illumination.contrast_score,
            "shadow_clip_fraction": roi_illumination.shadow_clip_fraction,
            "highlight_clip_fraction": roi_illumination.highlight_clip_fraction,
        }
    return result


def measure_population(
    records: list[LicensedImageRecord],
    derived: list[dict],
    raw_root: Path = RAW_DIR,
) -> list[dict]:
    """Measure every base and derived sample belonging to the given records."""
    wanted = {record.image_id: record for record in records}
    samples: list[dict] = []

    base_images: dict[str, np.ndarray] = {}
    for record in records:
        path = local_path_for(record.image_id, record.file_extension, raw_root)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CorpusError(f"missing local bytes for {record.image_id}")
        base_images[record.image_id] = image
        samples.append({
            "sample_id": record.image_id,
            "base_image_id": record.image_id,
            "kind": "base",
            "transformation": "none",
            "level": 1.0,
            "fruit_type": record.fruit_type,
            "track": record.track,
            "split": record.split,
            "source_group_id": record.source_group_id,
            "expected_verdict": BASE_EXPECTED[record.track].value,
            "width": record.width,
            "height": record.height,
            **_measure_one(image),
        })

    for entry in derived:
        if entry["base_image_id"] not in wanted:
            continue
        image = regenerate(
            base_images[entry["base_image_id"]],
            entry["transformation"],
            float(entry["parameters"]["level"]),
            int(entry["seed"]),
        )
        # The manifest pins what was generated when the ladder was built; a
        # mismatch means the transform or the base changed underneath us, and
        # measuring it anyway would silently compare different populations.
        if derived_digest(image) != entry["derived_sha256"]:
            raise CorpusError(
                f"regenerated {entry['derived_id']} does not match its recorded hash"
            )
        height, width = image.shape[:2]
        samples.append({
            "sample_id": entry["derived_id"],
            "base_image_id": entry["base_image_id"],
            "kind": "derived",
            "transformation": entry["transformation"],
            "level": entry["parameters"]["level"],
            "fruit_type": entry["fruit_type"],
            "track": entry["track"],
            "split": entry["split"],
            "source_group_id": entry["source_group_id"],
            "expected_verdict": entry["expected_verdict"],
            "width": int(width),
            "height": int(height),
            **_measure_one(image),
        })

    return samples


# --- analyses ----------------------------------------------------------------


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def sensitivity_analysis(samples: list[dict], metric: str, family: str) -> dict:
    """How far a metric moves across one degradation family on the same bases.

    Compares medians rung by rung against the undegraded base, restricted to
    samples whose base is itself undegraded and measurable, so the only thing
    varying is the transformation level.
    """
    by_level: dict[float, list[float]] = defaultdict(list)
    base_values = [
        s["roi"][metric] for s in samples
        if s["kind"] == "base" and s.get("roi_available")
    ]
    if base_values:
        by_level[1.0] = base_values
    for sample in samples:
        if sample["kind"] != "derived" or sample["transformation"] != family:
            continue
        if not sample.get("roi_available"):
            continue
        by_level[float(sample["level"])].append(sample["roi"][metric])

    medians = {level: _median(values) for level, values in sorted(by_level.items())}
    finite = [value for value in medians.values() if np.isfinite(value) and value > 0]
    ratio = (max(finite) / min(finite)) if len(finite) >= 2 else float("nan")
    return {
        "family": family,
        "medians_by_level": {str(k): v for k, v in medians.items()},
        "max_over_min_ratio": ratio,
        "n_by_level": {str(k): len(v) for k, v in sorted(by_level.items())},
    }


def distribution_breakdown(samples: list[dict], metric: str) -> dict:
    """Metric distribution sliced the ways Phase 1 and 2b said would matter."""
    def summarise(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
        }

    undegraded = [s for s in samples if s["kind"] == "base" and s.get("roi_available")]
    by_fruit: dict[str, list[float]] = defaultdict(list)
    by_resolution: dict[str, list[float]] = defaultdict(list)
    by_track: dict[str, list[float]] = defaultdict(list)
    for sample in undegraded:
        value = sample["roi"][metric]
        by_fruit[sample["fruit_type"]].append(value)
        by_track[sample["track"]].append(value)
        smallest = min(sample["width"], sample["height"])
        bucket = ("<1000" if smallest < 1000 else
                  "1000-1999" if smallest < 2000 else
                  "2000-2999" if smallest < 3000 else ">=3000")
        by_resolution[bucket].append(value)

    by_blur: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.get("roi_available") and sample["transformation"] == "gaussian_blur":
            by_blur[f"sigma={sample['level']}"].append(sample["roi"][metric])
    if undegraded:
        by_blur["sigma=0.0"] = [s["roi"][metric] for s in undegraded]

    return {
        "by_fruit": {k: summarise(v) for k, v in sorted(by_fruit.items())},
        "by_track": {k: summarise(v) for k, v in sorted(by_track.items())},
        "by_resolution": {k: summarise(v) for k, v in sorted(by_resolution.items())},
        "by_blur_level": {k: summarise(v) for k, v in sorted(by_blur.items())},
        "by_subject_scale": {
            f"scale={level}": summarise([
                s["roi"][metric] for s in samples
                if s.get("roi_available") and s["transformation"] == "shrink_subject"
                and s["level"] == level
            ])
            for level in sorted({
                s["level"] for s in samples if s["transformation"] == "shrink_subject"
            })
        },
        "by_exposure_level": {
            f"{family}={level}": summarise([
                s["roi"][metric] for s in samples
                if s.get("roi_available") and s["transformation"] == family
                and s["level"] == level
            ])
            for family in ("underexpose", "overexpose")
            for level in sorted({
                s["level"] for s in samples if s["transformation"] == family
            })
        },
    }


def _scored(samples: list[dict], metric_path: tuple[str, str]) -> tuple[list[float], list[float]]:
    """Values for the two scored classes only. BORDERLINE and UNLABELLED excluded."""
    scope, metric = metric_path
    acceptable, unusable = [], []
    for sample in samples:
        if not sample.get("roi_available"):
            continue
        value = sample[scope][metric]
        if sample["expected_verdict"] == ExpectedVerdict.ACCEPTABLE.value:
            acceptable.append(value)
        elif sample["expected_verdict"] == ExpectedVerdict.UNUSABLE.value:
            unusable.append(value)
    return acceptable, unusable


def evaluate_focus_metrics(samples: list[dict]) -> dict:
    """Test every candidate focus metric against the preregistered rule."""
    blur_samples = [
        s for s in samples
        if s["kind"] == "base" or s["transformation"] == "gaussian_blur"
    ]
    findings: dict[str, dict] = {}

    for metric in FOCUS_METRICS:
        acceptable, unusable = _scored(blur_samples, ("roi", metric))
        choice, sweep = calibrate_threshold(acceptable, unusable, "floor")
        exposure = [
            sensitivity_analysis(samples, metric, family)
            for family in ("underexpose", "overexpose")
        ]
        scale = sensitivity_analysis(samples, metric, "shrink_subject")

        disqualifiers: list[str] = list(choice.reasons)
        for analysis in exposure:
            ratio = analysis["max_over_min_ratio"]
            if np.isfinite(ratio) and ratio > MAX_EXPOSURE_SENSITIVITY:
                disqualifiers.append(
                    f"{analysis['family']} moves the metric by {ratio:.2f}x, "
                    f"above the {MAX_EXPOSURE_SENSITIVITY}x limit"
                )
        scale_ratio = scale["max_over_min_ratio"]
        if np.isfinite(scale_ratio) and scale_ratio > MAX_SCALE_SENSITIVITY:
            disqualifiers.append(
                f"subject scale moves the metric by {scale_ratio:.2f}x, "
                f"above the {MAX_SCALE_SENSITIVITY}x limit"
            )

        findings[metric] = {
            "threshold": choice.to_dict(),
            "exposure_sensitivity": exposure,
            "scale_sensitivity": scale,
            "distributions": distribution_breakdown(samples, metric),
            "usable_as_global_threshold": not disqualifiers,
            "disqualifiers": disqualifiers,
            "sweep_points": len(sweep),
        }
    return findings


def evaluate_exposure_thresholds(samples: list[dict]) -> dict:
    """Calibrate each illumination threshold on its own targeted ladder."""
    def population(families: tuple[str, ...], metric: str) -> tuple[list[float], list[float]]:
        subset = [
            s for s in samples
            if s["kind"] == "base" or s["transformation"] in families
        ]
        return _scored(subset, ("roi", metric))

    plan = {
        "underexposed_mean_luminance": (("underexpose",), "mean_luminance", "floor"),
        "overexposed_mean_luminance": (("overexpose",), "mean_luminance", "ceiling"),
        "low_contrast_limit": (("reduce_contrast",), "contrast_score", "floor"),
        "shadow_clip_fraction_limit": (("underexpose",), "shadow_clip_fraction", "ceiling"),
        "highlight_clip_fraction_limit": (("overexpose", "glare"), "highlight_clip_fraction", "ceiling"),
    }
    findings = {}
    for name, (families, metric, direction) in plan.items():
        acceptable, unusable = population(families, metric)
        choice, _ = calibrate_threshold(acceptable, unusable, direction)
        findings[name] = {"metric": metric, "families": list(families),
                          **choice.to_dict()}
    return findings


def _code_fingerprint() -> str:
    """Identify the working tree the calibration ran against."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    return f"{head}{'+dirty' if dirty else ''}"


def _manifest_fingerprint(path: Path) -> str:
    import hashlib

    if not path.is_file():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --- applying a candidate policy back to the measured population -------------


def _namespace(values: dict):
    from types import SimpleNamespace

    return SimpleNamespace(**values)


def gate_sample(sample: dict, policy: RoiQualityPolicy, scope: str = "roi") -> dict:
    """Run one measured sample through a policy and report the outcome."""
    if not sample.get("roi_available") and scope == "roi":
        return {"assessable": False, "flags": [], "blocked": None}
    metrics = sample[scope]
    flags = __import__(
        "competition.vision.roi_policy", fromlist=["apply_roi_policy"]
    ).apply_roi_policy(
        _namespace({"width": sample["width"], "height": sample["height"]}),
        _namespace(metrics),
        _namespace(metrics),
        policy,
    )
    return {"assessable": True, "flags": flags, "blocked": bool(flags)}


def confusion_matrix(samples: list[dict], policy: RoiQualityPolicy, scope: str = "roi") -> dict:
    """Confusion against the preregistered rubric. BORDERLINE reported, not scored."""
    counts = {
        "true_block": 0, "false_accept": 0, "true_accept": 0, "false_block": 0,
        "unassessable": 0,
    }
    borderline = {"blocked": 0, "accepted": 0}
    unlabelled = {"blocked": 0, "accepted": 0}

    for sample in samples:
        outcome = gate_sample(sample, policy, scope)
        expected = sample["expected_verdict"]
        if not outcome["assessable"]:
            counts["unassessable"] += 1
            continue
        blocked = outcome["blocked"]
        if expected == ExpectedVerdict.UNUSABLE.value:
            counts["true_block" if blocked else "false_accept"] += 1
        elif expected == ExpectedVerdict.ACCEPTABLE.value:
            counts["false_block" if blocked else "true_accept"] += 1
        elif expected == ExpectedVerdict.BORDERLINE.value:
            borderline["blocked" if blocked else "accepted"] += 1
        else:
            unlabelled["blocked" if blocked else "accepted"] += 1

    scored_unusable = counts["true_block"] + counts["false_accept"]
    scored_acceptable = counts["true_accept"] + counts["false_block"]
    return {
        "scope": scope,
        **counts,
        "false_accept_rate": (counts["false_accept"] / scored_unusable) if scored_unusable else None,
        "false_block_rate": (counts["false_block"] / scored_acceptable) if scored_acceptable else None,
        "scored_unusable": scored_unusable,
        "scored_acceptable": scored_acceptable,
        "borderline": borderline,
        "unlabelled": unlabelled,
    }


def segmentation_summary(samples: list[dict]) -> dict:
    """Foreground behaviour, from geometry alone - no ground-truth masks exist."""
    base = [s for s in samples if s["kind"] == "base"]
    by_track: dict[str, dict] = {}
    for track in (CorpusTrack.CLEAN_BASE.value, CorpusTrack.NATURAL_SCENE.value):
        subset = [s for s in base if s["track"] == track]
        if not subset:
            continue
        valid = [s for s in subset if s.get("segmentation_valid")]
        reasons: dict[str, int] = defaultdict(int)
        for sample in subset:
            for reason in sample.get("invalid_reasons", []):
                reasons[reason] += 1
        by_track[track] = {
            "n": len(subset),
            "valid": len(valid),
            "valid_rate": len(valid) / len(subset),
            "median_foreground_fraction": _median(
                [s["foreground_fraction"] for s in subset if "foreground_fraction" in s]
            ),
            "median_border_contact": _median(
                [s["border_contact_fraction"] for s in subset if "border_contact_fraction" in s]
            ),
            "median_component_count": _median(
                [float(s["component_count"]) for s in subset if "component_count" in s]
            ),
            "invalid_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        }
    return {
        "note": (
            "geometric validity only; no annotated foreground masks exist, so no "
            "IoU or Dice figure is reported and none should be inferred"
        ),
        "by_track": by_track,
    }


def clipping_scope_comparison(samples: list[dict]) -> dict:
    """Whole-image versus ROI clipping - the measurement Phase 2b turned on."""
    base = [s for s in samples if s["kind"] == "base" and s.get("roi_available")]
    if not base:
        return {}
    return {
        "n": len(base),
        "whole_image_median_highlight_clip": _median(
            [s["whole_image"]["highlight_clip_fraction"] for s in base]),
        "roi_median_highlight_clip": _median(
            [s["roi"]["highlight_clip_fraction"] for s in base]),
        "whole_image_median_shadow_clip": _median(
            [s["whole_image"]["shadow_clip_fraction"] for s in base]),
        "roi_median_shadow_clip": _median(
            [s["roi"]["shadow_clip_fraction"] for s in base]),
        "whole_image_median_laplacian": _median(
            [s["whole_image"]["laplacian_variance"] for s in base]),
        "roi_median_laplacian": _median(
            [s["roi"]["laplacian_variance"] for s in base]),
    }


def choose_focus_metric(findings: dict) -> tuple[str | None, float | None, list[str]]:
    """Pick the focus metric by the preregistered rule, or none at all.

    Tie-break is the separation AUC, then the false-block rate. Both were fixed
    in advance; neither is 'whichever makes the gate look best'.
    """
    admissible = [
        (name, finding) for name, finding in findings.items()
        if finding["usable_as_global_threshold"]
    ]
    if not admissible:
        return None, None, [
            f"{name}: {'; '.join(finding['disqualifiers'])}"
            for name, finding in findings.items()
        ]
    admissible.sort(
        key=lambda item: (
            -item[1]["threshold"]["separation_auc"],
            item[1]["threshold"]["false_block_rate"],
        )
    )
    name, finding = admissible[0]
    return name, finding["threshold"]["threshold"], []


def run_calibration() -> dict:
    view = calibration_view()
    if not len(view):
        raise CorpusError("the calibration split is empty")
    derived = [
        entry for entry in load_derived()
        if entry["split"] == "calibration"
    ]
    samples = measure_population(view.records, derived)

    focus_findings = evaluate_focus_metrics(samples)
    exposure_findings = evaluate_exposure_thresholds(samples)
    metric, floor, disqualifiers = choose_focus_metric(focus_findings)

    calibrated_fields: dict[str, float] = {}
    provisional_fields: dict[str, dict] = {}

    def threshold_or_default(name: str, default: float) -> float:
        """Take a calibrated value, or keep the provisional one and say so.

        A threshold that failed the preregistered rule is *not* quietly replaced
        by a fitted number. It keeps its Phase 1 value and is recorded as
        provisional, because a threshold nobody could justify is better left
        visibly unjustified than given a figure with a decimal point on it.
        """
        finding = exposure_findings.get(name, {})
        value = finding.get("threshold")
        if finding.get("admissible") and value is not None:
            calibrated_fields[name] = float(value)
            return float(value)
        provisional_fields[name] = {
            "retained_value": default,
            "source": "Phase 1 provisional whole-image value",
            "reasons": finding.get("reasons", ["no candidate threshold was admissible"]),
            "best_candidate": value,
            "separation_auc": finding.get("separation_auc"),
        }
        return default

    candidate = RoiQualityPolicy(
        focus_metric=metric or "laplacian_variance",
        focus_floor=floor,
        underexposed_mean_luminance=threshold_or_default("underexposed_mean_luminance", 0.25),
        overexposed_mean_luminance=threshold_or_default("overexposed_mean_luminance", 0.80),
        low_contrast_limit=threshold_or_default("low_contrast_limit", 0.20),
        shadow_clip_fraction_limit=threshold_or_default("shadow_clip_fraction_limit", 0.05),
        highlight_clip_fraction_limit=threshold_or_default("highlight_clip_fraction_limit", 0.05),
    )

    if metric is not None:
        calibrated_fields["focus_floor"] = float(floor)
        calibrated_fields["focus_metric"] = metric
    else:
        provisional_fields["focus_floor"] = {
            "retained_value": None,
            "source": "no focus gate: every candidate metric failed the rule",
            "reasons": disqualifiers,
        }

    status = STATUS_CALIBRATED if not provisional_fields else STATUS_PARTIALLY_CALIBRATED

    split_document = load_split()
    locked = candidate.locked({
        "calibrated_on": date.today().isoformat(),
        "calibration_version": CALIBRATION_VERSION,
        "corpus_manifest_fingerprint": _manifest_fingerprint(CORPUS_MANIFEST),
        "derived_manifest_fingerprint": _manifest_fingerprint(DERIVED_MANIFEST),
        "split_fingerprint": split_document["fingerprint"],
        "code_fingerprint": _code_fingerprint(),
        "calibration_image_count": len(view),
        "calibration_sample_count": len(samples),
        "focus_metric_selected": metric,
        "focus_metric_rejected": disqualifiers,
        "calibrated_thresholds": calibrated_fields,
        "retained_provisional_thresholds": provisional_fields,
        "criteria": {
            "min_separation_auc": MIN_SEPARATION_AUC,
            "max_false_accept_rate": MAX_FALSE_ACCEPT_RATE,
            "max_false_block_rate": MAX_FALSE_BLOCK_RATE,
            "max_exposure_sensitivity": MAX_EXPOSURE_SENSITIVITY,
            "max_scale_sensitivity": MAX_SCALE_SENSITIVITY,
        },
        "claim_boundary": (
            "calibrated on licensed real fruit photographs with controlled "
            "degradations; not calibrated on phone-camera deployment captures"
        ),
    }, status=status)

    report = {
        "calibration_version": CALIBRATION_VERSION,
        "opencv_version": cv2.__version__,
        "split": "calibration only; held-out groups were never opened",
        "population": {
            "base_images": len(view.records),
            "derived_samples": len(derived),
            "measured_samples": len(samples),
            "roi_available": sum(1 for s in samples if s.get("roi_available")),
        },
        "focus_metrics": focus_findings,
        "focus_selection": {
            "selected": metric,
            "floor": floor,
            "gates_on_focus": floor is not None,
            "rejections": disqualifiers,
        },
        "illumination_thresholds": exposure_findings,
        "segmentation": segmentation_summary(samples),
        "scope_comparison": clipping_scope_comparison(samples),
        "confusion_matrix_roi": confusion_matrix(samples, locked, "roi"),
        "confusion_matrix_whole_image": confusion_matrix(samples, locked, "whole_image"),
        "locked_policy": locked.to_dict(),
        "threshold_provenance": {
            "calibrated": calibrated_fields,
            "retained_provisional": provisional_fields,
            "status": status,
        },
    }
    return {"report": report, "policy": locked, "samples": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.calibrate_roi_policy",
        description="Calibrate the ROI capture-quality policy on calibration groups only.",
    )
    parser.parse_args(argv)

    outcome = run_calibration()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_REPORT.write_text(
        json.dumps(outcome["report"], indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    LOCKED_POLICY_PATH.write_text(
        json.dumps(outcome["policy"].to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "focus_metric": outcome["policy"].focus_metric,
        "focus_floor": outcome["policy"].focus_floor,
        "gates_on_focus": outcome["policy"].gates_on_focus,
        "lock_fingerprint": outcome["policy"].lock_fingerprint(),
        "confusion_roi": outcome["report"]["confusion_matrix_roi"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
