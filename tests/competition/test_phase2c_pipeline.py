"""Regression guards for the Phase 2c-B pipeline and the Phase 2b fix it forced."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from competition.data.licensed_sources import CorpusTrack, ProvenanceType
from competition.evaluation.calibrate_roi_policy import (
    MAX_EXPOSURE_SENSITIVITY,
    MAX_FALSE_ACCEPT_RATE,
    MAX_SCALE_SENSITIVITY,
    MIN_SEPARATION_AUC,
    calibrate_threshold,
    choose_focus_metric,
    separation_auc,
)
from competition.evaluation.degrade_licensed_corpus import (
    DEGRADATION_LADDER,
    ExpectedVerdict,
    derived_digest,
    regenerate,
)
from competition.vision.degradation import SUBJECT_SCALES, shrink_subject
from competition.vision.foreground import (
    CLEANUP_KERNEL_FRACTION,
    DEFAULT_GUARDS,
    cleanup_kernel_size,
    evaluate_mask,
)


# --- the Phase 2b resolution bug this phase found -----------------------------


def test_cleanup_kernel_is_unchanged_at_the_phase2b_fixture_size():
    """7 px at 256 px: Phase 2b's own constant, expressed relatively."""
    assert cleanup_kernel_size((256, 256)) == 7
    assert CLEANUP_KERNEL_FRACTION == 7.0 / 256.0


def test_cleanup_kernel_scales_with_the_image():
    small = cleanup_kernel_size((256, 256))
    large = cleanup_kernel_size((1440, 1920))
    assert large > small
    assert large / 1440 == pytest.approx(small / 256, rel=0.1)


def test_cleanup_kernel_is_always_odd_and_at_least_three():
    for shape in [(8, 8), (64, 64), (256, 256), (1080, 1920), (3000, 4000)]:
        size = cleanup_kernel_size(shape)
        assert size >= 3 and size % 2 == 1, shape


def test_speckle_does_not_count_as_subject_fragmentation():
    """The defect that rejected 71% of full-resolution samples as fragmented."""
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.circle(mask, (500, 500), 260, 255, -1)          # the subject
    rng = np.random.default_rng(0)
    for _ in range(300):                                 # thresholding speckle
        y, x = int(rng.integers(0, 1000)), int(rng.integers(0, 1000))
        mask[y:y + 2, x:x + 2] = 255

    valid, reasons, metrics = evaluate_mask(
        np.zeros((1000, 1000, 3), np.uint8), mask, "test", DEFAULT_GUARDS
    )
    assert "MASK_FRAGMENTED" not in reasons
    assert metrics["component_count"] <= DEFAULT_GUARDS.max_component_count
    assert metrics["speckle_component_count"] > 0


def test_a_genuinely_shattered_subject_is_still_flagged():
    """The guard must keep working, not merely stop firing."""
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    for row in range(5):
        for column in range(5):
            cv2.circle(mask, (150 + column * 170, 150 + row * 170), 60, 255, -1)
    _, reasons, _ = evaluate_mask(
        np.zeros((1000, 1000, 3), np.uint8), mask, "test", DEFAULT_GUARDS
    )
    assert "MASK_FRAGMENTED" in reasons


def test_speckle_still_counts_toward_foreground_area():
    """Excluded from the count, not from the area - otherwise area would lie."""
    plain = np.zeros((600, 600), dtype=np.uint8)
    cv2.circle(plain, (300, 300), 150, 255, -1)
    speckled = plain.copy()
    speckled[10:12, 10:12] = 255

    image = np.zeros((600, 600, 3), np.uint8)
    _, _, before = evaluate_mask(image, plain, "t", DEFAULT_GUARDS)
    _, _, after = evaluate_mask(image, speckled, "t", DEFAULT_GUARDS)
    assert after["foreground_fraction"] > before["foreground_fraction"]


# --- subject-scale degradation ------------------------------------------------


def test_shrink_subject_keeps_the_frame_size():
    image = np.full((400, 600, 3), 200, dtype=np.uint8)
    shrunk = shrink_subject(image, 0.5)
    assert shrunk.shape == image.shape


def test_shrink_subject_is_identity_at_full_scale():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert np.array_equal(shrink_subject(image, 1.0), image)


def test_subject_scale_ladder_starts_from_identity():
    assert SUBJECT_SCALES[0] == 1.0
    assert all(0.0 < level <= 1.0 for level in SUBJECT_SCALES)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_invalid_subject_scales_are_rejected(bad):
    from competition.vision.degradation import DegradationError

    with pytest.raises(DegradationError):
        shrink_subject(np.zeros((10, 10, 3), np.uint8), bad)


# --- the preregistered rubric -------------------------------------------------


def test_the_ladder_covers_every_scored_verdict():
    verdicts = {rung.expected for rung in DEGRADATION_LADDER}
    assert ExpectedVerdict.UNUSABLE in verdicts
    assert ExpectedVerdict.BORDERLINE in verdicts


def test_subject_scale_rungs_are_deliberately_unlabelled():
    """A small subject is a harder capture, not a defective one."""
    for rung in DEGRADATION_LADDER:
        if rung.kind == "shrink_subject":
            assert rung.expected is ExpectedVerdict.UNLABELLED


def test_severity_ordering_within_each_family_is_monotonic():
    """A milder rung may never be labelled worse than a harsher one."""
    order = {ExpectedVerdict.BORDERLINE: 0, ExpectedVerdict.UNUSABLE: 1}
    by_family: dict[str, list] = {}
    for rung in DEGRADATION_LADDER:
        if rung.expected in order:
            by_family.setdefault(rung.kind, []).append(rung)

    for family, rungs in by_family.items():
        # Blur, glare and occlusion worsen as the level rises; the exposure and
        # contrast families worsen as it falls away from 1.0.
        ascending = family in {"gaussian_blur", "glare", "occlude", "overexpose"}
        ordered = sorted(rungs, key=lambda r: r.level, reverse=not ascending)
        severities = [order[rung.expected] for rung in ordered]
        assert severities == sorted(severities), family


# --- derived samples are reproducible, not stored -----------------------------


def test_regenerating_a_derived_sample_is_deterministic():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    first = regenerate(base, "gaussian_blur", 4.0, 7)
    second = regenerate(base, "gaussian_blur", 4.0, 7)
    assert derived_digest(first) == derived_digest(second)


def test_different_levels_produce_different_derived_hashes():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    assert derived_digest(regenerate(base, "gaussian_blur", 2.0, 7)) != derived_digest(
        regenerate(base, "gaussian_blur", 4.0, 7)
    )


def test_seeded_transforms_depend_on_the_seed():
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    assert derived_digest(regenerate(base, "occlude", 0.2, 1)) != derived_digest(
        regenerate(base, "occlude", 0.2, 2)
    )


# --- the calibration decision rule --------------------------------------------


def test_separation_auc_is_one_for_perfectly_separated_classes():
    assert separation_auc([10.0, 11.0], [1.0, 2.0], "floor") == 1.0


def test_separation_auc_is_half_for_identical_distributions():
    assert separation_auc([5.0, 5.0], [5.0, 5.0], "floor") == 0.5


def test_a_threshold_is_rejected_when_no_candidate_holds_false_accepts_down():
    """Overlapping classes must produce no threshold, not a hopeful one."""
    acceptable = [1.0, 2.0, 3.0, 4.0, 5.0]
    unusable = [1.0, 2.0, 3.0, 4.0, 5.0]
    choice, _ = calibrate_threshold(acceptable, unusable, "floor")
    assert not choice.admissible
    assert choice.reasons


def test_an_admissible_threshold_respects_the_false_accept_ceiling():
    acceptable = [10.0] * 20
    unusable = [1.0] * 20
    choice, _ = calibrate_threshold(acceptable, unusable, "floor")
    assert choice.admissible
    assert choice.false_accept_rate <= MAX_FALSE_ACCEPT_RATE
    assert 1.0 < choice.threshold < 10.0


def test_choose_focus_metric_returns_nothing_when_every_metric_is_disqualified():
    findings = {
        "laplacian_variance": {
            "usable_as_global_threshold": False,
            "disqualifiers": ["exposure sensitivity"],
            "threshold": {"separation_auc": 1.0, "false_block_rate": 0.0},
        }
    }
    metric, floor, reasons = choose_focus_metric(findings)
    assert metric is None and floor is None
    assert reasons


def test_the_preregistered_criteria_are_what_the_phase_note_states():
    """These constants are the phase's preregistration; changing one is a decision."""
    assert MIN_SEPARATION_AUC == 0.95
    assert MAX_FALSE_ACCEPT_RATE == 0.10
    assert MAX_EXPOSURE_SENSITIVITY == 1.5
    assert MAX_SCALE_SENSITIVITY == 3.0


# --- synthetic stress tier ----------------------------------------------------


def test_every_stress_case_composes_deterministically():
    from competition.evaluation.build_synthetic_stress import STRESS_CASES, compose

    for case in STRESS_CASES[:4]:
        first, second = compose(case, 11), compose(case, 11)
        assert np.array_equal(first, second), case.name


def test_stress_records_declare_generated_provenance():
    manifest = Path("competition/data/licensed_real/manifests/synthetic_stress.json")
    if not manifest.is_file():
        pytest.skip("synthetic tier not built in this checkout")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["records"]
    for record in document["records"]:
        assert record["provenance"] == ProvenanceType.SYNTHETIC_GENERATED.value
        assert record["track"] == CorpusTrack.SYNTHETIC_STRESS.value
    assert "no image-generation model was used" in document["generator_note"]


def test_synthetic_results_are_labelled_exploratory():
    results = Path("competition/evaluation/results/phase2c_synthetic_stress.json")
    if not results.is_file():
        pytest.skip("synthetic tier not evaluated in this checkout")
    assert json.loads(results.read_text(encoding="utf-8"))["status"] == "EXPLORATORY"


# --- claim boundary in the emitted results ------------------------------------


def test_results_files_carry_no_food_safety_vocabulary():
    from competition.models.ontology import PROHIBITED_CLAIM_TERMS

    for path in Path("competition/evaluation/results").glob("phase2c*.json"):
        payload = path.read_text(encoding="utf-8").lower()
        for term in PROHIBITED_CLAIM_TERMS:
            assert term not in payload, f"{term!r} in {path.name}"


def test_held_out_results_never_claim_deployment_calibration():
    path = Path("competition/evaluation/results/phase2c_heldout.json")
    if not path.is_file():
        pytest.skip("held-out evaluation not run in this checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert "not phone-camera deployment captures" in document["claim_boundary"]
    assert document["condition_model"]["visible_condition_accuracy"] is None


# --- the Phase 2c-B contrast consistency audit --------------------------------


def test_low_contrast_is_advisory_not_blocking():
    """Pins the Phase 1b decision the audit traced the zero escalations to."""
    from competition.agent.inspection import ADVISORY_FLAGS, DEFAULT_BLOCKING_FLAGS
    from competition.vision.evidence import QualityFlag

    assert QualityFlag.LOW_CONTRAST.value in ADVISORY_FLAGS
    assert QualityFlag.LOW_CONTRAST.value not in DEFAULT_BLOCKING_FLAGS


def test_contrast_reduction_lowers_the_roi_contrast_metric():
    """The metric responds: the audit's first stage."""
    from competition.vision.config import DEFAULT_POLICY
    from competition.vision.degradation import reduce_contrast
    from competition.vision.quality import measure_illumination

    rng = np.random.default_rng(4)
    image = cv2.GaussianBlur(
        rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), (5, 5), 1.0
    )
    before = measure_illumination(image, DEFAULT_POLICY).contrast_score
    after = measure_illumination(reduce_contrast(image, 0.2), DEFAULT_POLICY).contrast_score
    assert after < before / 2


def test_the_calibrated_contrast_threshold_raises_the_flag():
    """The threshold fires: the audit's second and third stages."""
    from types import SimpleNamespace

    from competition.vision.roi_policy import RoiQualityPolicy, apply_roi_policy

    policy = RoiQualityPolicy(low_contrast_limit=0.1314, focus_floor=None)
    properties = SimpleNamespace(width=1200, height=900)
    collapsed = SimpleNamespace(
        mean_luminance=0.5, contrast_score=0.09,
        shadow_clip_fraction=0.0, highlight_clip_fraction=0.0,
    )
    intact = SimpleNamespace(
        mean_luminance=0.5, contrast_score=0.56,
        shadow_clip_fraction=0.0, highlight_clip_fraction=0.0,
    )
    assert "LOW_CONTRAST" in apply_roi_policy(properties, collapsed, collapsed, policy)
    assert "LOW_CONTRAST" not in apply_roi_policy(properties, intact, intact, policy)


def test_a_low_contrast_only_capture_stays_eligible_end_to_end():
    """The audit's conclusion: flagged, and deliberately not blocked."""
    from competition.evaluation.evaluate_locked_policy import gate_outcome
    from competition.vision.roi_policy import RoiQualityPolicy

    policy = RoiQualityPolicy(low_contrast_limit=0.1314, focus_floor=None)
    sample = {
        "width": 1200, "height": 900, "roi_available": True,
        "roi": {"laplacian_variance": 500.0, "tenengrad": 5000.0,
                "normalised_gradient_energy": 25.0, "high_frequency_ratio": 0.6,
                "mean_luminance": 0.5, "contrast_score": 0.09,
                "shadow_clip_fraction": 0.0, "highlight_clip_fraction": 0.0},
    }
    outcome = gate_outcome(sample, policy)
    assert "LOW_CONTRAST" in outcome["flags"]
    assert outcome["outcome"] == "ELIGIBLE"
    assert outcome["eligible"] is True


def test_the_committed_audit_records_the_mechanism():
    path = Path("competition/evaluation/results/phase2c_contrast_audit.json")
    if not path.is_file():
        pytest.skip("contrast audit not run in this checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    contrast = document["by_family"]["reduce_contrast"]
    assert contrast["block_rate"] == 0.0
    assert contrast["detection_rate_including_advisory"] > 0.8
    assert document["by_family"]["glare"]["detection_rate_including_advisory"] < 0.5
    assert "calibration groups only" in document["split"]
