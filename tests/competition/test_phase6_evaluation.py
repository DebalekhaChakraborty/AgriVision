"""Phase 6: the confirmatory evaluation's own guarantees.

These tests do not check that the numbers are good. They check that the numbers
mean what they claim to mean: that the system was frozen, that the sources are
independent, that expectations were fixed before execution, that a failure
cannot be quietly rewritten into a success, and that no photograph or local path
reaches version control.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from competition.data.licence_validation import (
    ALLOWED_LICENCE_IDS,
    LicenceAssertion,
    LicenceVerdict,
    evaluate_licence,
)
from competition.evaluation.phase6_metrics import (
    CATEGORIES,
    ConfirmatoryStatusError,
    assert_frozen,
    classify_track_c,
)
from competition.evaluation.phase6_source_pool import (
    PHASE6_SOURCES,
    PRIOR_CATEGORIES,
    TARGET_PER_FRUIT,
)
from competition.evaluation.phase6_track_c import (
    VARIANT_KEYS,
    VARIANT_SPECS,
    TrackCError,
    preregister,
    select_bases,
)

RESULTS = Path("competition/evaluation/results/phase6")


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen() -> dict:
    return _load("frozen_system.json")


@pytest.fixture(scope="module")
def source_manifest() -> dict:
    return _load("phase6_source_manifest.json")


@pytest.fixture(scope="module")
def expected_actions() -> dict:
    return _load("phase6_expected_actions.json")


@pytest.fixture(scope="module")
def track_r() -> dict:
    return _load("track_r_results.json")


@pytest.fixture(scope="module")
def track_c() -> dict:
    return _load("track_c_results.json")


# --- 1: the frozen system is identified, and identifies the right things ------


def test_frozen_system_carries_every_required_field(frozen):
    assert frozen["git"]["commit"]
    assert frozen["policy"]["fingerprints"]
    assert frozen["policy"]["versions"]["state_machine"]
    assert frozen["model"]["manifest_fingerprint"]
    assert frozen["dependencies"]["opencv"]
    assert frozen["deployed_revision"]["image_digest"].startswith("sha256:")
    assert len(frozen["frozen_fingerprint"]) == 16


def test_frozen_fingerprint_is_derived_not_stored(frozen):
    """Recomputing from the recorded parts must reproduce the fingerprint."""
    import hashlib

    payload = json.dumps(
        {
            "commit": frozen["git"]["commit"],
            "fingerprints": frozen["policy"]["fingerprints"],
            "versions": frozen["policy"]["versions"],
            "model": frozen["model"]["manifest_fingerprint"],
            "budget": frozen["policy"]["budget"],
        },
        sort_keys=True,
    )
    recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert recomputed == frozen["frozen_fingerprint"]


def test_fallback_was_disabled_during_the_evaluation(frozen):
    assert frozen["policy"]["foreground_fallback_enabled"] is False


# --- 2: independence from every prior corpus ----------------------------------


def test_phase6_categories_never_overlap_prior_harvests():
    used = {source.category for source in PHASE6_SOURCES}
    assert not (used & PRIOR_CATEGORIES)


def test_no_source_group_appears_in_any_prior_corpus(source_manifest):
    from competition.evaluation.phase6_source_pool import prior_source_groups

    groups = {r["source_group_id"] for r in source_manifest["records"]}
    assert not (groups & prior_source_groups())


def test_no_image_id_was_ever_screened_before(source_manifest):
    from competition.evaluation.phase6_source_pool import prior_image_ids

    ids = {r["image_id"] for r in source_manifest["records"]}
    assert not (ids & prior_image_ids())


def test_one_image_per_source_group(source_manifest):
    records = source_manifest["records"]
    assert len({r["source_group_id"] for r in records}) == len(records)


# --- 3: duplicates ------------------------------------------------------------


def test_every_content_hash_is_unique(source_manifest):
    hashes = [r["content_sha256"] for r in source_manifest["records"]]
    assert len(set(hashes)) == len(hashes)


def test_no_content_hash_collides_with_a_prior_corpus(source_manifest):
    from competition.evaluation.phase6_source_pool import prior_content_hashes

    hashes = {r["content_sha256"] for r in source_manifest["records"]}
    assert not (hashes & prior_content_hashes())


# --- 4: forbidden licences ----------------------------------------------------


@pytest.mark.parametrize("terms", [
    "CC BY-NC 4.0", "CC BY-ND 3.0", "CC BY-NC-SA 4.0",
    "All rights reserved", "Fair use", "",
])
def test_non_permissive_licences_are_refused(terms):
    decision = evaluate_licence(LicenceAssertion(short_name=terms, usage_terms=terms))
    assert decision.verdict is not LicenceVerdict.ALLOWED


def test_every_phase6_licence_is_on_the_allow_list(source_manifest):
    for record in source_manifest["records"]:
        assert record["licence_id"] in ALLOWED_LICENCE_IDS


def test_every_phase6_image_permits_commercial_use_and_derivatives(source_manifest):
    for record in source_manifest["records"]:
        assert record["commercial_use_allowed"]
        assert record["derivatives_allowed"]


def test_attribution_text_present_wherever_required(source_manifest):
    for record in source_manifest["records"]:
        if record["attribution_required"]:
            assert record["attribution_text"].strip()


# --- 5: expected actions were frozen before execution -------------------------


def test_preregistration_carries_a_fingerprint_over_the_scenarios(expected_actions):
    import hashlib

    payload = json.dumps(
        expected_actions["scenarios"], sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert recomputed == expected_actions["expected_actions_fingerprint"]


def test_preregistration_covers_every_base_and_variant(expected_actions):
    bases = {s["base_id"] for s in expected_actions["scenarios"]}
    assert len(expected_actions["scenarios"]) == len(bases) * len(VARIANT_KEYS)


def test_severe_blur_preregisters_that_the_model_must_not_run():
    blur = next(s for s in VARIANT_SPECS if s.key == "SEVERE_BLUR")
    assert blur.condition_model_must_run is False
    assert blur.expected_first_action == "REQUEST_RECAPTURE"


def test_both_denominators_were_fixed_in_advance(expected_actions):
    assert "primary" in expected_actions["denominators"]
    assert "secondary" in expected_actions["denominators"]


def test_observed_results_match_the_preregistered_fingerprint(track_c, expected_actions):
    assert (track_c["expected_actions_fingerprint"]
            == expected_actions["expected_actions_fingerprint"])


def test_every_scored_scenario_keeps_its_preregistered_expectation(
        track_c, expected_actions):
    """The result file must not have drifted from what was preregistered."""
    prereg = {
        (s["base_id"], s["variant"]): s["expected_first_action"]
        for s in expected_actions["scenarios"]
    }
    for scenario in track_c["per_scenario"]:
        key = (scenario["base_id"], scenario["variant"])
        assert scenario["expected_first_action"] == prereg[key]


# --- 6 and 7: results must have come from the frozen system -------------------


def test_scoring_refuses_observations_from_a_different_policy(frozen):
    tampered = [{
        "http_status": 200, "image_id": "WC-test",
        "policy_fingerprints": {"roi_policy_threshold": "deadbeefdeadbeef"},
    }]
    with pytest.raises(ConfirmatoryStatusError):
        assert_frozen(tampered, frozen)


def test_scoring_accepts_observations_carrying_the_frozen_fingerprints(frozen):
    matching = [{
        "http_status": 200, "image_id": "WC-test",
        "policy_fingerprints": frozen["policy"]["fingerprints"],
    }]
    assert_frozen(matching, frozen)  # must not raise


def test_deployed_responses_carried_the_frozen_policy(track_r, track_c):
    assert track_r["agent_safety"]["policy_fingerprints_consistent"]
    assert track_c["policy_fingerprints_consistent"]


def test_observability_audit_confirms_the_fingerprints_matched():
    audit = _load("observability_audit.json")
    check = audit["checks"]["policy_fingerprints_match_frozen_system"]
    assert check["verified"]
    assert check["matching"] == check["expected"]


# --- 8: unsafe inference ------------------------------------------------------


def test_unsafe_inference_is_zero_on_both_tracks(track_r, track_c):
    assert track_r["agent_safety"]["UNSAFE_INFERENCE_RATE"]["value"] == 0.0
    assert track_c["UNSAFE_INFERENCE_RATE"]["value"] == 0.0


def test_unsafe_inference_denominator_is_every_run(track_c):
    metric = track_c["UNSAFE_INFERENCE_RATE"]
    assert metric["denominator"] == len(track_c["per_scenario"])


def test_no_scenario_ran_the_model_after_a_blocking_terminal(track_c):
    blocking = {"REQUEST_RECAPTURE", "REQUEST_REPOSITION_LIGHT",
                "REQUEST_HUMAN_REVIEW", "FAILED_SAFE"}
    for scenario in track_c["per_scenario"]:
        if scenario["observed_terminal"] in blocking:
            assert not scenario["condition_model_invoked"], scenario["base_id"]


# --- 9: denominators ----------------------------------------------------------


def test_action_selection_denominator_is_all_scenarios(track_c):
    assert (track_c["ACTION_SELECTION_ACCURACY"]["denominator"]
            == len(track_c["per_scenario"]))


def test_secondary_denominator_is_a_strict_subset_of_the_primary(track_c):
    primary = track_c["ACTION_SELECTION_ACCURACY"]["denominator"]
    secondary = track_c["ACTION_SELECTION_ACCURACY_secondary"]["denominator"]
    assert 0 < secondary <= primary


def test_fruit_type_denominator_is_model_invocations_not_the_pool(track_r):
    metric = track_r["model"]["FRUIT_TYPE_ACCURACY"]
    assert metric["denominator"] == track_r["model"]["MODEL_INVOKED_COUNT"]
    assert metric["denominator"] <= track_r["counts"]["images"]


def test_every_reported_metric_states_its_denominator(track_r, track_c):
    def check(block):
        for key, value in block.items():
            if isinstance(value, dict) and "value" in value:
                assert "numerator" in value and "denominator" in value, key
                assert value["definition"], key

    check(track_r["quality_agent"])
    check(track_r["input_foreground"])
    for key in ("ACTION_SELECTION_ACCURACY", "TASK_SUCCESS_RATE",
                "DECISION_ATTRIBUTION_RATE", "FAIL_SAFE_RATE"):
        assert track_c[key]["denominator"] > 0


def test_remediation_acceptance_denominator_is_attempts_not_images(track_r):
    acceptance = track_r["quality_agent"]["REMEDIATION_ACCEPTANCE_RATE"]
    attempts = track_r["quality_agent"]["REMEDIATION_ATTEMPT_RATE"]
    assert acceptance["denominator"] == attempts["numerator"]


# --- 10: a failure stays a failure --------------------------------------------


def test_failed_scenarios_are_preserved_in_the_results(track_c):
    failures = [s for s in track_c["per_scenario"] if not s["task_success"]]
    assert failures, "a result file with no failures would be the suspicious one"
    for scenario in failures:
        assert scenario["expected_first_action"]
        assert "observed_first_action" in scenario


def test_the_underexposed_negative_finding_is_still_recorded(track_c):
    """The 2/12 result must survive in the artifact, not be smoothed away."""
    metric = track_c["per_variant"]["action_selection"]["UNDEREXPOSED_RECOVERABLE"]
    assert metric["numerator"] < metric["denominator"]


def test_every_failure_is_classified_and_none_is_collapsed():
    taxonomy = _load("failure_taxonomy.json")
    assert taxonomy["rows"]
    for row in taxonomy["rows"]:
        assert row["category"] in CATEGORIES
        assert row["detail"]
    assert len(taxonomy["counts"]) > 1, "everything in one bucket is not a taxonomy"


def test_a_mislabelled_scenario_is_not_charged_to_the_agent():
    scenario = {
        "task_success": False, "foreground_valid": True, "variant":
        "UNDEREXPOSED_RECOVERABLE", "condition_model_must_run": True,
        "condition_model_invoked": False, "remediation_action": "",
        "remediation_accepted": False, "observed_first_action": "REQUEST_RECAPTURE",
        "expected_first_action": "APPLY_GAMMA", "terminal_reason_code": "",
    }
    category, _ = classify_track_c(scenario, {"zone": "SHADOWS_CLIPPED"})
    assert category == "SCENARIO_CONSTRUCTION_DEFECT"


def test_a_genuine_false_block_is_still_charged_to_the_agent():
    scenario = {
        "task_success": False, "foreground_valid": True, "variant":
        "LOW_CONTRAST_RECOVERABLE", "condition_model_must_run": True,
        "condition_model_invoked": False, "remediation_action": "",
        "remediation_accepted": False, "observed_first_action": "REQUEST_RECAPTURE",
        "expected_first_action": "APPLY_CLAHE", "terminal_reason_code": "",
    }
    category, _ = classify_track_c(scenario, {"zone": "MEASURED"})
    assert category == "QUALITY_FALSE_BLOCK"


def test_a_successful_scenario_is_never_classified():
    assert classify_track_c({"task_success": True}, None) == ("", "")


# --- 11: the tracks cannot be merged ------------------------------------------


def test_the_two_tracks_are_reported_as_separate_documents(track_r, track_c):
    assert track_r["track"] == "R"
    assert track_c["track"] == "C"
    assert track_r["claim_boundary"] != track_c["claim_boundary"]


def test_agent_metrics_keeps_the_tracks_in_separate_blocks():
    agent = _load("agent_metrics.json")
    assert set(agent) >= {"track_r", "track_c"}
    assert not (set(agent["track_r"]) & {"ACTION_SELECTION_ACCURACY"})
    assert "COMPLETE_INSPECTION_RATE" not in agent["track_c"]


def test_track_c_claim_boundary_states_it_is_not_prevalence(track_c):
    boundary = track_c["claim_boundary"].lower()
    assert "controlled degradation" in boundary
    assert "not real camera-failure prevalence" in boundary


def test_complete_inspection_rate_is_not_called_accuracy(track_r):
    definition = track_r["quality_agent"]["COMPLETE_INSPECTION_RATE"]["definition"]
    assert "NOT accuracy" in definition


def test_exploratory_addendum_is_marked_non_confirmatory():
    exploratory = _load("exploratory_underexposure.json")
    assert exploratory["status"] == "EXPLORATORY_NOT_CONFIRMATORY"
    assert "not preregistered" in exploratory["banner"].lower()


# --- 12: no visible-condition accuracy without labels -------------------------


def test_visible_condition_accuracy_is_not_emitted(track_r):
    assert track_r["model"]["visible_condition_accuracy"] is None
    assert "no independent visible-condition label" in (
        track_r["model"]["visible_condition_note"])


def test_no_result_file_reports_a_visible_condition_accuracy_value():
    """The key may exist; a number in it may not."""
    for path in RESULTS.glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    lowered = key.lower()
                    if ("condition" in lowered or "freshness" in lowered
                            or "rotten" in lowered) and "accuracy" in lowered:
                        assert value is None, f"{key} has a value in {path.name}"
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(document)


def test_source_manifest_states_the_label_boundary(source_manifest):
    assert "none was invented" in source_manifest["label_boundary"]


# --- 13 and 14: nothing local, and nothing binary, reaches the repository -----


def test_no_committed_phase6_artifact_carries_a_local_filesystem_path():
    """Repository-relative paths are fine; absolute local ones are not."""
    markers = ("/home/", "/opt/", "/tmp/", "/var/", "/Users/", "C:\\")
    for path in RESULTS.rglob("*.json"):
        payload = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker not in payload, f"{marker} in {path.name}"


def test_no_committed_phase6_artifact_carries_an_aws_identifier():
    import re

    # Twelve digits standing alone. Bounded against alphanumerics and dots so a
    # run of digits inside a hex run id or a float is not mistaken for one.
    account = re.compile(r"(?<![0-9A-Za-z.])\d{12}(?![0-9A-Za-z.])")
    for path in RESULTS.rglob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert "arn:aws:" not in payload, path.name
        assert not account.search(payload), path.name


def test_phase6_image_directories_are_gitignored():
    """The trailing slash matters.

    The .gitignore entries are directory patterns, and a directory pattern
    cannot match a path that does not exist on disk. Querying without the
    slash passes in a working checkout, where the directory happens to exist,
    and fails in a fresh clone - which is where this check actually matters.
    """
    for directory in ("competition/data/licensed_real/phase6_raw/",
                      "competition/data/licensed_real/phase6_variants/"):
        result = subprocess.run(
            ["git", "check-ignore", directory],
            capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"{directory} is not gitignored"


def test_git_tracks_no_phase6_photograph():
    result = subprocess.run(
        ["git", "ls-files", "competition/data/licensed_real/"],
        capture_output=True, text=True, cwd=Path.cwd())
    for line in result.stdout.splitlines():
        assert not line.lower().endswith((".jpg", ".jpeg", ".png", ".webp")), line


def test_licence_manifest_carries_provenance_but_no_paths():
    licences = _load("phase6_licence_manifest.json")
    assert licences["all_permit_commercial_use"]
    assert licences["all_permit_derivatives"]
    for entry in licences["entries"]:
        assert entry["source_url"].startswith("https://")
        assert entry["track_membership"]
        assert "path" not in entry


# --- selection discipline -----------------------------------------------------


def test_base_selection_is_deterministic(source_manifest):
    records = source_manifest["records"]
    assert select_bases(records) == select_bases(records)


def test_base_selection_is_stratified_and_group_disjoint(source_manifest):
    bases = select_bases(source_manifest["records"])
    fruits = {b["fruit_type"] for b in bases}
    assert len(bases) == len(fruits) * 4
    assert len({b["source_group_id"] for b in bases}) == len(bases)


def test_base_selection_refuses_a_shared_source_group():
    duplicated = [
        {"image_id": f"WC-{i}", "fruit_type": "apple", "source_group_id": "SG-same"}
        for i in range(8)
    ]
    with pytest.raises(TrackCError):
        select_bases(duplicated, per_fruit=4)


def test_preregistration_is_pure(source_manifest):
    """Preregistering must not depend on anything measured."""
    bases = select_bases(source_manifest["records"])
    assert preregister(bases) == preregister(bases)


def test_target_per_fruit_is_the_documented_fifteen():
    assert TARGET_PER_FRUIT == 15


# --- denominator discipline, guarded permanently ------------------------------


def test_fruit_type_accuracy_is_never_restated_over_the_whole_pool(track_r):
    """27/28 is a rate among model invocations. 27/38 would be a different claim.

    Ten images were blocked before the model ran. Folding them into the
    denominator would silently convert a model measurement into an end-to-end
    one, which nothing here supports.
    """
    metric = track_r["model"]["FRUIT_TYPE_ACCURACY"]
    assert metric["denominator"] == track_r["model"]["MODEL_INVOKED_COUNT"]
    assert metric["denominator"] < track_r["counts"]["images"]
    assert "model actually ran" in metric["definition"]


def test_no_artifact_or_document_claims_end_to_end_accuracy():
    roots = [RESULTS, Path("docs/competition"), Path("README.md")]
    forbidden = ("27/38", "end-to-end accuracy", "end to end accuracy")
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*.md")) + list(
            root.rglob("*.json"))
        for path in paths:
            if not path.is_file():
                continue
            payload = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                assert phrase not in payload, f"{phrase!r} in {path}"


def test_confirmatory_and_exploratory_underexposure_stay_separate():
    """3/3 is a diagnostic. It must never stand in for the 2/12 that was measured."""
    exploratory = _load("exploratory_underexposure.json")
    track_c = _load("track_c_results.json")

    assert exploratory["status"] == "EXPLORATORY_NOT_CONFIRMATORY"
    assert exploratory["apply_gamma_count"] == 3
    assert exploratory["band_reachable_count"] == 3

    confirmatory = track_c["per_variant"]["action_selection"]["UNDEREXPOSED_RECOVERABLE"]
    assert (confirmatory["numerator"], confirmatory["denominator"]) == (2, 12)

    # The exploratory numbers must not have leaked into the scored track.
    payload = json.dumps(track_c)
    assert "EXPLORATORY" not in payload
    assert "band_reachable" not in payload


def test_observability_is_not_marked_complete_without_evidence():
    audit = _load("observability_audit.json")
    logging_check = audit["checks"]["structured_json_logging_emitted"]
    if not logging_check["verified"]:
        assert logging_check["status"] == "NOT VERIFIED HERE"
        assert logging_check["to_close_this_gap"]
