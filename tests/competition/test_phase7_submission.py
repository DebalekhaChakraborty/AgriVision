"""Phase 7: the submission package's own guarantees.

These do not check that the numbers are good. They check that the ablation
cannot be mistaken for confirmatory evidence, that the serving runtime has not
acquired research dependencies, that no licence was invented, that the
documentation states the things it is required to state, and that no claim the
evidence does not support has crept into a judge-facing document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RESULTS = Path("competition/evaluation/results/phase7")
DOCS = Path("docs/competition")


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def _doc(name: str) -> str:
    path = DOCS / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ablation() -> dict:
    return _load("baseline_ablation.json")


# --- the ablation must never pass as confirmatory ------------------------------


def test_ablation_is_labelled_retrospective_and_exploratory(ablation):
    assert ablation["status"] == "RETROSPECTIVE_EXPLORATORY_ABLATION"
    banner = ablation["banner"].lower()
    assert "not preregistered" in banner
    assert "not confirmatory" in banner


def test_ablation_states_it_was_defined_after_seeing_results(ablation):
    assert "after the phase 6 results were seen" in ablation["banner"].lower()


def test_ablation_uses_the_same_model_artifact(ablation):
    model = ablation["model"]
    assert model["artifact"].endswith("mobilenetv3_large_v2exp004")
    assert model["runtime"] == "opencv_dnn"
    assert "nothing was trained, tuned or replaced" in model["unchanged"].lower()


def test_ablation_validity_was_measured_not_assumed(ablation):
    """The baseline runs locally; the agentic figures came from AWS."""
    agreement = ablation["track_r"]["local_baseline_reproduces_deployed_predictions"]
    assert agreement["denominator"] > 0
    assert agreement["numerator"] == agreement["denominator"], (
        "local and deployed predictions must agree for the comparison to mean "
        "anything")


def test_coverage_and_accuracy_are_reported_separately(ablation):
    for system in ("model_only", "agentic"):
        block = ablation["track_r"][system]
        assert "INFERENCE_COVERAGE" in block
        assert "FRUIT_TYPE_ACCURACY" in block
        assert block["INFERENCE_COVERAGE"]["denominator"] >= (
            block["FRUIT_TYPE_ACCURACY"]["denominator"])
    assert "never combined" in ablation["reporting_rule"].lower()


def test_no_weighted_score_combines_the_two_systems(ablation):
    payload = json.dumps(ablation).lower()
    for invented in ("composite_score", "weighted_score", "overall_score",
                     "combined_metric"):
        assert invented not in payload


def test_agentic_fruit_type_denominator_is_invocations_not_the_pool(ablation):
    agentic = ablation["track_r"]["agentic"]
    assert (agentic["FRUIT_TYPE_ACCURACY"]["denominator"]
            == agentic["INFERENCE_COVERAGE"]["numerator"])
    assert "must not be restated over it" in (
        agentic["FRUIT_TYPE_ACCURACY"]["definition"])


# --- Track C metric definitions -----------------------------------------------


def test_policy_bypass_denominator_excludes_scenarios_needing_no_action(ablation):
    metric = ablation["track_c"]["POLICY_BYPASS_RATE"]
    total = len(ablation["track_c"]["per_scenario"])
    assert metric["denominator"] < total, (
        "REFERENCE scenarios preregister NONE; proceeding is correct there and "
        "they must not be counted as bypasses")
    assert "excludes" in metric["definition"].lower()


def test_unsafe_inference_is_reserved_for_blocking_scenarios(ablation):
    unsafe = ablation["track_c"]["UNSAFE_INFERENCE"]
    definition = unsafe["baseline"]["definition"].lower()
    assert "blocked" in definition
    assert "not called unsafe" in definition
    assert unsafe["agentic"]["numerator"] == 0


def test_recoverable_conditions_are_counted_separately_from_unsafe(ablation):
    uncorrected = ablation["track_c"]["UNCORRECTED_RECOVERABLE"]
    assert "not unsafe" in uncorrected["definition"].lower()


def test_capability_differences_are_labelled_structural_not_measured(ablation):
    note = ablation["track_c"]["capability_comparison"]["note"].lower()
    assert "structural, not measured" in note


# --- observability -------------------------------------------------------------


def test_cloudwatch_audit_found_no_sensitive_content():
    audit = _load("cloudwatch_audit.json")
    assert audit["no_sensitive_content"]["verified"]
    assert audit["no_sensitive_content"]["violations"] == []
    assert len(audit["no_sensitive_content"]["patterns_checked"]) >= 6


def test_cloudwatch_audit_used_a_separate_read_only_identity():
    audit = _load("cloudwatch_audit.json")
    note = audit["audit_identity_note"].lower()
    assert "not widened" in note
    assert "agrivision-auditor" in audit["audit_identity"]


def test_cloudwatch_records_carry_the_causal_fields():
    audit = _load("cloudwatch_audit.json")
    emitted = audit["structured_logging_emitted"]
    assert emitted["verified"]
    for field in ("run_id", "state", "tool", "action", "reason_code", "duration_ms"):
        assert field in emitted["required_fields"]
        assert emitted["field_coverage"][field] == emitted["complete_step_records"]


def test_cloudwatch_evidence_is_tied_to_a_recorded_inspection():
    audit = _load("cloudwatch_audit.json")
    tied = audit["tied_to_known_inspections"]
    assert tied["verified"]
    assert tied["targeted_records_found"] > 0


# --- dependency freeze and SBOM ------------------------------------------------


def test_serving_runtime_has_no_research_dependencies():
    freeze = _load("dependency_freeze.json")
    assert freeze["research_contamination_check"]["clean"]
    installed = {name.lower() for name in freeze["frozen"]}
    for forbidden in ("torch", "torchvision", "onnx", "onnxruntime",
                      "matplotlib", "pytest"):
        assert forbidden not in installed


def test_dependency_freeze_comes_from_the_image_not_the_requirements_file():
    freeze = _load("dependency_freeze.json")
    assert "built container image" in freeze["source_of_truth"]
    assert freeze["pip_check_clean"]


def test_every_direct_dependency_is_pinned():
    freeze = _load("dependency_freeze.json")
    assert freeze["pinned"]
    for name, version in freeze["pinned"].items():
        assert re.match(r"^\d+\.\d+", version), f"{name} is not a pinned version"


def test_sbom_never_invents_a_licence():
    sbom = _load("sbom.json")
    assert "no value is inferred" in sbom["licence_rule"]
    for component in sbom["components"]:
        if component["licence"] is None:
            assert component["source"], "an unknown licence must say why"
        else:
            assert "package metadata" in component["source"]


def test_sbom_is_honest_about_not_being_a_standard_sbom():
    sbom = _load("sbom.json")
    assert "not cyclonedx or spdx" in sbom["format"].lower()


# --- security ------------------------------------------------------------------


def test_security_audit_is_clean():
    audit = _load("security_audit.json")
    assert audit["clean"]
    assert audit["repository"]["tracked_file_scan"]["findings"] == []
    assert audit["repository"]["history_scan"]["findings"] == []
    assert audit["repository"]["env_and_key_files"]["tracked_secret_files"] == []


def test_security_audit_records_metadata_only():
    audit = _load("security_audit.json")
    assert "no secret value is read" in audit["disclosure_policy"].lower()


def test_security_audit_does_not_over_redact_account_identifiers():
    audit = _load("security_audit.json")
    assert "not treated as a secret" in audit["disclosure_policy"].lower()


def test_deployment_posture_holds():
    deployment = _load("security_audit.json")["deployment"]
    assert deployment["root_persistent_access_keys_zero"]
    assert deployment["root_mfa_enabled"]
    assert deployment["model_bucket_public_access_blocked"]["all_four_blocks_on"]
    assert deployment["runtime_uses_role_not_keys"]
    assert deployment["long_lived_credentials_in_container"] is False


def test_history_scan_states_its_own_bound():
    history = _load("security_audit.json")["repository"]["history_scan"]
    assert "not the entire repository history" in history["scope_note"]


# --- reproducibility -----------------------------------------------------------


def test_clean_clone_rehearsal_covered_every_documented_step():
    record = _load("clean_clone_reproduction.json")
    assert len(record["steps"]) == 13
    assert all(step["result"].startswith("PASS") for step in record["steps"])


def test_clean_clone_reproduced_the_deployed_policy_fingerprint():
    record = _load("clean_clone_reproduction.json")
    assert record["policy_fingerprint_reproduced"]["roi_policy_threshold"] == (
        "78b1e2151773787a")


def test_artifact_independent_mode_has_no_failures():
    record = _load("clean_clone_reproduction.json")
    independent = record["test_classification"]["artifact_independent"]
    assert independent["failed"] == 0
    assert independent["skipped"] > 0, "skips are the point of this mode"


def test_the_third_party_artifact_limitation_is_stated():
    record = _load("clean_clone_reproduction.json")
    assert "cannot obtain the model artifact unaided" in (
        record["third_party_limitation"])


def test_rehearsal_defects_were_recorded_not_quietly_fixed():
    record = _load("clean_clone_reproduction.json")
    assert len(record["defects_found_and_fixed"]) >= 3
    for defect in record["defects_found_and_fixed"]:
        assert defect["impact"] and defect["fix"]


# --- infrastructure ------------------------------------------------------------


def test_iac_declares_no_unused_cloud_services():
    check = _load("iac_check.json")
    assert check["template"]["no_cloud_service_stuffing"]
    assert check["template"]["forbidden_services_present"] == []
    assert check["template"]["required_resource_types_present"]


def test_iac_check_created_nothing():
    check = _load("iac_check.json")
    assert check["no_resources_created"] is True


def test_template_still_describes_the_running_service():
    check = _load("iac_check.json")
    assert check["live_comparison"]["all_match"], "template has drifted from live"


# --- live final ----------------------------------------------------------------


def test_live_final_smoke_passed_every_scenario():
    smoke = _load("live_final_smoke.json")["smoke"]
    assert smoke["all_http_200"]
    assert smoke["all_traces_retrievable"]
    assert smoke["all_fingerprints_match"]
    assert smoke["all_expectations_met"]


def test_live_ui_surface_still_carries_its_boundaries():
    surface = _load("live_final_smoke.json")["smoke"]["ui_surface"]
    assert surface["responsible_use_present"]
    assert surface["one_primary_fruit_contract_present"]
    assert surface["no_food_safety_claim"]


def test_final_demo_revision_is_recorded_by_digest():
    stability = _load("live_final_smoke.json")["stability"]
    revision = stability["FINAL_DEMO_REVISION"]
    assert revision["image_digest"].startswith("sha256:")
    assert revision["root_required"] is False
    assert stability["service_running"]
    assert stability["no_pending_failed_deployment"]


# --- documentation discipline ---------------------------------------------------


def test_technical_report_defines_its_result_labels():
    report = _doc("TECHNICAL_REPORT.md")
    for label in ("DEVELOPMENT", "CONFIRMATORY", "EXPLORATORY",
                  "RETROSPECTIVE ABLATION"):
        assert label in report


def test_technical_report_gives_underexposure_its_own_section():
    report = _doc("TECHNICAL_REPORT.md")
    assert "15.1" in report
    assert "2/12" in report
    assert "No post-confirmatory retuning was performed" in report


def test_technical_report_states_the_model_trade_off_plainly():
    report = _doc("TECHNICAL_REPORT.md")
    assert "not a claim that it is the best model" in report
    assert "15.9 points worse" in report


def test_technical_report_states_the_segmentation_contract():
    report = _doc("TECHNICAL_REPORT.md")
    assert "one primary produce item" in report.lower()
    assert "PROVISIONAL and disabled by default" in report


def test_technical_report_keeps_confidence_unqualified():
    report = _doc("TECHNICAL_REPORT.md")
    assert "UNQUALIFIED for policy" in report
    assert "No confidence threshold exists" in report


def test_illustrative_example_is_labelled_as_such():
    report = _doc("TECHNICAL_REPORT.md")
    assert "chosen to be legible, not because it is representative" in report


# A prohibited phrase is allowed to appear where it is being prohibited or
# denied. "It does not determine whether food is safe to eat" is the required
# disclaimer, and a "do not say this" table has to quote the thing not to say.
_NEGATING = ("not say", "do not", "don't", "does not", "must not", "never",
             "instead", "rather than", "forbid", "prohibit")


def test_no_judge_facing_document_claims_end_to_end_accuracy():
    forbidden = ("27/38", "end-to-end accuracy", "96.4% accurate",
                 "determines whether food is safe to eat")
    targets = [Path("README.md")] + sorted(DOCS.glob("*.md"))
    for path in targets:
        if not path.is_file():
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if any(marker in lowered for marker in _NEGATING):
                continue
            for phrase in forbidden:
                if phrase not in line:
                    continue
                # A phrase inside quotation marks is being quoted, not
                # asserted - the "do not say this" table has to name the thing
                # it is banning.
                quoted = f'"{phrase}"' in line or f"'{phrase}'" in line
                assert quoted, f"{phrase!r} asserted at {path.name}:{number}"


def test_the_claim_guard_would_catch_a_real_claim(tmp_path):
    """A guard that cannot fail is not a guard."""
    offender = tmp_path / "bad.md"
    offender.write_text("AgriVision determines whether food is safe to eat.\n")
    line = offender.read_text().splitlines()[0]
    phrase = "determines whether food is safe to eat"
    assert not any(marker in line.lower() for marker in _NEGATING)
    assert phrase in line
    assert f'"{phrase}"' not in line, "an unquoted assertion must not be excused"


def test_demo_script_forbids_the_phrases_that_overclaim():
    script = _doc("JUDGE_DEMO_SCRIPT.md")
    assert "Things that must not be said" in script
    assert "27 of 28 where the model ran" in script
    assert "controlled demonstration" in script.lower()


def test_video_checklist_covers_credential_exposure():
    checklist = _doc("VIDEO_RECORDING_CHECKLIST.md")
    for item in ("No AWS console visible", "access key", "local filesystem paths"):
        assert item.lower() in checklist.lower()


def test_attribution_file_does_not_dump_the_whole_corpus():
    attributions = _doc("IMAGE_ATTRIBUTIONS.md")
    assert "No third-party photograph is displayed" in attributions
    assert attributions.count("WC-") == 0, (
        "the evaluation corpus is not shown publicly and must not be listed "
        "here as though it were")


def test_submission_checklist_marks_outstanding_items_honestly():
    checklist = _doc("SUBMISSION_CHECKLIST.md")
    assert "NOT DONE" in checklist
    assert "Video recorded" in checklist
    assert "an item is only ticked when a named artifact backs it" in checklist.lower()


def test_architecture_diagram_lists_only_deployed_services():
    architecture = _doc("ARCHITECTURE.md")
    for absent in ("Bedrock", "SageMaker", "API Gateway", "Lambda", "Cognito"):
        assert f"{absent}</" not in architecture
    assert "Not present, on purpose" in architecture


def test_workflow_diagram_shows_both_bounds():
    workflow = _doc("AGENT_WORKFLOW_DIAGRAM.md")
    assert "MAX 1 ATTEMPT" in workflow
    assert "MAX 1 INVOCATION" in workflow


# --- a correction, locked so it cannot silently revert -------------------------


def test_the_deployer_log_access_correction_is_stated_not_reverted():
    """Phase 6 claimed the deployer was denied FilterLogEvents. It was not.

    The claim came from probing DescribeLogGroups - an account-wide list call
    needing Resource "*" - and generalising to the whole logs: namespace. The
    deployer has held scoped /aws/apprunner/* read since Phase 4. The gap Phase
    6 reported was real; the reason it gave was wrong, and the correction is
    recorded rather than quietly dropped.
    """
    audit = _load("security_audit.json")["deployment"]
    assert "scoped_deployer_allowed_logs" in audit
    assert "FilterLogEvents" in audit["scoped_deployer_allowed_logs"]
    assert "logs:FilterLogEvents" not in audit["scoped_deployer_denied"], (
        "the deployer is not denied FilterLogEvents; claiming so was the error")
    assert "logs:DescribeLogGroups" in audit["scoped_deployer_denied"]


def test_the_auditor_is_described_as_separation_not_necessity():
    audit = _load("cloudwatch_audit.json")
    note = audit["audit_identity_note"].lower()
    assert "not, however, denied log reads" in note
    assert "misdiagnosis" in note
    assert "not widened" in note


def test_architecture_document_carries_the_same_correction():
    architecture = _doc("ARCHITECTURE.md")
    assert "separation of duties" in architecture
    assert "since Phase 4" in architecture


def test_the_repo_policy_document_matches_the_live_deployer_grant():
    """The IaC was right all along; only the prose was wrong."""
    policy = json.loads(
        Path("infrastructure/aws/policies/deployer-policy.json")
        .read_text(encoding="utf-8"))
    statements = {s.get("Sid"): s for s in policy["Statement"]}
    logs = statements.get("ReadDeploymentLogs")
    assert logs is not None, "the deployer policy has always granted scoped log read"
    assert "logs:FilterLogEvents" in logs["Action"]
    assert "/aws/apprunner/" in logs["Resource"]
