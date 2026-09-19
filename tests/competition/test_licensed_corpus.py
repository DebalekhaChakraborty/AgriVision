"""Licensing gate, provenance integrity and split isolation for Phase 2c-B.

These tests protect three things that cannot be checked by reading the output
of a run: that no image entered the corpus without a verified permissive
licence, that provenance survives every transformation, and that held-out
material stays shut until a policy is frozen.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from competition.data.licence_validation import (
    ALLOWED_LICENCE_IDS,
    FORBIDDEN_LICENCE_IDS,
    LicenceAssertion,
    LicenceVerdict,
    evaluate_licence,
    normalise_licence,
    reconcile_equivalent,
)
from competition.data.licensed_sources import (
    SYNTHETIC_LICENCE_ID,
    CorpusError,
    CorpusTrack,
    DerivedSampleRecord,
    LicensedImageRecord,
    ProvenanceType,
    Split,
    capture_family_id,
    derive_from,
    source_group_id,
)
from competition.evaluation.split_licensed_corpus import (
    DEFAULT_SEED,
    CorpusView,
    HeldOutAccessError,
    apply_split,
    assign_group_splits,
    build_split_document,
    calibration_view,
    held_out_view,
    split_fingerprint,
    verify_no_group_crosses_split,
)


def record(**overrides) -> LicensedImageRecord:
    base = dict(
        image_id="WC-abc123def456",
        source_group_id="SG-000000000001",
        capture_family_id="CF-0000000001",
        fruit_type="apple",
        provenance=ProvenanceType.LICENSED_REAL.value,
        track=CorpusTrack.CLEAN_BASE.value,
        source_name="Wikimedia Commons",
        source_url="https://commons.wikimedia.org/wiki/File:Example.jpg",
        creator="A Photographer",
        licence_id="CC-BY-SA-4.0",
        licence_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution_text='A Photographer, "Example", CC-BY-SA-4.0, via Wikimedia Commons',
        commercial_use_allowed=True,
        derivatives_allowed=True,
        attribution_required=True,
        share_alike_required=True,
        retrieved_at="2026-09-18",
        content_sha256="a" * 64,
        width=1600,
        height=1200,
        file_extension=".jpg",
    )
    base.update(overrides)
    return LicensedImageRecord(**base)


# --- 1. unknown licence rejected ---------------------------------------------


def test_unknown_licence_is_rejected():
    decision = evaluate_licence(LicenceAssertion(short_name="Some Custom Terms"))
    assert decision.verdict is LicenceVerdict.REJECTED_UNKNOWN
    assert not decision.allowed


def test_absent_licence_metadata_is_rejected_not_assumed_free():
    assert evaluate_licence(LicenceAssertion()).verdict is LicenceVerdict.REJECTED_UNKNOWN


@pytest.mark.parametrize("text", ["GFDL 1.2", "Fair use", "Attribution", "", None])
def test_unrecognised_statements_never_resolve_to_a_licence(text):
    assert normalise_licence(text) == "UNKNOWN"


# --- 2. NonCommercial rejected ------------------------------------------------


@pytest.mark.parametrize("short_name", [
    "CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC BY-NC 2.0", "CC BY-NC-ND 4.0",
])
def test_non_commercial_licences_are_rejected(short_name):
    decision = evaluate_licence(LicenceAssertion(short_name=short_name))
    assert decision.verdict is LicenceVerdict.REJECTED_NON_COMMERCIAL
    assert "NonCommercial" in decision.reasons[0]


def test_non_commercial_prose_is_rejected():
    decision = evaluate_licence(LicenceAssertion(
        usage_terms="Creative Commons Attribution-NonCommercial 4.0"))
    assert decision.verdict is LicenceVerdict.REJECTED_NON_COMMERCIAL


# --- 3. NoDerivatives rejected ------------------------------------------------


@pytest.mark.parametrize("short_name", ["CC BY-ND 4.0", "CC BY-ND 3.0"])
def test_no_derivatives_licences_are_rejected(short_name):
    decision = evaluate_licence(LicenceAssertion(short_name=short_name))
    assert decision.verdict is LicenceVerdict.REJECTED_NO_DERIVATIVES


def test_nc_and_nd_qualifiers_are_not_swallowed_by_the_plain_patterns():
    """`by-nc-sa` must not normalise to `by-sa`; the pattern order is load-bearing."""
    assert normalise_licence("CC BY-NC-SA 4.0") == "CC-BY-NC-SA-4.0"
    assert normalise_licence("CC BY-NC-ND 4.0") == "CC-BY-NC-ND-4.0"
    assert set(FORBIDDEN_LICENCE_IDS) & ALLOWED_LICENCE_IDS == set()


# --- 4. permitted licence accepted --------------------------------------------


@pytest.mark.parametrize("short_name,expected", [
    ("CC0", "CC0-1.0"), ("Public domain", "public-domain"),
    ("CC BY 4.0", "CC-BY-4.0"), ("CC BY-SA 3.0", "CC-BY-SA-3.0"),
    ("CC BY 2.0", "CC-BY-2.0"),
])
def test_permitted_licences_are_accepted(short_name, expected):
    decision = evaluate_licence(LicenceAssertion(short_name=short_name))
    assert decision.allowed
    assert decision.licence_id == expected
    assert decision.terms.commercial_use_allowed
    assert decision.terms.derivatives_allowed


def test_every_allowed_licence_permits_commercial_use_and_derivatives():
    """The corpus is adapted and publicly submitted; neither is optional."""
    from competition.data.licence_validation import ALLOWED_LICENCES

    for licence_id, terms in ALLOWED_LICENCES.items():
        assert terms.commercial_use_allowed, licence_id
        assert terms.derivatives_allowed, licence_id


def test_conflicting_fields_are_rejected():
    decision = evaluate_licence(LicenceAssertion(
        short_name="CC BY-SA 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0",
    ))
    assert decision.verdict is LicenceVerdict.REJECTED_CONFLICTING


def test_cc0_and_public_domain_are_reconciled_not_treated_as_a_conflict():
    """Commons states one CC0 grant three ways; that is not a rights conflict."""
    decision = evaluate_licence(LicenceAssertion(
        short_name="CC0",
        usage_terms="Creative Commons Zero, Public Domain Dedication",
        licence_url="http://creativecommons.org/publicdomain/zero/1.0/deed.en",
    ))
    assert decision.allowed
    assert decision.licence_id == "CC0-1.0"


def test_reconciliation_never_crosses_a_rights_boundary():
    assert reconcile_equivalent({"CC-BY-SA-4.0", "CC-BY-4.0"}) is None
    assert reconcile_equivalent({"CC0-1.0", "CC-BY-NC-4.0"}) is None


def test_subject_level_restrictions_override_a_permissive_licence():
    decision = evaluate_licence(LicenceAssertion(
        short_name="CC BY-SA 4.0", restrictions="trademarked"))
    assert decision.verdict is LicenceVerdict.REJECTED_RESTRICTED


# --- 5. image-level attribution preserved -------------------------------------


def test_attribution_text_is_required_when_the_licence_demands_it():
    with pytest.raises(CorpusError) as error:
        record(attribution_text="   ").validate()
    assert "attribution" in str(error.value)


def test_attribution_is_per_image_not_per_source():
    first = record(image_id="WC-1", creator="Ada", attribution_text="Ada, CC-BY-SA-4.0")
    second = record(image_id="WC-2", creator="Grace", attribution_text="Grace, CC-BY-SA-4.0")
    assert first.attribution_text != second.attribution_text
    assert first.creator != second.creator


def test_declared_permissions_must_match_the_licence():
    with pytest.raises(CorpusError) as error:
        record(share_alike_required=False).validate()
    assert "contradict" in str(error.value)


def test_a_real_image_cannot_carry_a_licence_outside_the_allow_list():
    with pytest.raises(CorpusError):
        record(licence_id="CC-BY-NC-4.0").validate()


# --- 6. source groups cannot cross the split ----------------------------------


def test_verify_detects_a_group_spanning_both_splits():
    records = [
        record(image_id="WC-1", source_group_id="SG-x", split=Split.CALIBRATION.value),
        record(image_id="WC-2", source_group_id="SG-x", split=Split.HELD_OUT.value),
    ]
    with pytest.raises(CorpusError) as error:
        verify_no_group_crosses_split(records)
    assert "span both splits" in str(error.value)


def test_a_built_split_never_lets_a_group_cross():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 9}",
               fruit_type=["apple", "banana", "orange"][i % 3],
               track=[CorpusTrack.CLEAN_BASE.value, CorpusTrack.NATURAL_SCENE.value][i % 2])
        for i in range(36)
    ]
    document = build_split_document(records)
    verify_no_group_crosses_split(apply_split(records, document["assignment"]))


def test_every_image_of_a_group_lands_in_one_split():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 5}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(30)
    ]
    assignment = assign_group_splits(records)
    stamped = apply_split(records, assignment)
    by_group: dict[str, set[str]] = {}
    for item in stamped:
        by_group.setdefault(item.source_group_id, set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_group.values())


# --- 7. deterministic split ---------------------------------------------------


def test_split_is_deterministic_for_a_seed():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 7}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(28)
    ]
    assert assign_group_splits(records, 7) == assign_group_splits(records, 7)


def test_different_seeds_generally_differ():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(30)
    ]
    assert assign_group_splits(records, 1) != assign_group_splits(records, 2)


def test_split_order_does_not_matter():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 6}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(24)
    ]
    assert assign_group_splits(records, DEFAULT_SEED) == assign_group_splits(
        list(reversed(records)), DEFAULT_SEED
    )


def test_edited_assignment_breaks_the_fingerprint():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 6}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(24)
    ]
    document = build_split_document(records)
    assert document["fingerprint"] == split_fingerprint(document)
    tampered = {**document, "assignment": {**document["assignment"], "SG-0": "calibration"}}
    if tampered["assignment"] != document["assignment"]:
        assert split_fingerprint(tampered) != document["fingerprint"]


def test_empty_corpus_is_an_error_not_an_empty_split():
    with pytest.raises(CorpusError):
        assign_group_splits([])


# --- 8. derived samples inherit provenance ------------------------------------


def test_derived_sample_inherits_group_split_licence_and_creator():
    base = record(split=Split.HELD_OUT.value, source_group_id="SG-parent")
    derived = derive_from(base, "gaussian_blur", {"level": 4.0}, 11, "b" * 64)
    assert derived.source_group_id == "SG-parent"
    assert derived.split == Split.HELD_OUT.value
    assert derived.licence_id == base.licence_id
    assert derived.creator == base.creator
    assert derived.attribution_text == base.attribution_text
    assert derived.base_content_sha256 == base.content_sha256


def test_derived_ids_are_stable_and_transformation_specific():
    base = record()
    first = derive_from(base, "gaussian_blur", {"level": 4.0}, 11, "b" * 64)
    again = derive_from(base, "gaussian_blur", {"level": 4.0}, 11, "b" * 64)
    other = derive_from(base, "gaussian_blur", {"level": 7.0}, 11, "b" * 64)
    assert first.derived_id == again.derived_id
    assert first.derived_id != other.derived_id


def test_a_derived_sample_declares_derived_provenance():
    derived = derive_from(record(), "occlude", {"level": 0.5}, 1, "c" * 64)
    assert derived.provenance == ProvenanceType.DERIVED_DEGRADED.value


def test_a_derived_record_cannot_drop_its_source_group():
    with pytest.raises(CorpusError):
        DerivedSampleRecord(
            derived_id="d", base_image_id="b", base_content_sha256="a" * 64,
            source_group_id="", fruit_type="apple",
            track=CorpusTrack.CLEAN_BASE.value, split=Split.CALIBRATION.value,
            transformation="gaussian_blur", parameters={}, seed=0,
            derived_sha256="b" * 64, licence_id="CC-BY-SA-4.0",
            creator="A", attribution_text="A",
        ).validate()


# --- 9. generated images cannot masquerade as real ----------------------------


def synthetic(**overrides) -> LicensedImageRecord:
    base = dict(
        provenance=ProvenanceType.SYNTHETIC_GENERATED.value,
        track=CorpusTrack.SYNTHETIC_STRESS.value,
        licence_id=SYNTHETIC_LICENCE_ID,
        licence_url="",
        creator="procedurally composed",
        attribution_text="",
        commercial_use_allowed=True,
        derivatives_allowed=True,
        attribution_required=False,
        share_alike_required=False,
        source_name="AgriVision synthetic stress set",
        source_url="",
    )
    base.update(overrides)
    return record(**base)


def test_a_valid_synthetic_record_validates():
    synthetic().validate()


def test_a_generated_image_cannot_claim_a_real_licence():
    with pytest.raises(CorpusError) as error:
        synthetic(licence_id="CC-BY-SA-4.0").validate()
    assert "may not carry a real-image licence" in str(error.value)


def test_a_generated_image_cannot_sit_in_a_real_image_track():
    with pytest.raises(CorpusError):
        synthetic(track=CorpusTrack.CLEAN_BASE.value).validate()


def test_a_real_image_cannot_sit_in_the_synthetic_track():
    with pytest.raises(CorpusError) as error:
        record(track=CorpusTrack.SYNTHETIC_STRESS.value).validate()
    assert "reserved for generated images" in str(error.value)


def test_the_synthetic_licence_sentinel_is_not_an_allowed_licence():
    assert SYNTHETIC_LICENCE_ID not in ALLOWED_LICENCE_IDS


def test_a_generated_image_cannot_be_degraded_into_the_real_distribution():
    with pytest.raises(CorpusError) as error:
        derive_from(synthetic(), "gaussian_blur", {"level": 4.0}, 1, "b" * 64)
    assert "LICENSED_REAL" in str(error.value)


# --- 10. no absolute paths leak ------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("source_url", "/home/someone/pictures/apple.jpg"),
    ("notes", "stored at /home/someone/corpus"),
    ("creator", "/Users/someone"),
])
def test_absolute_paths_are_rejected_in_records(field, value):
    with pytest.raises(CorpusError):
        record(**{field: value}).validate()


def test_https_urls_are_not_mistaken_for_paths():
    record(source_url="https://commons.wikimedia.org/wiki/File:Home_apple.jpg").validate()


def test_no_committed_manifest_contains_a_machine_path():
    manifests = Path("competition/data/licensed_real/manifests")
    if not manifests.is_dir():
        pytest.skip("corpus manifests not built in this checkout")
    for path in manifests.glob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert "/home/" not in payload, path.name
        assert "/Users/" not in payload, path.name


def test_record_has_no_path_carrying_field():
    assert not any("path" in name for name in record().to_dict())


# --- 11. raw image bytes stay out of version control --------------------------


@pytest.mark.parametrize("relative", [
    "competition/data/licensed_real/raw/WC-x.jpg",
    "competition/data/licensed_real/cache/WC-x.jpg",
    "competition/data/licensed_real/derived/abc.png",
    "competition/data/licensed_real/synthetic/abc.png",
])
def test_image_directories_are_gitignored(relative):
    result = subprocess.run(["git", "check-ignore", relative],
                            capture_output=True, text=True)
    assert result.returncode == 0, f"{relative} is not ignored"


def test_manifests_are_not_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "competition/data/licensed_real/manifests/licensed_corpus.json"],
        capture_output=True, text=True)
    assert result.returncode != 0, "manifests must stay in version control"


# --- 12. calibration code cannot reach held-out groups ------------------------


def test_a_calibration_view_refuses_to_hold_held_out_records():
    with pytest.raises(HeldOutAccessError):
        CorpusView([record(split=Split.HELD_OUT.value)], Split.CALIBRATION.value)


def test_a_held_out_view_requires_a_policy_fingerprint():
    with pytest.raises(HeldOutAccessError) as error:
        CorpusView([record(split=Split.HELD_OUT.value)], Split.HELD_OUT.value)
    assert "frozen policy fingerprint" in str(error.value)


def test_held_out_view_rejects_an_empty_fingerprint():
    with pytest.raises(HeldOutAccessError):
        held_out_view("")


def test_calibration_view_contains_only_calibration_records():
    records = [
        record(image_id=f"WC-{i}", source_group_id=f"SG-{i % 6}",
               fruit_type=["apple", "banana", "orange"][i % 3])
        for i in range(24)
    ]
    document = build_split_document(records)
    view = calibration_view(records, document)
    assert view.records
    assert all(item.split == Split.CALIBRATION.value for item in view.records)


# --- 13/14. held-out runs are ledgered ----------------------------------------


def test_locked_policy_is_required_before_held_out_evaluation(tmp_path):
    from competition.evaluation.evaluate_locked_policy import load_locked_policy

    with pytest.raises(CorpusError) as error:
        load_locked_policy(tmp_path / "absent.json")
    assert "frozen policy" in str(error.value)


def test_an_edited_locked_policy_file_is_detected(tmp_path):
    from competition.evaluation.evaluate_locked_policy import load_locked_policy
    from competition.vision.roi_policy import RoiQualityPolicy

    policy = RoiQualityPolicy(focus_metric="tenengrad", focus_floor=500.0)
    document = policy.to_dict()
    document["focus_floor"] = 5.0  # someone relaxes the gate by hand
    path = tmp_path / "locked.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CorpusError) as error:
        load_locked_policy(path)
    assert "fingerprint mismatch" in str(error.value)


def test_first_run_is_confirmatory():
    from competition.evaluation.evaluate_locked_policy import CONFIRMATORY, classify_run
    from competition.vision.roi_policy import RoiQualityPolicy

    status, _ = classify_run(RoiQualityPolicy(), [])
    assert status == CONFIRMATORY


def test_changing_thresholds_after_a_held_out_run_makes_the_next_run_exploratory():
    from competition.evaluation.evaluate_locked_policy import EXPLORATORY, classify_run
    from competition.vision.roi_policy import RoiQualityPolicy

    first = RoiQualityPolicy(focus_metric="tenengrad", focus_floor=500.0)
    ledger = [{"threshold_fingerprint": first.threshold_fingerprint()}]
    retuned = RoiQualityPolicy(focus_metric="tenengrad", focus_floor=50.0)
    status, reason = classify_run(retuned, ledger)
    assert status == EXPLORATORY
    assert "already opened" in reason


def test_rerunning_identical_thresholds_stays_confirmatory():
    from competition.evaluation.evaluate_locked_policy import CONFIRMATORY, classify_run
    from competition.vision.roi_policy import RoiQualityPolicy

    policy = RoiQualityPolicy(focus_metric="tenengrad", focus_floor=500.0)
    ledger = [{"threshold_fingerprint": policy.threshold_fingerprint()}]
    status, _ = classify_run(policy, ledger)
    assert status == CONFIRMATORY


# --- 15. duplicate content detected -------------------------------------------


def test_identical_content_hashes_are_detectable_across_records():
    first = record(image_id="WC-1", content_sha256="d" * 64)
    second = record(image_id="WC-2", content_sha256="d" * 64)
    seen: dict[str, str] = {}
    duplicates = []
    for item in (first, second):
        if item.content_sha256 in seen:
            duplicates.append((seen[item.content_sha256], item.image_id))
        seen[item.content_sha256] = item.image_id
    assert duplicates == [("WC-1", "WC-2")]


def test_the_built_corpus_has_no_duplicate_content():
    manifest = Path("competition/data/licensed_real/manifests/licensed_corpus.json")
    if not manifest.is_file():
        pytest.skip("corpus not built in this checkout")
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    hashes = [item["content_sha256"] for item in records]
    assert len(hashes) == len(set(hashes))


def test_an_invalid_content_hash_is_rejected():
    with pytest.raises(CorpusError):
        record(content_sha256="not-a-hash").validate()


# --- 16. source families and groups are respected -----------------------------


def test_indexed_filenames_collapse_to_one_capture_family():
    assert capture_family_id("Apple Red 01.jpg") == capture_family_id("Apple Red 07.jpg")
    assert capture_family_id("Apple Red (3).jpg") == capture_family_id("Apple Red 11.jpg")


def test_different_subjects_do_not_share_a_capture_family():
    assert capture_family_id("Apple Red 01.jpg") != capture_family_id("Banana Yellow 01.jpg")


def test_one_creator_is_one_source_group_regardless_of_formatting():
    assert source_group_id("Ada Lovelace") == source_group_id("  ada   lovelace ")


def test_different_creators_are_different_source_groups():
    assert source_group_id("Ada Lovelace") != source_group_id("Grace Hopper")


def test_a_source_group_cannot_be_formed_without_a_creator():
    with pytest.raises(CorpusError):
        source_group_id("   ")


def test_built_corpus_keeps_every_group_within_one_split():
    manifest = Path("competition/data/licensed_real/manifests/corpus_split.json")
    if not manifest.is_file():
        pytest.skip("split not built in this checkout")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["fingerprint"] == split_fingerprint(document)
    assert set(document["assignment"].values()) <= {
        Split.CALIBRATION.value, Split.HELD_OUT.value
    }
