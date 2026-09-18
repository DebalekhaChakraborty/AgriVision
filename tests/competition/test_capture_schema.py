"""Capture schema: record validation, taxonomy locking, claim boundary."""

from __future__ import annotations

import json

import pytest

from competition.data.capture_schema import (
    CONDITION_MATRIX,
    CONDITION_SPEC_BY_ID,
    MINIMUM,
    REQUIRED_CONDITIONS,
    SAMPLED_CONDITIONS,
    STRETCH,
    SUPPORTED_EXTENSIONS,
    TARGET,
    CaptureCondition,
    CaptureRecord,
    CaptureSchemaError,
    QualityTarget,
    SceneCase,
    Split,
    VisibleConditionAnnotation,
    expected_item_ids,
    parse_item_id,
    protocol_summary,
)
from competition.models.ontology import PROHIBITED_CLAIM_TERMS, FruitType


def good_record(**overrides) -> CaptureRecord:
    base = dict(
        image_id="APL-001__REFERENCE__1",
        item_id="APL-001",
        fruit_type=FruitType.APPLE,
        capture_condition=CaptureCondition.REFERENCE,
        replicate_id=1,
        split=Split.CALIBRATION,
        device_id="PHONE_A",
        quality_target=QualityTarget.ACCEPTABLE,
        content_sha256="a" * 64,
        width=3024,
        height=4032,
    )
    base.update(overrides)
    return CaptureRecord(**base)


# --- required test 1: a good record validates --------------------------------


def test_valid_record_passes():
    good_record().validate()


def test_record_serialises_to_json():
    restored = json.loads(json.dumps(good_record().to_dict()))
    assert restored["fruit_type"] == "apple"
    assert restored["capture_condition"] == "REFERENCE"
    assert restored["split"] == "calibration"


# --- required test 2: unsupported fruit rejected ------------------------------


@pytest.mark.parametrize("bad", ["MNG-001", "APPLE-001", "APL-1", "apl-001", "", "APL-0001"])
def test_invalid_item_ids_are_rejected(bad):
    with pytest.raises(CaptureSchemaError):
        parse_item_id(bad)


def test_item_prefix_must_agree_with_declared_fruit():
    with pytest.raises(CaptureSchemaError) as error:
        good_record(item_id="BAN-001", fruit_type=FruitType.APPLE).validate()
    assert "encodes" in str(error.value)


def test_only_the_three_supported_fruits_have_prefixes():
    prefixes = {item[:3] for item in expected_item_ids(TARGET)}
    assert prefixes == {"APL", "BAN", "ORG"}


# --- required test 3: unknown condition rejected ------------------------------


def test_unknown_condition_is_rejected():
    with pytest.raises(CaptureSchemaError):
        good_record(capture_condition="SUPER_DARK").validate()  # type: ignore[arg-type]


def test_condition_taxonomy_is_locked():
    assert {c.value for c in CaptureCondition} == {
        "REFERENCE", "REFERENCE_REPEAT", "DIM", "DARK_SEVERE", "OVEREXPOSED",
        "GLARE", "DEFOCUS_MILD", "DEFOCUS_SEVERE", "MOTION_BLUR",
        "CLUTTERED_BACKGROUND", "PARTIAL_OCCLUSION", "SMALL_SUBJECT",
    }


def test_every_condition_has_a_complete_spec():
    assert len(CONDITION_MATRIX) == len(CaptureCondition)
    for spec in CONDITION_MATRIX:
        assert spec.purpose and spec.procedure
        assert spec.tests in {"quality gate", "segmentation", "both"}
        assert isinstance(spec.quality_target, QualityTarget)


def test_required_and_sampled_conditions_partition_the_matrix():
    assert set(REQUIRED_CONDITIONS) | set(SAMPLED_CONDITIONS) == set(CaptureCondition)
    assert not set(REQUIRED_CONDITIONS) & set(SAMPLED_CONDITIONS)


def test_reference_is_required_for_every_item():
    assert CaptureCondition.REFERENCE in REQUIRED_CONDITIONS
    assert CaptureCondition.REFERENCE_REPEAT in REQUIRED_CONDITIONS


def test_quality_target_must_match_the_preregistered_condition():
    """A per-record override would make the rubric unfalsifiable."""
    with pytest.raises(CaptureSchemaError) as error:
        good_record(
            capture_condition=CaptureCondition.DEFOCUS_SEVERE,
            quality_target=QualityTarget.ACCEPTABLE,
        ).validate()
    assert "preregistered" in str(error.value)


def test_severe_degradations_are_preregistered_as_recapture():
    for condition in (CaptureCondition.DEFOCUS_SEVERE, CaptureCondition.DARK_SEVERE,
                      CaptureCondition.MOTION_BLUR):
        spec = CONDITION_SPEC_BY_ID[condition]
        assert spec.quality_target is QualityTarget.RECAPTURE_REQUIRED
        assert spec.recapture_preferred
        assert not spec.remediation_expected


def test_recoverable_exposure_conditions_expect_remediation():
    for condition in (CaptureCondition.DIM, CaptureCondition.OVEREXPOSED):
        assert CONDITION_SPEC_BY_ID[condition].remediation_expected


# --- structural validation ----------------------------------------------------


@pytest.mark.parametrize("overrides", [
    {"content_sha256": "short"},
    {"width": 0},
    {"height": -1},
    {"replicate_id": 0},
    {"device_id": ""},
])
def test_structurally_invalid_records_are_rejected(overrides):
    with pytest.raises(CaptureSchemaError):
        good_record(**overrides).validate()


# --- required test 8: no filesystem paths in the committed representation ----


def test_record_carries_no_filesystem_path():
    payload = json.dumps(good_record().to_dict())
    assert "/" not in payload.replace("\\/", "")
    assert "path" not in payload.lower()


def test_record_fields_exclude_path():
    assert not any("path" in field for field in good_record().to_dict())


# --- claim boundary -----------------------------------------------------------


def test_visible_condition_vocabulary_is_appearance_only():
    assert {v.value for v in VisibleConditionAnnotation} == {
        "FRESH_APPEARING", "VISIBLY_DEGRADED", "UNCERTAIN"
    }


def test_no_food_safety_vocabulary_anywhere_in_the_schema():
    payload = json.dumps(protocol_summary()).lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in payload, f"prohibited term {term!r} in the capture protocol"


def test_schema_has_no_safe_or_edible_labels():
    values = {v.value for v in VisibleConditionAnnotation} | {q.value for q in QualityTarget}
    for forbidden in ("SAFE", "UNSAFE", "EDIBLE", "CONTAMINATED"):
        assert forbidden not in values


# --- collection targets -------------------------------------------------------


def test_target_allocation_matches_the_brief():
    assert TARGET.total_items == 24
    assert TARGET.calibration_items == 15
    assert TARGET.evaluation_items == 9
    assert TARGET.items_per_fruit == 8


def test_minimum_target_is_smaller_and_documents_weaker_power():
    assert MINIMUM.total_items == 18
    assert MINIMUM.items_per_fruit == 6
    assert "caveat" in MINIMUM.note or "atypical" in MINIMUM.note


def test_stretch_target_is_larger_but_optional():
    assert STRETCH.total_items == 30
    assert "not required" in STRETCH.note


def test_expected_item_ids_are_balanced_across_fruits():
    ids = expected_item_ids(TARGET)
    assert len(ids) == 24
    for prefix in ("APL", "BAN", "ORG"):
        assert sum(1 for item in ids if item.startswith(prefix)) == 8


def test_capture_count_includes_scene_images():
    """Regression: the earlier total of 180 omitted the six scene captures."""
    assert TARGET.total_capture_count == 186
    assert TARGET.total_capture_count != 180


# --- protocol summary ---------------------------------------------------------


def test_protocol_status_is_capture_pending():
    assert protocol_summary()["status"] == "PROTOCOL LOCKED - CAPTURE PENDING"


def test_protocol_summary_is_serialisable():
    assert json.loads(json.dumps(protocol_summary()))["schema_version"]


def test_scene_cases_are_separate_from_item_conditions():
    assert {case.value for case in SceneCase} == {"MIXED_FRUIT", "NON_PRODUCE"}
    assert not {case.value for case in SceneCase} & {c.value for c in CaptureCondition}


def test_supported_extensions_are_image_types():
    assert SUPPORTED_EXTENSIONS == {".jpg", ".jpeg", ".png"}


# --- corrected capture counts -------------------------------------------------


def test_target_total_is_186_including_scene_captures():
    """144 required + 36 extended + 6 scene. The scene images were previously omitted."""
    from competition.data.capture_schema import SCENE_CAPTURE_COUNT

    assert TARGET.required_capture_count == 144
    assert TARGET.extended_capture_count == 36
    assert SCENE_CAPTURE_COUNT == 6
    assert TARGET.total_capture_count == 186


def test_total_capture_count_is_the_sum_of_its_parts():
    from competition.data.capture_schema import SCENE_CAPTURE_COUNT

    for target in (TARGET, MINIMUM, STRETCH):
        assert target.total_capture_count == (
            target.required_capture_count
            + target.extended_capture_count
            + SCENE_CAPTURE_COUNT
        )


def test_extended_condition_items_is_two_per_fruit_at_target():
    assert TARGET.extended_condition_items == 6


def test_target_dict_exposes_the_count_breakdown():
    data = TARGET.to_dict()
    assert data["total_capture_count"] == 186
    assert data["required_capture_count"] == 144
    assert data["extended_capture_count"] == 36
    assert data["scene_capture_count"] == 6


# --- procedures must be distinguishable --------------------------------------


def test_dim_and_dark_severe_procedures_are_distinct():
    """Conflating them would blur the remediable/recapture boundary."""
    dim = CONDITION_SPEC_BY_ID[CaptureCondition.DIM].procedure
    dark = CONDITION_SPEC_BY_ID[CaptureCondition.DARK_SEVERE].procedure
    assert "CLEARLY VISIBLE" in dim
    assert "NEAR-DARK" in dark
    assert "DARK_SEVERE" in dim  # dim explicitly points away from darkness
    assert "do not use flash" in dim.lower()


def test_overexposed_and_glare_procedures_are_distinct():
    over = CONDITION_SPEC_BY_ID[CaptureCondition.OVEREXPOSED].procedure
    glare = CONDITION_SPEC_BY_ID[CaptureCondition.GLARE].procedure
    assert "DIFFUSE" in over
    assert "not create a concentrated specular hotspot" in over.lower()
    assert "DIRECTIONAL" in glare
    assert "specular highlight" in glare


def test_defocus_mild_and_severe_procedures_are_distinct():
    mild = CONDITION_SPEC_BY_ID[CaptureCondition.DEFOCUS_MILD].procedure
    severe = CONDITION_SPEC_BY_ID[CaptureCondition.DEFOCUS_SEVERE].procedure
    assert "RECOGNISABLE" in mild
    assert "destroy inspection detail" in severe


def test_small_subject_requires_physical_distance_not_cropping():
    procedure = CONDITION_SPEC_BY_ID[CaptureCondition.SMALL_SUBJECT].procedure
    assert "CAMERA DISTANCE" in procedure
    assert "not crop or digitally resize" in procedure.lower()


def test_motion_blur_keeps_the_subject_in_frame():
    procedure = CONDITION_SPEC_BY_ID[CaptureCondition.MOTION_BLUR].procedure
    assert "CAMERA" in procedure
    assert "in frame" in procedure


def test_cluttered_background_keeps_the_subject_well_captured():
    procedure = CONDITION_SPEC_BY_ID[CaptureCondition.CLUTTERED_BACKGROUND].procedure
    assert "SUBJECT itself must remain reasonably captured" in procedure


def test_partial_occlusion_does_not_alter_the_fruit():
    procedure = CONDITION_SPEC_BY_ID[CaptureCondition.PARTIAL_OCCLUSION].procedure
    assert "only occlude it" in procedure.lower()
