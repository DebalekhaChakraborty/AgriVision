"""Capture-set validation: leakage, duplicates, privacy, and access discipline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from competition.data.capture_schema import (
    REQUIRED_CONDITIONS,
    TARGET,
    CaptureSchemaError,
    Split,
    expected_item_ids,
)
from competition.evaluation.calibrate_quality_policy import (
    CALIBRATION_METHOD,
    MIN_CALIBRATION_CAPTURES,
    CalibrationNotReady,
    assert_ready,
    load_calibration_captures,
)
from competition.evaluation.prepare_capture_manifest import parse_filename
from competition.evaluation.split_capture_items import (
    DEFAULT_SEED,
    build_split_document,
)
from competition.evaluation.validate_capture_set import SENSITIVE_EXIF_TAGS, scan_exif, validate

RAW_DIR = Path("competition/data/self_captured/raw")
LOCAL_MANIFEST = Path("competition/data/capture_manifest.local.json")


@pytest.fixture(name="split_file")
def _split_file(tmp_path):
    document = build_split_document(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    path = tmp_path / "capture_split.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def manifest_record(item_id, condition, split, digest="a" * 64, **overrides):
    record = {
        "image_id": f"{item_id}__{condition}__1",
        "item_id": item_id,
        "fruit_type": {"APL": "apple", "BAN": "banana", "ORG": "orange"}[item_id[:3]],
        "capture_condition": condition,
        "replicate_id": 1,
        "split": split,
        "device_id": "PHONE_A",
        "quality_target": "ACCEPTABLE",
        "content_sha256": digest,
        "width": 3024,
        "height": 4032,
    }
    record.update(overrides)
    return record


def write_manifest(tmp_path, document, records):
    path = tmp_path / "capture_manifest.json"
    path.write_text(json.dumps({
        "schema_version": "phase2c-capture-schema-1.0.0",
        "split_fingerprint": document["fingerprint"],
        "captures": records,
        "scenes": [],
    }), encoding="utf-8")
    return path


def codes(findings):
    return {finding["code"] for finding in findings}


# --- required test 10: raw images are gitignored -----------------------------


def test_raw_capture_directory_is_gitignored():
    """Must hold before any photograph is placed there."""
    probe = RAW_DIR / "APL-001__REFERENCE__1.jpg"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)], capture_output=True, check=False
    )
    assert result.returncode == 0, f"{probe} would be committed"


def test_local_manifest_is_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(LOCAL_MANIFEST)], capture_output=True, check=False
    )
    assert result.returncode == 0, "the path-carrying manifest would be committed"


def test_schema_and_split_are_not_ignored():
    for path in ("competition/data/capture_schema.py", "competition/data/capture_split.json"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", path], capture_output=True, check=False
        )
        assert result.returncode != 0, f"{path} should be tracked"


# --- required test 5: cross-split item leakage -------------------------------


def test_cross_split_item_leakage_is_detected(tmp_path, split_file):
    """The same fruit in both splits is the failure this protocol exists to stop."""
    split_path, document = split_file
    item = next(iter(document["assignment"]))
    records = [
        manifest_record(item, "REFERENCE", Split.CALIBRATION.value, "a" * 64),
        manifest_record(item, "DIM", Split.EVALUATION.value, "b" * 64),
    ]
    manifest = write_manifest(tmp_path, document, records)
    found = codes(validate(manifest, split_path, tmp_path / "nothing", tmp_path / "nolocal"))
    assert "CROSS_SPLIT_ITEM_LEAKAGE" in found


def test_split_disagreement_with_the_manifest_is_detected(tmp_path, split_file):
    split_path, document = split_file
    item = next(i for i, s in document["assignment"].items() if s == Split.EVALUATION.value)
    records = [manifest_record(item, "REFERENCE", Split.CALIBRATION.value)]
    manifest = write_manifest(tmp_path, document, records)
    assert "SPLIT_DISAGREEMENT" in codes(
        validate(manifest, split_path, tmp_path / "nothing", tmp_path / "nolocal"))


# --- required test 4: duplicate content --------------------------------------


def test_duplicate_image_hash_is_detected(tmp_path, split_file):
    split_path, document = split_file
    item = next(i for i, s in document["assignment"].items() if s == Split.CALIBRATION.value)
    records = [
        manifest_record(item, "REFERENCE", Split.CALIBRATION.value, "c" * 64),
        manifest_record(item, "DIM", Split.CALIBRATION.value, "c" * 64),
    ]
    manifest = write_manifest(tmp_path, document, records)
    assert "DUPLICATE_CONTENT" in codes(
        validate(manifest, split_path, tmp_path / "nothing", tmp_path / "nolocal"))


# --- required test 11: missing required conditions ---------------------------


def test_missing_required_condition_is_detected(tmp_path, split_file):
    split_path, document = split_file
    item = next(i for i, s in document["assignment"].items() if s == Split.CALIBRATION.value)
    records = [manifest_record(item, "REFERENCE", Split.CALIBRATION.value)]
    found = codes(validate(manifest_path=write_manifest(tmp_path, document, records),
                           split_path=split_path, raw_dir=tmp_path / "nothing",
                           local_manifest_path=tmp_path / "nolocal"))
    assert "MISSING_REQUIRED_CONDITION" in found


def test_uncaptured_items_are_reported(tmp_path, split_file):
    split_path, document = split_file
    manifest = write_manifest(tmp_path, document, [])
    assert "ITEM_NOT_CAPTURED" in codes(
        validate(manifest, split_path, tmp_path / "nothing", tmp_path / "nolocal"))


def test_fruit_mismatch_is_detected(tmp_path, split_file):
    split_path, document = split_file
    item = next(i for i in document["assignment"] if i.startswith("APL"))
    records = [manifest_record(item, "REFERENCE", document["assignment"][item],
                               fruit_type="banana")]
    assert "FRUIT_MISMATCH" in codes(
        validate(write_manifest(tmp_path, document, records), split_path,
                 tmp_path / "nothing", tmp_path / "nolocal"))


def test_implausible_dimensions_are_detected(tmp_path, split_file):
    split_path, document = split_file
    item = next(iter(document["assignment"]))
    records = [manifest_record(item, "REFERENCE", document["assignment"][item],
                               width=10, height=10)]
    assert "IMPLAUSIBLE_DIMENSIONS" in codes(
        validate(write_manifest(tmp_path, document, records), split_path,
                 tmp_path / "nothing", tmp_path / "nolocal"))


def test_manifest_built_against_another_split_is_detected(tmp_path, split_file):
    split_path, document = split_file
    path = tmp_path / "capture_manifest.json"
    path.write_text(json.dumps({"split_fingerprint": "0000000000000000", "captures": []}),
                    encoding="utf-8")
    assert "SPLIT_FINGERPRINT_MISMATCH" in codes(
        validate(path, split_path, tmp_path / "nothing", tmp_path / "nolocal"))


# --- required tests 12-13: unreadable and unsupported files ------------------


def test_unsupported_file_type_is_detected(tmp_path, split_file):
    split_path, document = split_file
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "APL-001__REFERENCE__1.bmp").write_bytes(b"not a supported type")
    assert "UNSUPPORTED_EXTENSION" in codes(
        validate(write_manifest(tmp_path, document, []), split_path, raw,
                 tmp_path / "nolocal", check_exif=False))


def test_corrupted_image_is_detected_by_the_manifest_builder(tmp_path):
    from competition.evaluation.prepare_capture_manifest import read_dimensions

    broken = tmp_path / "APL-001__REFERENCE__1.jpg"
    broken.write_bytes(b"this is not a JPEG")
    with pytest.raises(CaptureSchemaError):
        read_dimensions(broken)


def test_image_on_disk_but_absent_from_manifest_is_reported(tmp_path, split_file):
    split_path, document = split_file
    raw = tmp_path / "raw"
    raw.mkdir()
    cv2.imwrite(str(raw / "APL-001__REFERENCE__1.jpg"),
                np.full((64, 64, 3), 128, dtype=np.uint8))
    assert "IMAGE_NOT_IN_MANIFEST" in codes(
        validate(write_manifest(tmp_path, document, []), split_path, raw,
                 tmp_path / "nolocal", check_exif=False))


def test_manifest_image_missing_from_disk_is_reported(tmp_path, split_file):
    split_path, document = split_file
    raw = tmp_path / "raw"
    raw.mkdir()
    item = next(iter(document["assignment"]))
    records = [manifest_record(item, "REFERENCE", document["assignment"][item])]
    assert "MANIFEST_IMAGE_MISSING" in codes(
        validate(write_manifest(tmp_path, document, records), split_path, raw,
                 tmp_path / "nolocal", check_exif=False))


# --- filename parsing ---------------------------------------------------------


def test_filename_parsing_extracts_the_three_parts():
    assert parse_filename("APL-001__REFERENCE__1.jpg") == ("APL-001", "REFERENCE", 1)
    assert parse_filename("SCENE__MIXED_FRUIT__2.png") == ("SCENE", "MIXED_FRUIT", 2)


@pytest.mark.parametrize("bad", [
    "APL-001-REFERENCE-1.jpg", "APL-001__REFERENCE.jpg",
    "APL-001__REFERENCE__x.jpg", "photo.jpg",
])
def test_malformed_filenames_are_rejected(bad):
    with pytest.raises(CaptureSchemaError):
        parse_filename(bad)


# --- required test 9: EXIF privacy -------------------------------------------


def test_gps_exif_is_detected(tmp_path):
    from PIL import Image

    path = tmp_path / "APL-001__REFERENCE__1.jpg"
    image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    exif = image.getexif()
    from PIL.TiffImagePlugin import IFDRational

    exif[0x8825] = {
        1: "N",
        2: (IFDRational(51, 1), IFDRational(30, 1), IFDRational(0, 1)),
    }  # GPSInfo IFD
    image.save(path, exif=exif)

    assert "GPSInfo" in scan_exif(path)


def test_clean_image_reports_no_sensitive_exif(tmp_path):
    path = tmp_path / "APL-001__REFERENCE__1.jpg"
    cv2.imwrite(str(path), np.full((64, 64, 3), 128, dtype=np.uint8))
    assert scan_exif(path) == []


def test_sensitive_tag_list_covers_location_and_identity():
    for tag in ("GPSInfo", "Artist", "BodySerialNumber", "CameraOwnerName"):
        assert tag in SENSITIVE_EXIF_TAGS


# --- required test 14: evaluation split cannot enter calibration -------------


def test_calibration_loader_returns_only_calibration_records(tmp_path, split_file):
    split_path, document = split_file
    calibration_item = next(
        i for i, s in document["assignment"].items() if s == Split.CALIBRATION.value)
    evaluation_item = next(
        i for i, s in document["assignment"].items() if s == Split.EVALUATION.value)
    records = [
        manifest_record(calibration_item, "REFERENCE", Split.CALIBRATION.value, "a" * 64),
        manifest_record(evaluation_item, "REFERENCE", Split.EVALUATION.value, "b" * 64),
    ]
    loaded = load_calibration_captures(
        write_manifest(tmp_path, document, records), split_path)
    assert len(loaded) == 1
    assert loaded[0]["item_id"] == calibration_item


def test_record_mislabelled_as_calibration_is_rejected(tmp_path, split_file):
    """An evaluation item cannot sneak in by relabelling its split field."""
    split_path, document = split_file
    evaluation_item = next(
        i for i, s in document["assignment"].items() if s == Split.EVALUATION.value)
    records = [manifest_record(evaluation_item, "REFERENCE", Split.CALIBRATION.value)]
    with pytest.raises(CalibrationNotReady) as error:
        load_calibration_captures(write_manifest(tmp_path, document, records), split_path)
    assert "assigned elsewhere" in str(error.value)


# --- required test 15: calibration cannot proceed without evidence -----------


def test_calibration_refuses_when_no_manifest_exists(tmp_path, split_file):
    split_path, _ = split_file
    with pytest.raises(CalibrationNotReady) as error:
        load_calibration_captures(tmp_path / "absent.json", split_path)
    assert "not been collected" in str(error.value)


def test_calibration_refuses_on_too_few_captures():
    with pytest.raises(CalibrationNotReady) as error:
        assert_ready([{"quality_target": "ACCEPTABLE"}] * 3)
    assert str(MIN_CALIBRATION_CAPTURES) in str(error.value)


def test_calibration_refuses_without_both_ends_of_the_rubric():
    only_acceptable = [{"quality_target": "ACCEPTABLE"}] * (MIN_CALIBRATION_CAPTURES + 1)
    with pytest.raises(CalibrationNotReady) as error:
        assert_ready(only_acceptable)
    assert "RECAPTURE_REQUIRED" in str(error.value)


def test_calibration_method_is_locked_and_invents_no_thresholds():
    payload = json.dumps(CALIBRATION_METHOD)
    assert "false-accept" in payload or "FALSE ACCEPT" in payload
    assert CALIBRATION_METHOD["metrics"]["roi_sharpness"]["single_threshold_is_not_assumed"]
    # No concrete threshold value may appear in the methodology.
    for metric in CALIBRATION_METHOD["metrics"].values():
        assert "threshold_value" not in metric
        assert "selected_threshold" not in metric


def test_calibration_method_documents_the_locking_rule():
    assert "held-out" in CALIBRATION_METHOD["locking"]
    assert "exploratory" in CALIBRATION_METHOD["locking"]
