"""Evidence contract: determinism, serialisation, and path hygiene."""

from __future__ import annotations

import json

import pytest

from competition.vision import assess_capture_quality
from competition.vision.config import THRESHOLD_STATUS, ThresholdPolicy
from competition.vision.evidence import PIPELINE_VERSION, QualityFlag
from competition.vision.fixtures import all_fixtures, checkerboard, textured_object


@pytest.fixture(name="fixtures")
def _fixtures():
    return all_fixtures()


# --- required test 1: identical input -> identical evidence -------------------


def test_identical_input_produces_identical_evidence(fixtures):
    """Determinism excludes timing, which is wall-clock and legitimately varies."""
    for name, image in fixtures.items():
        first = assess_capture_quality(image)
        second = assess_capture_quality(image)
        assert first.deterministic_payload() == second.deterministic_payload(), (
            f"non-deterministic evidence for fixture {name!r}"
        )


def test_determinism_holds_across_many_repeats():
    image = textured_object()
    baseline = assess_capture_quality(image).deterministic_payload()
    for _ in range(10):
        assert assess_capture_quality(image).deterministic_payload() == baseline


def test_timing_is_excluded_from_deterministic_payload():
    evidence = assess_capture_quality(checkerboard())
    assert "processing_ms" not in evidence.deterministic_payload()
    assert "processing_ms" in evidence.to_dict(include_timing=True)


def test_content_hash_distinguishes_different_images():
    a = assess_capture_quality(checkerboard())
    b = assess_capture_quality(textured_object())
    assert a.image.content_sha256 != b.image.content_sha256


# --- required test 2: JSON serialisation succeeds -----------------------------


def test_evidence_serialises_to_json(fixtures):
    for name, image in fixtures.items():
        evidence = assess_capture_quality(image)
        payload = evidence.to_json()
        restored = json.loads(payload)
        assert restored["pipeline_version"] == PIPELINE_VERSION
        assert isinstance(restored["quality_flags"], list)
        assert restored["image"]["width"] > 0
        assert "laplacian_variance" in restored["sharpness"]
        assert "mean_luminance" in restored["illumination"]


def test_json_is_stable_across_runs():
    image = checkerboard()
    first = assess_capture_quality(image).to_json(include_timing=False)
    second = assess_capture_quality(image).to_json(include_timing=False)
    assert first == second


def test_quality_flags_are_plain_strings():
    """Flags must survive a JSON round trip as strings, not enum reprs."""
    evidence = assess_capture_quality(checkerboard())
    restored = json.loads(evidence.to_json())
    for flag in restored["quality_flags"]:
        assert isinstance(flag, str)
        assert flag == QualityFlag(flag).value


# --- required test 9: no absolute filesystem path in output -------------------


def test_evidence_contains_no_absolute_filesystem_path(fixtures):
    """Guards the mistake already present in the V2 research result files.

    Several committed research metadata files embed absolute paths from an
    earlier directory layout, which became stale when the repository moved. The
    perception layer identifies images by content hash instead, and this test
    keeps it that way.
    """
    for name, image in fixtures.items():
        payload = assess_capture_quality(image).to_json()
        assert "/home/" not in payload, f"absolute path leaked for {name!r}"
        assert "/Users/" not in payload
        assert "C:\\" not in payload
        # No POSIX-absolute value anywhere in the record.
        for value in json.loads(payload).values():
            if isinstance(value, str):
                assert not value.startswith("/"), f"absolute-looking value: {value!r}"


# --- provenance ---------------------------------------------------------------


def test_evidence_records_policy_provenance():
    evidence = assess_capture_quality(checkerboard())
    assert evidence.threshold_policy_status == THRESHOLD_STATUS
    assert len(evidence.threshold_policy_fingerprint) == 16


def test_policy_fingerprint_changes_when_thresholds_change():
    strict = ThresholdPolicy(blur_variance_floor=100.0)
    lenient = ThresholdPolicy(blur_variance_floor=5.0)
    assert strict.fingerprint() != lenient.fingerprint()


def test_policy_fingerprint_is_stable_for_equal_policies():
    assert ThresholdPolicy().fingerprint() == ThresholdPolicy().fingerprint()


def test_is_suitable_for_inspection_tracks_flags():
    evidence = assess_capture_quality(textured_object())
    assert evidence.is_suitable_for_inspection == (not evidence.quality_flags)
