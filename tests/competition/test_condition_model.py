"""Condition-model adapter, evidence contract and ontology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from competition.models.adapter import (
    RUNTIME_ONNXRUNTIME,
    RUNTIME_OPENCV_DNN,
    array_fingerprint,
    load_condition_model,
    predict_condition,
)
from competition.models.evidence import ConditionModelError
from competition.models.ontology import (
    PROHIBITED_CLAIM_TERMS,
    SUPPORTED_RESEARCH_LABELS,
    FruitType,
    OntologyError,
    VisibleCondition,
    map_research_label,
)
from competition.models.preprocessing import DEFAULT_CONTRACT
from competition.vision.fixtures import textured_object

ARTIFACT_DIR = Path("competition/models/artifacts/mobilenetv3_large_v2exp004")

# The exported graph is a gitignored build product, so a clean clone will not
# have it. Tests that need real inference skip rather than fail.
requires_artifact = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.onnx").is_file(),
    reason="exported ONNX artifact absent; run `python -m competition.models.export`",
)


@pytest.fixture(name="model")
def _model():
    return load_condition_model(ARTIFACT_DIR, runtime=RUNTIME_OPENCV_DNN)


# --- ontology -----------------------------------------------------------------


def test_ontology_covers_exactly_the_research_labels():
    assert set(SUPPORTED_RESEARCH_LABELS) == {
        "fresh_apple", "fresh_banana", "fresh_orange",
        "rotten_apple", "rotten_banana", "rotten_orange",
    }


def test_research_labels_map_to_fruit_and_condition():
    fruit, condition = map_research_label("rotten_banana")
    assert fruit is FruitType.BANANA
    assert condition is VisibleCondition.DETERIORATED


def test_unsupported_label_raises_rather_than_guessing():
    """A label the model was never trained on must not be coerced into one."""
    with pytest.raises(OntologyError):
        map_research_label("rotten_mango")


def test_condition_vocabulary_is_visible_condition_only():
    assert {c.value for c in VisibleCondition} == {"fresh", "deteriorated"}


# --- required test 13: no food-safety language in machine output --------------


@requires_artifact
def test_no_prohibited_claim_language_in_evidence(model):
    evidence = predict_condition(model, textured_object())
    payload = json.dumps(evidence.to_dict()).lower()
    for term in PROHIBITED_CLAIM_TERMS:
        assert term not in payload, f"prohibited claim term {term!r} in model output"


def test_prohibited_terms_absent_from_ontology_descriptions():
    from competition.models.ontology import CONDITION_DESCRIPTION

    for text in CONDITION_DESCRIPTION.values():
        lowered = text.lower()
        for term in PROHIBITED_CLAIM_TERMS:
            assert term not in lowered


# --- required tests 1-5: evidence contract ------------------------------------


@requires_artifact
def test_evidence_serialises_to_json(model):
    evidence = predict_condition(model, textured_object())
    restored = json.loads(evidence.to_json())
    assert restored["predicted_research_label"] in SUPPORTED_RESEARCH_LABELS
    assert 0.0 <= restored["confidence"] <= 1.0
    assert len(restored["class_probabilities"]) == 6
    assert restored["runtime"] == RUNTIME_OPENCV_DNN


@requires_artifact
def test_no_filesystem_path_leaks_in_evidence(model):
    payload = predict_condition(model, textured_object()).to_json()
    assert "/home/" not in payload
    assert "/Users/" not in payload
    assert "C:\\" not in payload
    assert ".onnx" not in payload
    assert "competition/models/artifacts" not in payload


@requires_artifact
def test_input_content_hash_is_recorded(model):
    image = textured_object()
    evidence = predict_condition(model, image)
    assert evidence.input_sha256 == array_fingerprint(image)
    assert len(evidence.input_sha256) == 64


@requires_artifact
def test_different_inputs_have_different_hashes(model):
    from competition.vision.fixtures import checkerboard

    a = predict_condition(model, textured_object())
    b = predict_condition(model, checkerboard())
    assert a.input_sha256 != b.input_sha256


@requires_artifact
def test_model_artifact_fingerprint_is_recorded(model):
    evidence = predict_condition(model, textured_object())
    assert evidence.model_artifact_fingerprint == model.artifact_fingerprint
    assert len(evidence.model_artifact_fingerprint) == 32


@requires_artifact
def test_identical_input_produces_identical_deterministic_evidence(model):
    image = textured_object()
    first = predict_condition(model, image)
    second = predict_condition(model, image)
    assert first.deterministic_payload() == second.deterministic_payload()
    assert "inference_ms" not in first.deterministic_payload()


@requires_artifact
def test_probabilities_sum_to_one(model):
    evidence = predict_condition(model, textured_object())
    assert sum(evidence.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


@requires_artifact
def test_preprocessing_version_is_recorded(model):
    evidence = predict_condition(model, textured_object())
    assert evidence.preprocessing_version == DEFAULT_CONTRACT.version


# --- required test 12: labels stay inside the supported ontology --------------


@requires_artifact
def test_predicted_label_is_always_within_the_ontology(model):
    from competition.vision.fixtures import all_fixtures

    for image in all_fixtures().values():
        evidence = predict_condition(model, image)
        assert evidence.predicted_research_label in SUPPORTED_RESEARCH_LABELS
        assert isinstance(evidence.fruit_type, FruitType)
        assert isinstance(evidence.visible_condition, VisibleCondition)


# --- required test 8: corrupt or missing artifacts fail safely ---------------


def test_missing_artifact_directory_raises(tmp_path):
    with pytest.raises(ConditionModelError):
        load_condition_model(tmp_path / "nowhere")


def test_manifest_without_graph_raises(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_id": "x", "model_family": "y", "model_version": "1",
        "class_names": list(SUPPORTED_RESEARCH_LABELS),
    }), encoding="utf-8")
    with pytest.raises(ConditionModelError) as error:
        load_condition_model(tmp_path)
    assert "model.onnx" in str(error.value)


def test_corrupt_graph_fails_safely(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_id": "x", "model_family": "y", "model_version": "1",
        "class_names": list(SUPPORTED_RESEARCH_LABELS),
    }), encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"this is not an ONNX graph")
    with pytest.raises(ConditionModelError):
        load_condition_model(tmp_path)


@requires_artifact
def test_fingerprint_mismatch_is_detected(tmp_path):
    import shutil

    shutil.copy(ARTIFACT_DIR / "model.onnx", tmp_path / "model.onnx")
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["onnx_sha256_prefix"] = "0" * 32
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConditionModelError) as error:
        load_condition_model(tmp_path)
    assert "fingerprint" in str(error.value)


def test_unsupported_runtime_is_rejected():
    with pytest.raises(ConditionModelError):
        load_condition_model(ARTIFACT_DIR, runtime="tensorflow")


@requires_artifact
def test_preprocessing_mismatch_blocks_loading(tmp_path):
    import shutil

    from competition.models.preprocessing import PreprocessingContract

    shutil.copy(ARTIFACT_DIR / "model.onnx", tmp_path / "model.onnx")
    shutil.copy(ARTIFACT_DIR / "manifest.json", tmp_path / "manifest.json")
    with pytest.raises(ConditionModelError) as error:
        load_condition_model(tmp_path, contract=PreprocessingContract(image_size=128))
    assert "preprocessing" in str(error.value)


# --- required test 14: alternate-runtime parity -------------------------------


@requires_artifact
def test_opencv_dnn_and_onnxruntime_agree():
    """Both runtimes execute the same graph; agreement must be numerical noise."""
    onnxruntime = pytest.importorskip("onnxruntime")  # noqa: F841

    opencv_model = load_condition_model(ARTIFACT_DIR, runtime=RUNTIME_OPENCV_DNN)
    onnx_model = load_condition_model(ARTIFACT_DIR, runtime=RUNTIME_ONNXRUNTIME)

    from competition.vision.fixtures import all_fixtures

    for image in all_fixtures().values():
        a = predict_condition(opencv_model, image)
        b = predict_condition(onnx_model, image)
        assert a.predicted_research_label == b.predicted_research_label
        for label, probability in a.class_probabilities.items():
            assert b.class_probabilities[label] == pytest.approx(probability, abs=1e-4)


# --- required test 15: research artifacts are not mutated ---------------------


def test_research_checkpoints_are_not_modified_by_the_competition_line():
    """Checkpoint mtimes and sizes must be untouched by export or inference."""
    checkpoint = Path("v2/checkpoints/best_mobilenetv3.pt")
    if not checkpoint.is_file():
        pytest.skip("research checkpoint not present locally")
    before = checkpoint.stat()
    if (ARTIFACT_DIR / "model.onnx").is_file():
        model = load_condition_model(ARTIFACT_DIR)
        predict_condition(model, textured_object())
    after = checkpoint.stat()
    assert before.st_mtime == after.st_mtime
    assert before.st_size == after.st_size


def test_competition_sources_do_not_import_research_packages():
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in (root / "competition").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("from v2", "import v2", "from src", "import src ", "import tensorflow", "import keras"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, offenders
