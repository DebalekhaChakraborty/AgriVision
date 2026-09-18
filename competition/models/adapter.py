"""Condition-model adapter.

The rest of the system talks to `ConditionModel`, never to a PyTorch class. That
decoupling is the point: the selected runtime executes an ONNX graph through
`cv2.dnn`, so the serving path needs neither torch nor torchvision, and a future
change of runtime touches only this module.

A model artifact is a directory containing:

    model.onnx      the exported graph (gitignored; a build product)
    manifest.json   provenance: source checkpoint, classes, preprocessing,
                    fingerprints

The manifest is committed; the weights are not. The research repository already
excludes all trained weights from version control, and the models here are
fine-tuned on a dataset whose licence is recorded as Unknown with no
redistribution grant, so that boundary is preserved rather than quietly crossed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from competition.models.evidence import ConditionModelError, ConditionModelEvidence
from competition.models.ontology import map_research_label
from competition.models.preprocessing import (
    DEFAULT_CONTRACT,
    PreprocessingContract,
    contract_matches_checkpoint,
    preprocess_bgr,
)

ADAPTER_VERSION = "phase2-adapter-1.0.0"

RUNTIME_OPENCV_DNN = "opencv_dnn"
RUNTIME_ONNXRUNTIME = "onnxruntime"
SUPPORTED_RUNTIMES = (RUNTIME_OPENCV_DNN, RUNTIME_ONNXRUNTIME)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def array_fingerprint(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


@dataclass
class ConditionModel:
    """A loaded condition model, independent of any training framework."""

    model_id: str
    model_family: str
    model_version: str
    runtime: str
    class_names: list[str]
    contract: PreprocessingContract
    artifact_fingerprint: str
    manifest: dict
    _session: object

    # -- inference -----------------------------------------------------------

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        if self.runtime == RUNTIME_OPENCV_DNN:
            self._session.setInput(tensor)
            return np.asarray(self._session.forward()).reshape(-1)
        if self.runtime == RUNTIME_ONNXRUNTIME:
            name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {name: tensor})
            return np.asarray(outputs[0]).reshape(-1)
        raise ConditionModelError(f"unsupported runtime {self.runtime!r}")

    def predict(self, image_bgr: np.ndarray) -> ConditionModelEvidence:
        return predict_condition(self, image_bgr)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "runtime": self.runtime,
            "class_names": list(self.class_names),
            "artifact_fingerprint": self.artifact_fingerprint,
            "preprocessing": self.contract.to_dict(),
            "adapter_version": ADAPTER_VERSION,
        }


def load_condition_model(
    artifact_dir: str | Path,
    runtime: str = RUNTIME_OPENCV_DNN,
    contract: PreprocessingContract | None = None,
) -> ConditionModel:
    """Load a condition model from an artifact directory.

    Fails loudly rather than degrading: a missing artifact, an unreadable graph
    or a preprocessing mismatch between the contract and the source checkpoint
    all raise instead of producing predictions that look fine and are not.
    """
    if runtime not in SUPPORTED_RUNTIMES:
        raise ConditionModelError(
            f"unsupported runtime {runtime!r}; supported: {', '.join(SUPPORTED_RUNTIMES)}"
        )

    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    onnx_path = artifact_dir / "model.onnx"

    if not manifest_path.is_file():
        raise ConditionModelError(f"manifest not found in artifact {artifact_dir.name}")
    if not onnx_path.is_file():
        raise ConditionModelError(
            f"model.onnx not found in artifact {artifact_dir.name}; "
            "run the Phase 2 export step first"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = contract or DEFAULT_CONTRACT

    checkpoint_preprocessing = manifest.get("source_preprocessing", {})
    if checkpoint_preprocessing:
        matches, differences = contract_matches_checkpoint(
            contract, checkpoint_preprocessing
        )
        if not matches:
            raise ConditionModelError(
                "preprocessing contract does not match the source checkpoint: "
                + "; ".join(differences)
            )

    fingerprint = file_fingerprint(onnx_path)
    recorded = manifest.get("onnx_sha256_prefix")
    if recorded and recorded != fingerprint:
        raise ConditionModelError(
            "model.onnx does not match the fingerprint recorded in the manifest"
        )

    if runtime == RUNTIME_OPENCV_DNN:
        try:
            session = cv2.dnn.readNetFromONNX(str(onnx_path))
        except cv2.error as error:
            raise ConditionModelError(
                f"cv2.dnn could not read the ONNX graph: {error}"
            ) from None
    else:
        try:
            import onnxruntime
        except ImportError:
            raise ConditionModelError(
                "onnxruntime is not installed; it is a comparison runtime only"
            ) from None
        try:
            session = onnxruntime.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        except Exception as error:  # pragma: no cover - environment dependent
            raise ConditionModelError(f"onnxruntime could not load: {error}") from None

    return ConditionModel(
        model_id=manifest["model_id"],
        model_family=manifest["model_family"],
        model_version=manifest["model_version"],
        runtime=runtime,
        class_names=list(manifest["class_names"]),
        contract=contract,
        artifact_fingerprint=fingerprint,
        manifest=manifest,
        _session=session,
    )


def predict_condition(
    model: ConditionModel, image_bgr: np.ndarray
) -> ConditionModelEvidence:
    """Run condition inference on a decoded BGR image.

    The image is expected in OpenCV's native BGR order; conversion to the RGB
    the model was trained on happens inside the preprocessing contract.
    """
    started = time.perf_counter()

    input_hash = array_fingerprint(image_bgr)
    tensor = preprocess_bgr(image_bgr, model.contract)
    logits = model._forward(tensor)

    if logits.shape[0] != len(model.class_names):
        raise ConditionModelError(
            f"model returned {logits.shape[0]} logits for "
            f"{len(model.class_names)} classes"
        )

    probabilities = _softmax(logits.astype(np.float64))
    index = int(np.argmax(probabilities))
    label = model.class_names[index]
    fruit_type, visible_condition = map_research_label(label)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return ConditionModelEvidence(
        model_id=model.model_id,
        model_family=model.model_family,
        model_version=model.model_version,
        runtime=model.runtime,
        input_sha256=input_hash,
        predicted_research_label=label,
        fruit_type=fruit_type,
        visible_condition=visible_condition,
        confidence=float(probabilities[index]),
        class_probabilities={
            name: float(probability)
            for name, probability in zip(model.class_names, probabilities)
        },
        preprocessing_version=model.contract.version,
        model_artifact_fingerprint=model.artifact_fingerprint,
        inference_ms=elapsed_ms,
    )
