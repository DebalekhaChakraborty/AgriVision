"""Checkpoint-guarded zero-adaptation evaluation and analysis for Experiment 013."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from v2.src.datasets.fruit_dataset import CLASS_NAMES, default_transform
from v2.src.datasets.transfer_transforms import build_transfer_transform
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.foundation.prompt_registry import encode_class_prototypes, load_prompt_registry
from v2.src.foundation.registry import build_foundation_encoder
from v2.src.foundation.similarity_classifier import build_zero_shot_model
from v2.src.models.transfer_registry import build_transfer_model
from v2.src.training.train import build_model as build_cnn
from v2.src.training.utils import load_config, resolve_device, save_json, set_global_seed, sha256_file, utc_timestamp


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "v2/results/experiment_013_cross_domain"
CACHE = ROOT / "v2/cache/experiment_013_cross_domain"
CLASS_NAMES = tuple(CLASS_NAMES)
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}

SYSTEMS = {
    "custom_cnn": {"condition": "013-A", "name": "Custom CNN", "kind": "cnn", "config": "v2/configs/baseline_cnn.yaml", "source": "v2/results/baseline_test_metrics.json"},
    "resnet50": {"condition": "013-B", "name": "ResNet50", "kind": "transfer", "config": "v2/configs/resnet50.yaml", "source": "v2/results/experiment_002/metrics.json"},
    "efficientnet_b0": {"condition": "013-C", "name": "EfficientNet-B0", "kind": "transfer", "config": "v2/configs/efficientnet.yaml", "source": "v2/results/experiment_003/metrics.json"},
    "mobilenetv3_large": {"condition": "013-D", "name": "MobileNetV3-Large", "kind": "transfer", "config": "v2/configs/mobilenetv3.yaml", "source": "v2/results/experiment_004/metrics.json"},
    "dinov2_linear_probe": {"condition": "013-E", "name": "DINOv2 + linear probe", "kind": "probe", "config": "v2/configs/dinov2_linear_probe.yaml", "source": "v2/results/experiment_005/test_metrics.json"},
    "clip_linear_probe": {"condition": "013-F", "name": "CLIP + linear probe", "kind": "probe", "config": "v2/configs/clip_linear_probe.yaml", "source": "v2/results/experiment_006/test_metrics.json"},
    "siglip2_linear_probe": {"condition": "013-G", "name": "SigLIP2 + linear probe", "kind": "probe", "config": "v2/configs/siglip2_linear_probe.yaml", "source": "v2/results/experiment_007/test_metrics.json"},
    "clip_strict_zero_shot": {"condition": "013-H", "name": "CLIP strict P1 zero-shot", "kind": "zero", "config": "v2/configs/clip_zero_shot.yaml", "source": "v2/results/experiment_008/strict_zero_shot_test_metrics.json"},
    "siglip2_strict_zero_shot": {"condition": "013-I", "name": "SigLIP2 strict P1 zero-shot", "kind": "zero", "config": "v2/configs/siglip2_zero_shot.yaml", "source": "v2/results/experiment_009/strict_zero_shot_test_metrics.json"},
}


def path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(value: str | Path) -> Any:
    with path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ManifestDataset(Dataset):
    def __init__(self, manifest: dict[str, Any], root: Path, transform: Callable) -> None:
        self.records = manifest["images"]
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        with Image.open(self.root / record["identifier"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_TO_IDX[record["class"]], index


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path(config["outputs"]["manifest"])
    hash_line = path(config["outputs"]["manifest_hash"]).read_text(encoding="utf-8").split()[0]
    actual = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    if actual != hash_line or manifest["status"] != "frozen_before_model_evaluation" or len(manifest["images"]) != 1200:
        raise RuntimeError("Frozen external manifest identity or count changed.")
    return manifest


def verify_checkpoints(config: dict[str, Any]) -> None:
    output = RESULTS / "checkpoint_verification.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = verify_manifest(config)
    records: dict[str, Any] = {}
    failures: list[str] = []
    for system_id, spec in SYSTEMS.items():
        model_config = load_config(path(spec["config"]))
        source = read_json(spec["source"])
        kind = spec["kind"]
        record: dict[str, Any] = {
            "condition": spec["condition"], "name": spec["name"], "configuration": spec["config"],
            "configuration_sha256": sha256_file(path(spec["config"])), "source_metrics": spec["source"],
            "source_experiment_id": source.get("experiment_id"), "preprocessing_identity": None,
        }
        if kind in {"cnn", "transfer"}:
            checkpoint_path = path(model_config["outputs"]["checkpoint"])
            expected = source["checkpoint"]["sha256"]
            actual = sha256_file(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            record.update({
                "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256_expected": expected,
                "checkpoint_sha256_actual": actual, "checkpoint_experiment_id": checkpoint["experiment_id"],
                "preprocessing_identity": checkpoint.get("preprocessing", checkpoint.get("model_config")),
            })
            passed = actual == expected and checkpoint["experiment_id"] == model_config["experiment"]["id"] and checkpoint["class_to_idx"] == CLASS_TO_IDX
        elif kind == "probe":
            checkpoint_path = path(model_config["outputs"]["probe_checkpoint"])
            expected = source["probe_checkpoint"]["sha256"]
            actual = sha256_file(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            encoder_expected = source["encoder"]["checkpoint_file"]["sha256"]
            encoder_actual = sha256_file(Path(source["encoder"]["checkpoint_file"]["path"]))
            record.update({
                "probe_checkpoint": str(checkpoint_path.relative_to(ROOT)), "probe_checkpoint_sha256_expected": expected,
                "probe_checkpoint_sha256_actual": actual, "encoder_checkpoint_sha256_expected": encoder_expected,
                "encoder_checkpoint_sha256_actual": encoder_actual, "encoder_revision": model_config["encoder"]["revision"],
                "checkpoint_encoder_revision": checkpoint["encoder_revision"],
                "preprocessing_identity": model_config["encoder"]["preprocessing_identity"],
            })
            passed = actual == expected and encoder_actual == encoder_expected and checkpoint["experiment_id"] == model_config["experiment"]["id"] and checkpoint["encoder_revision"] == model_config["encoder"]["revision"] and checkpoint["class_mapping"] == CLASS_TO_IDX
        else:
            registry_path = path(model_config["prompts"]["registry_path"])
            expected_registry = model_config["prompts"]["registry_sha256"]
            source_metadata = read_json(f"v2/results/experiment_{int(model_config['experiment']['number']):03d}/metadata.json")
            checkpoint_path = Path(source_metadata["model"]["checkpoint_file"]["path"])
            checkpoint_actual = sha256_file(checkpoint_path)
            record.update({
                "encoder_checkpoint": str(checkpoint_path), "encoder_checkpoint_sha256_expected": model_config["model"]["checkpoint_sha256"],
                "encoder_checkpoint_sha256_actual": checkpoint_actual, "encoder_revision": model_config["model"]["revision"],
                "prompt_registry": model_config["prompts"]["registry_path"], "prompt_registry_sha256_expected": expected_registry,
                "prompt_registry_sha256_actual": sha256_file(registry_path), "prompt_family": model_config["prompts"]["strict_family"],
                "preprocessing_identity": model_config["model"]["preprocessing_identity"], "classifier_training": False,
            })
            passed = checkpoint_actual == model_config["model"]["checkpoint_sha256"] and sha256_file(registry_path) == expected_registry and model_config["prompts"]["strict_family"] == "P1"
        record["status"] = "PASS" if passed else "FAIL"
        records[system_id] = record
        if not passed:
            failures.append(system_id)
    payload = {
        "schema_version": 1, "experiment_id": config["experiment"]["id"], "verified_at": utc_timestamp(),
        "external_manifest_sha256": sha256_file(path(config["outputs"]["manifest"])),
        "all_nine_conditions_passed": not failures, "failed_conditions": failures, "systems": records,
        "status": "PASS" if not failures else "BLOCKED",
    }
    save_json(payload, output)
    if failures:
        raise RuntimeError(f"Checkpoint verification failed: {failures}")
    print("Verified frozen identities for all nine systems.")


def run_batches(model_fn: Callable[[torch.Tensor], torch.Tensor], dataset: ManifestDataset, device: torch.device, score_kind: str) -> tuple[list[dict[str, Any]], torch.Tensor, float, float]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    embeddings_or_logits: list[torch.Tensor] = []
    forward_seconds = 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device)
            start = time.perf_counter()
            output = model_fn(images)
            forward_seconds += time.perf_counter() - start
            output_cpu = output.detach().cpu().float()
            embeddings_or_logits.append(output_cpu)
            top = torch.topk(output_cpu, k=2, dim=1)
            predictions = top.indices[:, 0]
            for label, prediction, index, best, second in zip(labels.tolist(), predictions.tolist(), indices.tolist(), top.values[:, 0].tolist(), top.values[:, 1].tolist()):
                rows.append({"index": index, "true_index": label, "predicted_index": prediction, "score": best, "margin": best - second, "score_kind": score_kind})
    return rows, torch.cat(embeddings_or_logits), forward_seconds, time.perf_counter() - wall_start


def system_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true = [item["true_index"] for item in rows]
    pred = [item["predicted_index"] for item in rows]
    labels = list(range(6))
    report = classification_report(true, pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(true, pred, labels=labels)
    return {
        "samples": len(rows), "accuracy": float(np.mean(np.asarray(true) == np.asarray(pred))),
        "macro_precision": float(report["macro avg"]["precision"]), "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]), "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]), "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CLASS_NAMES}, "confusion_matrix": matrix.tolist(),
    }


def semantic_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({"condition_error": 0, "fruit_identity_error": 0, "combined_error": 0})
    pairs = Counter()
    for item in rows:
        true_name, pred_name = CLASS_NAMES[item["true_index"]], CLASS_NAMES[item["predicted_index"]]
        if true_name == pred_name:
            continue
        true_condition, true_fruit = true_name.split("_", 1)
        pred_condition, pred_fruit = pred_name.split("_", 1)
        category = "condition_error" if true_fruit == pred_fruit else "fruit_identity_error" if true_condition == pred_condition else "combined_error"
        counts[category] += 1
        pairs[f"{true_name} -> {pred_name}"] += 1
    errors = sum(counts.values())
    return {"total_errors": errors, "counts": dict(counts), "percent_of_errors": {key: value * 100 / errors if errors else 0.0 for key, value in counts.items()}, "top_confusion_pairs": dict(pairs.most_common(15))}


def plot_confusion(matrix: list[list[int]], system_name: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Ground truth", title=f"External benchmark — {system_name}")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    threshold = np.asarray(matrix).max() / 2
    for row in range(6):
        for col in range(6):
            ax.text(col, row, matrix[row][col], ha="center", va="center", color="white" if matrix[row][col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def infer_one(system_id: str, spec: dict[str, Any], manifest: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_config(path(spec["config"]))
    kind = spec["kind"]
    embeddings = None
    extra: dict[str, Any] = {}
    if kind == "cnn":
        transform = default_transform(int(cfg["model"]["input_size"]))
        dataset = ManifestDataset(manifest, path("v2/data/external/sultana_2022/Original Image"), transform)
        model = build_cnn(cfg["model"]).to(device)
        checkpoint = torch.load(path(cfg["outputs"]["checkpoint"]), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        rows, _, forward, wall = run_batches(model, dataset, device, "uncalibrated_logit")
    elif kind == "transfer":
        transform = build_transfer_transform("test", cfg["preprocessing"])
        dataset = ManifestDataset(manifest, path("v2/data/external/sultana_2022/Original Image"), transform)
        model = build_transfer_model(cfg["model"], load_pretrained=False).to(device)
        checkpoint = torch.load(path(cfg["outputs"]["checkpoint"]), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        rows, _, forward, wall = run_batches(model, dataset, device, "uncalibrated_logit")
    elif kind == "probe":
        encoder = build_foundation_encoder(cfg["encoder"], device)
        dataset = ManifestDataset(manifest, path("v2/data/external/sultana_2022/Original Image"), encoder.preprocess)
        raw_rows, embeddings, encoder_forward, wall = run_batches(lambda images: F.normalize(encoder.encode(images).float(), p=2, dim=1), dataset, device, "embedding_component_not_a_score")
        checkpoint = torch.load(path(cfg["outputs"]["probe_checkpoint"]), map_location=device, weights_only=True)
        probe = LinearProbe(int(cfg["encoder"]["embedding_dim"]), 6).to(device)
        probe.load_state_dict(checkpoint["model_state_dict"]); probe.eval()
        start = time.perf_counter()
        logits = probe(embeddings.to(device)).detach().cpu()
        probe_forward = time.perf_counter() - start
        top = torch.topk(logits, 2, dim=1)
        rows = []
        for index, (label, prediction, best, second) in enumerate(zip([item["true_index"] for item in raw_rows], top.indices[:, 0].tolist(), top.values[:, 0].tolist(), top.values[:, 1].tolist())):
            rows.append({"index": index, "true_index": label, "predicted_index": prediction, "score": best, "margin": best-second, "score_kind": "uncalibrated_linear_probe_logit"})
        forward = encoder_forward + probe_forward
        extra = {"encoder_forward_seconds": encoder_forward, "probe_forward_seconds": probe_forward}
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "labels": torch.tensor([row["true_index"] for row in rows]), "sample_ids": [record["identifier"] for record in manifest["images"]]}, CACHE / f"{system_id}_external.pt")
    else:
        model = build_zero_shot_model(cfg["model"], device)
        registry = load_prompt_registry(path(cfg["prompts"]["registry_path"]), cfg["prompts"]["registry_sha256"])
        prototypes, prototype_meta = encode_class_prototypes(model, registry, "P1")
        dataset = ManifestDataset(manifest, path("v2/data/external/sultana_2022/Original Image"), model.preprocess)
        def score(images: torch.Tensor) -> torch.Tensor:
            embedded = F.normalize(model.encode_images(images).float(), p=2, dim=1)
            return embedded @ prototypes.to(device).t()
        rows, _, forward, wall = run_batches(score, dataset, device, "cosine_similarity")
        extra = {"prompt_family": "P1", "prompt_registry_sha256": registry.sha256, "prototype_generation": prototype_meta, "classifier_training": False}
    metrics = system_metrics(rows)
    metrics.update({"condition": spec["condition"], "system": system_id, "display_name": spec["name"], "evaluated_at": utc_timestamp(), "single_external_pass": True, "zero_target_domain_adaptation": True, "timing": {"device": str(device), "model_forward_seconds": forward, "model_forward_ms_per_image": 1000 * forward / len(rows), "evaluation_loop_wall_seconds": wall, "scope": "Model forward after deterministic preprocessing; decoding/transforms excluded from model-forward time."}, **extra})
    plot_confusion(metrics["confusion_matrix"], spec["name"], RESULTS / f"confusion_matrix_{spec['condition'].lower().replace('-', '_')}_{system_id}.png")
    del dataset
    if "model" in locals(): del model
    if "encoder" in locals(): del encoder
    gc.collect()
    return rows, metrics


def infer_all(config: dict[str, Any]) -> None:
    verification = read_json(RESULTS / "checkpoint_verification.json")
    if verification["status"] != "PASS":
        raise RuntimeError("All checkpoint identities must pass before inference.")
    manifest = verify_manifest(config)
    ledger_path = RESULTS / "external_inference_access.json"
    metrics_path = RESULTS / "metrics_by_system.json"
    predictions_path = RESULTS / "predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Refusing to rerun completed external evaluation.")
    ledger = read_json(ledger_path) if ledger_path.exists() else {"schema_version": 1, "experiment_id": config["experiment"]["id"], "manifest_sha256": sha256_file(path(config["outputs"]["manifest"])), "started_at": utc_timestamp(), "completed_systems": []}
    CACHE.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config["evaluation"]["device"]))
    set_global_seed(int(config["evaluation"]["seed"]))
    all_rows: dict[str, list[dict[str, Any]]] = {}
    all_metrics: dict[str, Any] = {}
    for system_id, spec in SYSTEMS.items():
        cache_result = CACHE / f"{system_id}_result.json"
        if system_id in ledger["completed_systems"]:
            cached = read_json(cache_result); all_rows[system_id] = cached["rows"]; all_metrics[system_id] = cached["metrics"]
            continue
        print(f"Running the sole external pass for {spec['condition']} {spec['name']}...", flush=True)
        rows, metrics = infer_one(system_id, spec, manifest, device)
        save_json({"rows": rows, "metrics": metrics}, cache_result)
        ledger["completed_systems"].append(system_id)
        ledger["last_completed_at"] = utc_timestamp()
        save_json(ledger, ledger_path)
        all_rows[system_id], all_metrics[system_id] = rows, metrics
        print(f"{system_id}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}", flush=True)
    fieldnames = ["condition", "system", "image_identifier", "ground_truth_class", "predicted_class", "correct", "fruit_identity_correct", "freshness_condition_correct", "score", "margin", "score_kind", "score_warning"]
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for system_id, spec in SYSTEMS.items():
            for item in all_rows[system_id]:
                record = manifest["images"][item["index"]]; true_name = CLASS_NAMES[item["true_index"]]; pred_name = CLASS_NAMES[item["predicted_index"]]
                writer.writerow({"condition": spec["condition"], "system": system_id, "image_identifier": record["identifier"], "ground_truth_class": true_name, "predicted_class": pred_name, "correct": true_name == pred_name, "fruit_identity_correct": true_name.split("_",1)[1] == pred_name.split("_",1)[1], "freshness_condition_correct": true_name.split("_",1)[0] == pred_name.split("_",1)[0], "score": item["score"], "margin": item["margin"], "score_kind": item["score_kind"], "score_warning": "Uncalibrated and not comparable across systems."})
    save_json({"schema_version": 1, "experiment_id": config["experiment"]["id"], "manifest_sha256": ledger["manifest_sha256"], "systems": all_metrics}, metrics_path)
    save_json({"schema_version": 1, "systems": {system_id: semantic_errors(rows) for system_id, rows in all_rows.items()}}, RESULTS / "semantic_error_analysis.json")
    ledger.update({"completed_at": utc_timestamp(), "status": "COMPLETE_ONE_PASS_PER_SYSTEM", "external_passes_per_system": 1})
    save_json(ledger, ledger_path)


def source_summary(spec: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    source = read_json(spec["source"])
    report = source.get("classification_report", source.get("per_class"))
    weighted_f1 = source.get("weighted_f1")
    if weighted_f1 is None:
        weighted_f1 = source["classification_report"]["weighted avg"]["f1-score"]
    recalls = {name: float(report[name]["recall"]) for name in CLASS_NAMES}
    return float(source["accuracy"]), float(weighted_f1), recalls


def bootstrap_metrics(true: np.ndarray, pred: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    accuracies = np.empty(len(indices)); macro_f1 = np.empty(len(indices))
    for pos, sample in enumerate(indices):
        yt, yp = true[sample], pred[sample]
        accuracies[pos] = np.mean(yt == yp)
        matrix = np.bincount(yt * 6 + yp, minlength=36).reshape(6, 6)
        tp = np.diag(matrix).astype(float); fp = matrix.sum(0) - tp; fn = matrix.sum(1) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros(6), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(6), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(6), where=(precision + recall) != 0)
        macro_f1[pos] = f1.mean()
    return accuracies, macro_f1


def summarize_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {"mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)), "q05": float(np.quantile(array,.05)), "q95": float(np.quantile(array,.95)), "minimum": float(array.min()), "maximum": float(array.max())}
    return {"samples": len(records), "width": stats([r["width"] for r in records]), "height": stats([r["height"] for r in records]), "aspect_ratio": stats([r["aspect_ratio"] for r in records]), "brightness": stats([r["profile"]["brightness_mean"] for r in records]), "contrast": stats([r["profile"]["contrast_std"] for r in records]), "rgb": {channel: stats([r["profile"]["rgb_mean"][idx] for r in records]) for idx, channel in enumerate(("red","green","blue"))}, "formats": dict(Counter(str(r["format"]) for r in records))}


def scan_profile(root: Path) -> list[dict[str, Any]]:
    records = []
    for class_name in CLASS_NAMES:
        for file in sorted((root / "test" / class_name).iterdir()):
            if not file.is_file(): continue
            with Image.open(file) as image:
                fmt = image.format; rgb = image.convert("RGB"); width, height = rgb.size
                pixels = np.asarray(rgb.resize((64,64), Image.Resampling.BOX), dtype=float) / 255
                gray = .2126*pixels[:,:,0] + .7152*pixels[:,:,1] + .0722*pixels[:,:,2]
            records.append({"width":width,"height":height,"aspect_ratio":width/height,"format":fmt,"profile":{"brightness_mean":float(gray.mean()),"contrast_std":float(gray.std()),"rgb_mean":pixels.mean((0,1)).tolist()}})
    return records


def pca_plot() -> dict[str, Any]:
    models = [("DINOv2", "dinov2_linear_probe", "v2/cache/dinov2/test.pt"), ("SigLIP2", "siglip2_linear_probe", "v2/cache/siglip2/test.pt")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5)); result = {}
    rng = np.random.default_rng(42)
    for ax, (name, system_id, source_file) in zip(axes, models):
        source = torch.load(path(source_file), map_location="cpu", weights_only=False); external = torch.load(CACHE / f"{system_id}_external.pt", map_location="cpu", weights_only=False)
        selected = []
        labels = source["labels"].numpy()
        for class_index in range(6): selected.extend(rng.choice(np.where(labels == class_index)[0], 200, replace=False).tolist())
        src = source["embeddings"][selected].numpy(); ext = external["embeddings"].numpy(); combined = np.vstack([src, ext])
        reducer = PCA(n_components=2, random_state=42)
        projection = reducer.fit_transform(combined)
        ax.scatter(projection[:len(src),0], projection[:len(src),1], s=7, alpha=.35, label="Source test")
        ax.scatter(projection[len(src):,0], projection[len(src):,1], s=7, alpha=.35, label="External")
        ax.set(title=f"{name} representation", xlabel="PC1", ylabel="PC2"); ax.legend(markerscale=2)
        result[name.lower()] = {"source_samples":len(src),"external_samples":len(ext),"explained_variance_ratio":reducer.explained_variance_ratio_.tolist()}
    fig.suptitle("Descriptive PCA of frozen representations by domain"); fig.tight_layout(); fig.savefig(RESULTS / "pca_domain_shift.png", dpi=170); plt.close(fig)
    result["interpretation_boundary"] = "Exploratory descriptive visualization only; not used for selection or adaptation."
    return result


def analyze(config: dict[str, Any]) -> None:
    metrics = read_json(RESULTS / "metrics_by_system.json")["systems"]
    manifest = verify_manifest(config)
    rows: dict[str, list[dict[str, str]]] = {key: [] for key in SYSTEMS}
    with (RESULTS / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle): rows[row["system"]].append(row)
    gaps = {}; recall_shift = {}
    for system_id, spec in SYSTEMS.items():
        source_acc, source_f1, source_recalls = source_summary(spec); external = metrics[system_id]
        gaps[system_id] = {"condition":spec["condition"],"display_name":spec["name"],"source_accuracy":source_acc,"external_accuracy":external["accuracy"],"accuracy_gap_external_minus_source":external["accuracy"]-source_acc,"accuracy_retention":external["accuracy"]/source_acc,"source_weighted_f1":source_f1,"external_weighted_f1":external["weighted_f1"],"weighted_f1_gap_external_minus_source":external["weighted_f1"]-source_f1,"weighted_f1_retention":external["weighted_f1"]/source_f1}
        recall_shift[system_id] = {name:{"source_recall":source_recalls[name],"external_recall":external["per_class"][name]["recall"],"shift_external_minus_source":external["per_class"][name]["recall"]-source_recalls[name]} for name in CLASS_NAMES}
    save_json({"schema_version":1,"gap_definition":"external - source; negative is degradation","systems":gaps}, RESULTS / "domain_gap_analysis.json")
    save_json({"schema_version":1,"systems":recall_shift}, RESULTS / "class_recall_shift.json")

    true = np.asarray([CLASS_TO_IDX[record["class"]] for record in manifest["images"]]); class_indices = [np.where(true == c)[0] for c in range(6)]
    rng = np.random.default_rng(int(config["uncertainty"]["bootstrap_seed"])); samples = np.concatenate([rng.choice(index, size=(5000,len(index)), replace=True) for index in class_indices], axis=1)
    bootstrap = {}; boot_acc = {}
    for system_id in SYSTEMS:
        pred = np.asarray([CLASS_TO_IDX[row["predicted_class"]] for row in rows[system_id]])
        acc, mf1 = bootstrap_metrics(true, pred, samples); boot_acc[system_id] = acc
        bootstrap[system_id] = {"accuracy":{"estimate":metrics[system_id]["accuracy"],"lower":float(np.quantile(acc,.025)),"upper":float(np.quantile(acc,.975))},"macro_f1":{"estimate":metrics[system_id]["macro_f1"],"lower":float(np.quantile(mf1,.025)),"upper":float(np.quantile(mf1,.975))}}
    save_json({"schema_version":1,"seed":42,"resamples":5000,"method":"95% class-stratified bootstrap interval over the external evaluation sample.","systems":bootstrap}, RESULTS / "bootstrap_intervals.json")
    pairs = {}
    for left, right in config["uncertainty"]["paired_comparisons"]:
        delta = boot_acc[left] - boot_acc[right]
        pairs[f"{left}_minus_{right}"] = {"left":left,"right":right,"observed_accuracy_difference":metrics[left]["accuracy"]-metrics[right]["accuracy"],"mean_paired_bootstrap_accuracy_difference":float(delta.mean()),"lower":float(np.quantile(delta,.025)),"upper":float(np.quantile(delta,.975))}
    save_json({"schema_version":1,"seed":42,"resamples":5000,"method":"Paired 95% class-stratified bootstrap interval over the external evaluation sample.","comparisons":pairs}, RESULTS / "paired_bootstrap_comparisons.json")

    external_records = read_json(RESULTS / "external_dataset_audit.json")["records_for_manifest"]
    source_records = scan_profile(path(config["source_dataset"]["root"]))
    profile = {"schema_version":1,"source_test":summarize_profile(source_records),"external_originals":summarize_profile(external_records),"processing":"Decoded RGB; 64x64 BOX-resized descriptive summaries only; evaluation images were not altered.","aggregate_shift_not_causal_attribution":True}
    save_json(profile, RESULTS / "domain_profile.json")
    save_json(pca_plot(), RESULTS / "pca_analysis.json")
    print("Completed domain gaps, recall shifts, 5,000-resample intervals, paired comparisons, profile, and PCA.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", choices=("verify","infer","analyze")); parser.add_argument("--config", default="v2/configs/cross_domain_generalization.yaml"); args = parser.parse_args()
    config = load_config(path(args.config))
    {"verify":verify_checkpoints,"infer":infer_all,"analyze":analyze}[args.stage](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
