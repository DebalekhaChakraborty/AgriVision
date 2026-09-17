"""Evaluate one validation-selected Phase 2 checkpoint on the frozen test set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.datasets.transfer_transforms import build_transfer_transform
from v2.src.models.transfer_registry import build_transfer_model
from v2.src.training.utils import (
    load_config,
    resolve_device,
    resolve_repo_path,
    save_json,
    sha256_file,
    utc_timestamp,
)
from v2.src.visualization.transfer_plots import plot_transfer_confusion_matrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_transfer(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_config = config["outputs"]
    metadata_path = resolve_repo_path(REPOSITORY_ROOT, output_config["metadata"])
    if not metadata_path.is_file():
        raise FileNotFoundError("Training metadata is missing; train before evaluation.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("test_accessed"):
        raise RuntimeError("Test evaluation is already recorded for this experiment.")
    if metadata["experiment_id"] != config["experiment"]["id"]:
        raise ValueError("Configuration and metadata experiment IDs differ.")

    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    preprocessing = config["preprocessing"]
    device = resolve_device(str(training_config["device"]))
    dataset_root = resolve_repo_path(REPOSITORY_ROOT, str(data_config["root"]))
    test_dataset = FruitFreshnessDataset(
        dataset_root,
        str(data_config["test_split"]),
        image_size=int(preprocessing["image_size"]),
        transform=build_transfer_transform("test", preprocessing),
    )
    test_loader = create_data_loader(
        test_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        seed=int(training_config["seed"]),
        num_workers=int(training_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = resolve_repo_path(REPOSITORY_ROOT, output_config["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != metadata["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs from the training metadata.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["class_to_idx"] != test_dataset.class_to_idx:
        raise ValueError("Checkpoint and test class mappings differ.")
    model = build_transfer_model(model_config, load_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    total_loss = 0.0
    total_samples = 0
    model_inference_seconds = 0.0
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _synchronize(device)
            inference_started = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            model_inference_seconds += time.perf_counter() - inference_started
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())

    ordered_names = list(test_dataset.class_names)
    labels = list(range(len(ordered_names)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=ordered_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))
    metrics = {
        "experiment_id": config["experiment"]["id"],
        "evaluated_at": utc_timestamp(),
        "split": data_config["test_split"],
        "single_test_pass": True,
        "samples": total_samples,
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_mapping": test_dataset.class_to_idx,
        "parameters": {
            "total": model.total_parameters,
            "trainable": model.trainable_parameters,
        },
        "timing": {
            "training_seconds": metadata["training"]["duration_seconds"],
            "model_inference_seconds": model_inference_seconds,
            "inference_seconds_per_image": model_inference_seconds / total_samples,
            "inference_timing_scope": "Model forward pass only; image decoding and transforms excluded.",
            "device": str(device),
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": checkpoint_sha256,
            "selected_epoch": int(checkpoint["epoch"]),
        },
    }
    metrics_path = resolve_repo_path(REPOSITORY_ROOT, output_config["test_metrics"])
    confusion_path = resolve_repo_path(
        REPOSITORY_ROOT, output_config["confusion_matrix"]
    )
    plot_transfer_confusion_matrix(
        matrix,
        ordered_names,
        confusion_path,
        title=f"{config['experiment']['plot_title']}: frozen V1 test split",
    )
    save_json(metrics, metrics_path)
    metadata["status"] = "complete"
    metadata["test_accessed"] = True
    metadata["test_evaluation"] = {
        "evaluated_at": metrics["evaluated_at"],
        "samples": total_samples,
        "loss": metrics["loss"],
        "accuracy": accuracy,
        "inference_seconds_per_image": metrics["timing"]["inference_seconds_per_image"],
        "metrics_path": str(metrics_path.relative_to(REPOSITORY_ROOT)),
        "confusion_matrix_path": str(confusion_path.relative_to(REPOSITORY_ROOT)),
    }
    save_json(metadata, metadata_path)
    print(
        f"Test samples={total_samples}, loss={metrics['loss']:.4f}, "
        f"accuracy={accuracy:.4f}, "
        f"inference_ms_per_image={metrics['timing']['inference_seconds_per_image'] * 1000:.3f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_transfer(resolve_repo_path(REPOSITORY_ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
