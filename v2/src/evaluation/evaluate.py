"""Evaluate the validation-selected Phase 1 checkpoint once on frozen V1 test data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.models.cnn_baseline import CNNBaseline
from v2.src.training.utils import (
    load_config,
    resolve_device,
    resolve_repo_path,
    save_json,
    sha256_file,
    utc_timestamp,
)
from v2.src.visualization.plots import plot_confusion_matrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _build_model(model_config: dict[str, Any]) -> CNNBaseline:
    return CNNBaseline(
        image_size=int(model_config["input_size"]),
        input_channels=int(model_config["input_channels"]),
        conv_channels=[int(value) for value in model_config["conv_channels"]],
        kernel_size=int(model_config["kernel_size"]),
        pool_size=int(model_config["pool_size"]),
        dense_units=int(model_config["dense_units"]),
        num_classes=int(model_config["num_classes"]),
    )


def evaluate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_config = config["outputs"]
    metadata_path = resolve_repo_path(REPOSITORY_ROOT, output_config["metadata"])
    if not metadata_path.is_file():
        raise FileNotFoundError("Training metadata is missing; train before evaluation.")

    import json

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("test_accessed"):
        raise RuntimeError("Test evaluation is already recorded for this experiment.")

    training_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]
    device = resolve_device(str(training_config["device"]))
    dataset_root = resolve_repo_path(REPOSITORY_ROOT, str(data_config["root"]))
    test_dataset = FruitFreshnessDataset(
        dataset_root,
        str(data_config["test_split"]),
        image_size=int(model_config["input_size"]),
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["class_to_idx"] != test_dataset.class_to_idx:
        raise ValueError("Checkpoint and test class mappings differ.")
    model = _build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
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
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "selected_epoch": int(checkpoint["epoch"]),
        },
    }

    metrics_path = resolve_repo_path(REPOSITORY_ROOT, output_config["test_metrics"])
    confusion_path = resolve_repo_path(
        REPOSITORY_ROOT, output_config["confusion_matrix"]
    )
    plot_confusion_matrix(matrix, ordered_names, confusion_path)
    save_json(metrics, metrics_path)

    metadata["status"] = "complete"
    metadata["test_accessed"] = True
    metadata["test_evaluation"] = {
        "evaluated_at": metrics["evaluated_at"],
        "samples": total_samples,
        "loss": metrics["loss"],
        "accuracy": accuracy,
        "metrics_path": str(metrics_path.relative_to(REPOSITORY_ROOT)),
        "confusion_matrix_path": str(confusion_path.relative_to(REPOSITORY_ROOT)),
    }
    save_json(metadata, metadata_path)
    print(
        f"Test samples={total_samples}, loss={metrics['loss']:.4f}, "
        f"accuracy={accuracy:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/baseline_cnn.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(resolve_repo_path(REPOSITORY_ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
