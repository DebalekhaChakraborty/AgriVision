"""Train the single registered V2 Phase 1 baseline experiment."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.models.cnn_baseline import CNNBaseline
from v2.src.training.utils import (
    environment_record,
    git_record,
    load_config,
    resolve_device,
    resolve_repo_path,
    run_epoch,
    save_json,
    set_global_seed,
    sha256_file,
    utc_timestamp,
)
from v2.src.visualization.plots import plot_training_curves


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_model(model_config: dict[str, Any]) -> CNNBaseline:
    return CNNBaseline(
        image_size=int(model_config["input_size"]),
        input_channels=int(model_config["input_channels"]),
        conv_channels=[int(value) for value in model_config["conv_channels"]],
        kernel_size=int(model_config["kernel_size"]),
        pool_size=int(model_config["pool_size"]),
        dense_units=int(model_config["dense_units"]),
        num_classes=int(model_config["num_classes"]),
    )


def train(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    training_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]
    output_config = config["outputs"]

    seed = int(training_config["seed"])
    set_global_seed(seed)
    device = resolve_device(str(training_config["device"]))
    dataset_root = resolve_repo_path(REPOSITORY_ROOT, str(data_config["root"]))

    train_dataset = FruitFreshnessDataset(
        dataset_root,
        str(data_config["train_split"]),
        image_size=int(model_config["input_size"]),
    )
    validation_dataset = FruitFreshnessDataset(
        dataset_root,
        str(data_config["validation_split"]),
        image_size=int(model_config["input_size"]),
    )
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError("Train and validation class mappings differ.")

    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    if str(training_config["optimizer"]).lower() != "adam":
        raise ValueError("Phase 1 configuration requires the Adam optimizer.")
    learning_rate = float(training_config["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = resolve_repo_path(REPOSITORY_ROOT, output_config["checkpoint"])
    mapping_path = resolve_repo_path(REPOSITORY_ROOT, output_config["class_mapping"])
    history_path = resolve_repo_path(REPOSITORY_ROOT, output_config["history"])
    metadata_path = resolve_repo_path(REPOSITORY_ROOT, output_config["metadata"])
    training_curve_path = resolve_repo_path(
        REPOSITORY_ROOT, output_config["training_curve"]
    )
    validation_curve_path = resolve_repo_path(
        REPOSITORY_ROOT, output_config["validation_curve"]
    )

    save_json(train_dataset.class_to_idx, mapping_path)
    started_at = utc_timestamp()
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Trainable parameters: {model.trainable_parameters:,}")

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_record)

        improved = validation_metrics["accuracy"] > best_validation_accuracy or (
            validation_metrics["accuracy"] == best_validation_accuracy
            and validation_metrics["loss"] < best_validation_loss
        )
        if improved:
            best_epoch = epoch
            best_validation_accuracy = validation_metrics["accuracy"]
            best_validation_loss = validation_metrics["loss"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "experiment_id": config["experiment"]["id"],
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "class_to_idx": train_dataset.class_to_idx,
                    "model_config": model_config,
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                    "seed": seed,
                },
                checkpoint_path,
            )

        print(
            "Epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f} "
            "train_acc={train_accuracy:.4f} val_loss={validation_loss:.4f} "
            "val_acc={validation_accuracy:.4f}{marker}".format(
                epochs=training_config["epochs"],
                marker=" *" if improved else "",
                **epoch_record,
            ),
            flush=True,
        )

    save_json(history, history_path)
    plot_training_curves(history, training_curve_path, validation_curve_path)
    completed_at = utc_timestamp()
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "objective": config["experiment"]["objective"],
        "status": "training_complete_test_closed",
        "started_at": started_at,
        "completed_at": completed_at,
        "environment": environment_record(device),
        "repository": git_record(REPOSITORY_ROOT),
        "configuration_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "configuration": config,
        "dataset": {
            "name": data_config["dataset_name"],
            "author": data_config["dataset_author"],
            "version": data_config["dataset_version"],
            "location": str(dataset_root),
            "split_policy": data_config["split_policy"],
            "split_report_sha256": data_config["split_report_sha256"],
            "class_mapping": train_dataset.class_to_idx,
            "train_counts": train_dataset.class_counts(),
            "validation_counts": validation_dataset.class_counts(),
            "train_total": len(train_dataset),
            "validation_total": len(validation_dataset),
        },
        "model": {
            "architecture": "CNNBaseline",
            "trainable_parameters": model.trainable_parameters,
            "configuration": model_config,
        },
        "training": {
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": training_config["optimizer"],
            "loss": training_config["loss"],
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(checkpoint_path),
        },
        "artifacts": {
            "class_mapping": str(mapping_path.relative_to(REPOSITORY_ROOT)),
            "history": str(history_path.relative_to(REPOSITORY_ROOT)),
            "training_curve": str(training_curve_path.relative_to(REPOSITORY_ROOT)),
            "validation_curve": str(validation_curve_path.relative_to(REPOSITORY_ROOT)),
        },
        "test_accessed": False,
    }
    save_json(metadata, metadata_path)
    print(
        f"Selected epoch {best_epoch}: validation_accuracy="
        f"{best_validation_accuracy:.4f}, validation_loss={best_validation_loss:.4f}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="v2/configs/baseline_cnn.yaml",
        help="Configuration path relative to the repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(REPOSITORY_ROOT, args.config)
    train(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
