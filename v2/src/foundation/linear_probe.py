"""Train the identical validation-selected linear probe for one cached representation."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from v2.src.visualization.transfer_plots import plot_transfer_training_curves


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_embedding_cache(
    path: Path,
    expected_split: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    if payload["split"] != expected_split:
        raise ValueError(f"Expected {expected_split} cache, got {payload['split']}.")
    if identity["experiment_id"] != config["experiment"]["id"]:
        raise ValueError("Embedding cache belongs to another experiment.")
    if identity["dataset_split_hash"] != config["data"]["split_report_sha256"]:
        raise ValueError("Embedding cache uses a different dataset split.")
    if identity["encoder_revision"] != config["encoder"]["revision"]:
        raise ValueError("Embedding cache uses a different encoder revision.")
    if not identity.get("l2_normalized"):
        raise ValueError("Linear probes require L2-normalized embeddings.")
    if len(payload["sample_ids"]) != payload["embeddings"].shape[0]:
        raise ValueError("Cache sample identifiers and embeddings are misaligned.")
    if payload["labels"].shape[0] != payload["embeddings"].shape[0]:
        raise ValueError("Cache labels and embeddings are misaligned.")
    return payload


def probe_loader(
    payload: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(payload["embeddings"], payload["labels"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def train_linear_probe(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    probe_config = config["probe"]
    extraction = config["extraction"]
    output_config = config["outputs"]
    seed = int(probe_config["seed"])
    set_global_seed(seed)
    device = resolve_device(str(probe_config["device"]))
    extraction_metadata_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["extraction_metadata"])
    )
    with extraction_metadata_path.open("r", encoding="utf-8") as handle:
        extraction_metadata = json.load(handle)
    if extraction_metadata.get("test_accessed"):
        raise RuntimeError("Development extraction metadata unexpectedly accessed test.")

    train_path = resolve_repo_path(
        REPOSITORY_ROOT, str(extraction["cache_files"][config["data"]["train_split"]])
    )
    validation_path = resolve_repo_path(
        REPOSITORY_ROOT,
        str(extraction["cache_files"][config["data"]["validation_split"]]),
    )
    train_cache = load_embedding_cache(
        train_path, str(config["data"]["train_split"]), config
    )
    validation_cache = load_embedding_cache(
        validation_path, str(config["data"]["validation_split"]), config
    )
    if train_cache["class_mapping"] != validation_cache["class_mapping"]:
        raise ValueError("Train and validation cache class mappings differ.")
    if set(train_cache["sample_ids"]) & set(validation_cache["sample_ids"]):
        raise ValueError("Train and validation cache identifiers overlap.")

    batch_size = int(probe_config["batch_size"])
    train_loader = probe_loader(train_cache, batch_size, True, seed)
    validation_loader = probe_loader(validation_cache, batch_size, False, seed)
    model = LinearProbe(
        int(config["encoder"]["embedding_dim"]),
        int(config["probe"]["num_classes"]),
    ).to(device)
    if any(
        isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.ReLU))
        for module in model.modules()
    ):
        raise RuntimeError("The registered linear probe contains a forbidden layer.")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(probe_config["learning_rate"])
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["probe_checkpoint"])
    )
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite probe checkpoint: {checkpoint_path}")

    started_at = utc_timestamp()
    training_started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    for epoch in range(1, int(probe_config["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(record)
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
                    "embedding_dim": int(config["encoder"]["embedding_dim"]),
                    "num_classes": int(probe_config["num_classes"]),
                    "class_mapping": train_cache["class_mapping"],
                    "encoder_revision": config["encoder"]["revision"],
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
                epochs=probe_config["epochs"],
                marker=" *" if improved else "",
                **record,
            ),
            flush=True,
        )
    training_seconds = time.perf_counter() - training_started
    history_path = resolve_repo_path(REPOSITORY_ROOT, str(output_config["history"]))
    save_json(history, history_path)
    plot_transfer_training_curves(
        history,
        resolve_repo_path(REPOSITORY_ROOT, str(output_config["training_curve"])),
        resolve_repo_path(REPOSITORY_ROOT, str(output_config["validation_curve"])),
        str(config["experiment"]["plot_title"]),
    )
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "objective": config["experiment"]["objective"],
        "status": "probe_complete_test_closed",
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "environment": environment_record(device),
        "repository": git_record(REPOSITORY_ROOT),
        "configuration_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "configuration": config,
        "dataset": {
            "split_report_sha256": config["data"]["split_report_sha256"],
            "train_samples": int(train_cache["labels"].shape[0]),
            "validation_samples": int(validation_cache["labels"].shape[0]),
            "class_mapping": train_cache["class_mapping"],
        },
        "encoder": extraction_metadata["encoder"],
        "development_extraction": extraction_metadata["splits"],
        "embedding_caches": {
            "train": {
                "path": str(train_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(train_path),
            },
            "validation": {
                "path": str(validation_path.relative_to(REPOSITORY_ROOT)),
                "sha256": sha256_file(validation_path),
            },
        },
        "probe": {
            "architecture": "nn.Linear(embedding_dim, 6)",
            "embedding_dim": int(config["encoder"]["embedding_dim"]),
            "trainable_parameters": model.trainable_parameters,
            "optimizer": probe_config["optimizer"],
            "learning_rate": float(probe_config["learning_rate"]),
            "batch_size": batch_size,
            "epochs_completed": len(history),
            "seed": seed,
            "duration_seconds": training_seconds,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(checkpoint_path),
        },
        "test_accessed": False,
    }
    metadata_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["metadata"])
    )
    save_json(metadata, metadata_path)
    print(
        f"Selected epoch {best_epoch}: validation_accuracy="
        f"{best_validation_accuracy:.4f}, validation_loss={best_validation_loss:.4f}"
    )
    print(f"Probe training duration: {training_seconds:.3f} seconds")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_linear_probe(resolve_repo_path(REPOSITORY_ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
