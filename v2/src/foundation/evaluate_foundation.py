"""Extract test embeddings once and evaluate a selected Phase 3A linear probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.foundation.extract_features import peak_rss_mb, synchronize
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.foundation.registry import build_foundation_encoder
from v2.src.training.utils import (
    load_config,
    resolve_device,
    resolve_repo_path,
    save_json,
    set_global_seed,
    sha256_file,
    utc_timestamp,
)
from v2.src.visualization.transfer_plots import plot_transfer_confusion_matrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def evaluate_foundation(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_config = config["outputs"]
    metadata_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["metadata"])
    )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("test_accessed"):
        raise RuntimeError("Test evaluation is already recorded for this experiment.")
    if metadata["status"] != "probe_complete_test_closed":
        raise RuntimeError("Validation-selected probe metadata is not test-ready.")

    extraction = config["extraction"]
    data_config = config["data"]
    probe_config = config["probe"]
    test_split = str(data_config["test_split"])
    test_cache_path = resolve_repo_path(
        REPOSITORY_ROOT, str(extraction["cache_files"][test_split])
    )
    if test_cache_path.exists():
        raise FileExistsError(
            f"Refusing an unregistered repeated test cache: {test_cache_path}"
        )
    seed = int(probe_config["seed"])
    set_global_seed(seed)
    device = resolve_device(str(probe_config["device"]))

    checkpoint_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["probe_checkpoint"])
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != metadata["checkpoint"]["sha256"]:
        raise ValueError("Probe checkpoint hash differs from training metadata.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["encoder_revision"] != config["encoder"]["revision"]:
        raise ValueError("Probe checkpoint and encoder revision differ.")
    probe = LinearProbe(
        int(checkpoint["embedding_dim"]), int(checkpoint["num_classes"])
    ).to(device)
    probe.load_state_dict(checkpoint["model_state_dict"])
    probe.eval()
    encoder = build_foundation_encoder(config["encoder"], device)
    encoder_metadata = encoder.runtime_metadata()
    if encoder_metadata["checkpoint_file"]["sha256"] != metadata["encoder"][
        "checkpoint_file"
    ]["sha256"]:
        raise ValueError("Evaluation encoder file differs from development extraction.")

    dataset_root = resolve_repo_path(REPOSITORY_ROOT, str(data_config["root"]))
    test_dataset = FruitFreshnessDataset(
        dataset_root,
        test_split,
        image_size=int(config["encoder"]["input_resolution"]),
        transform=encoder.preprocess,
    )
    if test_dataset.class_to_idx != checkpoint["class_mapping"]:
        raise ValueError("Probe and test class mappings differ.")
    test_loader = create_data_loader(
        test_dataset,
        batch_size=int(extraction["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(extraction["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    predicted_labels: list[int] = []
    total_loss = 0.0
    total_samples = 0
    encoder_seconds = 0.0
    probe_seconds = 0.0
    full_forward_seconds = 0.0
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            synchronize(device)
            full_started = time.perf_counter()
            encoder_started = time.perf_counter()
            embeddings = encoder.encode(images)
            synchronize(device)
            encoder_seconds += time.perf_counter() - encoder_started
            embeddings = functional.normalize(embeddings.float(), p=2, dim=1)
            synchronize(device)
            probe_started = time.perf_counter()
            logits = probe(embeddings)
            synchronize(device)
            probe_seconds += time.perf_counter() - probe_started
            full_forward_seconds += time.perf_counter() - full_started
            loss = criterion(logits, labels_device)
            predictions = logits.argmax(dim=1)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.long().cpu())
            predicted_labels.extend(predictions.cpu().tolist())
    wall_seconds = time.perf_counter() - wall_started
    embeddings = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    sample_ids = test_dataset.relative_paths()
    expected_labels = torch.tensor(
        [label for _, label in test_dataset.samples], dtype=torch.long
    )
    if not torch.equal(labels_tensor, expected_labels):
        raise RuntimeError("Test feature order became detached from labels.")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError("Test embeddings are not L2-normalized.")

    cache_identity = {
        "experiment_id": config["experiment"]["id"],
        "dataset_split_hash": data_config["split_report_sha256"],
        "encoder_checkpoint": config["encoder"]["checkpoint_id"],
        "encoder_revision": config["encoder"]["revision"],
        "embedding_dim": int(config["encoder"]["embedding_dim"]),
        "preprocessing_identity": config["encoder"]["preprocessing_identity"],
        "l2_normalized": True,
    }
    test_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "created_at": utc_timestamp(),
            "split": test_split,
            "sample_ids": sample_ids,
            "labels": labels_tensor,
            "embeddings": embeddings,
            "class_mapping": test_dataset.class_to_idx,
            "identity": cache_identity,
        },
        test_cache_path,
    )

    true_labels = labels_tensor.tolist()
    class_names = list(test_dataset.class_names)
    numeric_labels = list(range(len(class_names)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=numeric_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=numeric_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(
        np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))
    )
    evaluated_at = utc_timestamp()
    classification_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["classification_report"])
    )
    save_json(report, classification_path)
    confusion_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["confusion_matrix"])
    )
    plot_transfer_confusion_matrix(
        matrix,
        class_names,
        confusion_path,
        f"{config['experiment']['plot_title']}: frozen V1 test split",
    )
    metrics = {
        "experiment_id": config["experiment"]["id"],
        "evaluated_at": evaluated_at,
        "single_test_pass": True,
        "split": test_split,
        "samples": total_samples,
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "classification_report": report,
        "classification_report_path": str(
            classification_path.relative_to(REPOSITORY_ROOT)
        ),
        "confusion_matrix": matrix.tolist(),
        "class_mapping": test_dataset.class_to_idx,
        "encoder": encoder_metadata,
        "probe": {
            "architecture": "nn.Linear(embedding_dim, 6)",
            "trainable_parameters": probe.trainable_parameters,
            "selected_epoch": int(checkpoint["epoch"]),
        },
        "timing": {
            "scope": "Model forward only after preprocessing; includes L2 normalization in the full path.",
            "device": str(device),
            "test_loop_wall_seconds": wall_seconds,
            "encoder_forward_seconds": encoder_seconds,
            "encoder_forward_ms_per_image": encoder_seconds * 1000.0 / total_samples,
            "probe_forward_seconds": probe_seconds,
            "probe_forward_ms_per_image": probe_seconds * 1000.0 / total_samples,
            "image_representation_probe_ms_per_image": full_forward_seconds
            * 1000.0
            / total_samples,
            "peak_process_rss_mb": peak_rss_mb(),
        },
        "probe_checkpoint": {
            "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "test_embedding_cache": {
            "path": str(test_cache_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(test_cache_path),
            "size_bytes": test_cache_path.stat().st_size,
        },
    }
    metrics_path = resolve_repo_path(
        REPOSITORY_ROOT, str(output_config["test_metrics"])
    )
    save_json(metrics, metrics_path)
    metadata["status"] = "complete"
    metadata["test_accessed"] = True
    metadata["test_evaluation"] = {
        "evaluated_at": evaluated_at,
        "samples": total_samples,
        "loss": metrics["loss"],
        "accuracy": accuracy,
        "metrics_path": str(metrics_path.relative_to(REPOSITORY_ROOT)),
        "classification_report_path": str(
            classification_path.relative_to(REPOSITORY_ROOT)
        ),
        "confusion_matrix_path": str(confusion_path.relative_to(REPOSITORY_ROOT)),
        "test_embedding_cache": metrics["test_embedding_cache"],
        "timing": metrics["timing"],
    }
    save_json(metadata, metadata_path)
    print(
        f"Test samples={total_samples}, loss={metrics['loss']:.4f}, "
        f"accuracy={accuracy:.4f}, full_forward_ms_per_image="
        f"{metrics['timing']['image_representation_probe_ms_per_image']:.3f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_foundation(resolve_repo_path(REPOSITORY_ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
