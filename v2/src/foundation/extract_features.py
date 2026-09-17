"""Extract deterministic train/validation embeddings for one Phase 3A encoder."""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.foundation.base_encoder import BaseFoundationEncoder
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def extract_split(
    encoder: BaseFoundationEncoder,
    dataset: FruitFreshnessDataset,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cache_path: Path,
    cache_identity: dict[str, Any],
) -> dict[str, Any]:
    if cache_path.exists():
        raise FileExistsError(f"Refusing to overwrite embedding cache: {cache_path}")
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    encoder_forward_seconds = 0.0
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            synchronize(device)
            forward_started = time.perf_counter()
            embeddings = encoder.encode(images)
            synchronize(device)
            encoder_forward_seconds += time.perf_counter() - forward_started
            embeddings = functional.normalize(embeddings.float(), p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.long().cpu())
    wall_seconds = time.perf_counter() - wall_started
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sample_ids = dataset.relative_paths()
    expected_labels = torch.tensor(
        [label for _, label in dataset.samples], dtype=torch.long
    )
    if not torch.equal(labels, expected_labels):
        raise RuntimeError("Feature order became detached from dataset labels.")
    if embeddings.shape != (len(dataset), int(cache_identity["embedding_dim"])):
        raise ValueError(f"Unexpected cache shape: {tuple(embeddings.shape)}")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError("Cached embeddings are not L2-normalized.")
    payload = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "split": dataset.split,
        "sample_ids": sample_ids,
        "labels": labels,
        "embeddings": embeddings,
        "class_mapping": dataset.class_to_idx,
        "identity": cache_identity,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return {
        "split": dataset.split,
        "samples": len(dataset),
        "embedding_dim": embeddings.shape[1],
        "cache_path": str(cache_path.relative_to(REPOSITORY_ROOT)),
        "cache_sha256": sha256_file(cache_path),
        "wall_seconds": wall_seconds,
        "wall_ms_per_image": wall_seconds * 1000.0 / len(dataset),
        "encoder_forward_seconds": encoder_forward_seconds,
        "encoder_forward_ms_per_image": encoder_forward_seconds
        * 1000.0
        / len(dataset),
        "peak_process_rss_mb": peak_rss_mb(),
        "l2_normalized": True,
        "deterministic_order": True,
    }


def extract_development_features(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    extraction = config["extraction"]
    data_config = config["data"]
    training_config = config["probe"]
    seed = int(training_config["seed"])
    set_global_seed(seed)
    device = resolve_device(str(training_config["device"]))
    encoder = build_foundation_encoder(config["encoder"], device)
    encoder_metadata = encoder.runtime_metadata()
    dataset_root = resolve_repo_path(REPOSITORY_ROOT, str(data_config["root"]))
    cache_identity = {
        "experiment_id": config["experiment"]["id"],
        "dataset_split_hash": data_config["split_report_sha256"],
        "encoder_checkpoint": config["encoder"]["checkpoint_id"],
        "encoder_revision": config["encoder"]["revision"],
        "embedding_dim": int(config["encoder"]["embedding_dim"]),
        "preprocessing_identity": config["encoder"]["preprocessing_identity"],
        "l2_normalized": True,
    }
    results: dict[str, Any] = {}
    for split in (str(data_config["train_split"]), str(data_config["validation_split"])):
        dataset = FruitFreshnessDataset(
            dataset_root,
            split,
            image_size=int(config["encoder"]["input_resolution"]),
            transform=encoder.preprocess,
        )
        loader = create_data_loader(
            dataset,
            batch_size=int(extraction["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=int(extraction["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        cache_path = resolve_repo_path(
            REPOSITORY_ROOT, str(extraction["cache_files"][split])
        )
        print(f"Extracting {split}: {len(dataset)} images", flush=True)
        results[split] = extract_split(
            encoder, dataset, loader, device, cache_path, cache_identity
        )
        print(
            f"{split}: encoder_forward_ms_per_image="
            f"{results[split]['encoder_forward_ms_per_image']:.3f}",
            flush=True,
        )
    summary = {
        "experiment_id": config["experiment"]["id"],
        "completed_at": utc_timestamp(),
        "device": str(device),
        "seed": seed,
        "test_accessed": False,
        "encoder": encoder_metadata,
        "cache_identity": cache_identity,
        "splits": results,
    }
    output_path = resolve_repo_path(
        REPOSITORY_ROOT, str(config["outputs"]["extraction_metadata"])
    )
    save_json(summary, output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extract_development_features(resolve_repo_path(REPOSITORY_ROOT, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
