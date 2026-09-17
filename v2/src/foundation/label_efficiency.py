"""Run the pre-registered Phase 3C frozen-representation label-efficiency study."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.training.utils import (
    load_config,
    run_epoch,
    save_json,
    set_global_seed,
    sha256_file,
    utc_timestamp,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
T_CRITICAL_DF4_95 = 2.7764451051977987
EXPECTED_CLASSES = (
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative(path: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT))


def assert_master() -> None:
    import subprocess

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if branch != "master":
        raise RuntimeError(f"Phase 3C is restricted to master; current branch is {branch!r}.")


def phase3a_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return load_config(repo_path(model_config["phase_3a_config"]))


def cache_path(base_config: dict[str, Any], split: str) -> Path:
    return repo_path(base_config["extraction"]["cache_files"][split])


def validate_cache(
    path: Path,
    split: str,
    base_config: dict[str, Any],
    recorded_hash: str,
) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    if actual_hash != recorded_hash:
        raise ValueError(f"Cache hash mismatch for {path}: {actual_hash}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload["identity"]
    expected = {
        "experiment_id": base_config["experiment"]["id"],
        "dataset_split_hash": base_config["data"]["split_report_sha256"],
        "encoder_checkpoint": base_config["encoder"]["checkpoint_id"],
        "encoder_revision": base_config["encoder"]["revision"],
        "embedding_dim": int(base_config["encoder"]["embedding_dim"]),
        "preprocessing_identity": base_config["encoder"]["preprocessing_identity"],
        "l2_normalized": True,
    }
    if payload["split"] != split or identity != expected:
        raise ValueError(f"Cache identity mismatch for {path}.")
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    if len(payload["sample_ids"]) != embeddings.shape[0] or labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"Misaligned samples in {path}.")
    if embeddings.shape[1] != expected["embedding_dim"]:
        raise ValueError(f"Embedding dimension mismatch in {path}.")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError(f"Non-normalized embeddings in {path}.")
    return payload


def validate_foundation_evidence(
    config: dict[str, Any], include_test: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    split_report = repo_path(config["data"]["split_report_path"])
    if sha256_file(split_report) != config["data"]["split_report_sha256"]:
        raise ValueError("Frozen split-report hash mismatch.")

    caches: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "verified_at": utc_timestamp(),
        "split_report": {
            "path": str(split_report),
            "sha256": sha256_file(split_report),
        },
        "models": {},
    }
    common_order: dict[str, list[str]] = {}
    common_labels: dict[str, list[int]] = {}
    splits = ["train", "validation"] + (["test"] if include_test else [])
    for model_key, model_config in config["models"].items():
        base = phase3a_config(model_config)
        extraction = load_json(base["outputs"]["extraction_metadata"])
        phase3a_metrics = load_json(model_config["full_data_metrics"])
        checkpoint_record = extraction["encoder"]["checkpoint_file"]
        checkpoint_file = Path(checkpoint_record["path"])
        if sha256_file(checkpoint_file) != checkpoint_record["sha256"]:
            raise ValueError(f"Foundation checkpoint hash mismatch for {model_key}.")
        model_audit = {
            "checkpoint_id": base["encoder"]["checkpoint_id"],
            "revision": base["encoder"]["revision"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "checkpoint_size_bytes": checkpoint_record["size_bytes"],
            "preprocessing_identity": base["encoder"]["preprocessing_identity"],
            "embedding_dim": base["encoder"]["embedding_dim"],
            "all_parameters_frozen": extraction["encoder"]["all_parameters_frozen"],
            "caches": {},
        }
        caches[model_key] = {}
        for split in splits:
            path = cache_path(base, split)
            if split in extraction["splits"]:
                recorded_hash = extraction["splits"][split]["cache_sha256"]
            else:
                recorded_hash = phase3a_metrics["test_embedding_cache"]["sha256"]
            payload = validate_cache(path, split, base, recorded_hash)
            caches[model_key][split] = payload
            model_audit["caches"][split] = {
                "path": relative(path),
                "sha256": recorded_hash,
                "samples": len(payload["sample_ids"]),
            }
            ids = list(payload["sample_ids"])
            labels = payload["labels"].tolist()
            if split not in common_order:
                common_order[split] = ids
                common_labels[split] = labels
            elif ids != common_order[split] or labels != common_labels[split]:
                raise ValueError(f"{split} sample ordering differs across encoder caches.")
        audit["models"][model_key] = model_audit

    train = caches[next(iter(caches))]["train"]
    mapping = train["class_mapping"]
    if tuple(name for name, _ in sorted(mapping.items(), key=lambda item: item[1])) != EXPECTED_CLASSES:
        raise ValueError("Frozen class mapping differs from the registered six-class order.")
    dataset = FruitFreshnessDataset(
        repo_path(config["data"]["root"]), "train", image_size=224
    )
    if dataset.relative_paths() != common_order["train"]:
        raise ValueError("Train cache ordering differs from the frozen dataset ordering.")
    expected_labels = [label for _, label in dataset.samples]
    if expected_labels != common_labels["train"]:
        raise ValueError("Train cache labels differ from the frozen dataset labels.")
    counts = {name: int((train["labels"] == index).sum().item()) for name, index in mapping.items()}
    if sum(counts.values()) != int(config["data"]["train_samples"]):
        raise ValueError("Frozen training sample count mismatch.")
    if any(count < max(config["protocol"]["examples_per_class"]) for count in counts.values()):
        raise RuntimeError(f"A training class cannot support 100 examples: {counts}")
    audit["class_mapping"] = mapping
    audit["training_class_capacity"] = counts
    audit["test_cache_loaded"] = include_test
    return caches, audit


def manifests(config: dict[str, Any]) -> None:
    assert_master()
    output_dir = repo_path(config["outputs"]["manifest_directory"])
    registry_path = output_dir / "manifest_hashes.json"
    if registry_path.exists():
        raise FileExistsError(f"Refusing to replace frozen manifests: {registry_path}")
    caches, audit = validate_foundation_evidence(config, include_test=False)
    reference = caches[next(iter(caches))]["train"]
    mapping = reference["class_mapping"]
    budgets = [int(value) for value in config["protocol"]["examples_per_class"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(audit, output_dir / "preflight_audit.json")
    manifest_records = []
    for subset_seed in config["protocol"]["subset_seeds"]:
        rng = random.Random(int(subset_seed))
        entries = []
        for class_name, class_index in sorted(mapping.items(), key=lambda item: item[1]):
            class_ids = sorted(
                sample_id
                for sample_id, label in zip(reference["sample_ids"], reference["labels"].tolist())
                if label == class_index
            )
            rng.shuffle(class_ids)
            for rank, sample_id in enumerate(class_ids[: max(budgets)], start=1):
                entries.append(
                    {
                        "class": class_name,
                        "class_index": class_index,
                        "sample_id": sample_id,
                        "selection_rank": rank,
                        "budget_membership": [budget for budget in budgets if rank <= budget],
                    }
                )
        payload = {
            "schema_version": 1,
            "phase": "V2 Phase 3C",
            "subset_seed": int(subset_seed),
            "dataset_split": "train",
            "dataset_split_hash": config["data"]["split_report_sha256"],
            "sampling": "Per-class lexicographic lists shuffled sequentially in class-index order by Python random.Random(subset_seed); first k items form each nested budget.",
            "replacement": False,
            "class_balanced": True,
            "budgets_examples_per_class": budgets,
            "entries": entries,
        }
        path = output_dir / f"subset_seed_{subset_seed}.json"
        save_json(payload, path)
        manifest_records.append(
            {"subset_seed": int(subset_seed), "path": relative(path), "sha256": sha256_file(path)}
        )
    registry = {
        "schema_version": 1,
        "status": "frozen_before_probe_training",
        "frozen_at": utc_timestamp(),
        "dataset_split_hash": config["data"]["split_report_sha256"],
        "subset_seeds": config["protocol"]["subset_seeds"],
        "budgets_examples_per_class": budgets,
        "manifests": manifest_records,
    }
    save_json(registry, registry_path)
    (output_dir / "manifest_hashes.sha256").write_text(
        f"{sha256_file(registry_path)}  manifest_hashes.json\n", encoding="utf-8"
    )
    print(f"Frozen {len(manifest_records)} nested subset manifests after capacity and cache validation.")


def verify_manifest_registry(config: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    output_dir = repo_path(config["outputs"]["manifest_directory"])
    registry_path = output_dir / "manifest_hashes.json"
    sidecar = (output_dir / "manifest_hashes.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(registry_path) != sidecar:
        raise ValueError("Manifest registry changed after freezing.")
    registry = load_json(registry_path)
    loaded = {}
    for record in registry["manifests"]:
        path = repo_path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen subset manifest changed: {path}")
        loaded[int(record["subset_seed"])] = load_json(path)
    return registry, loaded


def subset_indices(manifest: dict[str, Any], budget: int, sample_ids: list[str]) -> list[int]:
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    selected = [entry for entry in manifest["entries"] if entry["selection_rank"] <= budget]
    counts = {name: 0 for name in EXPECTED_CLASSES}
    indices = []
    for entry in selected:
        counts[entry["class"]] += 1
        indices.append(index[entry["sample_id"]])
    if any(value != budget for value in counts.values()) or len(indices) != budget * 6:
        raise ValueError(f"Manifest is not exactly balanced at budget {budget}: {counts}")
    if len(set(indices)) != len(indices):
        raise ValueError("Manifest selected a sample more than once.")
    return indices


def loader(embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        TensorDataset(embeddings, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def execution_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "cpu_count": __import__("os").cpu_count(),
    }


def train(config: dict[str, Any]) -> None:
    assert_master()
    registry, manifest_payloads = verify_manifest_registry(config)
    caches, audit = validate_foundation_evidence(config, include_test=False)
    budgets = [int(value) for value in config["protocol"]["examples_per_class"]]
    subset_seeds = [int(value) for value in config["protocol"]["subset_seeds"]]
    checkpoint_root = repo_path(config["outputs"]["checkpoint_directory"])
    checkpoint_registry_path = repo_path(config["outputs"]["manifest_directory"]) / "checkpoint_registry.json"
    if checkpoint_registry_path.exists() or checkpoint_root.exists():
        raise FileExistsError("Refusing to overwrite Phase 3C checkpoints or frozen checkpoint registry.")
    all_records = []
    for model_key, model_config in config["models"].items():
        base = phase3a_config(model_config)
        train_cache = caches[model_key]["train"]
        validation_cache = caches[model_key]["validation"]
        result_root = repo_path(f"v2/results/experiment_{int(model_config['experiment_number']):03d}")
        for budget in budgets:
            for subset_seed in subset_seeds:
                started = utc_timestamp()
                wall_started = time.perf_counter()
                manifest_record = next(
                    item for item in registry["manifests"] if int(item["subset_seed"]) == subset_seed
                )
                indices = subset_indices(
                    manifest_payloads[subset_seed], budget, list(train_cache["sample_ids"])
                )
                subset_embeddings = train_cache["embeddings"][indices]
                subset_labels = train_cache["labels"][indices]
                set_global_seed(int(config["protocol"]["optimization_seed"]))
                model = LinearProbe(int(base["encoder"]["embedding_dim"]), 6)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=float(config["protocol"]["learning_rate"])
                )
                criterion = nn.CrossEntropyLoss()
                train_loader = loader(
                    subset_embeddings,
                    subset_labels,
                    min(int(config["protocol"]["batch_size"]), len(indices)),
                    True,
                )
                validation_loader = loader(
                    validation_cache["embeddings"],
                    validation_cache["labels"],
                    int(config["protocol"]["batch_size"]),
                    False,
                )
                best_state = None
                best_epoch = 0
                best_accuracy = -1.0
                best_loss = float("inf")
                history = []
                for epoch in range(1, int(config["protocol"]["maximum_epochs"]) + 1):
                    train_metrics = run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer)
                    validation_metrics = run_epoch(
                        model, validation_loader, criterion, torch.device("cpu"), None
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_metrics["loss"],
                            "train_accuracy": train_metrics["accuracy"],
                            "validation_loss": validation_metrics["loss"],
                            "validation_accuracy": validation_metrics["accuracy"],
                        }
                    )
                    improved = validation_metrics["accuracy"] > best_accuracy or (
                        validation_metrics["accuracy"] == best_accuracy
                        and validation_metrics["loss"] < best_loss
                    )
                    if improved:
                        best_epoch = epoch
                        best_accuracy = validation_metrics["accuracy"]
                        best_loss = validation_metrics["loss"]
                        best_state = copy.deepcopy(model.state_dict())
                checkpoint_path = (
                    checkpoint_root
                    / f"experiment_{int(model_config['experiment_number']):03d}"
                    / f"budget_{budget:03d}"
                    / f"seed_{subset_seed}.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema_version": 1,
                        "experiment_id": model_config["experiment_id"],
                        "model": model_key,
                        "encoder_revision": base["encoder"]["revision"],
                        "embedding_dim": int(base["encoder"]["embedding_dim"]),
                        "num_classes": 6,
                        "examples_per_class": budget,
                        "subset_seed": subset_seed,
                        "optimization_seed": int(config["protocol"]["optimization_seed"]),
                        "manifest_sha256": manifest_record["sha256"],
                        "selected_epoch": best_epoch,
                        "validation_accuracy": best_accuracy,
                        "validation_loss": best_loss,
                        "class_mapping": train_cache["class_mapping"],
                        "model_state_dict": best_state,
                    },
                    checkpoint_path,
                )
                development_path = (
                    result_root / "development_records" / f"budget_{budget:03d}" / f"seed_{subset_seed}.json"
                )
                record = {
                    "schema_version": 1,
                    "status": "validation_selected_test_closed",
                    "model": model_config["display_name"],
                    "model_key": model_key,
                    "model_revision": base["encoder"]["revision"],
                    "experiment_id": model_config["experiment_id"],
                    "subset_seed": subset_seed,
                    "optimization_seed": int(config["protocol"]["optimization_seed"]),
                    "examples_per_class": budget,
                    "total_labelled_examples": budget * 6,
                    "manifest": manifest_record,
                    "probe": {
                        "architecture": "nn.Linear(embedding_dim, 6)",
                        "loss": "CrossEntropyLoss",
                        "optimizer": "Adam",
                        "learning_rate": float(config["protocol"]["learning_rate"]),
                        "maximum_epochs": int(config["protocol"]["maximum_epochs"]),
                        "batch_size": int(config["protocol"]["batch_size"]),
                        "effective_train_batch_size": min(64, budget * 6),
                        "selected_epoch": best_epoch,
                        "validation_accuracy": best_accuracy,
                        "validation_loss": best_loss,
                    },
                    "checkpoint": {
                        "path": relative(checkpoint_path),
                        "sha256": sha256_file(checkpoint_path),
                    },
                    "history": history,
                    "execution": {
                        **execution_metadata(),
                        "started_at": started,
                        "completed_at": utc_timestamp(),
                        "training_seconds": time.perf_counter() - wall_started,
                    },
                    "test_accessed": False,
                }
                save_json(record, development_path)
                all_records.append(
                    {
                        "model": model_key,
                        "examples_per_class": budget,
                        "subset_seed": subset_seed,
                        "development_record_path": relative(development_path),
                        "development_record_sha256": sha256_file(development_path),
                        "checkpoint_path": relative(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "selected_epoch": best_epoch,
                        "validation_accuracy": best_accuracy,
                        "validation_loss": best_loss,
                        "manifest_sha256": manifest_record["sha256"],
                    }
                )
                print(
                    f"trained {model_key} budget={budget:03d} seed={subset_seed}: "
                    f"epoch={best_epoch} val_acc={best_accuracy:.4f}", flush=True
                )
    expected = len(config["models"]) * len(budgets) * len(subset_seeds)
    if len(all_records) != expected:
        raise RuntimeError(f"Expected {expected} checkpoints, produced {len(all_records)}.")
    checkpoint_registry = {
        "schema_version": 1,
        "status": "all_validation_checkpoints_frozen_test_closed",
        "frozen_at": utc_timestamp(),
        "planned_runs": expected,
        "completed_training_runs": len(all_records),
        "manifest_registry_sha256": sha256_file(
            repo_path(config["outputs"]["manifest_directory"]) / "manifest_hashes.json"
        ),
        "development_audit": audit,
        "records": all_records,
    }
    save_json(checkpoint_registry, checkpoint_registry_path)
    sidecar = checkpoint_registry_path.with_suffix(".sha256")
    sidecar.write_text(
        f"{sha256_file(checkpoint_registry_path)}  {checkpoint_registry_path.name}\n",
        encoding="utf-8",
    )
    print(f"Frozen {len(all_records)} validation-selected checkpoints; test remains closed.")


def verify_checkpoint_registry(config: dict[str, Any]) -> dict[str, Any]:
    path = repo_path(config["outputs"]["manifest_directory"]) / "checkpoint_registry.json"
    sidecar_hash = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != sidecar_hash:
        raise ValueError("Checkpoint registry changed after freezing.")
    registry = load_json(path)
    if registry["planned_runs"] != 90 or len(registry["records"]) != 90:
        raise ValueError("Checkpoint registry does not account for all 90 planned runs.")
    for record in registry["records"]:
        if sha256_file(repo_path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after freezing: {record['checkpoint_path']}")
        if sha256_file(repo_path(record["development_record_path"])) != record["development_record_sha256"]:
            raise ValueError(f"Development metadata changed after freezing: {record['development_record_path']}")
    return registry


def metric_payload(true: np.ndarray, predicted: np.ndarray, loss: float, class_mapping: dict[str, int]) -> dict[str, Any]:
    names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    labels = list(range(len(names)))
    report = classification_report(
        true, predicted, labels=labels, target_names=names, output_dict=True, zero_division=0
    )
    return {
        "loss": loss,
        "accuracy": float(np.mean(true == predicted)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in names
        },
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }


def locked_test(config: dict[str, Any]) -> None:
    assert_master()
    verify_manifest_registry(config)
    registry = verify_checkpoint_registry(config)
    access_path = repo_path(config["outputs"]["manifest_directory"]) / "locked_test_access.json"
    if access_path.exists():
        raise FileExistsError("Locked Phase 3C test evaluation has already begun; refusing a second pass.")
    access = {
        "schema_version": 1,
        "status": "in_progress",
        "started_at": utc_timestamp(),
        "planned_selected_probes": 90,
        "completed_selected_probes": 0,
        "test_cache_loaded_after_checkpoint_freeze": True,
        "completed_keys": [],
    }
    save_json(access, access_path)
    caches, audit = validate_foundation_evidence(config, include_test=True)
    criterion = nn.CrossEntropyLoss()
    for record in registry["records"]:
        model_key = record["model"]
        model_config = config["models"][model_key]
        test_cache = caches[model_key]["test"]
        checkpoint = torch.load(repo_path(record["checkpoint_path"]), map_location="cpu", weights_only=True)
        probe = LinearProbe(int(checkpoint["embedding_dim"]), int(checkpoint["num_classes"]))
        probe.load_state_dict(checkpoint["model_state_dict"])
        probe.eval()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = probe(test_cache["embeddings"])
            loss = float(criterion(logits, test_cache["labels"]).item())
            predicted = logits.argmax(dim=1).numpy()
        inference_seconds = time.perf_counter() - started
        true = test_cache["labels"].numpy()
        development = load_json(record["development_record_path"])
        metrics = metric_payload(true, predicted, loss, test_cache["class_mapping"])
        run = {
            "schema_version": 1,
            "status": "complete",
            "model": model_config["display_name"],
            "model_key": model_key,
            "model_revision": development["model_revision"],
            "experiment_id": model_config["experiment_id"],
            "subset_seed": record["subset_seed"],
            "optimization_seed": int(config["protocol"]["optimization_seed"]),
            "examples_per_class": record["examples_per_class"],
            "total_labelled_examples": record["examples_per_class"] * 6,
            "manifest_hash": record["manifest_sha256"],
            "selected_epoch": record["selected_epoch"],
            "validation_metrics": {
                "accuracy": record["validation_accuracy"],
                "loss": record["validation_loss"],
            },
            "test_metrics": metrics,
            "checkpoint": {
                "path": record["checkpoint_path"],
                "sha256": record["checkpoint_sha256"],
            },
            "execution": {
                **development["execution"],
                "test_evaluated_at": utc_timestamp(),
                "probe_test_forward_seconds": inference_seconds,
                "probe_test_forward_ms_per_image": inference_seconds * 1000 / len(true),
            },
            "single_locked_test_evaluation": True,
        }
        run_path = repo_path(
            f"v2/results/experiment_{int(model_config['experiment_number']):03d}/runs/"
            f"budget_{int(record['examples_per_class']):03d}/seed_{int(record['subset_seed'])}.json"
        )
        if run_path.exists():
            raise FileExistsError(f"Refusing to overwrite test result: {run_path}")
        save_json(run, run_path)
        key = f"{model_key}:{record['examples_per_class']}:{record['subset_seed']}"
        access["completed_selected_probes"] += 1
        access["completed_keys"].append(key)
        save_json(access, access_path)
        print(f"tested {key}: accuracy={metrics['accuracy']:.4f}", flush=True)
    if access["completed_selected_probes"] != 90:
        raise RuntimeError("Locked test stage did not evaluate all 90 selected probes.")
    access["status"] = "complete_single_locked_pass"
    access["completed_at"] = utc_timestamp()
    access["test_audit"] = audit
    save_json(access, access_path)
    print("Completed exactly one locked test evaluation for each of 90 selected probes.")


def statistics(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = T_CRITICAL_DF4_95 * std / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": std,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "interval_95_across_pre_registered_subset_draws": [mean - half, mean + half],
        "n_subset_draws": len(values),
        "interval_method": "Two-sided t interval for the mean, df=4, t*=2.776445; not complete predictive uncertainty.",
    }


def aggregate(config: dict[str, Any]) -> None:
    assert_master()
    verify_manifest_registry(config)
    verify_checkpoint_registry(config)
    access_path = repo_path(config["outputs"]["manifest_directory"]) / "locked_test_access.json"
    access = load_json(access_path)
    if access["status"] != "complete_single_locked_pass" or access["completed_selected_probes"] != 90:
        raise RuntimeError("Cannot aggregate before the locked 90-probe test pass is complete.")
    budgets = [int(value) for value in config["protocol"]["examples_per_class"]]
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    )
    combined = {
        "schema_version": 1,
        "phase": "V2 Phase 3C",
        "generated_at": utc_timestamp(),
        "interval_label": "95% interval across pre-registered subset draws",
        "interval_scope_warning": "This is not complete predictive uncertainty.",
        "zero_to_one_boundary_warning": "Zero-shot points use frozen text prototypes; supervised points use trained linear probes. The 0-to-1 transition changes both supervision amount and downstream classifier.",
        "models": {},
    }
    for model_key, model_config in config["models"].items():
        full = load_json(model_config["full_data_metrics"])
        zero = load_json(model_config["strict_zero_shot_metrics"]) if model_config["strict_zero_shot_metrics"] else None
        model_summary = {
            "experiment_id": model_config["experiment_id"],
            "display_name": model_config["display_name"],
            "aggregates": {},
            "full_data_endpoint": {
                "source_experiment": model_config["phase_3a_experiment"],
                "training_images": int(config["data"]["train_samples"]),
                "accuracy": full["accuracy"],
                "weighted_f1": full["classification_report"]["weighted avg"]["f1-score"],
                "rotten_apple_recall": full["classification_report"]["rotten_apple"]["recall"],
                "reused_not_rerun": True,
            },
            "strict_zero_shot_endpoint": (
                {
                    "accuracy": zero["accuracy"],
                    "weighted_f1": zero["weighted_f1"],
                    "rotten_apple_recall": zero["per_class"]["rotten_apple"]["recall"],
                    "decision_mechanism": "frozen text prototypes",
                    "reused_not_rerun": True,
                }
                if zero
                else {"applicable": False, "reason": "DINOv2 has no registered text classifier."}
            ),
        }
        for budget in budgets:
            runs = [
                load_json(
                    f"v2/results/experiment_{int(model_config['experiment_number']):03d}/runs/"
                    f"budget_{budget:03d}/seed_{seed}.json"
                )
                for seed in config["protocol"]["subset_seeds"]
            ]
            aggregate_record = {
                metric: statistics([run["test_metrics"][metric] for run in runs])
                for metric in metric_names
            }
            aggregate_record["per_class"] = {
                class_name: {
                    metric: statistics(
                        [run["test_metrics"]["per_class"][class_name][metric] for run in runs]
                    )
                    for metric in ("precision", "recall", "f1")
                }
                for class_name in EXPECTED_CLASSES
            }
            mean_accuracy = aggregate_record["accuracy"]["mean"]
            full_accuracy = full["accuracy"]
            aggregate_record["full_performance_gap"] = full_accuracy - mean_accuracy
            aggregate_record["performance_retention"] = mean_accuracy / full_accuracy
            model_summary["aggregates"][str(budget)] = aggregate_record
        within_one = next(
            (budget for budget in budgets if full["accuracy"] - model_summary["aggregates"][str(budget)]["accuracy"]["mean"] <= 0.01),
            None,
        )
        ninety_five = next(
            (budget for budget in budgets if model_summary["aggregates"][str(budget)]["accuracy"]["mean"] >= 0.95 * full["accuracy"]),
            None,
        )
        model_summary["thresholds"] = {
            "within_one_percentage_point_of_full": within_one if within_one is not None else "Not reached.",
            "at_least_95_percent_of_full": ninety_five if ninety_five is not None else "Not reached.",
            "interpolation_used": False,
        }
        combined["models"][model_key] = model_summary
        summary_path = repo_path(f"v2/results/experiment_{int(model_config['experiment_number']):03d}/summary.json")
        save_json(model_summary, summary_path)
    combined["rankings_by_budget"] = {
        str(budget): sorted(
            (
                {
                    "model": model_key,
                    "mean_accuracy": combined["models"][model_key]["aggregates"][str(budget)]["accuracy"]["mean"],
                }
                for model_key in config["models"]
            ),
            key=lambda item: item["mean_accuracy"],
            reverse=True,
        )
        for budget in budgets
    }
    save_json(combined, config["outputs"]["combined_summary"])
    plots(config, combined)
    print("Aggregated all 90 runs and generated Phase 3C figures.")


def plot_metric(config: dict[str, Any], combined: dict[str, Any], metric: str, filename: str, ylabel: str) -> None:
    budgets = [int(value) for value in config["protocol"]["examples_per_class"]]
    colors = {"dinov2": "#4c78a8", "clip": "#f58518", "siglip2": "#54a24b"}
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for model_key, summary in combined["models"].items():
        if metric == "rotten_apple_recall":
            means = [summary["aggregates"][str(b)]["per_class"]["rotten_apple"]["recall"]["mean"] for b in budgets]
            stds = [summary["aggregates"][str(b)]["per_class"]["rotten_apple"]["recall"]["standard_deviation"] for b in budgets]
            full = summary["full_data_endpoint"]["rotten_apple_recall"]
            zero_key = "rotten_apple_recall"
        else:
            means = [summary["aggregates"][str(b)][metric]["mean"] for b in budgets]
            stds = [summary["aggregates"][str(b)][metric]["standard_deviation"] for b in budgets]
            full = summary["full_data_endpoint"][metric]
            zero_key = metric
        color = colors[model_key]
        axis.errorbar(budgets, means, yerr=stds, marker="o", capsize=3, label=summary["display_name"], color=color)
        axis.scatter([256.5], [full], marker="s", s=55, color=color)
        zero = summary["strict_zero_shot_endpoint"]
        if zero.get("applicable", True):
            axis.scatter([0], [zero[zero_key]], marker="X", s=95, color=color, edgecolor="black", linewidth=0.6)
    axis.set_xscale("symlog", linthresh=1)
    axis.set_xlabel("Examples per class (square = reused full-data endpoint at 1,539 total / 6)")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1.025)
    axis.grid(alpha=0.25)
    model_legend = axis.legend(loc="lower right")
    axis.add_artist(model_legend)
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="X", color="none", markerfacecolor="black",
                markersize=9, label="Strict zero-shot text prototype",
            ),
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor="black",
                markersize=8, label="Reused full-data linear probe",
            ),
        ],
        loc="lower left",
        fontsize=8,
    )
    axis.set_title("V2 Phase 3C label efficiency (error bars: subset-draw standard deviation)")
    fig.tight_layout()
    fig.savefig(repo_path(config["outputs"]["figure_directory"]) / filename, dpi=180)
    plt.close(fig)


def plots(config: dict[str, Any], combined: dict[str, Any]) -> None:
    plot_metric(config, combined, "accuracy", "label_efficiency_accuracy.png", "Mean held-out test accuracy")
    plot_metric(config, combined, "weighted_f1", "label_efficiency_weighted_f1.png", "Mean held-out weighted F1")
    plot_metric(config, combined, "rotten_apple_recall", "label_efficiency_rotten_apple_recall.png", "Mean rotten-apple recall")
    budgets = [int(value) for value in config["protocol"]["examples_per_class"]]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for model_key, summary in combined["models"].items():
        axis.plot(
            budgets,
            [summary["aggregates"][str(b)]["accuracy"]["standard_deviation"] for b in budgets],
            marker="o",
            label=summary["display_name"],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Examples per class")
    axis.set_ylabel("Accuracy standard deviation across five subset draws")
    axis.set_title("Phase 3C subset-selection stability")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(repo_path(config["outputs"]["figure_directory"]) / "label_efficiency_variance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, (model_key, summary) in zip(axes, combined["models"].items()):
        for class_name in EXPECTED_CLASSES:
            axis.plot(
                budgets,
                [summary["aggregates"][str(b)]["per_class"][class_name]["recall"]["mean"] for b in budgets],
                marker="o",
                label=class_name.replace("_", " "),
            )
        axis.set_xscale("log")
        axis.set_title(summary["display_name"])
        axis.set_xlabel("Examples per class")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean held-out recall")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 3C class-level learning curves")
    fig.tight_layout()
    fig.savefig(repo_path(config["outputs"]["figure_directory"]) / "label_efficiency_class_recall.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifests", "train", "test", "aggregate"))
    parser.add_argument("--config", default="v2/configs/label_efficiency.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(repo_path(args.config))
    {"manifests": manifests, "train": train, "test": locked_test, "aggregate": aggregate}[args.stage](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
