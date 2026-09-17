"""Run validation selection and one guarded Phase 3B zero-shot test pass."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import classification_report, confusion_matrix

from v2.src.datasets.fruit_dataset import FruitFreshnessDataset, create_data_loader
from v2.src.foundation.extract_features import peak_rss_mb, synchronize
from v2.src.foundation.prompt_registry import (
    FrozenPromptRegistry,
    encode_class_prototypes,
    file_sha256,
    load_prompt_registry,
)
from v2.src.foundation.similarity_classifier import (
    FrozenSimilarityModel,
    build_zero_shot_model,
)
from v2.src.training.utils import (
    environment_record,
    git_record,
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
ALLOWED_SPLITS = {"validation", "test"}


def validate_zero_shot_config(config: dict[str, Any]) -> None:
    forbidden = {"optimizer", "learning_rate", "epochs"}

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    present = forbidden & keys(config)
    if present:
        raise ValueError(f"Training fields are forbidden in zero-shot config: {present}")
    if not config["data"].get("train_split_prohibited"):
        raise ValueError("Zero-shot config must explicitly prohibit the train split.")
    if "train_split" in config["data"]:
        raise ValueError("Zero-shot config must not name a train split.")
    if config["evaluation"].get("classifier_training") is not False:
        raise ValueError("Classifier training must be false.")
    if config["prompts"]["strict_family"] != "P1":
        raise ValueError("Strict zero-shot family must be P1.")
    if config["prompts"]["candidate_families"] != ["P1", "P2", "P3", "P4"]:
        raise ValueError("Validation candidates must be exactly P1-P4.")


def load_registry(config: dict[str, Any]) -> FrozenPromptRegistry:
    path = resolve_repo_path(REPOSITORY_ROOT, config["prompts"]["registry_path"])
    return load_prompt_registry(path, config["prompts"]["registry_sha256"])


def build_dataset_loader(
    config: dict[str, Any],
    model: FrozenSimilarityModel,
    split: str,
) -> tuple[FruitFreshnessDataset, torch.utils.data.DataLoader]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Phase 3B cannot access split: {split}")
    root = resolve_repo_path(REPOSITORY_ROOT, config["data"]["root"])
    dataset = FruitFreshnessDataset(
        root,
        split,
        image_size=int(config["model"]["input_resolution"]),
        transform=model.preprocess,
    )
    expected = int(config["data"][f"{split}_samples"])
    if len(dataset) != expected:
        raise ValueError(f"Frozen {split} count changed: {len(dataset)} != {expected}")
    loader = create_data_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        seed=int(config["evaluation"]["seed"]),
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=model.device.type == "cuda",
    )
    return dataset, loader


def encode_images(
    model: FrozenSimilarityModel,
    dataset: FruitFreshnessDataset,
    loader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    encoder_seconds = 0.0
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(model.device, non_blocking=True)
            synchronize(model.device)
            started = time.perf_counter()
            batch_embeddings = model.encode_images(images)
            synchronize(model.device)
            encoder_seconds += time.perf_counter() - started
            batch_embeddings = functional.normalize(
                batch_embeddings.float(), p=2, dim=1
            )
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.long())
    wall_seconds = time.perf_counter() - wall_started
    all_embeddings = torch.cat(embeddings, dim=0)
    all_labels = torch.cat(labels, dim=0)
    expected_labels = torch.tensor(
        [label for _, label in dataset.samples], dtype=torch.long
    )
    if not torch.equal(all_labels, expected_labels):
        raise RuntimeError("Image embeddings became detached from dataset labels.")
    norms = torch.linalg.vector_norm(all_embeddings, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise ValueError("Image embeddings are not L2-normalized.")
    return all_embeddings, all_labels, {
        "samples": len(dataset),
        "image_encoder_seconds": encoder_seconds,
        "image_encoder_ms_per_image": encoder_seconds * 1000.0 / len(dataset),
        "image_loop_wall_seconds": wall_seconds,
        "image_loop_wall_ms_per_image": wall_seconds * 1000.0 / len(dataset),
        "peak_process_rss_mb": peak_rss_mb(),
        "scope": "Model forward after deterministic preprocessing; image embeddings are L2-normalized.",
    }


def semantic_error_analysis(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    counts = {"condition_errors": 0, "fruit_identity_errors": 0, "both_errors": 0}
    pairs: dict[str, int] = {}
    errors = 0
    for true_index, predicted_index in zip(true_labels, predicted_labels):
        if true_index == predicted_index:
            continue
        errors += 1
        true_name = class_names[true_index]
        predicted_name = class_names[predicted_index]
        true_condition, true_fruit = true_name.split("_", 1)
        predicted_condition, predicted_fruit = predicted_name.split("_", 1)
        if true_fruit == predicted_fruit:
            category = "condition_errors"
        elif true_condition == predicted_condition:
            category = "fruit_identity_errors"
        else:
            category = "both_errors"
        counts[category] += 1
        pair = f"{true_name} -> {predicted_name}"
        pairs[pair] = pairs.get(pair, 0) + 1
    total = len(true_labels)
    return {
        "total_predictions": total,
        "total_errors": errors,
        "counts": counts,
        "percent_of_all_predictions": {
            name: count * 100.0 / total for name, count in counts.items()
        },
        "percent_of_errors": {
            name: (count * 100.0 / errors if errors else 0.0)
            for name, count in counts.items()
        },
        "confusion_pairs": dict(
            sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def classification_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    true = labels.tolist()
    predicted = predictions.detach().cpu().tolist()
    numeric = list(range(len(class_names)))
    report = classification_report(
        true,
        predicted,
        labels=numeric,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true, predicted, labels=numeric)
    metrics = {
        "accuracy": float(np.mean(np.asarray(true) == np.asarray(predicted))),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": matrix.tolist(),
        "semantic_error_analysis": semantic_error_analysis(
            true, predicted, class_names
        ),
    }
    return metrics, report


def classify(
    model: FrozenSimilarityModel,
    image_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    synchronize(model.device)
    started = time.perf_counter()
    cosine = image_embeddings @ prototypes.to(model.device).t()
    logits = model.scaled_logits(cosine)
    predictions = logits.argmax(dim=1)
    synchronize(model.device)
    seconds = time.perf_counter() - started
    return predictions, cosine, logits, seconds


def plot_prompt_comparison(
    family_results: dict[str, Any], path: Path, model_label: str
) -> None:
    families = list(family_results)
    macro_f1 = [family_results[name]["metrics"]["macro_f1"] for name in families]
    accuracy = [family_results[name]["metrics"]["accuracy"] for name in families]
    positions = np.arange(len(families))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(positions - width / 2, macro_f1, width, label="Validation macro F1")
    axis.bar(positions + width / 2, accuracy, width, label="Validation accuracy")
    axis.set_xticks(positions, families)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(f"{model_label}: pre-registered prompt validation comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def select_prompt_family(
    family_results: dict[str, Any], registry: FrozenPromptRegistry
) -> str:
    rank = {name: index for index, name in enumerate(registry.simplicity_order)}
    return max(
        family_results,
        key=lambda name: (
            family_results[name]["metrics"]["macro_f1"],
            family_results[name]["metrics"]["accuracy"],
            -rank[name],
        ),
    )


def require_absent(paths: list[Path], purpose: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {purpose}: {existing}")


def run_validation(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_zero_shot_config(config)
    output = config["outputs"]
    metadata_path = resolve_repo_path(REPOSITORY_ROOT, output["metadata"])
    comparison_path = resolve_repo_path(
        REPOSITORY_ROOT, output["validation_comparison"]
    )
    figure_path = resolve_repo_path(REPOSITORY_ROOT, output["prompt_figure"])
    require_absent(
        [metadata_path, comparison_path, figure_path], "validation artifacts"
    )
    registry = load_registry(config)
    seed = int(config["evaluation"]["seed"])
    set_global_seed(seed)
    device = resolve_device(config["evaluation"]["device"])
    model = build_zero_shot_model(config["model"], device)
    model_metadata = model.runtime_metadata()
    dataset, loader = build_dataset_loader(
        config, model, config["data"]["validation_split"]
    )
    if tuple(dataset.class_names) != registry.class_order:
        raise ValueError("Dataset class order differs from the frozen prompt registry.")
    image_embeddings, labels, image_timing = encode_images(model, dataset, loader)
    family_results: dict[str, Any] = {}
    for family in config["prompts"]["candidate_families"]:
        prototypes, text_timing = encode_class_prototypes(model, registry, family)
        predictions, _, _, similarity_seconds = classify(
            model, image_embeddings, prototypes
        )
        metrics, _ = classification_metrics(
            labels, predictions, list(dataset.class_names)
        )
        family_results[family] = {
            "metrics": metrics,
            "text_prototype_generation": text_timing,
            "similarity_classification_seconds": similarity_seconds,
            "similarity_classification_ms_per_image": similarity_seconds
            * 1000.0
            / len(dataset),
        }
    selected = select_prompt_family(family_results, registry)
    comparison = {
        "experiment_id": config["experiment"]["id"],
        "evaluated_at": utc_timestamp(),
        "split": "validation",
        "samples": len(dataset),
        "prompt_registry": {
            "path": str(registry.path.relative_to(REPOSITORY_ROOT)),
            "sha256": registry.sha256,
            "frozen_before_validation": True,
        },
        "selection_rule": {
            "primary": "macro_f1",
            "first_tie_breaker": "accuracy",
            "second_tie_breaker": "simpler_prompt_strategy",
            "simplicity_order": list(registry.simplicity_order),
        },
        "families": family_results,
        "selected_prompt_family": selected,
        "strict_zero_shot_family": "P1",
        "image_encoding": image_timing,
        "train_split_accessed": False,
        "classifier_trained": False,
        "test_accessed": False,
    }
    save_json(comparison, comparison_path)
    plot_prompt_comparison(
        family_results, figure_path, config["experiment"]["model_label"]
    )
    metadata = {
        "experiment_id": config["experiment"]["id"],
        "status": "validation_complete_test_closed",
        "completed_at": utc_timestamp(),
        "configuration_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "configuration": config,
        "environment": environment_record(device),
        "repository": git_record(REPOSITORY_ROOT),
        "model": model_metadata,
        "prompt_registry": comparison["prompt_registry"],
        "validation": {
            "samples": len(dataset),
            "comparison_path": str(comparison_path.relative_to(REPOSITORY_ROOT)),
            "comparison_sha256": sha256_file(comparison_path),
            "figure_path": str(figure_path.relative_to(REPOSITORY_ROOT)),
            "selected_prompt_family": selected,
            "strict_zero_shot_family": "P1",
            "selected_macro_f1": family_results[selected]["metrics"]["macro_f1"],
            "selected_accuracy": family_results[selected]["metrics"]["accuracy"],
        },
        "integrity": {
            "train_split_accessed": False,
            "classifier_trained": False,
            "model_parameters_updated": False,
            "prompt_parameters_learned": False,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    save_json(metadata, metadata_path)
    print(
        f"Validation complete: selected={selected}, "
        f"macro_f1={family_results[selected]['metrics']['macro_f1']:.4f}, "
        f"accuracy={family_results[selected]['metrics']['accuracy']:.4f}"
    )
    return metadata


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "minimum": ordered[0],
        "p25": float(np.percentile(ordered, 25)),
        "p75": float(np.percentile(ordered, 75)),
        "maximum": ordered[-1],
    }


def margin_analysis(
    condition: str,
    cosine: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    cosine_values, cosine_indices = torch.topk(cosine, k=2, dim=1)
    logit_values, logit_indices = torch.topk(logits, k=2, dim=1)
    if not torch.equal(cosine_indices, logit_indices):
        raise RuntimeError("Native scaling changed class rank unexpectedly.")
    records: list[dict[str, Any]] = []
    correct_cosine: list[float] = []
    incorrect_cosine: list[float] = []
    correct_logit: list[float] = []
    incorrect_logit: list[float] = []
    labels_list = labels.tolist()
    for index, true_index in enumerate(labels_list):
        top1_index = int(cosine_indices[index, 0].item())
        top2_index = int(cosine_indices[index, 1].item())
        cosine_margin = float(
            (cosine_values[index, 0] - cosine_values[index, 1]).item()
        )
        logit_margin = float(
            (logit_values[index, 0] - logit_values[index, 1]).item()
        )
        correct = top1_index == true_index
        (correct_cosine if correct else incorrect_cosine).append(cosine_margin)
        (correct_logit if correct else incorrect_logit).append(logit_margin)
        records.append(
            {
                "sample_id": sample_ids[index],
                "true_class": class_names[true_index],
                "predicted_class": class_names[top1_index],
                "runner_up_class": class_names[top2_index],
                "correct": correct,
                "top1_cosine_similarity": float(cosine_values[index, 0].item()),
                "top2_cosine_similarity": float(cosine_values[index, 1].item()),
                "cosine_margin": cosine_margin,
                "top1_native_logit": float(logit_values[index, 0].item()),
                "top2_native_logit": float(logit_values[index, 1].item()),
                "native_logit_margin": logit_margin,
            }
        )
    return {
        "condition": condition,
        "definition": "top-1 minus top-2 class score",
        "cosine_margin": {
            "correct_predictions": describe(correct_cosine),
            "incorrect_predictions": describe(incorrect_cosine),
        },
        "native_logit_margin": {
            "correct_predictions": describe(correct_logit),
            "incorrect_predictions": describe(incorrect_logit),
        },
        "per_prediction": records,
    }


def condition_result(
    condition: str,
    family: str,
    model: FrozenSimilarityModel,
    image_embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    predictions, cosine, logits, similarity_seconds = classify(
        model, image_embeddings, prototypes
    )
    metrics, report = classification_metrics(labels, predictions, class_names)
    metrics.update(
        {
            "condition": condition,
            "prompt_family": family,
            "samples": int(labels.shape[0]),
            "classifier_trained": False,
            "similarity_classification_seconds": similarity_seconds,
            "similarity_classification_ms_per_image": similarity_seconds
            * 1000.0
            / labels.shape[0],
            "similarity_score_warning": "Cosine similarities and native logits are not calibrated probabilities.",
        }
    )
    return metrics, report, cosine, logits


def run_test(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_zero_shot_config(config)
    output = config["outputs"]
    metadata_path = resolve_repo_path(REPOSITORY_ROOT, output["metadata"])
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("test_accessed") or metadata["integrity"].get("test_accessed"):
        raise RuntimeError("Phase 3B test is already recorded for this experiment.")
    if metadata["status"] != "validation_complete_test_closed":
        raise RuntimeError("Validation prompt selection is not test-ready.")
    registry = load_registry(config)
    if registry.sha256 != metadata["prompt_registry"]["sha256"]:
        raise ValueError("Prompt registry hash changed after validation selection.")
    comparison_path = resolve_repo_path(
        REPOSITORY_ROOT, metadata["validation"]["comparison_path"]
    )
    if sha256_file(comparison_path) != metadata["validation"]["comparison_sha256"]:
        raise ValueError("Validation comparison changed after prompt selection.")
    selected_family = metadata["validation"]["selected_prompt_family"]
    if selected_family not in config["prompts"]["candidate_families"]:
        raise ValueError("Selected prompt family is not pre-registered.")
    test_paths = [
        resolve_repo_path(REPOSITORY_ROOT, output[key])
        for key in (
            "strict_metrics",
            "selected_metrics",
            "strict_report",
            "selected_report",
            "strict_confusion",
            "selected_confusion",
            "margins",
        )
    ]
    require_absent(test_paths, "test artifacts")
    seed = int(config["evaluation"]["seed"])
    set_global_seed(seed)
    device = resolve_device(config["evaluation"]["device"])
    model = build_zero_shot_model(config["model"], device)
    model_metadata = model.runtime_metadata()
    if model_metadata["checkpoint_file"] != metadata["model"]["checkpoint_file"]:
        raise ValueError("Test model checkpoint differs from validation checkpoint.")
    dataset, loader = build_dataset_loader(config, model, config["data"]["test_split"])
    if tuple(dataset.class_names) != registry.class_order:
        raise ValueError("Test class order differs from the prompt registry.")
    strict_prototypes, strict_text_timing = encode_class_prototypes(
        model, registry, "P1"
    )
    if selected_family == "P1":
        selected_prototypes = strict_prototypes
        selected_text_timing = dict(strict_text_timing)
        selected_text_timing["reused_strict_prototypes"] = True
    else:
        selected_prototypes, selected_text_timing = encode_class_prototypes(
            model, registry, selected_family
        )
    image_embeddings, labels, image_timing = encode_images(model, dataset, loader)
    class_names = list(dataset.class_names)
    strict_metrics, strict_report, strict_cosine, strict_logits = condition_result(
        "strict_zero_shot", "P1", model, image_embeddings, labels,
        strict_prototypes, class_names
    )
    selected_metrics, selected_report, selected_cosine, selected_logits = condition_result(
        "validation_selected_zero_shot", selected_family, model, image_embeddings,
        labels, selected_prototypes, class_names
    )
    strict_metrics["image_encoding"] = image_timing
    strict_metrics["text_prototype_generation"] = strict_text_timing
    selected_metrics["image_encoding"] = image_timing
    selected_metrics["text_prototype_generation"] = selected_text_timing
    strict_metrics["end_to_end_forward_ms_per_image"] = (
        image_timing["image_encoder_seconds"]
        + strict_metrics["similarity_classification_seconds"]
    ) * 1000.0 / len(dataset)
    selected_metrics["end_to_end_forward_ms_per_image"] = (
        image_timing["image_encoder_seconds"]
        + selected_metrics["similarity_classification_seconds"]
    ) * 1000.0 / len(dataset)
    sample_ids = dataset.relative_paths()
    margins = {
        "experiment_id": config["experiment"]["id"],
        "evaluated_at": utc_timestamp(),
        "warning": "Similarity scores and softmax-derived scores are not guaranteed to represent true predictive confidence.",
        "calibration_analysis": {
            "performed": False,
            "reason": "No learned calibration was permitted, and CLIP/SigLIP2 native logits do not define a directly comparable multiclass probability model.",
        },
        "strict": margin_analysis(
            "strict_zero_shot", strict_cosine, strict_logits, labels,
            sample_ids, class_names
        ),
        "selected": margin_analysis(
            "validation_selected_zero_shot", selected_cosine, selected_logits,
            labels, sample_ids, class_names
        ),
    }
    paths = {
        key: resolve_repo_path(REPOSITORY_ROOT, output[key])
        for key in (
            "strict_metrics", "selected_metrics", "strict_report",
            "selected_report", "strict_confusion", "selected_confusion", "margins"
        )
    }
    save_json(strict_metrics, paths["strict_metrics"])
    save_json(selected_metrics, paths["selected_metrics"])
    save_json(strict_report, paths["strict_report"])
    save_json(selected_report, paths["selected_report"])
    save_json(margins, paths["margins"])
    plot_transfer_confusion_matrix(
        np.asarray(strict_metrics["confusion_matrix"]), class_names,
        paths["strict_confusion"],
        f"{config['experiment']['model_label']}: strict P1 zero-shot test",
    )
    plot_transfer_confusion_matrix(
        np.asarray(selected_metrics["confusion_matrix"]), class_names,
        paths["selected_confusion"],
        f"{config['experiment']['model_label']}: selected {selected_family} zero-shot test",
    )
    metadata["status"] = "complete"
    metadata["test_accessed"] = True
    metadata["integrity"]["test_accessed"] = True
    metadata["integrity"]["single_locked_test_run"] = True
    metadata["test"] = {
        "evaluated_at": utc_timestamp(),
        "samples": len(dataset),
        "strict_prompt_family": "P1",
        "selected_prompt_family": selected_family,
        "strict_metrics_path": str(paths["strict_metrics"].relative_to(REPOSITORY_ROOT)),
        "selected_metrics_path": str(paths["selected_metrics"].relative_to(REPOSITORY_ROOT)),
        "margins_path": str(paths["margins"].relative_to(REPOSITORY_ROOT)),
        "strict_accuracy": strict_metrics["accuracy"],
        "strict_weighted_f1": strict_metrics["weighted_f1"],
        "selected_accuracy": selected_metrics["accuracy"],
        "selected_weighted_f1": selected_metrics["weighted_f1"],
        "image_encoding": image_timing,
        "strict_text_prototype_generation": strict_text_timing,
        "selected_text_prototype_generation": selected_text_timing,
        "strict_similarity_classification_seconds": strict_metrics[
            "similarity_classification_seconds"
        ],
        "selected_similarity_classification_seconds": selected_metrics[
            "similarity_classification_seconds"
        ],
    }
    metadata["artifacts"] = {
        str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
        for path in paths.values()
    }
    save_json(metadata, metadata_path)
    print(
        f"Locked test complete: strict_accuracy={strict_metrics['accuracy']:.4f}, "
        f"selected={selected_family}, selected_accuracy={selected_metrics['accuracy']:.4f}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("validation", "test"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_repo_path(REPOSITORY_ROOT, args.config)
    if args.stage == "validation":
        run_validation(config_path)
    else:
        run_test(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
