"""Post-hoc frozen-representation and perturbation study for Experiment 015."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, calinski_harabasz_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v2.src.datasets.transfer_transforms import build_transfer_transform
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.foundation.prompt_registry import encode_class_prototypes, load_prompt_registry
from v2.src.foundation.similarity_classifier import build_zero_shot_model
from v2.src.models.transfer_registry import build_transfer_model
from v2.src.training.utils import load_config, resolve_device, save_json, sha256_file, utc_timestamp


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "v2/results/experiment_015_interpretability"
PHASE13 = ROOT / "v2/results/experiment_013_cross_domain"
PHASE14 = ROOT / "v2/results/experiment_014_multi_domain"
SULTANA_ROOT = ROOT / "v2/data/external/sultana_2022/Original Image"
FRUITVISION_ROOT = ROOT / "v2/data/external/fruitvision/Fruits Original"
CLASS_NAMES = ["fresh_apple", "fresh_banana", "fresh_orange", "rotten_apple", "rotten_banana", "rotten_orange"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
DOMAINS = ["source", "sultana", "fruitvision"]
ENCODERS = ["dinov2", "clip", "siglip2"]
ATTRIBUTION_SYSTEMS = ["efficientnet_b0", "clip_linear_probe", "clip_strict_zero_shot", "siglip2_linear_probe", "siglip2_strict_zero_shot"]


def root_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read_json(value: str | Path) -> Any:
    with root_path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()


def load_caches(config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for encoder in ENCODERS:
        spec = config["encoders"][encoder]
        result[encoder] = {}
        for domain in DOMAINS:
            payload = torch.load(root_path(spec[f"{domain}_cache"]), map_location="cpu", weights_only=False)
            payload["embeddings"] = F.normalize(payload["embeddings"].float(), p=2, dim=1)
            result[encoder][domain] = payload
    return result


def verify(config: dict[str, Any]) -> None:
    output = RESULTS / "embedding_verification.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite Experiment 015 embedding verification.")
    caches = load_caches(config)
    sultana_manifest = read_json(config["frozen_manifests"]["sultana"])
    fruitvision_manifest = read_json(config["frozen_manifests"]["fruitvision"])
    expected_manifest_ids = {
        "sultana": [item["identifier"] for item in sultana_manifest["images"]],
        "fruitvision": [item["identifier"] for item in fruitvision_manifest["images"]],
    }
    expected_manifest_labels = {
        "sultana": [CLASS_TO_IDX[item["class"]] for item in sultana_manifest["images"]],
        "fruitvision": [CLASS_TO_IDX[item["class"]] for item in fruitvision_manifest["images"]],
    }
    failures: list[str] = []
    records: dict[str, Any] = {}
    source_ids = list(caches["dinov2"]["source"]["sample_ids"])
    source_labels = caches["dinov2"]["source"]["labels"].tolist()
    for encoder in ENCODERS:
        spec = config["encoders"][encoder]
        records[encoder] = {}
        for domain in DOMAINS:
            cache_path = root_path(spec[f"{domain}_cache"])
            payload = caches[encoder][domain]
            ids = list(payload["sample_ids"])
            labels = payload["labels"].tolist()
            expected_count = {"source": 2698, "sultana": 1200, "fruitvision": 4185}[domain]
            checks = {
                "sample_count": len(ids) == expected_count,
                "embedding_rows": payload["embeddings"].shape[0] == expected_count,
                "embedding_dimension": payload["embeddings"].shape[1] == int(spec["embedding_dim"]),
                "finite": bool(torch.isfinite(payload["embeddings"]).all()),
                "l2_normalized": bool(torch.allclose(payload["embeddings"].norm(dim=1), torch.ones(expected_count), atol=2e-4)),
                "sample_order": ids == (source_ids if domain == "source" else expected_manifest_ids[domain]),
                "label_order": labels == (source_labels if domain == "source" else expected_manifest_labels[domain]),
            }
            if domain == "source":
                identity = payload.get("identity", {})
                checks.update({
                    "source_split_identity": identity.get("dataset_split_hash") == config["frozen_manifests"]["source_split_identity"],
                    "encoder_revision": identity.get("encoder_revision") == spec["revision"],
                    "preprocessing_identity_present": bool(identity.get("preprocessing_identity")),
                })
            if not all(checks.values()):
                failures.append(f"{encoder}:{domain}")
            records[encoder][domain] = {
                "cache": str(cache_path.relative_to(ROOT)),
                "cache_sha256": sha256_file(cache_path),
                "samples": len(ids),
                "embedding_dimension": int(payload["embeddings"].shape[1]),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
    manifest_checks = {
        "sultana": sha256_file(root_path(config["frozen_manifests"]["sultana"])) == config["frozen_manifests"]["sultana_sha256"],
        "fruitvision": sha256_file(root_path(config["frozen_manifests"]["fruitvision"])) == config["frozen_manifests"]["fruitvision_sha256"],
        "prompt_registry": sha256_file(ROOT / "v2/configs/zero_shot_prompts.yaml") == "e902ba9a66fab36ff86c6440e34c241095d3f947e05c12d86a51d635ca2c67d7",
    }
    if not all(manifest_checks.values()):
        failures.append("frozen_manifest_or_prompt")
    save_json({
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "verified_at": utc_timestamp(),
        "records": records,
        "frozen_identity_checks": manifest_checks,
        "failures": failures,
        "status": "PASS" if not failures else "BLOCKED",
    }, output)
    if failures:
        raise RuntimeError(f"Embedding verification failed: {failures}")
    print("Verified nine frozen embedding caches and all manifest/order identities.")


def read_predictions(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["system"]][row["image_identifier"]] = row
    return dict(result)


def freeze_manifests(config: dict[str, Any]) -> None:
    balanced_path = root_path(config["outputs"]["balanced_manifest"])
    attribution_path = root_path(config["outputs"]["attribution_manifest"])
    if balanced_path.exists() or attribution_path.exists():
        raise FileExistsError("Refusing to overwrite frozen Experiment 015 manifests.")
    verification = read_json(RESULTS / "embedding_verification.json")
    if verification["status"] != "PASS":
        raise RuntimeError("Embedding verification must pass before manifest freeze.")
    caches = load_caches(config)
    seed = int(config["balanced_pool"]["seed"])
    per_cell = int(config["balanced_pool"]["samples_per_domain_class"])
    records: list[dict[str, Any]] = []
    reference = caches["dinov2"]
    for domain in DOMAINS:
        labels = reference[domain]["labels"].tolist()
        ids = list(reference[domain]["sample_ids"])
        for class_index, class_name in enumerate(CLASS_NAMES):
            indices = [i for i, value in enumerate(labels) if value == class_index]
            ordered = sorted(indices, key=lambda i: stable_key(seed, domain, class_name, ids[i]))
            if len(ordered) < per_cell:
                raise RuntimeError(f"Insufficient {domain}/{class_name}: {len(ordered)}")
            for index in ordered[:per_cell]:
                records.append({
                    "domain": domain,
                    "semantic_class": class_name,
                    "fruit_identity": class_name.split("_", 1)[1],
                    "freshness_condition": class_name.split("_", 1)[0],
                    "identifier": ids[index],
                    "cache_index": index,
                    "selection_sha256": stable_key(seed, domain, class_name, ids[index]),
                })
    save_json({
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "status": "frozen_before_geometry_analysis",
        "frozen_at": utc_timestamp(),
        "seed": seed,
        "selection_policy": config["balanced_pool"]["policy"],
        "samples_per_domain_class": per_cell,
        "total_samples": len(records),
        "records": records,
    }, balanced_path)
    (RESULTS / "balanced_analysis_manifest.sha256").write_text(f"{sha256_file(balanced_path)}  balanced_analysis_manifest.json\n", encoding="utf-8")

    phase_predictions = {
        "sultana": read_predictions(PHASE13 / "predictions.csv"),
        "fruitvision": read_predictions(PHASE14 / "fruitvision_predictions.csv"),
    }
    selection: list[dict[str, Any]] = []
    n_select = int(config["attribution"]["samples_per_domain_class_stratum"])
    for domain in config["attribution"]["domains"]:
        prediction_sets = phase_predictions[domain]
        common_ids = set.intersection(*(set(prediction_sets[system]) for system in ATTRIBUTION_SYSTEMS))
        for class_name in CLASS_NAMES:
            eligible = []
            for identifier in common_ids:
                rows = {system: prediction_sets[system][identifier] for system in ATTRIBUTION_SYSTEMS}
                if rows[ATTRIBUTION_SYSTEMS[0]]["ground_truth_class"] != class_name:
                    continue
                correct_count = sum(row["correct"] == "True" for row in rows.values())
                stratum = "majority_correct" if correct_count >= 3 else "majority_error"
                eligible.append((identifier, stratum, correct_count, rows))
            for stratum in ("majority_correct", "majority_error"):
                candidates = [item for item in eligible if item[1] == stratum]
                candidates.sort(key=lambda item: stable_key(seed, domain, class_name, stratum, item[0]))
                for identifier, _, correct_count, rows in candidates[:n_select]:
                    selection.append({
                        "domain": domain,
                        "semantic_class": class_name,
                        "fruit_identity": class_name.split("_", 1)[1],
                        "freshness_condition": class_name.split("_", 1)[0],
                        "correctness_stratum": stratum,
                        "correct_system_count": correct_count,
                        "identifier": identifier,
                        "selection_sha256": stable_key(seed, domain, class_name, stratum, identifier),
                        "frozen_predictions": {system: {
                            "predicted_class": rows[system]["predicted_class"],
                            "correct": rows[system]["correct"] == "True",
                            "margin": float(rows[system]["margin"]),
                        } for system in ATTRIBUTION_SYSTEMS},
                    })
    save_json({
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "status": "frozen_before_attribution_generation",
        "frozen_at": utc_timestamp(),
        "seed": seed,
        "domains": list(config["attribution"]["domains"]),
        "correctness_definition": config["attribution"]["correctness_stratum"],
        "selection_policy": config["attribution"]["selection_order"],
        "target_per_domain_class_stratum": n_select,
        "same_images_for_all_five_systems": True,
        "source_omission": "No frozen per-image source prediction record exists; source was not rescored because Experiments 001-014 are frozen.",
        "total_images": len(selection),
        "records": selection,
    }, attribution_path)
    (RESULTS / "attribution_subset_manifest.sha256").write_text(f"{sha256_file(attribution_path)}  attribution_subset_manifest.json\n", encoding="utf-8")
    print(f"Frozen balanced pool ({len(records)}) and attribution subset ({len(selection)}).")


def balanced_arrays(caches: dict[str, dict[str, dict[str, Any]]], manifest: dict[str, Any], encoder: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    xs = []
    labels: dict[str, list[Any]] = {"semantic_class": [], "fruit_identity": [], "freshness_condition": [], "domain": []}
    for item in manifest["records"]:
        xs.append(caches[encoder][item["domain"]]["embeddings"][item["cache_index"]].numpy())
        for key in labels:
            labels[key].append(item[key])
    return np.vstack(xs), {key: np.asarray(value) for key, value in labels.items()}


def cv_probe(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["analytical_probes"]
    splitter = StratifiedKFold(n_splits=int(settings["folds"]), shuffle=True, random_state=int(settings["seed"]))
    folds = []
    for fold, (train, test) in enumerate(splitter.split(x, y), 1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(settings["regularization_c"]), solver=str(settings["solver"]), max_iter=int(settings["max_iterations"]), random_state=int(settings["seed"])),
        )
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        folds.append({"fold": fold, "accuracy": float(accuracy_score(y[test], pred)), "macro_f1": float(f1_score(y[test], pred, average="macro"))})
    return {
        "folds": folds,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])),
        "accuracy_fold_std": float(np.std([f["accuracy"] for f in folds], ddof=1)),
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
        "macro_f1_fold_std": float(np.std([f["macro_f1"] for f in folds], ddof=1)),
        "interpretation": "Exploratory analytical probe only; not a predictive model candidate.",
    }


def geometry_and_probes(config: dict[str, Any], caches: dict[str, dict[str, dict[str, Any]]], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    geometry: dict[str, Any] = {"schema_version": 1, "balanced_manifest_sha256": sha256_file(RESULTS / "balanced_analysis_manifest.json"), "encoders": {}}
    separability: dict[str, Any] = {"schema_version": 1, "protocol": {"silhouette_metric": "cosine", "secondary_metric": "calinski_harabasz"}, "encoders": {}}
    domain_probes: dict[str, Any] = {"schema_version": 1, "protocol": config["analytical_probes"], "encoders": {}}
    factor_probes: dict[str, Any] = {"schema_version": 1, "protocol": config["analytical_probes"], "encoders": {}}
    pca_payload: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for encoder in ENCODERS:
        x, labels = balanced_arrays(caches, manifest, encoder)
        pca = PCA(n_components=2, random_state=int(config["geometry"]["pca_seed"]))
        projected = pca.fit_transform(x)
        pca_payload[encoder] = (projected, labels)
        geometry["encoders"][encoder] = {"samples": len(x), "embedding_dimension": x.shape[1], "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
        separability["encoders"][encoder] = {}
        for factor in ("semantic_class", "fruit_identity", "freshness_condition", "domain"):
            y = labels[factor]
            separability["encoders"][encoder][factor] = {
                "silhouette_cosine": float(silhouette_score(x, y, metric="cosine")),
                "calinski_harabasz": float(calinski_harabasz_score(x, y)),
                "boundary": "Unsupervised cluster statistic; not classification accuracy.",
            }
        domain_probes["encoders"][encoder] = cv_probe(x, labels["domain"], config)
        factor_probes["encoders"][encoder] = {
            "six_class": cv_probe(x, labels["semantic_class"], config),
            "fruit_identity": cv_probe(x, labels["fruit_identity"], config),
            "freshness_condition": cv_probe(x, labels["freshness_condition"], config),
        }
        fruit_acc = factor_probes["encoders"][encoder]["fruit_identity"]["accuracy_mean"]
        fresh_acc = factor_probes["encoders"][encoder]["freshness_condition"]["accuracy_mean"]
        factor_probes["encoders"][encoder]["freshness_minus_fruit_accuracy"] = fresh_acc - fruit_acc
        factor_probes["encoders"][encoder]["freshness_to_fruit_accuracy_ratio"] = fresh_acc / fruit_acc
    plot_pca(pca_payload)
    plot_factor_probes(factor_probes)
    return geometry, separability, domain_probes, factor_probes


def plot_pca(payload: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]) -> None:
    factors = {
        "class": ("semantic_class", CLASS_NAMES),
        "fruit": ("fruit_identity", ["apple", "banana", "orange"]),
        "freshness": ("freshness_condition", ["fresh", "rotten"]),
        "domain": ("domain", DOMAINS),
    }
    display = {"dinov2": "DINOv2", "clip": "CLIP", "siglip2": "SigLIP2"}
    for suffix, (factor, values) in factors.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, encoder in zip(axes, ENCODERS):
            points, labels = payload[encoder]
            for value in values:
                mask = labels[factor] == value
                ax.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.28, label=value)
            ax.set_title(display[encoder]); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        axes[-1].legend(fontsize=8, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle(f"Frozen representation PCA by {factor.replace('_', ' ')}")
        fig.tight_layout(); fig.savefig(RESULTS / f"representation_pca_by_{suffix}.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_factor_probes(results: dict[str, Any]) -> None:
    x = np.arange(3); width = 0.34
    fruit = [results["encoders"][e]["fruit_identity"]["accuracy_mean"] for e in ENCODERS]
    fresh = [results["encoders"][e]["freshness_condition"]["accuracy_mean"] for e in ENCODERS]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(x-width/2, fruit, width, label="Fruit identity"); ax.bar(x+width/2, fresh, width, label="Freshness")
    ax.set_xticks(x, ["DINOv2", "CLIP", "SigLIP2"]); ax.set_ylim(0, 1.05); ax.set_ylabel("5-fold CV accuracy"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(RESULTS / "factor_separability.png", dpi=180); plt.close(fig)


def centroid_direction(embeddings: torch.Tensor, labels: torch.Tensor, fruit: str) -> torch.Tensor:
    fresh_index = CLASS_TO_IDX[f"fresh_{fruit}"]; rotten_index = CLASS_TO_IDX[f"rotten_{fruit}"]
    direction = embeddings[labels == rotten_index].mean(0) - embeddings[labels == fresh_index].mean(0)
    return F.normalize(direction, p=2, dim=0)


def bootstrap_directions(embeddings: np.ndarray, labels: np.ndarray, fruit: str, resamples: int, seed: int) -> np.ndarray:
    groups = []
    rng = np.random.default_rng(seed)
    for class_name in (f"fresh_{fruit}", f"rotten_{fruit}"):
        values = embeddings[labels == CLASS_TO_IDX[class_name]]
        n = len(values)
        weights = rng.multinomial(n, np.full(n, 1/n), size=resamples).astype(np.float32)
        groups.append((weights @ values) / n)
    directions = groups[1] - groups[0]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.maximum(norms, 1e-12)


def direction_analysis(config: dict[str, Any], caches: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    result: dict[str, Any] = {"schema_version": 1, "definition": "empirical fresh-to-rotten centroid direction", "encoders": {}}
    boot: dict[str, Any] = {"schema_version": 1, "seed": 42, "resamples": int(config["directions"]["bootstrap_resamples"]), "method": "Independent class-stratified bootstrap within each domain/fruit condition; percentile 95% interval.", "encoders": {}}
    pairs = [("source", "sultana"), ("source", "fruitvision"), ("sultana", "fruitvision")]
    for encoder in ENCODERS:
        point: dict[tuple[str, str], np.ndarray] = {}
        bootstrap: dict[tuple[str, str], np.ndarray] = {}
        for domain in DOMAINS:
            payload = caches[encoder][domain]
            x = payload["embeddings"]
            y = payload["labels"]
            for fruit in ("apple", "banana", "orange"):
                point[(domain, fruit)] = centroid_direction(x, y, fruit).numpy()
                local_seed = int(stable_key(42, encoder, domain, fruit)[:8], 16)
                bootstrap[(domain, fruit)] = bootstrap_directions(x.numpy(), y.numpy(), fruit, int(config["directions"]["bootstrap_resamples"]), local_seed)
        per_fruit = {}; boot_fruit = {}
        for fruit in ("apple", "banana", "orange"):
            per_fruit[fruit] = {}; boot_fruit[fruit] = {}
            for first, second in pairs:
                key = f"{first}_vs_{second}"
                value = float(np.dot(point[(first, fruit)], point[(second, fruit)]))
                distribution = np.sum(bootstrap[(first, fruit)] * bootstrap[(second, fruit)], axis=1)
                per_fruit[fruit][key] = value
                boot_fruit[fruit][key] = {"estimate": value, "bootstrap_mean": float(distribution.mean()), "lower": float(np.quantile(distribution, .025)), "upper": float(np.quantile(distribution, .975))}
        cross_fruit = {}
        for domain in DOMAINS:
            values = {}
            for first, second in (("apple", "banana"), ("apple", "orange"), ("banana", "orange")):
                values[f"{first}_vs_{second}"] = float(np.dot(point[(domain, first)], point[(domain, second)]))
            cross_fruit[domain] = {"pairs": values, "mean": float(np.mean(list(values.values()))), "range": [float(np.min(list(values.values()))), float(np.max(list(values.values())))]}
        flat = [per_fruit[f][f"{a}_vs_{b}"] for f in per_fruit for a, b in pairs]
        result["encoders"][encoder] = {"cross_domain_by_fruit": per_fruit, "cross_domain_mean": float(np.mean(flat)), "cross_domain_range": [float(np.min(flat)), float(np.max(flat))], "cross_fruit_within_domain": cross_fruit}
        boot["encoders"][encoder] = boot_fruit
    plot_direction_alignment(result)
    return result, boot


def plot_direction_alignment(result: dict[str, Any]) -> None:
    pair_keys = ["source_vs_sultana", "source_vs_fruitvision", "sultana_vs_fruitvision"]
    x = np.arange(3); width = .25
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, encoder in zip((-width, 0, width), ENCODERS):
        values = [np.mean([result["encoders"][encoder]["cross_domain_by_fruit"][fruit][pair] for fruit in ("apple", "banana", "orange")]) for pair in pair_keys]
        ax.bar(x+offset, values, width, label=encoder.upper())
    ax.set_xticks(x, ["Source–Sultana", "Source–FruitVision", "Sultana–FruitVision"]); ax.set_ylabel("Mean direction cosine"); ax.axhline(0, color="black", lw=.8); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(RESULTS / "freshness_direction_alignment.png", dpi=180); plt.close(fig)


def centroid_analysis(caches: dict[str, dict[str, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    distances: dict[str, Any] = {"schema_version": 1, "metric": "cosine_distance", "encoders": {}}
    drift: dict[str, Any] = {"schema_version": 1, "metric": "cosine_distance", "encoders": {}}
    for encoder in ENCODERS:
        centroids: dict[str, np.ndarray] = {}
        distances["encoders"][encoder] = {}
        for domain in DOMAINS:
            payload = caches[encoder][domain]; x=payload["embeddings"]; y=payload["labels"]
            matrix=[]
            for i, name in enumerate(CLASS_NAMES):
                value=F.normalize(x[y==i].mean(0),p=2,dim=0).numpy(); centroids[f"{domain}:{name}"]=value
            for a in CLASS_NAMES:
                matrix.append([float(1-np.dot(centroids[f"{domain}:{a}"],centroids[f"{domain}:{b}"])) for b in CLASS_NAMES])
            within=[matrix[CLASS_TO_IDX[f"fresh_{fruit}"]][CLASS_TO_IDX[f"rotten_{fruit}"]] for fruit in ("apple","banana","orange")]
            between=[matrix[i][j] for i in range(6) for j in range(i+1,6) if CLASS_NAMES[i].split("_",1)[1]!=CLASS_NAMES[j].split("_",1)[1]]
            distances["encoders"][encoder][domain]={"matrix":matrix,"within_fruit_fresh_rotten":dict(zip(("apple","banana","orange"),within)),"within_fruit_mean":float(np.mean(within)),"between_fruit_mean":float(np.mean(between)),"within_minus_between":float(np.mean(within)-np.mean(between))}
        drift["encoders"][encoder]={}
        for domain in ("sultana","fruitvision"):
            values={name:float(1-np.dot(centroids[f"source:{name}"],centroids[f"{domain}:{name}"])) for name in CLASS_NAMES}
            drift["encoders"][encoder][f"source_to_{domain}"]={"by_class":values,"mean":float(np.mean(list(values.values()))),"range":[float(np.min(list(values.values()))),float(np.max(list(values.values())))]}
    plot_centroid_heatmaps(distances); plot_centroid_drift(drift)
    return distances,drift


def plot_centroid_heatmaps(result: dict[str, Any]) -> None:
    fig,axes=plt.subplots(3,3,figsize=(14,12),layout="constrained"); vmax=max(max(max(row) for row in result["encoders"][e][d]["matrix"]) for e in ENCODERS for d in DOMAINS)
    for i,e in enumerate(ENCODERS):
        for j,d in enumerate(DOMAINS):
            im=axes[i,j].imshow(result["encoders"][e][d]["matrix"],vmin=0,vmax=vmax,cmap="magma")
            axes[i,j].set_title(f"{e.upper()} — {d}"); axes[i,j].set_xticks(range(6),CLASS_NAMES,rotation=55,ha="right",fontsize=6)
            axes[i,j].set_yticks(range(6),CLASS_NAMES if j==0 else [],fontsize=6)
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Cosine distance",shrink=.7); fig.suptitle("Six-class centroid cosine-distance matrices"); fig.savefig(RESULTS/"class_centroid_distance_heatmap.png",dpi=180); plt.close(fig)


def plot_centroid_drift(result: dict[str, Any]) -> None:
    x=np.arange(3); width=.34; fig,ax=plt.subplots(figsize=(8,5))
    for offset,domain in zip((-width/2,width/2),("source_to_sultana","source_to_fruitvision")):
        ax.bar(x+offset,[result["encoders"][e][domain]["mean"] for e in ENCODERS],width,label=domain.replace("source_to_", "Source → ").title())
    ax.set_xticks(x,["DINOv2","CLIP","SigLIP2"]); ax.set_ylabel("Mean class-centroid cosine drift"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"centroid_drift.png",dpi=180); plt.close(fig)


def retrieval_analysis(caches: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any]={"schema_version":1,"gallery":"source","metric":"cosine_similarity","definitions":{"consistency_at_k":"mean fraction of the k retrieved neighbors matching the query factor","hit_at_k":"fraction of queries with at least one matching neighbor"},"encoders":{}}
    fieldnames=["encoder","query_domain","query_identifier","query_class","top1_identifier","top1_class","top1_similarity","fruit_consistency_at_1","fruit_consistency_at_5","fruit_consistency_at_10","freshness_consistency_at_1","freshness_consistency_at_5","freshness_consistency_at_10","six_class_consistency_at_1","six_class_consistency_at_5","six_class_consistency_at_10","top10_identifiers","top10_classes","top10_similarities"]
    with (RESULTS/"retrieval_records.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames); writer.writeheader()
        for encoder in ENCODERS:
            source=caches[encoder]["source"]; gallery=source["embeddings"]; gy=source["labels"].numpy(); gids=list(source["sample_ids"]); output["encoders"][encoder]={}
            for domain in ("sultana","fruitvision"):
                query=caches[encoder][domain]; qx=query["embeddings"]; qy=query["labels"].numpy(); qids=list(query["sample_ids"])
                sums={factor:{k:0.0 for k in (1,5,10)} for factor in ("fruit","freshness","six_class")}; hits={factor:{k:0 for k in (1,5,10)} for factor in sums}
                for start in range(0,len(qx),256):
                    values,indices=torch.topk(qx[start:start+256]@gallery.T,10,dim=1); indices=indices.numpy(); values=values.numpy()
                    for local,(neighbors,sims) in enumerate(zip(indices,values)):
                        qi=start+local; truth=CLASS_NAMES[int(qy[qi])]; neighbor_classes=[CLASS_NAMES[int(gy[idx])] for idx in neighbors]
                        row={"encoder":encoder,"query_domain":domain,"query_identifier":qids[qi],"query_class":truth,"top1_identifier":gids[neighbors[0]],"top1_class":neighbor_classes[0],"top1_similarity":float(sims[0]),"top10_identifiers":"|".join(gids[idx] for idx in neighbors),"top10_classes":"|".join(neighbor_classes),"top10_similarities":"|".join(f"{v:.8f}" for v in sims)}
                        query_fresh,query_fruit=truth.split("_",1)
                        factor_matches={
                            "fruit":np.asarray([name.split("_",1)[1]==query_fruit for name in neighbor_classes]),
                            "freshness":np.asarray([name.split("_",1)[0]==query_fresh for name in neighbor_classes]),
                            "six_class":np.asarray([name==truth for name in neighbor_classes]),
                        }
                        for factor,matches in factor_matches.items():
                            for k in (1,5,10):
                                consistency=float(matches[:k].mean()); sums[factor][k]+=consistency; hits[factor][k]+=int(matches[:k].any()); row[f"{factor}_consistency_at_{k}"]=consistency
                        writer.writerow(row)
                n=len(qx); output["encoders"][encoder][domain]={factor:{f"consistency_at_{k}":sums[factor][k]/n for k in (1,5,10)}|{f"hit_at_{k}":hits[factor][k]/n for k in (1,5,10)} for factor in sums}
    plot_retrieval(output); return output


def plot_retrieval(result: dict[str, Any]) -> None:
    factors=("fruit","freshness","six_class"); fig,axes=plt.subplots(1,3,figsize=(14,4.5)); x=np.arange(3); width=.35
    for ax,factor in zip(axes,factors):
        for offset,domain in zip((-width/2,width/2),("sultana","fruitvision")):
            ax.bar(x+offset,[result["encoders"][e][domain][factor]["consistency_at_10"] for e in ENCODERS],width,label=domain.title())
        ax.set_title(factor.replace("_"," ").title()); ax.set_xticks(x,["DINOv2","CLIP","SigLIP2"]); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.25)
    axes[0].set_ylabel("Source-neighbor consistency@10"); axes[-1].legend(); fig.tight_layout(); fig.savefig(RESULTS/"cross_domain_retrieval.png",dpi=180); plt.close(fig)


def disagreement_analysis() -> dict[str, Any]:
    sources={"sultana":read_predictions(PHASE13/"predictions.csv"),"fruitvision":read_predictions(PHASE14/"fruitvision_predictions.csv")}; output={"schema_version":1,"families":{}}
    pairs={"clip":("clip_linear_probe","clip_strict_zero_shot"),"siglip2":("siglip2_linear_probe","siglip2_strict_zero_shot")}
    for family,(probe,zero) in pairs.items():
        output["families"][family]={}
        for domain,preds in sources.items():
            ids=sorted(set(preds[probe])&set(preds[zero])); categories=defaultdict(list); per_class={name:defaultdict(int) for name in CLASS_NAMES}
            agreement=0
            for identifier in ids:
                p,z=preds[probe][identifier],preds[zero][identifier]; agreement+=p["predicted_class"]==z["predicted_class"]
                pc,zc=p["correct"]=="True",z["correct"]=="True"
                category="both_correct" if pc and zc else "probe_only_correct" if pc else "zero_shot_only_correct" if zc else "both_wrong"
                categories[category].append({"zero_margin":float(z["margin"]),"probe_margin":float(p["margin"])})
                per_class[p["ground_truth_class"]][category]+=1
            n=len(ids); summary={key:len(categories[key])/n for key in ("both_correct","probe_only_correct","zero_shot_only_correct","both_wrong")}
            margins={key:{"count":len(values),"zero_shot_margin_mean":float(np.mean([v["zero_margin"] for v in values])) if values else None,"zero_shot_margin_median":float(np.median([v["zero_margin"] for v in values])) if values else None,"probe_margin_mean":float(np.mean([v["probe_margin"] for v in values])) if values else None} for key,values in categories.items()}
            output["families"][family][domain]={"samples":n,"agreement_rate":agreement/n,**summary,"margin_by_outcome":margins,"per_class":{name:{key:value/ sum(per_class[name].values()) for key,value in per_class[name].items()} for name in CLASS_NAMES}}
    plot_disagreement(output); return output


def plot_disagreement(result: dict[str, Any]) -> None:
    labels=["Both correct","Probe only","Zero-shot only","Both wrong"]; keys=["both_correct","probe_only_correct","zero_shot_only_correct","both_wrong"]; groups=[("clip","sultana"),("clip","fruitvision"),("siglip2","sultana"),("siglip2","fruitvision")]; x=np.arange(4); bottom=np.zeros(4); fig,ax=plt.subplots(figsize=(10,5))
    for key,label in zip(keys,labels):
        values=[result["families"][f][d][key] for f,d in groups]; ax.bar(x,values,bottom=bottom,label=label); bottom+=values
    ax.set_xticks(x,[f"{f.upper()}\n{d.title()}" for f,d in groups]); ax.set_ylim(0,1); ax.set_ylabel("Fraction of images"); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(RESULTS/"probe_vs_zero_shot_disagreement.png",dpi=180); plt.close(fig)


def failure_associations(drift: dict[str, Any], directions: dict[str, Any], retrieval: dict[str, Any], disagreement: dict[str, Any]) -> dict[str, Any]:
    predictions={"sultana":read_predictions(PHASE13/"predictions.csv"),"fruitvision":read_predictions(PHASE14/"fruitvision_predictions.csv")}; rows=[]
    probe_system={"dinov2":"dinov2_linear_probe","clip":"clip_linear_probe","siglip2":"siglip2_linear_probe"}
    for encoder in ENCODERS:
        for domain in ("sultana","fruitvision"):
            pred=predictions[domain][probe_system[encoder]]
            for class_name in CLASS_NAMES:
                relevant=[r for r in pred.values() if r["ground_truth_class"]==class_name]; fruit=class_name.split("_",1)[1]
                rows.append({"encoder":encoder,"domain":domain,"class":class_name,"probe_error_rate":1-np.mean([r["correct"]=="True" for r in relevant]),"centroid_drift":drift["encoders"][encoder][f"source_to_{domain}"]["by_class"][class_name],"freshness_direction_alignment":directions["encoders"][encoder]["cross_domain_by_fruit"][fruit][f"source_vs_{domain}"],"retrieval_six_class_consistency_at_10":retrieval["encoders"][encoder][domain]["six_class"]["consistency_at_10"]})
    metrics={key:np.asarray([r[key] for r in rows]) for key in ("probe_error_rate","centroid_drift","freshness_direction_alignment","retrieval_six_class_consistency_at_10")}; correlations={}
    for key in ("centroid_drift","freshness_direction_alignment","retrieval_six_class_consistency_at_10"):
        stat=spearmanr(metrics["probe_error_rate"],metrics[key]); correlations[f"probe_error_rate_vs_{key}"]={"spearman_rho":float(stat.statistic),"p_value_descriptive":float(stat.pvalue),"n_cells":len(rows)}
    zero_margin={}
    for family in ("clip","siglip2"):
        zero_margin[family]={}
        for domain in ("sultana","fruitvision"):
            groups=disagreement["families"][family][domain]["margin_by_outcome"]
            zero_margin[family][domain]={"zero_shot_only_correct_mean_margin":groups.get("zero_shot_only_correct",{}).get("zero_shot_margin_mean"),"both_wrong_mean_margin":groups.get("both_wrong",{}).get("zero_shot_margin_mean"),"both_correct_mean_margin":groups.get("both_correct",{}).get("zero_shot_margin_mean")}
    return {"schema_version":1,"boundary":"Descriptive associations do not establish causal mechanisms.","cells":rows,"correlations":correlations,"zero_shot_margin_associations":zero_margin}


def analyze(config: dict[str, Any]) -> None:
    if any((RESULTS/name).exists() for name in ("embedding_geometry.json","retrieval_metrics.json")):
        raise FileExistsError("Refusing to overwrite completed Experiment 015 geometry analysis.")
    manifest=read_json(config["outputs"]["balanced_manifest"])
    if manifest["status"]!="frozen_before_geometry_analysis" or sha256_file(root_path(config["outputs"]["balanced_manifest"]))!=(RESULTS/"balanced_analysis_manifest.sha256").read_text().split()[0]: raise RuntimeError("Balanced manifest identity failed.")
    caches=load_caches(config)
    geometry,separability,domain_probes,factor_probes=geometry_and_probes(config,caches,manifest)
    save_json(geometry,RESULTS/"embedding_geometry.json"); save_json(separability,RESULTS/"separability_metrics.json"); save_json(domain_probes,RESULTS/"domain_probe_results.json"); save_json(factor_probes,RESULTS/"factor_probe_results.json")
    directions,boot=direction_analysis(config,caches); save_json(directions,RESULTS/"freshness_direction_alignment.json"); save_json(boot,RESULTS/"direction_bootstrap_intervals.json")
    distances,drift=centroid_analysis(caches); save_json(distances,RESULTS/"class_centroid_distances.json"); save_json(drift,RESULTS/"centroid_drift.json")
    retrieval=retrieval_analysis(caches); save_json(retrieval,RESULTS/"retrieval_metrics.json")
    disagreement=disagreement_analysis(); save_json(disagreement,RESULTS/"probe_zero_shot_disagreement.json")
    save_json(failure_associations(drift,directions,retrieval,disagreement),RESULTS/"failure_mode_correlations.json")
    print("Completed frozen-embedding geometry, probes, directions, centroids, retrieval, and disagreement analyses.")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("verify","manifests","analyze")); parser.add_argument("--config",default="v2/configs/representation_interpretability.yaml"); args=parser.parse_args(); config=load_config(root_path(args.config)); RESULTS.mkdir(parents=True,exist_ok=True)
    {"verify":verify,"manifests":freeze_manifests,"analyze":analyze}[args.stage](config); return 0


if __name__ == "__main__": raise SystemExit(main())
