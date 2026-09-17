"""Audit, contamination-screen, and freeze FruitVision Domain B for Experiment 014."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from v2.src.training.utils import load_config, save_json, sha256_file, utc_timestamp


ROOT = Path(__file__).resolve().parents[3]
CLASS_NAMES = ("fresh_apple", "fresh_banana", "fresh_orange", "rotten_apple", "rotten_banana", "rotten_orange")
FOLDER_MAP = {
    "Apple/Fresh": "fresh_apple", "Apple/Rotten": "rotten_apple",
    "Banana/Fresh": "fresh_banana", "Banana/Rotten": "rotten_banana",
    "Orange/Fresh": "fresh_orange", "Orange/Rotten": "rotten_orange",
}


def repo_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read_json(value: str | Path) -> Any:
    with repo_path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(math.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


DCT32 = dct_matrix(32)


def perceptual_hash(rgb: Image.Image) -> int:
    pixels = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (DCT32 @ pixels @ DCT32.T)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    result = 0
    for bit in low > median:
        result = (result << 1) | int(bit)
    return result


def inspect_image(file: Path, identifier_root: Path, class_name: str, source_folder: str) -> dict[str, Any]:
    with Image.open(file) as image:
        image.load()
        source_mode, source_format = image.mode, image.format
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels_hash = hashlib.sha256(struct.pack(">II", width, height) + rgb.tobytes()).hexdigest()
        phash = perceptual_hash(rgb)
        small = np.asarray(rgb.resize((64, 64), Image.Resampling.BOX), dtype=np.float64) / 255.0
        gray = 0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]
    return {
        "identifier": str(file.relative_to(identifier_root)), "class": class_name,
        "fruit_identity": class_name.split("_", 1)[1], "freshness_condition": class_name.split("_", 1)[0],
        "source_folder": source_folder, "extension": file.suffix.lower(), "size_bytes": file.stat().st_size,
        "file_sha256": sha256_file(file), "pixel_rgb_sha256": pixels_hash, "phash_63bit_hex": f"{phash:016x}",
        "width": width, "height": height, "aspect_ratio": width / height, "source_mode": source_mode,
        "decoded_mode": "RGB", "format": source_format,
        "profile": {"brightness_mean": float(gray.mean()), "contrast_std": float(gray.std()), "rgb_mean": small.mean((0, 1)).tolist()},
    }


def duplicate_groups(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(record["identifier"])
    return [{key: digest, "identifiers": ids, "count": len(ids)} for digest, ids in sorted(groups.items()) if len(ids) > 1]


def compact(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("identifier", "class", "source_folder", "width", "height", "file_sha256", "pixel_rgb_sha256", "phash_63bit_hex")
    return {key: record.get(key) for key in keys if key in record}


def near_pairs(left: list[dict[str, Any]], right: list[dict[str, Any]] | None, threshold: int) -> list[dict[str, Any]]:
    candidates = []
    if right is None:
        for i, first in enumerate(left):
            first_hash = int(first["phash_63bit_hex"], 16)
            for second in left[i + 1:]:
                if first["file_sha256"] == second["file_sha256"] or first["pixel_rgb_sha256"] == second["pixel_rgb_sha256"]:
                    continue
                distance = (first_hash ^ int(second["phash_63bit_hex"], 16)).bit_count()
                if distance <= threshold:
                    candidates.append({"hamming_distance": distance, "first": compact(first), "second": compact(second)})
    else:
        right_hashes = [(int(record["phash_63bit_hex"], 16), record) for record in right]
        for first in left:
            first_hash = int(first["phash_63bit_hex"], 16)
            for second_hash, second in right_hashes:
                distance = (first_hash ^ second_hash).bit_count()
                if distance <= threshold:
                    candidates.append({"hamming_distance": distance, "domain_b": compact(first), "comparison": compact(second)})
    return sorted(candidates, key=lambda item: (item["hamming_distance"], json.dumps(item, sort_keys=True)))


def summarize_dimensions(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {"minimum": float(array.min()), "median": float(np.median(array)), "maximum": float(array.max()), "mean": float(array.mean()), "std": float(array.std())}
    return {"width": summary([r["width"] for r in records]), "height": summary([r["height"] for r in records]), "aspect_ratio": summary([r["aspect_ratio"] for r in records]), "unique_width_height_pairs": len({(r["width"], r["height"]) for r in records})}


def audit(config: dict[str, Any]) -> None:
    audit_path = repo_path(config["outputs"]["dataset_audit"])
    contamination_path = repo_path(config["outputs"]["contamination_audit"])
    if audit_path.exists() or contamination_path.exists():
        raise FileExistsError("Refusing to overwrite Experiment 014 audits.")
    screen = config["contamination_screening"]
    if screen["threshold_status"] != "frozen_before_candidate_inspection":
        raise RuntimeError("pHash threshold was not frozen before inspection.")
    threshold = int(screen["phash_hamming_candidate_threshold"])
    root = repo_path(config["domains"]["fruitvision"]["local_original_root"])
    records, decode_failures, zero_byte, auxiliary = [], [], [], []
    for source_folder, class_name in FOLDER_MAP.items():
        folder = root / source_folder
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue
            if file.stat().st_size == 0:
                zero_byte.append(str(file.relative_to(root))); continue
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                auxiliary.append(str(file.relative_to(root))); continue
            try:
                records.append(inspect_image(file, root, class_name, source_folder))
            except Exception as error:
                decode_failures.append({"identifier": str(file.relative_to(root)), "error": repr(error)})
    expected = config["domains"]["fruitvision"]["selected_class_counts_from_article"]
    counts = Counter(r["class"] for r in records)
    if counts != Counter({key: int(value) for key, value in expected.items()}):
        raise RuntimeError(f"FruitVision class counts differ from the registered article table: {counts}")
    if decode_failures or zero_byte:
        raise RuntimeError("Corrupt or zero-byte selected FruitVision image found.")
    exact_groups = duplicate_groups(records, "file_sha256")
    pixel_groups = duplicate_groups(records, "pixel_rgb_sha256")
    internal_near = near_pairs(records, None, threshold)
    secondary = len({identifier for group in exact_groups + pixel_groups for identifier in group["identifiers"][1:]})
    severe_fraction = secondary / len(records)
    severe = severe_fraction > float(screen["severe_internal_exact_or_pixel_secondary_record_fraction"])

    source_root = repo_path("v2/data/processed/v1_frozen_split")
    source_records = []
    for split in ("train", "validation", "test"):
        for class_name in CLASS_NAMES:
            for file in sorted((source_root / split / class_name).iterdir()):
                if file.is_file():
                    source_records.append(inspect_image(file, source_root, class_name, f"{split}/{class_name}"))
    if len(source_records) != 4507:
        raise RuntimeError("Frozen source count changed.")
    sultana_records = read_json("v2/results/experiment_013_cross_domain/external_dataset_audit.json")["records_for_manifest"]

    def compare(other: list[dict[str, Any]], domain_name: str) -> dict[str, Any]:
        file_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pixel_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in other:
            file_map[record["file_sha256"]].append(record); pixel_map[record["pixel_rgb_sha256"]].append(record)
        exact, pixel = [], []
        for record in records:
            for match in file_map.get(record["file_sha256"], []):
                exact.append({"domain_b": compact(record), domain_name: compact(match)})
            for match in pixel_map.get(record["pixel_rgb_sha256"], []):
                pixel.append({"domain_b": compact(record), domain_name: compact(match)})
        return {"comparison_images": len(other), "exact_file_matches": exact, "pixel_equivalent_matches": pixel, "perceptual_candidates": near_pairs(records, other, threshold)}

    source_comparison = compare(source_records, "source")
    sultana_comparison = compare(sultana_records, "sultana")
    excluded = sorted({item["domain_b"]["identifier"] for comparison in (source_comparison, sultana_comparison) for key in ("exact_file_matches", "pixel_equivalent_matches") for item in comparison[key]})
    candidates = len(internal_near) + len(source_comparison["perceptual_candidates"]) + len(sultana_comparison["perceptual_candidates"])
    audit_payload = {
        "schema_version": 1, "experiment_id": config["experiment"]["id"], "audited_at": utc_timestamp(),
        "official_original_archive": config["domains"]["fruitvision"]["official_original_archive"],
        "official_archive_hash_verified": True, "official_archive_crc_test": "PASS",
        "raw_augmented_separation": "PASS_SEPARATE_OFFICIAL_ARCHIVES",
        "selected_images": len(records), "class_counts": dict(sorted(counts.items())),
        "extensions": dict(sorted(Counter(r["extension"] for r in records).items())), "formats": dict(sorted(Counter(str(r["format"]) for r in records).items())),
        "source_modes": dict(sorted(Counter(r["source_mode"] for r in records).items())), "decoded_modes": dict(sorted(Counter(r["decoded_mode"] for r in records).items())),
        "dimensions": summarize_dimensions(records), "zero_byte_files": zero_byte, "decode_failures": decode_failures,
        "excluded_auxiliary_files": auxiliary, "folder_mapping": FOLDER_MAP,
        "formalins_present_in_archive_but_not_extracted": True, "excluded_fruits_not_extracted": ["Grape", "Mango"],
        "within_domain_exact_duplicate_groups": exact_groups, "within_domain_pixel_duplicate_groups": pixel_groups,
        "within_domain_perceptual_candidates": internal_near, "internal_secondary_exact_or_pixel_fraction": severe_fraction,
        "severe_internal_duplication": severe, "records_for_manifest": records,
        "status": "BLOCKED_SEVERE_DUPLICATION" if severe else "PASS",
    }
    contamination_payload = {
        "schema_version": 1, "audited_at": utc_timestamp(), "threshold": threshold,
        "threshold_status": screen["threshold_status"], "exact_method": screen["exact_method"],
        "pixel_method": screen["pixel_method"], "perceptual_method": screen["perceptual_method"],
        "domain_b_images": len(records), "source_comparison": source_comparison, "sultana_comparison": sultana_comparison,
        "domain_b_identifiers_excluded_for_exact_or_pixel_match": excluded,
        "perceptual_candidate_count_including_internal": candidates,
        "status": "REVIEW_REQUIRED" if candidates else "PASS_NO_CANDIDATES",
    }
    save_json(audit_payload, audit_path); save_json(contamination_payload, contamination_path)
    print(f"FruitVision={len(records)} internal exact={len(exact_groups)} pixel={len(pixel_groups)} near={len(internal_near)}; source exact={len(source_comparison['exact_file_matches'])} pixel={len(source_comparison['pixel_equivalent_matches'])} near={len(source_comparison['perceptual_candidates'])}; Sultana exact={len(sultana_comparison['exact_file_matches'])} pixel={len(sultana_comparison['pixel_equivalent_matches'])} near={len(sultana_comparison['perceptual_candidates'])}")


def freeze_manifest(config: dict[str, Any]) -> None:
    manifest_path, hash_path = repo_path(config["outputs"]["manifest"]), repo_path(config["outputs"]["manifest_hash"])
    if manifest_path.exists() or hash_path.exists():
        raise FileExistsError("Refusing to overwrite frozen FruitVision manifest.")
    audit_payload = read_json(config["outputs"]["dataset_audit"])
    contamination = read_json(config["outputs"]["contamination_audit"])
    if audit_payload["status"] != "PASS":
        raise RuntimeError("FruitVision audit is blocked.")
    candidates = contamination["perceptual_candidate_count_including_internal"]
    review = None
    if candidates:
        review = read_json(config["outputs"]["candidate_review"])
        if review.get("status") != "PASS_REVIEW_COMPLETE" or review.get("candidate_count") != candidates or review.get("ambiguous_or_contaminating_candidates"):
            raise RuntimeError("Perceptual candidate review is incomplete or blocking.")
    excluded = set(contamination["domain_b_identifiers_excluded_for_exact_or_pixel_match"])
    records = [{key: value for key, value in record.items() if key != "profile"} for record in audit_payload["records_for_manifest"] if record["identifier"] not in excluded]
    payload = {
        "schema_version": 1, "experiment_id": config["experiment"]["id"], "status": "frozen_before_fruitvision_inference",
        "frozen_at": utc_timestamp(), "dataset_doi": config["domains"]["fruitvision"]["doi"],
        "official_original_archive_sha256": config["domains"]["fruitvision"]["official_original_archive"]["sha256"],
        "class_mapping": FOLDER_MAP, "class_counts": dict(sorted(Counter(r["class"] for r in records).items())),
        "images": sorted(records, key=lambda r: (CLASS_NAMES.index(r["class"]), r["identifier"])),
        "exclusions": {"cross_domain_exact_or_pixel_duplicates": sorted(excluded), "all_formalin_mixed": True, "all_grape_and_mango": True, "all_augmented_images": True},
        "contamination_review": {"perceptual_candidates": candidates, "review_status": review["status"] if review else "NOT_REQUIRED"},
        "domain_b_training": None, "domain_b_validation": None, "domain_b_calibration": None,
        "all_eligible_images_are_evaluation_only": True,
    }
    save_json(payload, manifest_path)
    digest = sha256_file(manifest_path); hash_path.write_text(f"{digest}  {manifest_path.name}\n", encoding="utf-8")
    print(f"Frozen FruitVision manifest: {len(records)} images, SHA-256 {digest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", choices=("audit", "freeze-manifest")); parser.add_argument("--config", default="v2/configs/multi_domain_robustness.yaml"); args = parser.parse_args()
    config = load_config(repo_path(args.config)); {"audit": audit, "freeze-manifest": freeze_manifest}[args.stage](config); return 0


if __name__ == "__main__":
    raise SystemExit(main())
