"""Audit and freeze the original-photo external benchmark for Experiment 013."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CLASSES = (
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


def read_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as handle:
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
    grayscale = rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    coefficients = DCT32 @ pixels @ DCT32.T
    low = coefficients[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def inspect_image(path: Path, identifier_root: Path, class_name: str) -> dict[str, Any]:
    raw_sha = sha256_file(path)
    with Image.open(path) as image:
        image.load()
        source_mode = image.mode
        source_format = image.format
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixel_digest = hashlib.sha256()
        pixel_digest.update(struct.pack(">II", width, height))
        pixel_digest.update(rgb.tobytes())
        phash = perceptual_hash(rgb)
        profile = rgb.resize((64, 64), Image.Resampling.BOX)
        profile_pixels = np.asarray(profile, dtype=np.float64) / 255.0
        grayscale = (
            0.2126 * profile_pixels[:, :, 0]
            + 0.7152 * profile_pixels[:, :, 1]
            + 0.0722 * profile_pixels[:, :, 2]
        )
    return {
        "identifier": str(path.relative_to(identifier_root)),
        "class": class_name,
        "fruit_identity": class_name.split("_", 1)[1],
        "freshness_condition": class_name.split("_", 1)[0],
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "file_sha256": raw_sha,
        "pixel_rgb_sha256": pixel_digest.hexdigest(),
        "phash_63bit_hex": f"{phash:016x}",
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "source_mode": source_mode,
        "decoded_mode": "RGB",
        "format": source_format,
        "profile": {
            "brightness_mean": float(grayscale.mean()),
            "contrast_std": float(grayscale.std()),
            "rgb_mean": profile_pixels.mean(axis=(0, 1)).tolist(),
        },
    }


def duplicate_groups(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record["identifier"])
    return [
        {key: value, "identifiers": identifiers, "count": len(identifiers)}
        for value, identifiers in sorted(grouped.items())
        if len(identifiers) > 1
    ]


def summarize_dimensions(records: list[dict[str, Any]]) -> dict[str, Any]:
    widths = np.asarray([record["width"] for record in records], dtype=float)
    heights = np.asarray([record["height"] for record in records], dtype=float)
    ratios = widths / heights
    return {
        "unique_width_height_pairs": len({(record["width"], record["height"]) for record in records}),
        "width": {"minimum": int(widths.min()), "median": float(np.median(widths)), "maximum": int(widths.max())},
        "height": {"minimum": int(heights.min()), "median": float(np.median(heights)), "maximum": int(heights.max())},
        "aspect_ratio": {"minimum": float(ratios.min()), "median": float(np.median(ratios)), "maximum": float(ratios.max())},
    }


def audit(config: dict[str, Any]) -> None:
    output_audit = repo_path(config["outputs"]["dataset_audit"])
    duplicate_output = repo_path(config["outputs"]["duplicate_audit"])
    if output_audit.exists() or duplicate_output.exists():
        raise FileExistsError("Refusing to overwrite an Experiment 013 dataset audit.")
    threshold = int(config["contamination_screening"]["phash_hamming_candidate_threshold"])
    if config["contamination_screening"]["threshold_status"] != "frozen_before_candidate_inspection":
        raise ValueError("Perceptual candidate threshold is not pre-registered.")
    external_root = repo_path(config["external_dataset"]["local_original_root"])
    mapping = config["external_dataset"]["class_mapping"]
    records: list[dict[str, Any]] = []
    decode_failures = []
    zero_byte = []
    unexpected = []
    for source_folder, canonical_class in mapping.items():
        folder = external_root / source_folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing official original-image folder: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.stat().st_size == 0:
                zero_byte.append(str(path.relative_to(external_root)))
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                unexpected.append(str(path.relative_to(external_root)))
                continue
            try:
                records.append(inspect_image(path, external_root, canonical_class))
            except Exception as error:
                decode_failures.append({"identifier": str(path.relative_to(external_root)), "error": repr(error)})
    counts = Counter(record["class"] for record in records)
    if set(counts) != set(CANONICAL_CLASSES) or any(counts[name] != 200 for name in CANONICAL_CLASSES):
        raise RuntimeError(f"Expected 200 decoded originals per selected class; found {dict(counts)}")
    if decode_failures or zero_byte or unexpected:
        raise RuntimeError("External original-image audit found invalid files; model evaluation is blocked.")

    source_root = repo_path(config["source_dataset"]["root"])
    source_records: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for class_name in CANONICAL_CLASSES:
            for path in sorted((source_root / split / class_name).iterdir()):
                if path.is_file():
                    source_records.append(inspect_image(path, source_root, class_name))
    if len(source_records) != int(config["source_dataset"]["total_images"]):
        raise RuntimeError("Frozen source-dataset file count changed.")

    source_file_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_pixel_hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        source_file_hashes[record["file_sha256"]].append(record)
        source_pixel_hashes[record["pixel_rgb_sha256"]].append(record)
    exact_matches = []
    pixel_matches = []
    excluded_external = set()
    for external in records:
        for source in source_file_hashes.get(external["file_sha256"], []):
            exact_matches.append({"source": source, "external": external})
            excluded_external.add(external["identifier"])
        for source in source_pixel_hashes.get(external["pixel_rgb_sha256"], []):
            pixel_matches.append({"source": source, "external": external})
            excluded_external.add(external["identifier"])

    source_phashes = [(int(record["phash_63bit_hex"], 16), record) for record in source_records]
    candidates = []
    for external in records:
        external_hash = int(external["phash_63bit_hex"], 16)
        for source_hash, source in source_phashes:
            distance = (external_hash ^ source_hash).bit_count()
            if distance <= threshold:
                candidates.append(
                    {
                        "hamming_distance": distance,
                        "source": {key: source[key] for key in ("identifier", "class", "width", "height", "file_sha256", "pixel_rgb_sha256", "phash_63bit_hex")},
                        "external": {key: external[key] for key in ("identifier", "class", "width", "height", "file_sha256", "pixel_rgb_sha256", "phash_63bit_hex")},
                    }
                )
    candidates.sort(key=lambda item: (item["hamming_distance"], item["source"]["identifier"], item["external"]["identifier"]))

    audit_payload = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "audited_at": utc_timestamp(),
        "official_release": config["external_dataset"],
        "original_image_identification": {
            "method": "Selected only the separately published official Mendeley file named Original Image.zip; Augmented Image.zip was neither downloaded nor extracted.",
            "official_archive_hash_verified_before_extraction": True,
            "official_archive_crc_test": "PASS",
            "selected_folder_mapping": mapping,
        },
        "selected_original_images": len(records),
        "class_counts": dict(sorted(counts.items())),
        "extensions": dict(sorted(Counter(record["extension"] for record in records).items())),
        "formats": dict(sorted(Counter(str(record["format"]) for record in records).items())),
        "source_modes": dict(sorted(Counter(record["source_mode"] for record in records).items())),
        "decoded_modes": dict(sorted(Counter(record["decoded_mode"] for record in records).items())),
        "dimensions": summarize_dimensions(records),
        "zero_byte_files": zero_byte,
        "decode_failures": decode_failures,
        "unexpected_files": unexpected,
        "within_external_exact_duplicate_groups": duplicate_groups(records, "file_sha256"),
        "within_external_pixel_duplicate_groups": duplicate_groups(records, "pixel_rgb_sha256"),
        "status": "PASS" if not (decode_failures or zero_byte or unexpected) else "BLOCKED",
        "records_for_manifest": records,
    }
    duplicate_payload = {
        "schema_version": 1,
        "audited_at": utc_timestamp(),
        "source_images_compared": len(source_records),
        "external_original_images_compared": len(records),
        "exact_file_method": config["contamination_screening"]["exact_method"],
        "pixel_equivalent_method": config["contamination_screening"]["pixel_method"],
        "perceptual_method": config["contamination_screening"]["perceptual_method"],
        "phash_hamming_candidate_threshold": threshold,
        "threshold_status": config["contamination_screening"]["threshold_status"],
        "exact_cross_dataset_matches": exact_matches,
        "pixel_equivalent_cross_dataset_matches": pixel_matches,
        "perceptual_near_duplicate_candidates": candidates,
        "external_identifiers_excluded_for_exact_or_pixel_match": sorted(excluded_external),
        "near_duplicate_candidates_require_review_before_manifest_freeze": bool(candidates),
        "status": "REVIEW_REQUIRED" if candidates else "PASS_NO_CANDIDATES",
    }
    save_json(audit_payload, output_audit)
    save_json(duplicate_payload, duplicate_output)
    print(
        f"Audited {len(records)} external originals and {len(source_records)} source images; "
        f"exact={len(exact_matches)} pixel={len(pixel_matches)} phash_candidates={len(candidates)}"
    )


def freeze_manifest(config: dict[str, Any]) -> None:
    manifest_path = repo_path(config["outputs"]["manifest"])
    hash_path = repo_path(config["outputs"]["manifest_hash"])
    if manifest_path.exists() or hash_path.exists():
        raise FileExistsError("Refusing to overwrite the frozen external manifest.")
    audit_payload = read_json(config["outputs"]["dataset_audit"])
    duplicate_payload = read_json(config["outputs"]["duplicate_audit"])
    if audit_payload["status"] != "PASS":
        raise RuntimeError("Dataset audit did not pass.")
    candidates = duplicate_payload["perceptual_near_duplicate_candidates"]
    candidate_review = None
    if candidates:
        review_path = repo_path(config["outputs"]["candidate_review"])
        if not review_path.exists():
            raise RuntimeError("Perceptual candidates require documented review before manifest freeze.")
        candidate_review = read_json(review_path)
        candidate_pairs = {
            (item["source"]["identifier"], item["external"]["identifier"], item["hamming_distance"])
            for item in candidates
        }
        reviewed_pairs = {
            (item["source_identifier"], item["external_identifier"], item["hamming_distance"])
            for item in candidate_review.get("candidate_reviews", [])
            if item.get("disposition") == "false_positive_not_same_photograph"
        }
        if candidate_review.get("status") != "PASS_FALSE_POSITIVES" or reviewed_pairs != candidate_pairs:
            raise RuntimeError("Perceptual candidate review is incomplete or does not match the frozen audit.")
    excluded = set(duplicate_payload["external_identifiers_excluded_for_exact_or_pixel_match"])
    records = [
        {key: value for key, value in record.items() if key != "profile"}
        for record in audit_payload["records_for_manifest"]
        if record["identifier"] not in excluded
    ]
    payload = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "status": "frozen_before_model_evaluation",
        "frozen_at": utc_timestamp(),
        "dataset_doi": config["external_dataset"]["doi"],
        "official_original_archive_sha256": config["external_dataset"]["official_original_archive"]["sha256"],
        "class_mapping": config["external_dataset"]["class_mapping"],
        "class_counts": dict(sorted(Counter(record["class"] for record in records).items())),
        "images": sorted(records, key=lambda record: (CANONICAL_CLASSES.index(record["class"]), record["identifier"])),
        "exclusions": {
            "exact_or_pixel_cross_dataset_duplicates": sorted(excluded),
            "augmentation_derived_images": "Entire official Augmented Image.zip archive excluded.",
            "non_overlapping_classes": sorted(set([
                "fresh_grape", "rotten_grape", "fresh_guava", "rotten_guava",
                "fresh_jujube", "rotten_jujube", "fresh_pomegranate", "rotten_pomegranate",
                "fresh_strawberry", "rotten_strawberry",
            ])),
        },
        "contamination_review": {
            "exact_cross_dataset_matches": len(duplicate_payload["exact_cross_dataset_matches"]),
            "pixel_equivalent_cross_dataset_matches": len(duplicate_payload["pixel_equivalent_cross_dataset_matches"]),
            "perceptual_candidates": len(candidates),
            "perceptual_review_status": candidate_review["status"] if candidate_review else "NOT_REQUIRED",
            "review_file": config["outputs"]["candidate_review"] if candidate_review else None,
        },
        "external_training_split": None,
        "external_validation_split": None,
        "all_eligible_images_are_evaluation_only": True,
    }
    save_json(payload, manifest_path)
    hash_path.write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8")
    print(f"Frozen external manifest with {len(records)} images: {sha256_file(manifest_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "freeze-manifest"))
    parser.add_argument("--config", default="v2/configs/cross_domain_generalization.yaml")
    args = parser.parse_args()
    config = load_config(repo_path(args.config))
    {"audit": audit, "freeze-manifest": freeze_manifest}[args.stage](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
