"""Validate the self-captured set before it is relied on.

Runs every structural check that can be made without looking at a single
capture-quality metric, so that collection mistakes surface while the fruit is
still on the table rather than after calibration has been attempted.

Checks: item catalogue completeness, fruit classes, condition coverage, missing
required captures, duplicate content, cross-split item leakage, unsupported
extensions, unreadable images, implausible dimensions, manifest/image
agreement, and EXIF privacy exposure.

    .venv-competition/bin/python -m competition.evaluation.validate_capture_set
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from competition.data.capture_schema import (
    COLLECTION_TARGETS,
    REQUIRED_CONDITIONS,
    SAMPLED_CONDITIONS,
    SCENE_CAPTURE_COUNT,
    SUPPORTED_EXTENSIONS,
    TARGET,
    CaptureSchemaError,
    Split,
    expected_item_ids,
    parse_item_id,
)
from competition.evaluation.prepare_capture_manifest import (
    DEFAULT_LOCAL_MANIFEST,
    DEFAULT_MANIFEST,
    DEFAULT_RAW_DIR,
)
from competition.evaluation.split_capture_items import (
    DEFAULT_SPLIT_PATH,
    extended_condition_item_ids,
    load_split,
)

# EXIF tags that can carry personal or device-identifying information.
SENSITIVE_EXIF_TAGS: tuple[str, ...] = (
    "GPSInfo", "GPSLatitude", "GPSLongitude", "GPSAltitude",
    "Artist", "XPAuthor", "Copyright", "OwnerName", "CameraOwnerName",
    "BodySerialNumber", "SerialNumber", "LensSerialNumber", "ImageUniqueID",
)


class Finding(dict):
    """A single validation finding: severity, code, and message."""

    def __init__(self, severity: str, code: str, message: str) -> None:
        super().__init__(severity=severity, code=code, message=message)


def scan_exif(path: Path) -> list[str]:
    """Return the names of sensitive EXIF tags present in an image."""
    try:
        from PIL import Image, ExifTags
    except ImportError:  # pragma: no cover - Pillow is a pinned dependency
        return []

    try:
        with Image.open(path) as handle:
            exif = handle.getexif()
            if not exif:
                return []
            names = {ExifTags.TAGS.get(tag, str(tag)) for tag in exif.keys()}
            # GPS lives in its own IFD and does not appear in the top-level keys.
            gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
            if gps_ifd:
                names.add("GPSInfo")
    except Exception:  # noqa: BLE001 - a corrupt file is reported elsewhere
        return []

    return sorted(name for name in names if name in SENSITIVE_EXIF_TAGS)


def validate(
    manifest_path: Path = DEFAULT_MANIFEST,
    split_path: Path = DEFAULT_SPLIT_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    local_manifest_path: Path = DEFAULT_LOCAL_MANIFEST,
    target_label: str = TARGET.label,
    check_exif: bool = True,
) -> list[Finding]:
    """Run every structural check. Returns findings; empty means clean."""
    findings: list[Finding] = []
    target = COLLECTION_TARGETS[target_label]

    # --- split -----------------------------------------------------------
    try:
        split_document = load_split(Path(split_path))
    except CaptureSchemaError as error:
        return [Finding("ERROR", "SPLIT_MISSING", str(error))]

    assignment = split_document["assignment"]
    for item_id in expected_item_ids(target):
        if item_id not in assignment:
            findings.append(Finding(
                "ERROR", "ITEM_NOT_IN_SPLIT",
                f"{item_id} is expected by target {target.label} but absent from the split"))

    # --- manifest --------------------------------------------------------
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        findings.append(Finding(
            "INFO", "MANIFEST_ABSENT",
            "No capture manifest yet. Expected before collection begins."))
        return findings

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = manifest.get("captures", [])

    if manifest.get("split_fingerprint") != split_document["fingerprint"]:
        findings.append(Finding(
            "ERROR", "SPLIT_FINGERPRINT_MISMATCH",
            "Manifest was built against a different split. Regenerate the manifest."))

    # --- per-record structure -------------------------------------------
    hashes: dict[str, list[str]] = defaultdict(list)
    item_splits: dict[str, set[str]] = defaultdict(set)
    by_item_condition: dict[str, set[str]] = defaultdict(set)

    for record in captures:
        image_id = record.get("image_id", "<unknown>")
        item_id = record.get("item_id", "")

        try:
            fruit = parse_item_id(item_id)
        except CaptureSchemaError as error:
            findings.append(Finding("ERROR", "BAD_ITEM_ID", f"{image_id}: {error}"))
            continue

        if record.get("fruit_type") != fruit.value:
            findings.append(Finding(
                "ERROR", "FRUIT_MISMATCH",
                f"{image_id}: item id encodes {fruit.value} but record says "
                f"{record.get('fruit_type')}"))

        expected_split = assignment.get(item_id)
        if expected_split and record.get("split") != expected_split:
            findings.append(Finding(
                "ERROR", "SPLIT_DISAGREEMENT",
                f"{image_id}: split {record.get('split')} contradicts the split "
                f"manifest ({expected_split})"))

        hashes[record.get("content_sha256", "")].append(image_id)
        item_splits[item_id].add(record.get("split", ""))
        by_item_condition[item_id].add(record.get("capture_condition", ""))

        width, height = record.get("width", 0), record.get("height", 0)
        if width < 64 or height < 64:
            findings.append(Finding(
                "ERROR", "IMPLAUSIBLE_DIMENSIONS",
                f"{image_id}: {width}x{height} is too small to be a real capture"))

    # --- cross-split item leakage ---------------------------------------
    for item_id, splits in item_splits.items():
        if len(splits) > 1:
            findings.append(Finding(
                "ERROR", "CROSS_SPLIT_ITEM_LEAKAGE",
                f"{item_id} appears in multiple splits {sorted(splits)}; every "
                "capture of one physical item must share a split"))

    # --- duplicate content ----------------------------------------------
    for digest, image_ids in hashes.items():
        if len(image_ids) > 1:
            findings.append(Finding(
                "ERROR", "DUPLICATE_CONTENT",
                f"identical image content in {sorted(image_ids)} - the same file "
                "was probably copied twice"))

    # --- required condition coverage ------------------------------------
    for item_id in sorted(assignment):
        present = by_item_condition.get(item_id)
        if not present:
            findings.append(Finding(
                "WARNING", "ITEM_NOT_CAPTURED",
                f"{item_id} has no captures yet"))
            continue
        for condition in REQUIRED_CONDITIONS:
            if condition.value not in present:
                findings.append(Finding(
                    "ERROR", "MISSING_REQUIRED_CONDITION",
                    f"{item_id} is missing required condition {condition.value}"))

    # --- extended conditions on the preassigned items only --------------
    extended_items = extended_condition_item_ids(split_document)
    if not extended_items:
        findings.append(Finding(
            "ERROR", "EXTENDED_ITEMS_NOT_PREASSIGNED",
            "The split manifest does not name the extended-condition items. "
            "Regenerate it so the choice is fixed before capture."))
    else:
        for item_id, present in sorted(by_item_condition.items()):
            sampled_present = {c.value for c in SAMPLED_CONDITIONS} & present
            if item_id in extended_items:
                for condition in SAMPLED_CONDITIONS:
                    if condition.value not in present:
                        findings.append(Finding(
                            "ERROR", "MISSING_EXTENDED_CONDITION",
                            f"{item_id} is a preassigned extended-condition item but "
                            f"is missing {condition.value}"))
            elif sampled_present:
                findings.append(Finding(
                    "WARNING", "UNEXPECTED_EXTENDED_CONDITION",
                    f"{item_id} is not a preassigned extended-condition item but has "
                    f"{sorted(sampled_present)}. Extra captures are harmless but the "
                    "six preassigned items are the ones the protocol expects."))

    # --- expected totals for the selected protocol mode -----------------
    scene_count = len(manifest.get("scenes", []))
    required_captures = sum(
        1 for record in captures
        if record.get("capture_condition") in {c.value for c in REQUIRED_CONDITIONS})
    extended_captures = sum(
        1 for record in captures
        if record.get("capture_condition") in {c.value for c in SAMPLED_CONDITIONS})
    total = required_captures + extended_captures + scene_count

    expected = {
        "required": target.required_capture_count,
        "extended": target.extended_capture_count,
        "scene": SCENE_CAPTURE_COUNT,
        "total": target.total_capture_count,
    }
    actual = {
        "required": required_captures,
        "extended": extended_captures,
        "scene": scene_count,
        "total": total,
    }
    if actual != expected:
        severity = "WARNING" if total < expected["total"] else "INFO"
        findings.append(Finding(
            severity, "CAPTURE_COUNT_MISMATCH",
            f"protocol {target.label} expects {expected} but the manifest has "
            f"{actual}. Collection may simply be incomplete."))

    # --- raw directory agreement and privacy ----------------------------
    raw_dir = Path(raw_dir)
    if raw_dir.is_dir():
        on_disk = [
            path for path in sorted(raw_dir.rglob("*"))
            if path.is_file() and not path.name.startswith(".")
        ]
        for path in on_disk:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                findings.append(Finding(
                    "ERROR", "UNSUPPORTED_EXTENSION",
                    f"{path.name}: {path.suffix!r} is not a supported image type"))
                continue
            if check_exif:
                sensitive = scan_exif(path)
                if sensitive:
                    findings.append(Finding(
                        "WARNING", "SENSITIVE_EXIF",
                        f"{path.name}: carries {', '.join(sensitive)}. Sanitise "
                        "before any public or committed use."))

        manifest_ids = {record.get("image_id") for record in captures}
        manifest_ids |= {record.get("image_id") for record in manifest.get("scenes", [])}
        disk_ids = {path.stem for path in on_disk
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS}
        for missing in sorted(disk_ids - manifest_ids):
            findings.append(Finding(
                "WARNING", "IMAGE_NOT_IN_MANIFEST",
                f"{missing} is on disk but absent from the manifest; rebuild it"))
        for absent in sorted(manifest_ids - disk_ids):
            findings.append(Finding(
                "ERROR", "MANIFEST_IMAGE_MISSING",
                f"{absent} is in the manifest but not on disk"))

    # --- local manifest must not be committed ---------------------------
    local_path = Path(local_manifest_path)
    if local_path.is_file():
        import subprocess

        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(local_path)],
                capture_output=True, check=False,
            )
            if result.returncode != 0:
                findings.append(Finding(
                    "ERROR", "LOCAL_MANIFEST_NOT_IGNORED",
                    f"{local_path} carries filesystem paths but is not gitignored"))
        except OSError:  # pragma: no cover - git unavailable
            pass

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.validate_capture_set",
        description="Validate the self-captured set before relying on it.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--target", default=TARGET.label, choices=sorted(COLLECTION_TARGETS))
    parser.add_argument("--no-exif-check", action="store_true")
    args = parser.parse_args(argv)

    findings = validate(
        Path(args.manifest), Path(args.split), Path(args.raw_dir),
        target_label=args.target, check_exif=not args.no_exif_check,
    )

    counts = Counter(finding["severity"] for finding in findings)
    for severity in ("ERROR", "WARNING", "INFO"):
        for finding in [f for f in findings if f["severity"] == severity][:40]:
            print(f"  [{severity}] {finding['code']}: {finding['message']}")

    print()
    print(f"errors: {counts['ERROR']}   warnings: {counts['WARNING']}   info: {counts['INFO']}")
    if not counts["ERROR"]:
        print("No blocking problems found.")
    return 1 if counts["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
