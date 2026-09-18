"""Build the capture manifest from locally stored photographs.

Reads the raw capture directory, derives what it can from filenames, hashes
every image, and writes two files:

* ``capture_manifest.json`` — no filesystem paths, safe to commit or report
* ``capture_manifest.local.json`` — LOCAL ONLY, carries paths for execution

The split is **read**, never decided here. Item assignment happens in
``split_capture_items`` before any image exists, and this tool refuses to invent
an assignment for an unknown item.

Expected filename form::

    <ITEM-ID>__<CONDITION>__<REPLICATE>.<ext>
    APL-001__REFERENCE__1.jpg
    BAN-004__DEFOCUS_SEVERE__1.jpg

Scene cases, which are not photographs of one catalogued item::

    SCENE__<CASE>__<REPLICATE>.<ext>
    SCENE__MIXED_FRUIT__1.jpg

    .venv-competition/bin/python -m competition.evaluation.prepare_capture_manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from competition.data.capture_schema import (
    CAPTURE_SCHEMA_VERSION,
    CONDITION_SPEC_BY_ID,
    SUPPORTED_EXTENSIONS,
    CaptureCondition,
    CaptureRecord,
    CaptureSchemaError,
    SceneCase,
    SceneRecord,
    Split,
    VisibleConditionAnnotation,
    parse_item_id,
)
from competition.evaluation.split_capture_items import DEFAULT_SPLIT_PATH, load_split

DEFAULT_RAW_DIR = Path("competition/data/self_captured/raw")
DEFAULT_MANIFEST = Path("competition/data/capture_manifest.json")
DEFAULT_LOCAL_MANIFEST = Path("competition/data/capture_manifest.local.json")
DEFAULT_DEVICE_ID = "PHONE_A"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filename(name: str) -> tuple[str, str, int]:
    """Split ``<SUBJECT>__<KIND>__<REPLICATE>`` into its parts."""
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) != 3:
        raise CaptureSchemaError(
            f"{name!r} does not match <ITEM-ID>__<CONDITION>__<REPLICATE>"
        )
    subject, kind, replicate = parts
    if not replicate.isdigit():
        raise CaptureSchemaError(f"{name!r} has a non-numeric replicate {replicate!r}")
    return subject, kind, int(replicate)


def read_dimensions(path: Path) -> tuple[int, int]:
    """Decode just far enough to get dimensions, and to prove the file is readable."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureSchemaError(f"could not decode {Path(path).name}")
    height, width = image.shape[:2]
    return width, height


def build_records(
    raw_dir: Path,
    split_document: dict,
    device_id: str = DEFAULT_DEVICE_ID,
) -> tuple[list[CaptureRecord], list[SceneRecord], list[dict], list[str]]:
    """Scan the raw directory. Returns (captures, scenes, local_entries, problems)."""
    raw_dir = Path(raw_dir)
    captures: list[CaptureRecord] = []
    scenes: list[SceneRecord] = []
    local: list[dict] = []
    problems: list[str] = []

    if not raw_dir.is_dir():
        return captures, scenes, local, [f"raw directory not found: {raw_dir}"]

    assignment = split_document["assignment"]

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            problems.append(f"{path.name}: unsupported extension {path.suffix!r}")
            continue

        try:
            subject, kind, replicate = parse_filename(path.name)
            width, height = read_dimensions(path)
            digest = file_sha256(path)
        except CaptureSchemaError as error:
            problems.append(str(error))
            continue

        image_id = f"{subject}__{kind}__{replicate}"

        if subject == "SCENE":
            try:
                case = SceneCase(kind)
            except ValueError:
                problems.append(f"{path.name}: unknown scene case {kind!r}")
                continue
            scenes.append(SceneRecord(
                image_id=image_id, scene_case=case, split=Split.CALIBRATION,
                device_id=device_id, content_sha256=digest, width=width, height=height,
            ))
            local.append({"image_id": image_id, "path": str(path)})
            continue

        try:
            fruit = parse_item_id(subject)
        except CaptureSchemaError as error:
            problems.append(f"{path.name}: {error}")
            continue

        if subject not in assignment:
            problems.append(
                f"{path.name}: item {subject} is not in the split manifest; "
                "regenerate the split or correct the item id"
            )
            continue

        try:
            condition = CaptureCondition(kind)
        except ValueError:
            problems.append(f"{path.name}: unknown capture condition {kind!r}")
            continue

        spec = CONDITION_SPEC_BY_ID[condition]
        record = CaptureRecord(
            image_id=image_id,
            item_id=subject,
            fruit_type=fruit,
            capture_condition=condition,
            replicate_id=replicate,
            split=Split(assignment[subject]),
            device_id=device_id,
            quality_target=spec.quality_target,
            content_sha256=digest,
            width=width,
            height=height,
            visible_condition=VisibleConditionAnnotation.UNCERTAIN,
            remediation_expected=spec.remediation_expected,
        )
        try:
            record.validate()
        except CaptureSchemaError as error:
            problems.append(f"{path.name}: {error}")
            continue

        captures.append(record)
        local.append({"image_id": image_id, "path": str(path)})

    return captures, scenes, local, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.prepare_capture_manifest",
        description="Build the capture manifest from locally stored photographs.",
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--local-manifest", default=str(DEFAULT_LOCAL_MANIFEST))
    args = parser.parse_args(argv)

    split_document = load_split(Path(args.split))
    captures, scenes, local, problems = build_records(
        Path(args.raw_dir), split_document, args.device_id
    )

    if not captures and not scenes:
        print(f"No captures found under {args.raw_dir}.")
        print("This is expected before collection begins; see "
              "docs/competition/PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md")
        for problem in problems[:10]:
            print(f"  - {problem}")
        return 1

    manifest = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "split_fingerprint": split_document["fingerprint"],
        "device_id": args.device_id,
        "capture_count": len(captures),
        "scene_count": len(scenes),
        "note": "Contains no filesystem paths; images are addressed by content hash.",
        "captures": [record.to_dict() for record in sorted(captures, key=lambda r: r.image_id)],
        "scenes": [record.to_dict() for record in sorted(scenes, key=lambda r: r.image_id)],
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    Path(args.local_manifest).write_text(json.dumps({
        "warning": "LOCAL ONLY - contains filesystem paths. Not for commit.",
        "split_fingerprint": split_document["fingerprint"],
        "entries": sorted(local, key=lambda e: e["image_id"]),
    }, indent=2, sort_keys=True), encoding="utf-8")

    print(f"captures: {len(captures)}   scenes: {len(scenes)}   problems: {len(problems)}")
    for problem in problems[:10]:
        print(f"  - {problem}")
    print(f"manifest: {args.manifest}")
    print(f"local:    {args.local_manifest} (gitignored)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
