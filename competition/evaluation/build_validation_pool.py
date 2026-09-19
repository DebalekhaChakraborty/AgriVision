"""A second, untouched real-image pool for confirmatory Phase 2d evaluation.

Why a new pool is required
--------------------------
The Phase 2c-B held-out groups have been spent. They were opened confirmatorily
under policy `78b1e2151773787a`, and Phase 2d changes the detector set. Any
re-run against those groups is therefore exploratory by construction, no matter
how carefully it is done - the numbers would be reported by someone who has
already seen how that data behaves.

So Phase 2d gets its own validation pool, built before its detectors exist and
never used to tune them.

Independence is enforced, not intended
--------------------------------------
Every candidate is rejected if its source group already appears in the Phase 2c-B
corpus. The source group is the creator identity, so this excludes not just the
same photograph but any other photograph by a contributor who is already
represented - which is what stops "a new image" from being another angle of a
fruit the calibration set already contains.

The same licence gate applies unchanged. Nothing enters without a per-file
permissive licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2

from competition.data.licence_validation import LICENCE_GATE_VERSION
from competition.data.licensed_sources import (
    CORPUS_SCHEMA_VERSION,
    CorpusError,
    CorpusTrack,
    LicensedImageRecord,
    ProvenanceType,
    Split,
    capture_family_id,
    source_group_id,
)
from competition.evaluation.build_licensed_corpus import (
    CACHE_DIR,
    CORPUS_ROOT,
    DOWNLOAD_DELAY_S,
    MANIFEST_DIR,
    SOURCE_NAME,
    Candidate,
    CategorySource,
    _download,
    _extension_for,
    _request,
    build_contact_sheets,
    load_corpus,
    rendition_url,
    screen_category,
)
from competition.models.ontology import FruitType

VALIDATION_RAW = CORPUS_ROOT / "validation_raw"
VALIDATION_CACHE = CACHE_DIR / "validation"
VALIDATION_POOL = MANIFEST_DIR / "validation_pool.json"
VALIDATION_CURATION = MANIFEST_DIR / "validation_curation.json"
VALIDATION_MANIFEST = MANIFEST_DIR / "validation_set.json"
VALIDATION_VERSION = "phase2d-validation-pool-1.0.0"

# Categories chosen so that **none** was used to build the Phase 2c-B corpus.
# Excluding shared source groups is necessary but not sufficient: paging deeper
# into the same categories mostly re-surfaces files that were screened and
# rejected the first time, which are eligible again because their creators never
# entered the corpus. Independence is much cleaner if the material was never
# looked at, so the pool is drawn from a disjoint part of the category tree and
# every previously screened file is excluded by id as well.
VALIDATION_SOURCES: tuple[CategorySource, ...] = (
    CategorySource("Malus domestica", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "botanical apple category, not used in Phase 2c-B"),
    CategorySource("Apple cultivars", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "named cultivars, typically single fruit on a plain surface"),
    CategorySource("Apples in Germany", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "regional apple photographs"),
    CategorySource("Musa acuminata", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "botanical banana category"),
    CategorySource("Bananas in India", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "bananas photographed in India, mixed settings"),
    CategorySource("Oranges in Spain", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "regional orange photographs"),
    CategorySource("Sliced oranges", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "orange close-ups; whole fruit kept, cut fruit curated out"),
    CategorySource("Fruit vendors", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "vendors and stalls, heavy clutter"),
    CategorySource("Farmers' markets", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "market scenes, ambient light"),
    CategorySource("Market stalls", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "general stall displays"),
    CategorySource("Bananas in the Philippines", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "bananas in markets and transport"),
    CategorySource("Bananas in Ecuador", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "banana production scenes"),
)


def existing_source_groups() -> set[str]:
    """Every source group already represented in the Phase 2c-B corpus."""
    return {record.source_group_id for record in load_corpus()}


def previously_screened_image_ids() -> set[str]:
    """Every file already looked at in Phase 2c-B, admitted or not.

    The Phase 2c-B candidate pool is far larger than the corpus it produced, and
    a file rejected then is still licence-clean now. Re-admitting one would be
    defensible on provenance grounds and poor on evidentiary ones: it is a photo
    whose appearance already influenced a curation judgement.
    """
    ids = {record.image_id for record in load_corpus()}
    pool_path = MANIFEST_DIR / "candidate_pool.json"
    if pool_path.is_file():
        document = json.loads(pool_path.read_text(encoding="utf-8"))
        ids |= {candidate["image_id"] for candidate in document["candidates"]}
    return ids


def fetch_category_files_paged(
    category: str, pages: int = 3, per_page: int = 200
) -> list[dict]:
    """Walk deeper into a category than a single request reaches.

    Requesting rendered thumbnails makes each response expensive, so the API
    truncates well short of the limit and returns a continuation token. Without
    following it, a second pool drawn from the same categories would be the same
    first fifty files.
    """
    collected: list[dict] = []
    continuation: dict[str, str] = {}
    for _ in range(pages):
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": str(per_page),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "400",
            "iiextmetadatafilter": (
                "LicenseShortName|UsageTerms|LicenseUrl|Artist|Credit|"
                "AttributionRequired|Restrictions|ObjectName"
            ),
            **continuation,
        }
        payload = _request(params)
        collected.extend(
            page for page in payload.get("query", {}).get("pages", [])
            if page.get("imageinfo")
        )
        cont = payload.get("continue")
        time.sleep(0.4)
        if not cont:
            break
        continuation = {k: v for k, v in cont.items() if k != "continue"}
        continuation["continue"] = cont["continue"]
    return collected


def screen_validation_pool(
    sources: tuple[CategorySource, ...] = VALIDATION_SOURCES,
    max_per_category: int = 14,
    pages: int = 3,
) -> tuple[list[Candidate], list[dict]]:
    """Licence-clean candidates whose source group is new."""
    import competition.evaluation.build_licensed_corpus as harvest

    blocked_groups = existing_source_groups()
    blocked_images = previously_screened_image_ids()
    pool: list[Candidate] = []
    rejections: list[dict] = []
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    seen_families: set[str] = set()

    original = harvest.fetch_category_files
    try:
        harvest.fetch_category_files = lambda category, limit=200: (
            fetch_category_files_paged(category, pages=pages, per_page=limit)
        )
        for source in sources:
            candidates, category_rejections = screen_category(source, limit=200)
            rejections.extend(category_rejections)
            kept = 0
            for candidate in candidates:
                if kept >= max_per_category:
                    break
                if candidate.image_id in seen_ids or candidate.image_id in blocked_images:
                    rejections.append({"title": candidate.title, "stage": "independence",
                                       "verdict": "REJECTED_ALREADY_SCREENED",
                                       "reasons": ["file was already screened in Phase 2c-B"]})
                    continue
                try:
                    group = source_group_id(candidate.creator)
                except CorpusError as error:
                    rejections.append({"title": candidate.title, "stage": "grouping",
                                       "verdict": "REJECTED_NO_GROUP", "reasons": [str(error)]})
                    continue
                if group in blocked_groups:
                    rejections.append({"title": candidate.title, "stage": "independence",
                                       "verdict": "REJECTED_SOURCE_GROUP_IN_CORPUS",
                                       "reasons": [f"creator already represented in Phase 2c-B"]})
                    continue
                if group in seen_groups:
                    rejections.append({"title": candidate.title, "stage": "diversity",
                                       "verdict": "SKIPPED_GROUP_ALREADY_POOLED",
                                       "reasons": ["one image per creator in the validation pool"]})
                    continue
                family = capture_family_id(candidate.title)
                if family in seen_families:
                    rejections.append({"title": candidate.title, "stage": "diversity",
                                       "verdict": "SKIPPED_CAPTURE_FAMILY", "reasons": [family]})
                    continue
                if not candidate.thumb_url:
                    continue

                pool.append(candidate)
                seen_ids.add(candidate.image_id)
                seen_groups.add(group)
                seen_families.add(family)
                kept += 1
            print(f"  {source.category:22} pooled {kept:3} / passed {len(candidates):4}",
                  file=sys.stderr)
    finally:
        harvest.fetch_category_files = original

    return pool, rejections


def pool_fingerprint(records: list[dict]) -> str:
    """Identity of the frozen validation set: content hashes and provenance."""
    payload = json.dumps(
        sorted(
            (r["image_id"], r["content_sha256"], r["source_group_id"],
             r["fruit_type"], r["track"])
            for r in records
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def harvest_validation(curation: dict[str, dict], pool: list[Candidate]) -> dict:
    by_id = {candidate.image_id: candidate for candidate in pool}
    accepted: list[LicensedImageRecord] = []
    rejections: list[dict] = []
    seen_hashes: dict[str, str] = {}
    retrieved_at = date.today().isoformat()
    corpus_groups = existing_source_groups()

    for image_id, decision in sorted(curation.items()):
        if not decision.get("keep"):
            continue
        candidate = by_id.get(image_id)
        if candidate is None:
            continue
        extension = _extension_for(candidate.mime, candidate.file_url)
        destination = VALIDATION_RAW / f"{candidate.image_id}{extension}"
        try:
            payload = (
                destination.read_bytes() if destination.is_file()
                else _download(rendition_url(candidate), destination, DOWNLOAD_DELAY_S)
            )
        except Exception as error:  # noqa: BLE001
            rejections.append({"image_id": image_id, "verdict": "REJECTED_DOWNLOAD",
                               "reasons": [str(error)]})
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            destination.unlink(missing_ok=True)
            continue
        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None:
            destination.unlink(missing_ok=True)
            continue
        height, width = image.shape[:2]
        group = source_group_id(candidate.creator)
        if group in corpus_groups:
            destination.unlink(missing_ok=True)
            rejections.append({"image_id": image_id,
                               "verdict": "REJECTED_SOURCE_GROUP_IN_CORPUS",
                               "reasons": ["independence check failed at harvest"]})
            continue

        record = LicensedImageRecord(
            image_id=candidate.image_id,
            source_group_id=group,
            capture_family_id=capture_family_id(candidate.title),
            fruit_type=decision.get("fruit", candidate.fruit.value),
            provenance=ProvenanceType.LICENSED_REAL.value,
            track=decision.get("track", candidate.track.value),
            source_name=SOURCE_NAME,
            source_url=candidate.description_url,
            creator=candidate.creator,
            licence_id=candidate.licence_id,
            licence_url=candidate.licence_url,
            attribution_text=candidate.attribution_text,
            commercial_use_allowed=candidate.terms.commercial_use_allowed,
            derivatives_allowed=candidate.terms.derivatives_allowed,
            attribution_required=candidate.terms.attribution_required,
            share_alike_required=candidate.terms.share_alike_required,
            retrieved_at=retrieved_at,
            content_sha256=digest,
            width=int(width), height=int(height), file_extension=extension,
            local_only=True,
            split="validation",
            notes=decision.get("note", "")[:200],
        )
        record.validate()
        accepted.append(record)
        seen_hashes[digest] = record.image_id

    records = [record.to_dict() for record in accepted]
    return {
        "validation_version": VALIDATION_VERSION,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "licence_gate_version": LICENCE_GATE_VERSION,
        "purpose": (
            "untouched confirmatory validation for Phase 2d; never used to "
            "develop or tune a detector"
        ),
        "independence": (
            "every source group is absent from the Phase 2c-B corpus, so no "
            "contributor is shared with calibration or held-out"
        ),
        "retrieved_at": retrieved_at,
        "count": len(records),
        "records": records,
        "harvest_rejections": rejections,
        "fingerprint": pool_fingerprint(records),
    }


def load_validation_set(path: Path = VALIDATION_MANIFEST) -> list[LicensedImageRecord]:
    if not path.is_file():
        raise CorpusError(f"{path.name} not found; build the validation pool first")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = pool_fingerprint(document["records"])
    if document.get("fingerprint") != expected:
        raise CorpusError(
            f"validation fingerprint mismatch: recorded {document.get('fingerprint')!r}, "
            f"recomputed {expected!r}; the frozen set has been edited"
        )
    return [LicensedImageRecord.from_dict(item) for item in document["records"]]


def validation_image_path(record: LicensedImageRecord) -> Path:
    return VALIDATION_RAW / f"{record.image_id}{record.file_extension}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.build_validation_pool",
        description="Independent validation pool for Phase 2d.",
    )
    parser.add_argument("stage", choices=["pool", "thumbnails", "harvest"])
    parser.add_argument("--max-per-category", type=int, default=14)
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args(argv)

    if args.stage == "pool":
        pool, rejections = screen_validation_pool(
            max_per_category=args.max_per_category, pages=args.pages
        )
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATION_POOL.write_text(json.dumps({
            "validation_version": VALIDATION_VERSION,
            "generated_on": date.today().isoformat(),
            "candidates": [
                {"image_id": c.image_id, "title": c.title, "file_url": c.file_url,
                 "thumb_url": c.thumb_url, "description_url": c.description_url,
                 "mime": c.mime, "creator": c.creator, "licence_id": c.licence_id,
                 "licence_url": c.licence_url, "attribution_text": c.attribution_text,
                 "candidate_fruit": c.fruit.value, "candidate_track": c.track.value,
                 "category": c.category, "declared_width": c.declared_width,
                 "declared_height": c.declared_height}
                for c in pool
            ],
            "excluded_source_groups": len(existing_source_groups()),
            "excluded_previously_screened_images": len(previously_screened_image_ids()),
            "rejections": len(rejections),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"pooled": len(pool), "rejected": len(rejections)}, indent=2))
        return 0

    from competition.evaluation.build_licensed_corpus import fetch_thumbnails, load_pool

    pool = load_pool(VALIDATION_POOL)
    if args.stage == "thumbnails":
        thumbnails = fetch_thumbnails(pool, VALIDATION_CACHE)
        sheets = build_contact_sheets(pool, thumbnails, VALIDATION_CACHE / "sheets")
        print(json.dumps({"thumbnails": len(thumbnails), "sheets": len(sheets)}, indent=2))
        return 0

    curation = json.loads(VALIDATION_CURATION.read_text(encoding="utf-8"))["decisions"]
    document = harvest_validation(curation, pool)
    VALIDATION_MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": document["count"],
                      "fingerprint": document["fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
