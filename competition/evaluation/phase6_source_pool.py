"""Phase 6 Track R: a third real-image pool, independent of everything spent.

Why a third pool
----------------
Phase 2c-B calibrated on its groups and opened its held-out split once. Phase 2d
opened its own validation set once. Both are spent: a number measured on either
would be reported by someone who has already seen how that data behaves, however
carefully the run is done. Phase 3b worked on the Phase 2c-B images. Phase 3's
scenarios are constructed fixtures, not photographs.

So the confirmatory evaluation gets material that no part of this project has
looked at, and the independence is enforced at three levels rather than
intended:

- **image** - every file id ever screened, admitted or rejected, is blocked
- **creator** - any source group present in either prior corpus is blocked
- **category** - the categories below were used by neither prior harvest

The third one matters more than it looks. Paging deeper into a category that has
already been harvested mostly re-surfaces files that were screened and rejected
the first time. Those are licence-clean and creator-clean, and still carry a
curation judgement someone already made while looking at them.

What Track R is for
-------------------
Natural-domain usability of the frozen system, under the deployment contract:
one primary produce item per capture. Natural backgrounds are wanted, not
avoided - a pool of studio-white cutouts would measure the easiest case and call
it the field. Market piles and crates are out of scope by contract, and they are
curated out here rather than counted as failures later.

Order of operations is the same as every earlier harvest and is not negotiable:
read metadata, run the licence gate, and only then fetch bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
    build_contact_sheets,
    fetch_thumbnails,
    load_corpus,
    rendition_url,
    screen_category,
)
from competition.evaluation.build_validation_pool import (
    VALIDATION_MANIFEST,
    VALIDATION_POOL,
)
from competition.models.ontology import FruitType

PHASE6_VERSION = "phase6-source-pool-1.0.0"

PHASE6_RAW = CORPUS_ROOT / "phase6_raw"
PHASE6_CACHE = CACHE_DIR / "phase6"
PHASE6_POOL = MANIFEST_DIR / "phase6_pool.json"
PHASE6_CURATION = MANIFEST_DIR / "phase6_curation.json"

RESULTS_DIR = Path("competition/evaluation/results/phase6")
SOURCE_MANIFEST = RESULTS_DIR / "phase6_source_manifest.json"

# Target allocation. 45 is what a single reviewer can adjudicate carefully in one
# sitting; a larger pool reviewed sloppily is worse evidence, not better.
TARGET_PER_FRUIT = 15
TARGET_TOTAL = TARGET_PER_FRUIT * 3

# Used by neither the Phase 2c-B harvest nor the Phase 2d validation harvest.
# Each was probed for existence and for holding a usable number of permissively
# licensed photographs before being listed here.
PHASE6_SOURCES: tuple[CategorySource, ...] = (
    CategorySource("Apples in Poland", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "regional apple photographs, domestic and outdoor settings"),
    CategorySource("Apples in the United States", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "regional apple photographs"),
    CategorySource("Golden Delicious", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "named cultivar, usually one or two fruit in frame"),
    CategorySource("Granny Smith", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "named cultivar, green skin; adds colour variation"),
    CategorySource("Gala (apple)", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "named cultivar"),
    CategorySource("Fuji (apple)", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "named cultivar"),
    CategorySource("Apples in Japan", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "regional photographs, retail and orchard settings"),
    CategorySource("Yellow apples", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "yellow-skinned apples; separates colour from cultivar"),

    CategorySource("Bananas in Indonesia", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "bananas in markets, homes and transport"),
    CategorySource("Bananas in Thailand", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Brazil", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Vietnam", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Tanzania", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "bananas in East African markets and smallholdings"),
    CategorySource("Bananas in Malaysia", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Kenya", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Musa (genus)", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "botanical category; fruit close-ups among plant photographs"),

    CategorySource("Blood oranges", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "cultivar close-ups; whole fruit kept, cut fruit curated out"),
    CategorySource("Oranges in Israel", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs, groves and packing"),
    CategorySource("Oranges in India", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs, markets"),
    CategorySource("Oranges in Tunisia", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Oranges in Turkey", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Oranges in China", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Oranges in Greece", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),

    # Second sweep. The first screen made a real property of the source visible:
    # under the one-primary-item contract, permissively licensed banana and
    # orange photography on Commons is overwhelmingly plantations, market piles
    # and cut fruit, while apples are routinely photographed one at a time. The
    # answer is more source breadth for the two short fruits, not a looser
    # curation rule - relaxing the contract would change what the evaluation
    # measures in order to make the count come out right.
    CategorySource("Bananas in Australia", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Nigeria", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Taiwan", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Japan", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs, retail settings"),
    CategorySource("Bananas in Bangladesh", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Musa balbisiana", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "botanical banana category with fruit close-ups"),
    CategorySource("Bananas in Cameroon", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Mexico", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in China", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Peru", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Spain", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Guatemala", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),
    CategorySource("Bananas in Colombia", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "regional banana photographs"),

    CategorySource("Oranges in Ukraine", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs, retail and domestic"),
    CategorySource("Oranges in Germany", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Orange orchards", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "orchard photographs; single fruit on the branch"),
    CategorySource("Oranges in France", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Oranges in Malaysia", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
    CategorySource("Oranges in Colombia", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "regional orange photographs"),
)

# Categories harvested in Phase 2c-B and Phase 2d. Listed so the disjointness
# claim is checkable by a reader and by a test, not merely asserted in prose.
PRIOR_CATEGORIES: frozenset[str] = frozenset({
    "Apples", "Red apples", "Green apples", "Bananas", "Red banana", "Oranges",
    "Orange fruit", "Citrus sinensis", "Navel oranges", "Apples in baskets",
    "Fruit bowls", "Bananas in Uganda", "Fruit stalls", "Greengrocers",
    "Fruit markets", "Fruit shops",
    "Malus domestica", "Apple cultivars", "Apples in Germany", "Musa acuminata",
    "Bananas in India", "Oranges in Spain", "Sliced oranges", "Fruit vendors",
    "Farmers' markets", "Market stalls", "Bananas in the Philippines",
    "Bananas in Ecuador",
})


class Phase6PoolError(RuntimeError):
    """Raised when independence or licensing cannot be established."""


# --- transport ----------------------------------------------------------------

# A full screen is ~70 API calls over several minutes. A single transient
# network failure part-way through would otherwise discard the whole run and
# send us back to Commons for everything again, which is both slow and impolite.
REQUEST_ATTEMPTS = 4
REQUEST_BACKOFF_S = 6.0


def _request_retrying(params: dict) -> dict:
    from competition.evaluation.build_licensed_corpus import _request

    last: Exception | None = None
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            return _request(params)
        except Exception as error:  # noqa: BLE001
            last = error
            if attempt == REQUEST_ATTEMPTS - 1:
                break
            wait = REQUEST_BACKOFF_S * (attempt + 1)
            print(f"    retry in {wait:.0f}s after {str(error)[:70]}", file=sys.stderr)
            time.sleep(wait)
    raise Phase6PoolError(f"Commons unreachable after {REQUEST_ATTEMPTS} attempts: {last}")


def fetch_category_files_paged(
    category: str, pages: int = 3, per_page: int = 200
) -> list[dict]:
    """Walk deeper into a category than one request reaches, with retries."""
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
        payload = _request_retrying(params)
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


# --- independence -------------------------------------------------------------


def _pool_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    document = json.loads(path.read_text(encoding="utf-8"))
    return {candidate["image_id"] for candidate in document.get("candidates", [])}


def _manifest_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def prior_source_groups() -> set[str]:
    """Creators represented in either spent corpus."""
    groups = {record.source_group_id for record in load_corpus()}
    groups |= {r["source_group_id"] for r in _manifest_records(VALIDATION_MANIFEST)}
    return groups


def prior_image_ids() -> set[str]:
    """Every file this project has ever screened, admitted or not."""
    ids = {record.image_id for record in load_corpus()}
    ids |= {r["image_id"] for r in _manifest_records(VALIDATION_MANIFEST)}
    ids |= _pool_ids(MANIFEST_DIR / "candidate_pool.json")
    ids |= _pool_ids(VALIDATION_POOL)
    return ids


def prior_content_hashes() -> set[str]:
    hashes = {record.content_sha256 for record in load_corpus()}
    hashes |= {r["content_sha256"] for r in _manifest_records(VALIDATION_MANIFEST)}
    return hashes


def independence_report() -> dict:
    return {
        "blocked_source_groups": len(prior_source_groups()),
        "blocked_image_ids": len(prior_image_ids()),
        "blocked_content_hashes": len(prior_content_hashes()),
        "phase6_categories": [source.category for source in PHASE6_SOURCES],
        "prior_categories": sorted(PRIOR_CATEGORIES),
        "category_overlap": sorted(
            {source.category for source in PHASE6_SOURCES} & PRIOR_CATEGORIES
        ),
    }


# --- screening ----------------------------------------------------------------


def screen_pool(
    sources: tuple[CategorySource, ...] = PHASE6_SOURCES,
    max_per_category: int = 10,
    pages: int = 3,
) -> tuple[list[Candidate], list[dict]]:
    """Licence-clean candidates whose creator and file are both new.

    One image per creator and one per capture family, so a contributor who
    uploaded forty photographs of the same apple cannot dominate the pool.
    """
    import competition.evaluation.build_licensed_corpus as harvest

    blocked_groups = prior_source_groups()
    blocked_images = prior_image_ids()
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
            if source.category in PRIOR_CATEGORIES:
                raise Phase6PoolError(
                    f"{source.category!r} was harvested before Phase 6"
                )
            candidates, category_rejections = screen_category(source, limit=200)
            rejections.extend(category_rejections)
            kept = 0
            for candidate in candidates:
                if kept >= max_per_category:
                    break
                if candidate.image_id in seen_ids or candidate.image_id in blocked_images:
                    rejections.append({
                        "title": candidate.title, "stage": "independence",
                        "verdict": "REJECTED_ALREADY_SCREENED",
                        "reasons": ["file was screened in an earlier phase"]})
                    continue
                try:
                    group = source_group_id(candidate.creator)
                except CorpusError as error:
                    rejections.append({
                        "title": candidate.title, "stage": "grouping",
                        "verdict": "REJECTED_NO_GROUP", "reasons": [str(error)]})
                    continue
                if group in blocked_groups:
                    rejections.append({
                        "title": candidate.title, "stage": "independence",
                        "verdict": "REJECTED_SOURCE_GROUP_IN_PRIOR_CORPUS",
                        "reasons": ["creator already represented in an earlier corpus"]})
                    continue
                if group in seen_groups:
                    rejections.append({
                        "title": candidate.title, "stage": "diversity",
                        "verdict": "SKIPPED_GROUP_ALREADY_POOLED",
                        "reasons": ["one image per creator in the Phase 6 pool"]})
                    continue
                family = capture_family_id(candidate.title)
                if family in seen_families:
                    rejections.append({
                        "title": candidate.title, "stage": "diversity",
                        "verdict": "SKIPPED_CAPTURE_FAMILY", "reasons": [family]})
                    continue
                if not candidate.thumb_url:
                    continue

                pool.append(candidate)
                seen_ids.add(candidate.image_id)
                seen_groups.add(group)
                seen_families.add(family)
                kept += 1
            print(f"  {source.category:30} pooled {kept:3} / passed {len(candidates):4}",
                  file=sys.stderr)
    finally:
        harvest.fetch_category_files = original

    return pool, rejections


def write_pool(pool: list[Candidate], rejections: list[dict]) -> Path:
    PHASE6_POOL.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "phase6_version": PHASE6_VERSION,
        "licence_gate_version": LICENCE_GATE_VERSION,
        "generated_on": date.today().isoformat(),
        "independence": independence_report(),
        "rejections": rejections,
        "candidates": [
            {
                "image_id": c.image_id, "title": c.title, "category": c.category,
                "candidate_fruit": c.fruit.value, "candidate_track": c.track.value,
                "creator": c.creator, "licence_id": c.licence_id,
                "licence_url": c.licence_url, "attribution_text": c.attribution_text,
                "description_url": c.description_url, "file_url": c.file_url,
                "thumb_url": c.thumb_url, "mime": c.mime,
                "declared_width": c.declared_width, "declared_height": c.declared_height,
            }
            for c in pool
        ],
    }
    PHASE6_POOL.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return PHASE6_POOL


def load_pool() -> list[Candidate]:
    """Rebuild Candidate objects from the stored pool, for the harvest step."""
    from competition.data.licence_validation import ALLOWED_LICENCES

    document = json.loads(PHASE6_POOL.read_text(encoding="utf-8"))
    pool: list[Candidate] = []
    for entry in document["candidates"]:
        # The gate already ran, on the full Commons rights metadata, at screening
        # time. Re-running it here against a licence id alone would be a lossy
        # round-trip that rejects grants the gate actually admitted. The
        # obligations are looked up by the id the gate resolved, which is exact.
        terms = ALLOWED_LICENCES.get(entry["licence_id"])
        if terms is None:
            raise Phase6PoolError(
                f"{entry['image_id']} carries {entry['licence_id']!r}, which is "
                "not on the allow-list"
            )
        pool.append(Candidate(
            image_id=entry["image_id"], title=entry["title"],
            file_url=entry["file_url"], description_url=entry["description_url"],
            mime=entry["mime"], creator=entry["creator"], credit="",
            licence_id=entry["licence_id"], licence_url=entry["licence_url"],
            attribution_text=entry["attribution_text"], terms=terms,
            fruit=FruitType(entry["candidate_fruit"]),
            track=CorpusTrack(entry["candidate_track"]),
            category=entry["category"],
            declared_width=entry["declared_width"],
            declared_height=entry["declared_height"],
            thumb_url=entry["thumb_url"],
        ))
    return pool


# --- harvest ------------------------------------------------------------------


def manifest_fingerprint(records: list[dict]) -> str:
    payload = json.dumps(
        sorted(
            (r["image_id"], r["content_sha256"], r["source_group_id"], r["fruit_type"])
            for r in records
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def harvest(curation: dict[str, dict], pool: list[Candidate]) -> dict:
    """Fetch bytes for curated keeps and build the frozen Track R manifest."""
    by_id = {candidate.image_id: candidate for candidate in pool}
    blocked_groups = prior_source_groups()
    blocked_hashes = prior_content_hashes()
    accepted: list[LicensedImageRecord] = []
    exclusions: list[dict] = []
    seen_hashes: dict[str, str] = {}
    retrieved_at = date.today().isoformat()
    PHASE6_RAW.mkdir(parents=True, exist_ok=True)

    for image_id, decision in sorted(curation.items()):
        if not decision.get("keep"):
            continue
        candidate = by_id.get(image_id)
        if candidate is None:
            exclusions.append({"image_id": image_id, "reason": "NOT_IN_POOL"})
            continue
        extension = _extension_for(candidate.mime, candidate.file_url)
        destination = PHASE6_RAW / f"{candidate.image_id}{extension}"
        try:
            payload = (
                destination.read_bytes() if destination.is_file()
                else _download(rendition_url(candidate), destination, DOWNLOAD_DELAY_S)
            )
        except Exception as error:  # noqa: BLE001
            exclusions.append({"image_id": image_id, "reason": "DOWNLOAD_FAILED",
                               "detail": str(error)[:200]})
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            destination.unlink(missing_ok=True)
            exclusions.append({"image_id": image_id, "reason": "DUPLICATE_IN_POOL",
                               "detail": f"same bytes as {seen_hashes[digest]}"})
            continue
        if digest in blocked_hashes:
            destination.unlink(missing_ok=True)
            exclusions.append({"image_id": image_id,
                               "reason": "DUPLICATE_OF_PRIOR_CORPUS"})
            continue
        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None:
            destination.unlink(missing_ok=True)
            exclusions.append({"image_id": image_id, "reason": "CORRUPT_DOWNLOAD"})
            continue
        height, width = image.shape[:2]
        group = source_group_id(candidate.creator)
        if group in blocked_groups:
            destination.unlink(missing_ok=True)
            exclusions.append({"image_id": image_id,
                               "reason": "SOURCE_GROUP_IN_PRIOR_CORPUS"})
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
            split=Split.PHASE6_CONFIRMATORY.value,
            notes=decision.get("note", "")[:200],
        )
        record.validate()
        accepted.append(record)
        seen_hashes[digest] = record.image_id

    records = [record.to_dict() for record in accepted]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["fruit_type"]] = counts.get(record["fruit_type"], 0) + 1
    return {
        "phase6_version": PHASE6_VERSION,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "licence_gate_version": LICENCE_GATE_VERSION,
        "retrieved_at": retrieved_at,
        "purpose": (
            "Track R: fresh licence-verified real photographs for the Phase 6 "
            "confirmatory evaluation of the frozen deployed system. Frozen "
            "before any AgriVision metric was computed on any of these images."
        ),
        "count": len(records),
        "counts_by_fruit": counts,
        "source_group_count": len({r["source_group_id"] for r in records}),
        "licence_distribution": _distribution(records, "licence_id"),
        "track_distribution": _distribution(records, "track"),
        "independence": independence_report(),
        "exclusions": exclusions,
        "manifest_fingerprint": manifest_fingerprint(records),
        "label_boundary": (
            "fruit_type comes from curated source identity and is independent of "
            "AgriVision. No visible-condition label exists for these images and "
            "none was invented, so no fresh-vs-degraded accuracy can be computed "
            "from this manifest."
        ),
        "records": records,
    }


def _distribution(records: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        out[record[key]] = out.get(record[key], 0) + 1
    return dict(sorted(out.items()))


def write_manifest(document: dict) -> Path:
    SOURCE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SOURCE_MANIFEST


def load_manifest(path: Path = SOURCE_MANIFEST) -> dict:
    if not path.is_file():
        raise Phase6PoolError(f"{path.name} not found; run the Phase 6 harvest first")
    return json.loads(path.read_text(encoding="utf-8"))


def local_path_for(record: dict) -> Path:
    """Reconstruct the local path from the id, so no path is ever committed."""
    return PHASE6_RAW / f"{record['image_id']}{record['file_extension']}"


# --- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "sheets", "harvest"))
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args(argv)

    if args.stage == "screen":
        report = independence_report()
        if report["category_overlap"]:
            raise Phase6PoolError(f"categories reused: {report['category_overlap']}")
        print(f"blocking {report['blocked_image_ids']} screened files and "
              f"{report['blocked_source_groups']} creators", file=sys.stderr)
        pool, rejections = screen_pool(
            max_per_category=args.max_per_category, pages=args.pages
        )
        path = write_pool(pool, rejections)
        print(f"\npooled {len(pool)} candidates -> {path}")
        return 0

    if args.stage == "sheets":
        pool = load_pool()
        thumbnails = fetch_thumbnails(pool, cache=PHASE6_CACHE)
        sheets = build_contact_sheets(pool, thumbnails, PHASE6_CACHE / "sheets")
        print(f"{len(sheets)} contact sheets in {PHASE6_CACHE / 'sheets'}")
        return 0

    curation = json.loads(PHASE6_CURATION.read_text(encoding="utf-8"))["decisions"]
    document = harvest(curation, load_pool())
    path = write_manifest(document)
    print(f"Track R: {document['count']} images, "
          f"{document['source_group_count']} source groups")
    print(f"by fruit: {document['counts_by_fruit']}")
    print(f"fingerprint: {document['manifest_fingerprint']}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
