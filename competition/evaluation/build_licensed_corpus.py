"""Harvest a licensing-verified real-image corpus from Wikimedia Commons.

Why Commons rather than a packaged dataset
------------------------------------------
Packaged fruit datasets carry one licence statement for the whole archive, which
is exactly the thing that cannot be verified per image. Commons publishes the
licence, the creator and the attribution requirement *per file*, through an API,
which makes the gate in `competition.data.licence_validation` actually
meaningful: every image in this corpus was admitted by reading its own rights
statement, not by inheriting a claim made about a zip file.

Order of operations, which is not negotiable
--------------------------------------------
1. read metadata
2. run the licence gate
3. **only then** download bytes

Downloading first and filtering later would mean holding bytes we have no
established right to hold. The rejection list is therefore built entirely from
metadata, and rejected files are never fetched.

What lands on disk
------------------
Image bytes go to a git-ignored directory. The committed artifact is the
manifest: content hashes, provenance, licence terms and attribution, with no
filesystem paths. Local paths are reconstructed from `image_id` on demand by
`local_path_for`, so the mapping is code, not data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from competition.data.licence_validation import (
    LICENCE_GATE_VERSION,
    LicenceAssertion,
    LicenceVerdict,
    evaluate_licence,
)
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
from competition.models.ontology import FruitType

CORPUS_ROOT = Path("competition/data/licensed_real")
RAW_DIR = CORPUS_ROOT / "raw"
CACHE_DIR = CORPUS_ROOT / "cache"
DERIVED_DIR = CORPUS_ROOT / "derived"
MANIFEST_DIR = CORPUS_ROOT / "manifests"
CORPUS_MANIFEST = MANIFEST_DIR / "licensed_corpus.json"
REJECTION_REPORT = MANIFEST_DIR / "licence_rejections.json"
CURATION_PATH = MANIFEST_DIR / "curation.json"
POOL_PATH = MANIFEST_DIR / "candidate_pool.json"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SOURCE_NAME = "Wikimedia Commons"

# Wikimedia requires a descriptive User-Agent identifying the client and a
# contact route. Anonymous or browser-spoofing agents are rate-limited or
# blocked, and rightly so.
USER_AGENT = (
    "AgriVision-Phase2cB/1.0 "
    "(OpenCV AI Competition 2026 research corpus; "
    "https://github.com/DebalekhaChakraborty/AgriVision)"
)

# Accepted encodings. TIFF and SVG are excluded: TIFF because the Commons copy
# is often a scan at a different colour depth, SVG because it is not a
# photograph at all.
ACCEPTED_MIME = {"image/jpeg", "image/png"}

# Minimum stored dimension. Laplacian variance is strongly resolution-dependent
# (Phase 1), so admitting thumbnails alongside full-resolution photographs would
# inject a resolution confound into the exact metric being calibrated.
MIN_SOURCE_DIMENSION = 640
MAX_SOURCE_BYTES = 30 * 1024 * 1024

# Diversity caps, applied per fruit and track.
MAX_PER_CAPTURE_FAMILY = 1
MAX_PER_SOURCE_GROUP = 2

REQUEST_DELAY_S = 0.35
# Full-resolution downloads are paced far more slowly than metadata calls and
# retried with backoff. Commons returns HTTP 429 to impolite clients, and the
# correct response to that is to slow down, not to route around it.
DOWNLOAD_DELAY_S = 1.5
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BACKOFF_S = 8.0

# Longest edge requested for corpus images. Asking for a bounded rendition costs
# Commons one cached thumbnail instead of serving a multi-megabyte original, and
# it bounds the resolution confound Phase 1 found in Laplacian variance: the
# corpus spans 640-1920 px rather than 640 px to whatever a contributor's camera
# produced. Resolution effects are measured across that range and no wider.
#
# Commons rejects arbitrary widths with HTTP 400 and serves only a standard
# ladder, so the width is snapped down to the largest permitted size rather than
# computed. Measured against the live service, not assumed.
TARGET_LONG_EDGE = 1920
ALLOWED_RENDITION_WIDTHS: tuple[int, ...] = (1920, 1280, 500)

# Curation is done on thumbnails: full-resolution bytes are fetched only for
# images that survive visual review, so rejected material is never stored.
THUMBNAIL_WIDTH = 400
CONTACT_SHEET_COLUMNS = 7
CONTACT_SHEET_ROWS = 5


@dataclass(frozen=True)
class CategorySource:
    """One curated Commons category and the role its files are *expected* to play.

    Both fields are proposals, not conclusions. Commons categories are organised
    by subject matter, not by photographic composition, so `Fruit stalls` is
    full of oranges, bananas and apples in one heap, and `Red apples` contains a
    wax model of an apple and a photograph of a car parade. Category membership
    therefore selects a candidate pool; the fruit and the track are confirmed
    per file at the curation stage, against the image itself.
    """

    category: str
    candidate_fruit: FruitType
    candidate_track: CorpusTrack
    rationale: str


# Curated by hand after probing the category tree: every entry below was checked
# to exist, to hold a usable number of permissively licensed files, and to
# contain photographs of the fruit rather than of trees, blossom or cooking.
CATEGORY_SOURCES: tuple[CategorySource, ...] = (
    # --- Track A pools: single subject, plain or simple background ------------
    CategorySource("Apples", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "general apple fruit photographs"),
    CategorySource("Red apples", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "single red apples, often on a plain surface"),
    CategorySource("Green apples", FruitType.APPLE, CorpusTrack.CLEAN_BASE,
                   "single green apples"),
    CategorySource("Bananas", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "general banana photographs, isolated and in hands"),
    CategorySource("Red banana", FruitType.BANANA, CorpusTrack.CLEAN_BASE,
                   "red banana cultivar, adds skin-colour variation"),
    CategorySource("Oranges", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "general orange photographs"),
    CategorySource("Orange fruit", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "fruit-specific orange category"),
    CategorySource("Citrus sinensis", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "sweet orange, botanical category with fruit close-ups"),
    CategorySource("Navel oranges", FruitType.ORANGE, CorpusTrack.CLEAN_BASE,
                   "navel cultivar close-ups"),
    # --- Track B pools: real scenes, cluttered and variably lit ---------------
    CategorySource("Apples in baskets", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "apples among apples, basket texture, ambient light"),
    CategorySource("Fruit bowls", FruitType.APPLE, CorpusTrack.NATURAL_SCENE,
                   "household table scenes with mixed produce"),
    CategorySource("Bananas in Uganda", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "bananas in markets and transport, non-studio lighting"),
    CategorySource("Fruit stalls", FruitType.BANANA, CorpusTrack.NATURAL_SCENE,
                   "market stalls, heavy clutter, mixed fruit"),
    CategorySource("Greengrocers", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "shop interiors, artificial lighting, dense displays"),
    CategorySource("Fruit markets", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "outdoor markets, strong sun and shade"),
    CategorySource("Fruit shops", FruitType.ORANGE, CorpusTrack.NATURAL_SCENE,
                   "small retail displays"),
)


class HarvestError(RuntimeError):
    """Raised when the source API cannot be used as expected."""


def _request(params: dict) -> dict:
    """One Commons API call. Errors loudly rather than returning a partial dict."""
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the operator
        raise HarvestError(f"Commons API request failed: {error}") from error
    if "error" in payload:
        raise HarvestError(f"Commons API error: {payload['error']}")
    return payload


def _strip_html(value: str | None) -> str:
    """Commons returns small HTML fragments for Artist and Credit."""
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = (
        text.replace("&amp;", "&").replace("&quot;", '"')
        .replace("&#039;", "'").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&nbsp;", " ")
    )
    return " ".join(text.split())


def fetch_category_files(category: str, limit: int = 200) -> list[dict]:
    """Metadata for files held directly in one category.

    Subcategories are deliberately not followed. Recursion on Commons reaches
    unrelated material within two hops (Apples -> Apple trees -> Orchards), and
    an unreviewed category tree is not a curated corpus.
    """
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": str(min(limit, 500)),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(THUMBNAIL_WIDTH),
        "iiextmetadatafilter": (
            "LicenseShortName|UsageTerms|LicenseUrl|Artist|Credit|"
            "AttributionRequired|Restrictions|ObjectName|DateTimeOriginal"
        ),
    }
    payload = _request(params)
    pages = payload.get("query", {}).get("pages", [])
    time.sleep(REQUEST_DELAY_S)
    return [page for page in pages if page.get("imageinfo")]


def image_id_for(canonical_url: str) -> str:
    """Stable, path-free, name-free identifier.

    Derived from the canonical file URL so it is reproducible from the manifest
    alone, and short enough to be a filename.
    """
    return "WC-" + hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:14]


def local_path_for(image_id: str, extension: str, root: Path = RAW_DIR) -> Path:
    """Reconstruct where an image lives. Never stored in a committed record."""
    return root / f"{image_id}{extension}"


def _extension_for(mime: str, url: str) -> str:
    if mime == "image/png":
        return ".png"
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return ".jpg" if suffix in {".jpg", ".jpeg", ""} else suffix


@dataclass
class Candidate:
    """A file that passed the licence gate and may now be downloaded."""

    image_id: str
    title: str
    file_url: str
    description_url: str
    mime: str
    creator: str
    credit: str
    licence_id: str
    licence_url: str
    attribution_text: str
    terms: object
    fruit: FruitType
    track: CorpusTrack
    category: str
    declared_width: int
    declared_height: int
    thumb_url: str = ""


def screen_category(
    source: CategorySource, limit: int = 200
) -> tuple[list[Candidate], list[dict]]:
    """Apply the licence gate and the technical filters to one category.

    Returns (candidates, rejections). No bytes are fetched here.
    """
    candidates: list[Candidate] = []
    rejections: list[dict] = []

    for page in fetch_category_files(source.category, limit=limit):
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})

        def field(name: str) -> str | None:
            entry = meta.get(name)
            return entry.get("value") if isinstance(entry, dict) else None

        title = page.get("title", "")
        record_stub = {
            "title": title,
            "category": source.category,
            "fruit": source.candidate_fruit.value,
            "track": source.candidate_track.value,
        }

        decision = evaluate_licence(
            LicenceAssertion(
                short_name=field("LicenseShortName"),
                usage_terms=_strip_html(field("UsageTerms")) or None,
                licence_url=field("LicenseUrl"),
                restrictions=field("Restrictions"),
                attribution_flag=field("AttributionRequired"),
            )
        )
        if not decision.allowed:
            rejections.append(
                {**record_stub, "stage": "licence",
                 "verdict": decision.verdict.value,
                 "licence_id": decision.licence_id,
                 "reasons": decision.reasons}
            )
            continue

        mime = info.get("mime", "")
        if mime not in ACCEPTED_MIME:
            rejections.append({**record_stub, "stage": "format",
                               "verdict": "REJECTED_FORMAT", "reasons": [f"mime {mime}"]})
            continue

        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        if min(width, height) < MIN_SOURCE_DIMENSION:
            rejections.append({**record_stub, "stage": "resolution",
                               "verdict": "REJECTED_RESOLUTION",
                               "reasons": [f"{width}x{height} below {MIN_SOURCE_DIMENSION}px"]})
            continue
        if int(info.get("size", 0)) > MAX_SOURCE_BYTES:
            rejections.append({**record_stub, "stage": "size",
                               "verdict": "REJECTED_SIZE",
                               "reasons": [f"{info.get('size')} bytes"]})
            continue

        creator = _strip_html(field("Artist"))
        if not creator:
            rejections.append({**record_stub, "stage": "attribution",
                               "verdict": "REJECTED_NO_CREATOR",
                               "reasons": ["no creator recorded; attribution cannot be given"]})
            continue

        credit = _strip_html(field("Credit"))
        description_url = info.get("descriptionurl", "")
        attribution = (
            f"{creator}, \"{_strip_html(field('ObjectName')) or title}\", "
            f"{decision.licence_id}, via {SOURCE_NAME} ({description_url})"
        )

        candidates.append(
            Candidate(
                image_id=image_id_for(info.get("url", description_url)),
                title=title,
                file_url=info.get("url", ""),
                description_url=description_url,
                mime=mime,
                creator=creator,
                credit=credit,
                licence_id=decision.licence_id,
                licence_url=decision.terms.licence_url,
                attribution_text=attribution,
                terms=decision.terms,
                fruit=source.candidate_fruit,
                track=source.candidate_track,
                thumb_url=info.get("thumburl", ""),
                category=source.category,
                declared_width=width,
                declared_height=height,
            )
        )

    return candidates, rejections


def _download(url: str, destination: Path, delay: float = REQUEST_DELAY_S) -> bytes:
    """Fetch one file, backing off rather than hammering on a rate limit."""
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 503):
                raise
            time.sleep(DOWNLOAD_BACKOFF_S * (2 ** attempt))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        time.sleep(delay)
        return payload
    raise RuntimeError(f"gave up after {DOWNLOAD_ATTEMPTS} attempts: {last_error}")


def rendition_url(candidate: "Candidate", long_edge: int = TARGET_LONG_EDGE) -> str:
    """URL for a width-bounded rendition, or the original when it is smaller.

    Commons thumbnail URLs carry the width as a path segment, so the bounded
    rendition is the same URL with that segment rewritten. Never asks for more
    pixels than the original has: Commons does not upscale, and requesting an
    upscale just fails.
    """
    if not candidate.thumb_url or candidate.declared_width <= long_edge:
        return candidate.file_url
    ceiling = min(long_edge, candidate.declared_width)
    permitted = [width for width in ALLOWED_RENDITION_WIDTHS if width <= ceiling]
    if not permitted:
        return candidate.file_url
    rewritten = re.sub(r"/\d+px-", f"/{max(permitted)}px-", candidate.thumb_url, count=1)
    return rewritten if rewritten != candidate.thumb_url else candidate.file_url


# --- stage 1: build a candidate pool from metadata only ----------------------


def screen_pool(
    sources: tuple[CategorySource, ...] = CATEGORY_SOURCES,
    limit_per_category: int = 200,
    max_per_category: int = 30,
) -> tuple[list[Candidate], list[dict]]:
    """Pool every licence-clean candidate across the curated categories.

    Diversity caps are applied here, while filling, rather than to the finished
    corpus. Applying them afterwards would mean the corpus is whatever the most
    prolific uploader happened to contribute, thinned at the end.
    """
    pool: list[Candidate] = []
    rejections: list[dict] = []
    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}

    for source in sources:
        candidates, category_rejections = screen_category(source, limit=limit_per_category)
        rejections.extend(category_rejections)
        kept = 0
        for candidate in candidates:
            if kept >= max_per_category:
                break
            if candidate.image_id in seen_ids:
                rejections.append({"title": candidate.title, "stage": "duplicate",
                                   "verdict": "SKIPPED_ALREADY_POOLED",
                                   "reasons": ["same file reached from another category"]})
                continue
            family = capture_family_id(candidate.title)
            if family_counts.get(family, 0) >= MAX_PER_CAPTURE_FAMILY:
                rejections.append({"title": candidate.title, "stage": "diversity",
                                   "verdict": "SKIPPED_CAPTURE_FAMILY",
                                   "reasons": [f"family {family} already represented"]})
                continue
            try:
                group = source_group_id(candidate.creator)
            except CorpusError as error:
                rejections.append({"title": candidate.title, "stage": "grouping",
                                   "verdict": "REJECTED_NO_GROUP", "reasons": [str(error)]})
                continue
            if group_counts.get(group, 0) >= MAX_PER_SOURCE_GROUP:
                rejections.append({"title": candidate.title, "stage": "diversity",
                                   "verdict": "SKIPPED_SOURCE_GROUP",
                                   "reasons": [f"group {group} already at cap"]})
                continue
            if not candidate.thumb_url:
                rejections.append({"title": candidate.title, "stage": "thumbnail",
                                   "verdict": "REJECTED_NO_THUMBNAIL",
                                   "reasons": ["no thumbnail available for curation"]})
                continue

            pool.append(candidate)
            seen_ids.add(candidate.image_id)
            family_counts[family] = family_counts.get(family, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            kept += 1

        print(f"  {source.category:22} pooled {kept:3} / passed {len(candidates):4}",
              file=sys.stderr)

    return pool, rejections


def save_pool(pool: list[Candidate], rejections: list[dict]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(
        json.dumps(
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "licence_gate_version": LICENCE_GATE_VERSION,
                "generated_on": date.today().isoformat(),
                "candidates": [
                    {
                        "image_id": c.image_id,
                        "title": c.title,
                        "file_url": c.file_url,
                        "thumb_url": c.thumb_url,
                        "description_url": c.description_url,
                        "mime": c.mime,
                        "creator": c.creator,
                        "licence_id": c.licence_id,
                        "licence_url": c.licence_url,
                        "attribution_text": c.attribution_text,
                        "candidate_fruit": c.fruit.value,
                        "candidate_track": c.track.value,
                        "category": c.category,
                        "declared_width": c.declared_width,
                        "declared_height": c.declared_height,
                    }
                    for c in pool
                ],
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    REJECTION_REPORT.write_text(
        json.dumps({"rejections": rejections, "count": len(rejections)},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_pool(path: Path = POOL_PATH) -> list[Candidate]:
    from competition.data.licence_validation import ALLOWED_LICENCES

    if not path.is_file():
        raise CorpusError(f"{path.name} not found; run the `pool` stage first")
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        Candidate(
            image_id=item["image_id"], title=item["title"], file_url=item["file_url"],
            description_url=item["description_url"], mime=item["mime"],
            creator=item["creator"], credit="", licence_id=item["licence_id"],
            licence_url=item["licence_url"], attribution_text=item["attribution_text"],
            terms=ALLOWED_LICENCES[item["licence_id"]],
            fruit=FruitType(item["candidate_fruit"]),
            track=CorpusTrack(item["candidate_track"]),
            category=item["category"],
            declared_width=item["declared_width"],
            declared_height=item["declared_height"],
            thumb_url=item["thumb_url"],
        )
        for item in document["candidates"]
    ]


# --- stage 2: curate on thumbnails -------------------------------------------


def fetch_thumbnails(pool: list[Candidate], cache: Path = CACHE_DIR) -> dict[str, Path]:
    """Cache a small thumbnail per candidate. Cheap, and re-runs are free."""
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for candidate in pool:
        destination = cache / f"{candidate.image_id}.jpg"
        if not destination.is_file():
            try:
                _download(candidate.thumb_url, destination)
            except Exception as error:  # noqa: BLE001
                print(f"  thumbnail failed {candidate.image_id}: {error}", file=sys.stderr)
                continue
        paths[candidate.image_id] = destination
    return paths


def build_contact_sheets(
    pool: list[Candidate],
    thumbnails: dict[str, Path],
    out_dir: Path,
    columns: int = CONTACT_SHEET_COLUMNS,
    rows: int = CONTACT_SHEET_ROWS,
    cell: int = 240,
) -> list[Path]:
    """Render numbered grids so every candidate is actually looked at.

    The index drawn on each cell is the candidate's position in the pool, which
    is what a curation decision refers to. Without it, reviewing a grid and
    writing down "the third one in row two" is not reproducible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = columns * rows
    sheets: list[Path] = []
    label_height = 22

    for start in range(0, len(pool), per_sheet):
        chunk = list(enumerate(pool[start:start + per_sheet], start=start))
        sheet = np.full(
            (rows * (cell + label_height), columns * cell, 3), 32, dtype=np.uint8
        )
        for offset, (index, candidate) in enumerate(chunk):
            row, column = divmod(offset, columns)
            y0 = row * (cell + label_height)
            x0 = column * cell
            path = thumbnails.get(candidate.image_id)
            thumb = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
            if thumb is not None:
                height, width = thumb.shape[:2]
                scale = min(cell / width, cell / height)
                resized = cv2.resize(
                    thumb, (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                rh, rw = resized.shape[:2]
                oy, ox = (cell - rh) // 2, (cell - rw) // 2
                sheet[y0 + oy:y0 + oy + rh, x0 + ox:x0 + ox + rw] = resized
            cv2.putText(
                sheet, f"{index}", (x0 + 6, y0 + cell + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                sheet, candidate.fruit.value[:3].upper(), (x0 + cell - 42, y0 + cell + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA,
            )
        destination = out_dir / f"contact_sheet_{start:03d}.png"
        cv2.imwrite(str(destination), sheet)
        sheets.append(destination)
    return sheets


def load_curation(path: Path = CURATION_PATH) -> dict[str, dict]:
    """Per-image keep/drop decisions with the confirmed fruit and track.

    This file is the record of the visual review. It is committed, because it is
    metadata and because it is the only evidence that curation happened at all.
    """
    if not path.is_file():
        raise CorpusError(f"{path.name} not found; the curation stage has not been recorded")
    document = json.loads(path.read_text(encoding="utf-8"))
    return document["decisions"]


# --- stage 3: fetch full resolution for keepers only -------------------------


def harvest_curated(
    pool: list[Candidate] | None = None,
    curation: dict[str, dict] | None = None,
) -> dict:
    """Download and manifest exactly the curated keepers."""
    pool = pool if pool is not None else load_pool()
    curation = curation if curation is not None else load_curation()
    by_id = {candidate.image_id: candidate for candidate in pool}

    accepted: list[LicensedImageRecord] = []
    rejections: list[dict] = []
    seen_hashes: dict[str, str] = {}
    retrieved_at = date.today().isoformat()

    for image_id, decision in sorted(curation.items()):
        if not decision.get("keep"):
            continue
        candidate = by_id.get(image_id)
        if candidate is None:
            rejections.append({"image_id": image_id, "stage": "curation",
                               "verdict": "REJECTED_NOT_IN_POOL",
                               "reasons": ["curation names an image absent from the pool"]})
            continue

        extension = _extension_for(candidate.mime, candidate.file_url)
        destination = local_path_for(candidate.image_id, extension)
        try:
            payload = (
                destination.read_bytes() if destination.is_file()
                else _download(rendition_url(candidate), destination, DOWNLOAD_DELAY_S)
            )
        except Exception as error:  # noqa: BLE001
            rejections.append({"image_id": image_id, "stage": "download",
                               "verdict": "REJECTED_DOWNLOAD", "reasons": [str(error)]})
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_hashes:
            destination.unlink(missing_ok=True)
            rejections.append({"image_id": image_id, "stage": "duplicate",
                               "verdict": "REJECTED_DUPLICATE_CONTENT",
                               "reasons": [f"identical bytes to {seen_hashes[digest]}"]})
            continue

        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            destination.unlink(missing_ok=True)
            rejections.append({"image_id": image_id, "stage": "decode",
                               "verdict": "REJECTED_UNDECODABLE",
                               "reasons": ["OpenCV could not decode the file"]})
            continue
        height, width = image.shape[:2]
        if min(width, height) < MIN_SOURCE_DIMENSION:
            destination.unlink(missing_ok=True)
            rejections.append({"image_id": image_id, "stage": "resolution",
                               "verdict": "REJECTED_RESOLUTION_DECODED",
                               "reasons": [f"decoded {width}x{height}"]})
            continue

        record = LicensedImageRecord(
            image_id=candidate.image_id,
            source_group_id=source_group_id(candidate.creator),
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
            width=int(width),
            height=int(height),
            file_extension=extension,
            local_only=True,
            split=Split.UNASSIGNED.value,
            notes=decision.get("note", "")[:200],
        )
        try:
            record.validate()
        except CorpusError as error:
            destination.unlink(missing_ok=True)
            rejections.append({"image_id": image_id, "stage": "schema",
                               "verdict": "REJECTED_SCHEMA", "reasons": [str(error)]})
            continue

        accepted.append(record)
        seen_hashes[digest] = record.image_id

    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "licence_gate_version": LICENCE_GATE_VERSION,
        "source_name": SOURCE_NAME,
        "retrieved_at": retrieved_at,
        "opencv_version": cv2.__version__,
        "curation": "visual file-by-file review on cached thumbnails",
        "rendition": {
            "target_long_edge": TARGET_LONG_EDGE,
            "allowed_widths": list(ALLOWED_RENDITION_WIDTHS),
            "note": (
                "images were retrieved as width-bounded renditions where the "
                "original was larger; resolution effects are measured across "
                "640-1920 px only"
            ),
        },
        "filters": {
            "accepted_mime": sorted(ACCEPTED_MIME),
            "min_source_dimension": MIN_SOURCE_DIMENSION,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "max_per_capture_family": MAX_PER_CAPTURE_FAMILY,
            "max_per_source_group": MAX_PER_SOURCE_GROUP,
        },
        "records": [record.to_dict() for record in accepted],
        "harvest_rejections": rejections,
    }


def write_manifest(document: dict) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_corpus(path: Path = CORPUS_MANIFEST) -> list[LicensedImageRecord]:
    if not path.is_file():
        raise CorpusError(
            f"{path.name} not found; run "
            "`python -m competition.evaluation.build_licensed_corpus harvest` first"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    return [LicensedImageRecord.from_dict(item) for item in document["records"]]


def load_image(record: LicensedImageRecord, root: Path = RAW_DIR) -> np.ndarray:
    path = local_path_for(record.image_id, record.file_extension, root)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CorpusError(f"could not read local bytes for {record.image_id}")
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.build_licensed_corpus",
        description="Licence-gated corpus harvest from Wikimedia Commons.",
    )
    parser.add_argument("stage", choices=["pool", "thumbnails", "harvest"])
    parser.add_argument("--limit-per-category", type=int, default=200)
    parser.add_argument("--max-per-category", type=int, default=30)
    parser.add_argument("--sheet-dir", default="")
    args = parser.parse_args(argv)

    if args.stage == "pool":
        pool, rejections = screen_pool(
            limit_per_category=args.limit_per_category,
            max_per_category=args.max_per_category,
        )
        save_pool(pool, rejections)
        print(json.dumps({"pooled": len(pool), "rejected": len(rejections)}, indent=2))
        return 0

    if args.stage == "thumbnails":
        pool = load_pool()
        thumbnails = fetch_thumbnails(pool)
        sheets = build_contact_sheets(
            pool, thumbnails, Path(args.sheet_dir or CACHE_DIR / "sheets")
        )
        print(json.dumps({"thumbnails": len(thumbnails), "sheets": len(sheets)}, indent=2))
        return 0

    document = harvest_curated()
    write_manifest(document)
    counts: dict[str, int] = {}
    for record in document["records"]:
        key = f"{record['fruit_type']}/{record['track']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"accepted": len(document["records"]), "by_bucket": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
