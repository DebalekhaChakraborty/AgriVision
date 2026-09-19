"""Provenance records for third-party and generated imagery.

Three tiers of evidence exist in this corpus and they must never be silently
mixed, because they support different claims:

`LICENSED_REAL`
    A photograph of real produce, taken by an identified person, under a
    verified permissive licence. This is the only tier that may support a
    statement about how the system behaves on real imagery.

`SYNTHETIC_GENERATED`
    An image produced by a generator. Useful as a stress case, worthless as
    evidence of real-world accuracy. The schema refuses to let one of these
    carry a real-image licence or pass as `LICENSED_REAL`.

`DERIVED_DEGRADED`
    A controlled transformation of a `LICENSED_REAL` base. It inherits the base
    image's licence, creator and source group — inheritance is enforced here
    rather than left to the caller, because a derived sample that lost its
    source group is exactly how a train/test leak happens.

Two rules are enforced structurally rather than by convention:

**No filesystem paths.** Committed records identify an image by content hash and
a stable id. A local path is a property of one machine and a privacy leak
(`/home/<name>/...`); it is reconstructed on demand and never stored.

**Source groups travel with the image.** Every record carries the
`source_group_id` of the physical subject or shoot it came from. Splitting
happens on that field, never on `image_id`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from competition.data.licence_validation import ALLOWED_LICENCE_IDS, ALLOWED_LICENCES
from competition.models.ontology import FruitType

CORPUS_SCHEMA_VERSION = "phase2c-licensed-corpus-1.0.0"

# Licence sentinel for generated imagery. Deliberately not a real licence id and
# deliberately not in ALLOWED_LICENCE_IDS, so a generated record can never be
# mistaken for a licensed photograph by anything that filters on licence.
SYNTHETIC_LICENCE_ID = "SYNTHETIC-GENERATED-NO-LICENCE"


class ProvenanceType(str, Enum):
    """Where an image actually came from. Closed set."""

    LICENSED_REAL = "LICENSED_REAL"
    DERIVED_DEGRADED = "DERIVED_DEGRADED"
    SYNTHETIC_GENERATED = "SYNTHETIC_GENERATED"


class CorpusTrack(str, Enum):
    """Role an image plays, assigned from source metadata, never from pixels.

    Track assignment is made from the category an image was curated out of, so
    it is fixed before any quality metric is computed. Deciding "this looks like
    a clean studio shot" by measuring the image would couple the corpus design
    to the measurements being calibrated.
    """

    CLEAN_BASE = "CLEAN_BASE"          # Track A: substrates for degradation
    NATURAL_SCENE = "NATURAL_SCENE"    # Track B: segmentation / gate behaviour
    SYNTHETIC_STRESS = "SYNTHETIC_STRESS"  # Tier 3: exploratory only


class Split(str, Enum):
    CALIBRATION = "calibration"
    HELD_OUT = "held_out"
    # Phase 2d's independent pool. A separate member rather than a reuse of
    # HELD_OUT, because the two are not interchangeable: the Phase 2c-B held-out
    # groups have been spent, and mixing the names would make it possible to
    # report a validation result as though it came from the original split.
    VALIDATION = "validation"
    # Phase 6's confirmatory pool, for the same reason again: Phase 2c-B's
    # held-out groups and Phase 2d's validation set are both spent, and a
    # confirmatory number must not be reportable as though it came from either.
    PHASE6_CONFIRMATORY = "phase6_confirmatory"
    UNASSIGNED = "unassigned"


class CorpusError(ValueError):
    """Raised when a record violates a provenance or hygiene rule."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH_MARKERS = ("/home/", "/Users/", "/root/", "/tmp/", "C:\\", "/var/")


def _reject_paths(name: str, value: str) -> None:
    """Refuse anything that looks like a local filesystem location.

    A URL is allowed; a path is not. The two are distinguished by scheme, not by
    guesswork, so an `https://` source URL containing `/home/` in its path is
    still accepted while a bare `/home/...` string is not.
    """
    if not value:
        return
    if value.startswith(("http://", "https://")):
        return
    if value.startswith("/") or value.startswith("\\\\"):
        raise CorpusError(f"{name} looks like an absolute path: {value!r}")
    for marker in _ABSOLUTE_PATH_MARKERS:
        if marker in value:
            raise CorpusError(f"{name} contains a machine-specific path fragment: {marker!r}")


def source_group_id(creator: str, shoot_hint: str = "") -> str:
    """Stable id for "images that plausibly share a physical subject or shoot".

    Derived from the creator identity, optionally narrowed by a shoot hint. The
    creator is the conservative unit: a single photographer's contributions to
    one fruit category are frequently several angles of the *same* fruit on the
    same afternoon, and nothing in the metadata distinguishes that from two
    unrelated shoots. Grouping at the creator level costs a little corpus
    diversity and removes the entire class of same-subject leakage.

    Hashed rather than stored verbatim so the group id is fixed-width and does
    not carry a display name into every derived record.
    """
    identity = " ".join(str(creator).strip().lower().split())
    if not identity:
        raise CorpusError("cannot form a source group without a creator identity")
    payload = f"{identity}|{shoot_hint.strip().lower()}"
    return "SG-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def capture_family_id(title: str) -> str:
    """Narrower grouping: one filename family, e.g. `Apple 01` .. `Apple 04`.

    Used to cap how many near-identical frames of one family enter the corpus.
    It is *not* the split unit — `source_group_id` is — because a family id
    cannot see that two differently-named files came from the same shoot.
    """
    stem = re.sub(r"\.[a-z0-9]+$", "", str(title).strip().lower())
    stem = re.sub(r"[\s_\-]*\(?\d{1,6}\)?$", "", stem)          # trailing index
    stem = re.sub(r"[\s_\-]*\d{4}[-_]\d{2}[-_]\d{2}.*$", "", stem)  # trailing date
    stem = " ".join(stem.split()) or str(title).strip().lower()
    return "CF-" + hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class LicensedImageRecord:
    """One image in the corpus, with everything needed to justify using it."""

    image_id: str
    source_group_id: str
    capture_family_id: str
    fruit_type: str
    provenance: str
    track: str
    source_name: str
    source_url: str
    creator: str
    licence_id: str
    licence_url: str
    attribution_text: str
    commercial_use_allowed: bool
    derivatives_allowed: bool
    attribution_required: bool
    share_alike_required: bool
    retrieved_at: str
    content_sha256: str
    width: int
    height: int
    file_extension: str
    local_only: bool = True
    split: str = Split.UNASSIGNED.value
    notes: str = ""
    schema_version: str = CORPUS_SCHEMA_VERSION

    def validate(self) -> None:
        if self.provenance not in {p.value for p in ProvenanceType}:
            raise CorpusError(f"unknown provenance {self.provenance!r}")
        if self.track not in {t.value for t in CorpusTrack}:
            raise CorpusError(f"unknown track {self.track!r}")
        if self.split not in {s.value for s in Split}:
            raise CorpusError(f"unknown split {self.split!r}")
        if self.fruit_type not in {f.value for f in FruitType}:
            raise CorpusError(f"unsupported fruit type {self.fruit_type!r}")
        if not _SHA256.match(self.content_sha256 or ""):
            raise CorpusError(f"content_sha256 must be 64 hex characters, got {self.content_sha256!r}")
        if self.width <= 0 or self.height <= 0:
            raise CorpusError(f"invalid dimensions {self.width}x{self.height}")
        if not self.image_id or not self.source_group_id:
            raise CorpusError("image_id and source_group_id are required")

        # The rule that stops a generated image being passed off as a photograph.
        if self.provenance == ProvenanceType.SYNTHETIC_GENERATED.value:
            if self.licence_id != SYNTHETIC_LICENCE_ID:
                raise CorpusError(
                    "a SYNTHETIC_GENERATED record may not carry a real-image licence "
                    f"(got {self.licence_id!r}); use {SYNTHETIC_LICENCE_ID!r}"
                )
            if self.track != CorpusTrack.SYNTHETIC_STRESS.value:
                raise CorpusError(
                    "SYNTHETIC_GENERATED images belong to the SYNTHETIC_STRESS track only"
                )
        else:
            if self.licence_id not in ALLOWED_LICENCE_IDS:
                raise CorpusError(
                    f"{self.licence_id!r} is not an allowed licence for a real image"
                )
            terms = ALLOWED_LICENCES[self.licence_id]
            if (
                self.commercial_use_allowed != terms.commercial_use_allowed
                or self.derivatives_allowed != terms.derivatives_allowed
                or self.share_alike_required != terms.share_alike_required
            ):
                raise CorpusError(
                    f"declared permissions contradict {self.licence_id} for {self.image_id}"
                )
            if self.attribution_required and not self.attribution_text.strip():
                raise CorpusError(f"{self.licence_id} requires attribution text for {self.image_id}")
            if self.track == CorpusTrack.SYNTHETIC_STRESS.value:
                raise CorpusError("the SYNTHETIC_STRESS track is reserved for generated images")

        for name in ("image_id", "source_url", "creator", "attribution_text", "notes"):
            _reject_paths(name, getattr(self, name))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LicensedImageRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class DerivedSampleRecord:
    """A controlled degradation of a licensed base image.

    Carries its parent's identity so provenance survives the transformation. A
    derived sample is never a new subject: it stays in its base image's source
    group and split, and that is asserted rather than assumed.
    """

    derived_id: str
    base_image_id: str
    base_content_sha256: str
    source_group_id: str
    fruit_type: str
    track: str
    split: str
    transformation: str
    parameters: dict
    seed: int
    derived_sha256: str
    licence_id: str
    creator: str
    attribution_text: str
    provenance: str = ProvenanceType.DERIVED_DEGRADED.value
    schema_version: str = CORPUS_SCHEMA_VERSION
    notes: str = field(default="")

    def validate(self) -> None:
        if self.provenance != ProvenanceType.DERIVED_DEGRADED.value:
            raise CorpusError("a derived sample must declare DERIVED_DEGRADED provenance")
        for name, value in (
            ("base_content_sha256", self.base_content_sha256),
            ("derived_sha256", self.derived_sha256),
        ):
            if not _SHA256.match(value or ""):
                raise CorpusError(f"{name} must be 64 hex characters")
        if self.split not in {s.value for s in Split}:
            raise CorpusError(f"unknown split {self.split!r}")
        if not self.source_group_id:
            raise CorpusError("a derived sample must inherit its base source group")
        if self.licence_id not in ALLOWED_LICENCE_IDS:
            raise CorpusError(
                f"a derived sample must inherit an allowed licence, got {self.licence_id!r}"
            )
        _reject_paths("notes", self.notes)

    def to_dict(self) -> dict:
        return asdict(self)


def derive_from(
    base: LicensedImageRecord,
    transformation: str,
    parameters: dict,
    seed: int,
    derived_sha256: str,
) -> DerivedSampleRecord:
    """Build a derived record that provably inherits its base's provenance.

    The only supported way to create a `DerivedSampleRecord`, so inheritance
    cannot be forgotten. A generated image has no licensed base and therefore
    cannot be degraded into the real-image distribution through this path.
    """
    if base.provenance != ProvenanceType.LICENSED_REAL.value:
        raise CorpusError(
            "controlled degradation applies to LICENSED_REAL bases only; "
            f"{base.image_id} is {base.provenance}"
        )
    suffix = hashlib.sha256(
        f"{base.image_id}|{transformation}|{sorted(parameters.items())}|{seed}".encode("utf-8")
    ).hexdigest()[:8]
    record = DerivedSampleRecord(
        derived_id=f"{base.image_id}~{transformation}~{suffix}",
        base_image_id=base.image_id,
        base_content_sha256=base.content_sha256,
        source_group_id=base.source_group_id,
        fruit_type=base.fruit_type,
        track=base.track,
        split=base.split,
        transformation=transformation,
        parameters=dict(parameters),
        seed=int(seed),
        derived_sha256=derived_sha256,
        licence_id=base.licence_id,
        creator=base.creator,
        attribution_text=base.attribution_text,
    )
    record.validate()
    return record
