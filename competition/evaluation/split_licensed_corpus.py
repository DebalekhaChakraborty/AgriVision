"""Source-group split for the licensed corpus, fixed before any metric is read.

The unit of splitting is the **source group**, never the image. Two photographs
by the same contributor in the same fruit category are frequently two angles of
the same physical apple on the same afternoon, and no field in the metadata
distinguishes that from two unrelated shoots. Splitting on `image_id` would put
one angle in calibration and the other in held-out, and the held-out numbers
would then measure memorisation of a specific apple rather than generalisation.

This is the same discipline as the self-capture protocol, where the unit is the
physical fruit. The reason it needs restating here is that public imagery hides
its grouping: a dataset of 100,000 frames can be 250 fruits filmed rotating, and
nothing in the file names says so.

Access control, not just convention
-----------------------------------
`CorpusView` is the only supported way to read records during calibration, and a
calibration view raises on any attempt to reach held-out material. A comment
saying "do not look at held-out" is not a control; an exception is. The view
also refuses to hand out held-out records until a policy fingerprint has been
supplied, which is what makes "the policy was frozen first" checkable rather
than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from competition.data.licensed_sources import (
    CorpusError,
    CorpusTrack,
    LicensedImageRecord,
    Split,
)
from competition.evaluation.build_licensed_corpus import (
    CORPUS_MANIFEST,
    MANIFEST_DIR,
    load_corpus,
)

SPLIT_PATH = MANIFEST_DIR / "corpus_split.json"
SPLIT_VERSION = "phase2c-source-group-split-1.0.0"
DEFAULT_SEED = 20260918
HELD_OUT_TARGET_FRACTION = 0.28


def _group_rank(group_id: str, seed: int) -> str:
    """Deterministic, order-independent rank for one source group.

    Salted with the seed and hashed so that adding a group later cannot shuffle
    the groups already assigned — the same property the capture split relies on.
    """
    return hashlib.sha256(f"{seed}:group:{group_id}".encode("utf-8")).hexdigest()


def assign_group_splits(
    records: list[LicensedImageRecord],
    seed: int = DEFAULT_SEED,
    held_out_fraction: float = HELD_OUT_TARGET_FRACTION,
) -> dict[str, str]:
    """Assign every source group to exactly one split.

    Stratification is approximate by necessity: a group can contain images from
    several fruit/track strata, and group integrity outranks exact stratum
    proportions. Strata are filled in a fixed order and each stops as soon as it
    reaches its held-out quota, so the shortfall lands on whichever stratum is
    processed last rather than being spread invisibly.
    """
    if not records:
        raise CorpusError("cannot split an empty corpus")
    if not 0.0 < held_out_fraction < 1.0:
        raise CorpusError(f"held_out_fraction must be in (0, 1), got {held_out_fraction}")

    by_stratum: dict[tuple[str, str], list[LicensedImageRecord]] = defaultdict(list)
    groups_in_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        stratum = (record.fruit_type, record.track)
        by_stratum[stratum].append(record)
        if record.source_group_id not in groups_in_stratum[stratum]:
            groups_in_stratum[stratum].append(record.source_group_id)

    group_sizes: Counter = Counter(record.source_group_id for record in records)
    assignment: dict[str, str] = {}

    for stratum in sorted(by_stratum):
        quota = int(round(len(by_stratum[stratum]) * held_out_fraction))
        held_out_so_far = sum(
            group_sizes[group] for group in groups_in_stratum[stratum]
            if assignment.get(group) == Split.HELD_OUT.value
        )
        candidates = sorted(
            (group for group in groups_in_stratum[stratum] if group not in assignment),
            key=lambda group: _group_rank(group, seed),
        )
        for group in candidates:
            if held_out_so_far >= quota:
                break
            assignment[group] = Split.HELD_OUT.value
            held_out_so_far += group_sizes[group]

    for group in group_sizes:
        assignment.setdefault(group, Split.CALIBRATION.value)
    return assignment


def apply_split(
    records: list[LicensedImageRecord], assignment: dict[str, str]
) -> list[LicensedImageRecord]:
    """Stamp each record with its group's split."""
    from dataclasses import replace

    stamped = []
    for record in records:
        split = assignment.get(record.source_group_id)
        if split is None:
            raise CorpusError(f"no split assigned for group {record.source_group_id}")
        stamped.append(replace(record, split=split))
    return stamped


def verify_no_group_crosses_split(records: list[LicensedImageRecord]) -> None:
    """The invariant this whole module exists to hold. Checked, not assumed."""
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_group[record.source_group_id].add(record.split)
    offenders = {
        group: sorted(splits) for group, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if offenders:
        raise CorpusError(f"source groups span both splits: {offenders}")


def split_fingerprint(document: dict) -> str:
    payload = json.dumps(document["assignment"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_split_document(
    records: list[LicensedImageRecord],
    seed: int = DEFAULT_SEED,
    held_out_fraction: float = HELD_OUT_TARGET_FRACTION,
) -> dict:
    assignment = assign_group_splits(records, seed, held_out_fraction)
    stamped = apply_split(records, assignment)
    verify_no_group_crosses_split(stamped)

    distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in stamped:
        distribution[record.split][f"{record.fruit_type}/{record.track}"] += 1

    document = {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "held_out_target_fraction": held_out_fraction,
        "rule": "split assigned per source group, never per image",
        "discipline": (
            "generated from provenance metadata alone, before any quality metric "
            "was computed on the corpus"
        ),
        "assignment": dict(sorted(assignment.items())),
        "group_count": len(assignment),
        "image_count": len(stamped),
        "distribution": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(distribution.items())
        },
        "images_per_split": dict(sorted(Counter(r.split for r in stamped).items())),
    }
    document["fingerprint"] = split_fingerprint(document)
    return document


def load_split(path: Path = SPLIT_PATH) -> dict:
    if not path.is_file():
        raise CorpusError(
            f"{path.name} not found; run "
            "`python -m competition.evaluation.split_licensed_corpus` first"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = split_fingerprint(document)
    if document.get("fingerprint") != expected:
        raise CorpusError(
            f"split fingerprint mismatch: recorded {document.get('fingerprint')!r}, "
            f"recomputed {expected!r}. The assignment has been edited since it was created."
        )
    return document


class HeldOutAccessError(RuntimeError):
    """Raised when held-out material is reached without a frozen policy."""


@dataclass
class CorpusView:
    """A split-scoped window onto the corpus.

    Constructed through `calibration_view` or `held_out_view`; a calibration
    view physically cannot return held-out records, so a calibration script
    cannot peek even by mistake.
    """

    records: list[LicensedImageRecord]
    split: str
    policy_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.split == Split.HELD_OUT.value and not self.policy_fingerprint:
            raise HeldOutAccessError(
                "held-out records require a frozen policy fingerprint; freeze the "
                "policy with calibrate_roi_policy before opening held-out data"
            )
        for record in self.records:
            if record.split != self.split:
                raise HeldOutAccessError(
                    f"{record.image_id} is {record.split}, not {self.split}"
                )

    def by_track(self, track: CorpusTrack) -> list[LicensedImageRecord]:
        return [record for record in self.records if record.track == track.value]

    def __len__(self) -> int:
        return len(self.records)


def split_records(
    records: list[LicensedImageRecord] | None = None,
    document: dict | None = None,
) -> list[LicensedImageRecord]:
    records = records if records is not None else load_corpus()
    document = document if document is not None else load_split()
    return apply_split(records, document["assignment"])


def calibration_view(
    records: list[LicensedImageRecord] | None = None,
    document: dict | None = None,
) -> CorpusView:
    stamped = split_records(records, document)
    return CorpusView(
        [r for r in stamped if r.split == Split.CALIBRATION.value],
        Split.CALIBRATION.value,
    )


def held_out_view(
    policy_fingerprint: str,
    records: list[LicensedImageRecord] | None = None,
    document: dict | None = None,
) -> CorpusView:
    if not policy_fingerprint:
        raise HeldOutAccessError("a frozen policy fingerprint is required")
    stamped = split_records(records, document)
    return CorpusView(
        [r for r in stamped if r.split == Split.HELD_OUT.value],
        Split.HELD_OUT.value,
        policy_fingerprint=policy_fingerprint,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.split_licensed_corpus",
        description="Deterministic source-group split of the licensed corpus.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--held-out-fraction", type=float, default=HELD_OUT_TARGET_FRACTION)
    args = parser.parse_args(argv)

    records = load_corpus(CORPUS_MANIFEST)
    document = build_split_document(records, args.seed, args.held_out_fraction)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "groups": document["group_count"],
        "images": document["image_count"],
        "per_split": document["images_per_split"],
        "fingerprint": document["fingerprint"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
