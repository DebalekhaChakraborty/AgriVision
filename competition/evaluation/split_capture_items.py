"""Deterministic item-level split for the self-captured set.

Assigns **physical items**, never photographs. Every capture of one fruit
inherits that fruit's split, which is what keeps the held-out set genuinely
held out: a dark photo of APL-003 in evaluation and a reference photo of the
same apple in calibration would leak the subject across the boundary.

**Run this before any image is examined.** The split must not be influenced,
even indirectly, by how the metrics turn out. The assignment depends only on the
item catalogue and the seed, so it can be regenerated and checked at any time.

    .venv-competition/bin/python -m competition.evaluation.split_capture_items
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from competition.data.capture_schema import (
    COLLECTION_TARGETS,
    ITEM_PREFIX_BY_FRUIT,
    TARGET,
    CaptureSchemaError,
    CollectionTarget,
    Split,
    expected_item_ids,
    parse_item_id,
)

DEFAULT_SPLIT_PATH = Path("competition/data/capture_split.json")
DEFAULT_SEED = 20260918
SPLIT_VERSION = "phase2c-item-split-1.0.0"


def _item_rank(item_id: str, seed: int) -> str:
    """Stable pseudo-random ordering key for an item.

    A hash of (seed, item_id) rather than a shuffled list, so an item's ordering
    key does not depend on which other items exist.
    """
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def assign_splits(
    item_ids: list[str], target: CollectionTarget, seed: int = DEFAULT_SEED
) -> dict[str, str]:
    """Assign each item to a split, stratified by fruit type.

    Returns {item_id: split}. Deterministic for a given (items, target, seed).

    Held-out membership is drawn only from the *declared catalogue* - the first
    ``target.items_per_fruit`` numbers for each fruit. Any item beyond that
    always joins calibration.

    That restriction is what makes the assignment genuinely stable. Ranking
    every item that happens to exist would let a later-added APL-009 hash ahead
    of APL-003 and displace it out of the held-out set, silently invalidating a
    held-out result that had already been reported. Surplus fruit strengthens
    calibration; it never disturbs the boundary.
    """
    by_fruit: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        fruit = parse_item_id(item_id)
        by_fruit[fruit.value].append(item_id)

    assignment: dict[str, str] = {}
    for fruit_value, items in by_fruit.items():
        wanted_evaluation = target.evaluation_per_fruit

        catalogue = [
            item for item in items
            if int(item.split("-")[1]) <= target.items_per_fruit
        ]
        surplus = [item for item in items if item not in set(catalogue)]

        if len(catalogue) < wanted_evaluation + 1:
            raise CaptureSchemaError(
                f"{fruit_value}: {len(catalogue)} catalogued items cannot fill an "
                f"evaluation split of {wanted_evaluation} and leave any for calibration"
            )

        ordered = sorted(catalogue, key=lambda item: _item_rank(item, seed))
        for index, item_id in enumerate(ordered):
            assignment[item_id] = (
                Split.EVALUATION.value if index < wanted_evaluation
                else Split.CALIBRATION.value
            )
        for item_id in surplus:
            assignment[item_id] = Split.CALIBRATION.value
    return assignment


def select_extended_condition_items(
    assignment: dict[str, str], target: CollectionTarget, seed: int = DEFAULT_SEED
) -> dict[str, list[str]]:
    """Choose which items receive the sampled (extended) conditions.

    Fixed here, before any fruit is bought, rather than left to the photographer
    afterwards. "Pick any two" would let the choice be influenced — even
    unconsciously — by how the fruit looked once it was on the table, which is
    exactly the kind of selection effect the item-level split exists to prevent.

    One calibration and one held-out item per fruit, so the extended conditions
    are represented on both sides of the boundary. Selection uses a salted hash
    of the item id, independent of appearance, condition and every metric, and
    it does **not** alter the calibration/held-out assignment it reads.
    """
    chosen: dict[str, list[str]] = {}
    per_fruit = target.sampled_items_per_fruit

    for fruit, prefix in ITEM_PREFIX_BY_FRUIT.items():
        items = sorted(item for item in assignment if item.startswith(prefix))
        picks: list[str] = []

        # One from each split first, so both sides are covered; any remaining
        # slots are filled from the combined pool in the same stable order.
        for split in (Split.CALIBRATION.value, Split.EVALUATION.value):
            pool = sorted(
                (item for item in items if assignment[item] == split),
                key=lambda item: _extended_rank(item, seed),
            )
            if pool and len(picks) < per_fruit:
                picks.append(pool[0])

        if len(picks) < per_fruit:
            remainder = sorted(
                (item for item in items if item not in picks),
                key=lambda item: _extended_rank(item, seed),
            )
            picks.extend(remainder[: per_fruit - len(picks)])

        chosen[fruit.value] = sorted(picks)

    return chosen


def _extended_rank(item_id: str, seed: int) -> str:
    """Ordering key for extended-condition selection.

    Salted differently from `_item_rank` so the choice is independent of where
    an item landed in the calibration/held-out split.
    """
    return hashlib.sha256(f"{seed}:extended:{item_id}".encode("utf-8")).hexdigest()


def build_split_document(
    item_ids: list[str], target: CollectionTarget, seed: int = DEFAULT_SEED
) -> dict:
    assignment = assign_splits(item_ids, target, seed)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item_id, split in assignment.items():
        counts[parse_item_id(item_id).value][split] += 1

    document = {
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "target_label": target.label,
        "target": target.to_dict(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "rule": (
            "Splits are assigned per physical item. Every capture of an item "
            "inherits the item's split. Assignment is a hash of (seed, item_id), "
            "so it is reproducible and independent of catalogue order."
        ),
        "discipline": (
            "Generated before any capture metric is examined. Evaluation items "
            "stay closed until a calibrated policy is locked and fingerprinted."
        ),
        "counts_by_fruit": {fruit: dict(split_counts) for fruit, split_counts in counts.items()},
        "assignment": dict(sorted(assignment.items())),
    }
    document["fingerprint"] = split_fingerprint(document)

    # Recorded after the fingerprint is computed, deliberately: the fingerprint
    # covers the calibration/held-out assignment only, so adding this selection
    # provably does not disturb split membership.
    extended = select_extended_condition_items(assignment, target, seed)
    document["extended_condition_items"] = extended
    document["extended_condition_note"] = (
        "Items receiving the sampled (extended) conditions. Preassigned before "
        "capture so the choice cannot be influenced by how the fruit looks. "
        "One calibration and one held-out item per fruit."
    )
    document["extended_condition_fingerprint"] = hashlib.sha256(
        json.dumps(extended, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return document


def extended_condition_item_ids(document: dict) -> set[str]:
    """Flat set of the preassigned extended-condition items."""
    return {
        item
        for items in document.get("extended_condition_items", {}).values()
        for item in items
    }


def split_fingerprint(document: dict) -> str:
    """Hash of the assignment itself, so a silently edited split is detectable."""
    payload = json.dumps(
        {
            "seed": document["seed"],
            "target_label": document["target_label"],
            "assignment": document["assignment"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_split(path: Path = DEFAULT_SPLIT_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise CaptureSchemaError(
            f"split manifest not found: {path.name}. Generate it with "
            "`python -m competition.evaluation.split_capture_items` before capturing."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    recorded = document.get("fingerprint")
    if recorded and recorded != split_fingerprint(document):
        raise CaptureSchemaError(
            "split manifest fingerprint does not match its assignment; "
            "the file has been edited after generation"
        )
    return document


def calibration_items(document: dict) -> set[str]:
    return {
        item for item, split in document["assignment"].items()
        if split == Split.CALIBRATION.value
    }


def evaluation_items(document: dict) -> set[str]:
    return {
        item for item, split in document["assignment"].items()
        if split == Split.EVALUATION.value
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.split_capture_items",
        description="Generate the deterministic item-level calibration/evaluation split.",
    )
    parser.add_argument("--target", default=TARGET.label, choices=sorted(COLLECTION_TARGETS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing split (invalidates any held-out claim)")
    args = parser.parse_args(argv)

    target = COLLECTION_TARGETS[args.target]
    output = Path(args.output)

    if output.exists() and not args.force:
        print(f"{output.name} already exists. Regenerating it would invalidate any "
              "held-out evaluation already performed. Pass --force if that is intended.")
        return 1

    document = build_split_document(expected_item_ids(target), target, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    print(f"target:       {target.label} ({target.total_items} items)")
    print(f"calibration:  {len(calibration_items(document))} items")
    print(f"evaluation:   {len(evaluation_items(document))} items (held out)")
    print(f"seed:         {args.seed}")
    print(f"fingerprint:  {document['fingerprint']}")
    print(f"captures:     {target.total_capture_count} "
          f"({target.required_capture_count} required + "
          f"{target.extended_capture_count} extended + 6 scene)")
    print("extended-condition items:")
    for fruit, items in sorted(document["extended_condition_items"].items()):
        labels = ", ".join(f"{i} ({document['assignment'][i]})" for i in items)
        print(f"  {fruit:8} {labels}")
    print(f"written:      {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
