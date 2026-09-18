"""Item-level split: determinism, stratification, and leakage prevention."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from competition.data.capture_schema import (
    MINIMUM,
    TARGET,
    CaptureSchemaError,
    Split,
    expected_item_ids,
    parse_item_id,
)
from competition.evaluation.split_capture_items import (
    DEFAULT_SEED,
    DEFAULT_SPLIT_PATH,
    assign_splits,
    build_split_document,
    calibration_items,
    evaluation_items,
    load_split,
    split_fingerprint,
)


# --- required test 6: deterministic for a given seed --------------------------


def test_split_is_deterministic_for_the_same_seed():
    items = expected_item_ids(TARGET)
    assert assign_splits(items, TARGET, 42) == assign_splits(items, TARGET, 42)


def test_different_seeds_generally_produce_different_splits():
    items = expected_item_ids(TARGET)
    assert assign_splits(items, TARGET, 1) != assign_splits(items, TARGET, 2)


def test_assignment_does_not_depend_on_catalogue_order():
    """Hashing (seed, item_id) means order cannot shift an item's split."""
    items = expected_item_ids(TARGET)
    assert assign_splits(items, TARGET, DEFAULT_SEED) == assign_splits(
        list(reversed(items)), TARGET, DEFAULT_SEED
    )


def test_adding_an_item_does_not_move_existing_items():
    """Adding a 25th fruit later must not silently reassign APL-003."""
    items = expected_item_ids(TARGET)
    before = assign_splits(items, TARGET, DEFAULT_SEED)
    extended = assign_splits(items + ["APL-009"], TARGET, DEFAULT_SEED)
    for item_id in items:
        assert extended[item_id] == before[item_id], item_id


# --- required test 7: stratification ------------------------------------------


def test_split_is_stratified_by_fruit():
    assignment = assign_splits(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    counts: Counter = Counter()
    for item_id, split in assignment.items():
        counts[(parse_item_id(item_id).value, split)] += 1

    for fruit in ("apple", "banana", "orange"):
        assert counts[(fruit, Split.EVALUATION.value)] == TARGET.evaluation_per_fruit
        assert counts[(fruit, Split.CALIBRATION.value)] == TARGET.calibration_per_fruit


def test_minimum_target_also_stratifies():
    assignment = assign_splits(expected_item_ids(MINIMUM), MINIMUM, DEFAULT_SEED)
    counts: Counter = Counter()
    for item_id, split in assignment.items():
        counts[(parse_item_id(item_id).value, split)] += 1
    for fruit in ("apple", "banana", "orange"):
        assert counts[(fruit, Split.EVALUATION.value)] == MINIMUM.evaluation_per_fruit


def test_every_item_receives_exactly_one_split():
    items = expected_item_ids(TARGET)
    assignment = assign_splits(items, TARGET, DEFAULT_SEED)
    assert set(assignment) == set(items)
    assert all(split in {s.value for s in Split} for split in assignment.values())


# --- required test 5: no item can span both splits ----------------------------


def test_calibration_and_evaluation_item_sets_are_disjoint():
    document = build_split_document(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    assert not (calibration_items(document) & evaluation_items(document))


def test_surplus_items_fall_into_calibration_not_evaluation():
    """Held-out size stays fixed; extra fruit strengthens calibration only."""
    items = expected_item_ids(TARGET) + ["APL-009", "APL-010"]
    assignment = assign_splits(items, TARGET, DEFAULT_SEED)
    evaluation_apples = [
        item for item, split in assignment.items()
        if split == Split.EVALUATION.value and item.startswith("APL")
    ]
    assert len(evaluation_apples) == TARGET.evaluation_per_fruit


def test_too_few_items_is_an_error_not_a_silent_shrink():
    with pytest.raises(CaptureSchemaError):
        assign_splits(["APL-001", "APL-002"], TARGET, DEFAULT_SEED)


# --- fingerprinting -----------------------------------------------------------


def test_document_carries_a_fingerprint_of_its_assignment():
    document = build_split_document(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    assert document["fingerprint"] == split_fingerprint(document)
    assert len(document["fingerprint"]) == 16


def test_edited_assignment_breaks_the_fingerprint(tmp_path):
    document = build_split_document(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    tampered = dict(document)
    tampered["assignment"] = dict(document["assignment"])
    victim = next(
        item for item, split in tampered["assignment"].items()
        if split == Split.EVALUATION.value
    )
    tampered["assignment"][victim] = Split.CALIBRATION.value

    path = tmp_path / "capture_split.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CaptureSchemaError) as error:
        load_split(path)
    assert "fingerprint" in str(error.value)


def test_missing_split_file_raises_with_guidance(tmp_path):
    with pytest.raises(CaptureSchemaError) as error:
        load_split(tmp_path / "absent.json")
    assert "split_capture_items" in str(error.value)


# --- the committed split ------------------------------------------------------


def test_committed_split_exists_and_verifies():
    """The split is generated before capture, so it must already be present."""
    document = load_split(DEFAULT_SPLIT_PATH)
    assert document["target_label"] == TARGET.label
    assert len(calibration_items(document)) == TARGET.calibration_items
    assert len(evaluation_items(document)) == TARGET.evaluation_items


def test_committed_split_records_its_discipline():
    document = load_split(DEFAULT_SPLIT_PATH)
    assert "before any capture metric is examined" in document["discipline"]
    assert "per physical item" in document["rule"]


def test_committed_split_contains_no_filesystem_paths():
    payload = Path(DEFAULT_SPLIT_PATH).read_text(encoding="utf-8")
    assert "/home/" not in payload
    assert "/Users/" not in payload


# --- extended-condition items are preassigned, not chosen later --------------


def test_extended_items_are_preassigned_in_the_committed_split():
    """Leaving "pick any two" to the photographer would allow selection effects."""
    from competition.evaluation.split_capture_items import extended_condition_item_ids

    document = load_split(DEFAULT_SPLIT_PATH)
    items = extended_condition_item_ids(document)
    assert len(items) == TARGET.extended_condition_items == 6


def test_extended_items_are_two_per_fruit():
    document = load_split(DEFAULT_SPLIT_PATH)
    chosen = document["extended_condition_items"]
    assert set(chosen) == {"apple", "banana", "orange"}
    for fruit, items in chosen.items():
        assert len(items) == 2, fruit
        prefix = {"apple": "APL", "banana": "BAN", "orange": "ORG"}[fruit]
        assert all(item.startswith(prefix) for item in items)


def test_extended_items_cover_both_splits_per_fruit():
    """One calibration and one held-out item per fruit."""
    document = load_split(DEFAULT_SPLIT_PATH)
    for fruit, items in document["extended_condition_items"].items():
        splits = {document["assignment"][item] for item in items}
        assert splits == {Split.CALIBRATION.value, Split.EVALUATION.value}, fruit


def test_extended_selection_is_deterministic():
    from competition.evaluation.split_capture_items import (
        assign_splits,
        select_extended_condition_items,
    )

    assignment = assign_splits(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    first = select_extended_condition_items(assignment, TARGET, DEFAULT_SEED)
    second = select_extended_condition_items(assignment, TARGET, DEFAULT_SEED)
    assert first == second


def test_extended_selection_does_not_alter_split_membership():
    """The fingerprint covers the assignment, so it must be untouched."""
    document = build_split_document(expected_item_ids(TARGET), TARGET, DEFAULT_SEED)
    assert document["fingerprint"] == split_fingerprint(document)
    assert len(calibration_items(document)) == TARGET.calibration_items
    assert len(evaluation_items(document)) == TARGET.evaluation_items


def test_extended_selection_has_its_own_fingerprint():
    document = load_split(DEFAULT_SPLIT_PATH)
    assert len(document["extended_condition_fingerprint"]) == 16


def test_committed_split_fingerprint_is_the_published_one():
    """The protocol document quotes this value; a change must be deliberate."""
    assert load_split(DEFAULT_SPLIT_PATH)["fingerprint"] == "2feb9360e1c2bdf5"
