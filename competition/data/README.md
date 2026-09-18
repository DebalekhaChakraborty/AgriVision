# Capture data directory

Schema, split and manifest for the self-captured deployment-domain image set.

**Status: PROTOCOL LOCKED — CAPTURE PENDING.** No photograph has been taken. No
threshold has been calibrated.

**186 photographs required**: 144 required (24 items × 6) + 36 extended
(6 preassigned items × 6) + 6 scene.

Extended-condition items, preassigned before capture:
**APL-002, APL-003, BAN-004, BAN-007, ORG-006, ORG-007**.

## What lives here

| Path | Committed | Purpose |
| --- | --- | --- |
| `capture_schema.py` | yes | Conditions, quality rubric, record types, collection targets |
| `capture_split.json` | yes | Deterministic item-level calibration/evaluation split |
| `capture_manifest.json` | yes, once captures exist | Records with **no filesystem paths** |
| `capture_manifest.local.json` | **no** — gitignored | Same records plus local paths, for execution |
| `self_captured/raw/` | **no** — gitignored | The photographs themselves |
| `self_captured/sanitised/` | **no** — gitignored | EXIF-stripped copies for later public use |

## Why the images are not committed

They are self-captured and therefore licensing-clean, which is the entire point
of collecting them. They are still kept local at this stage for two reasons:
the repository stays light, and no photograph reaches version control before its
EXIF has been checked. Phone cameras routinely embed GPS coordinates and device
serial numbers.

A later phase may commit a small sanitised demo subset. That is a deliberate
decision to be taken then, not a default.

## Naming

```
<ITEM-ID>__<CONDITION>__<REPLICATE>.jpg
APL-001__REFERENCE__1.jpg
BAN-004__DEFOCUS_SEVERE__1.jpg
SCENE__MIXED_FRUIT__1.jpg
```

Item ids are `APL-###`, `BAN-###`, `ORG-###`. The prefix encodes the fruit type
and is cross-checked against the record.

## The rule that matters most

**Splits are assigned per physical fruit, not per photograph.** Every capture of
apple `APL-003` belongs to whichever split `APL-003` belongs to. Mixing them
would leak the subject across the held-out boundary and make any evaluation
figure meaningless.

`capture_split.json` is generated **before** any image is examined, and carries a
fingerprint so a later edit is detectable.

## Order of operations

```bash
# 1. before buying fruit — fixes the split in advance
python -m competition.evaluation.split_capture_items

# 2. after photographing
python -m competition.evaluation.prepare_capture_manifest

# 3. before relying on anything
python -m competition.evaluation.validate_capture_set
```

Full protocol and the human capture checklist:
[`docs/competition/PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md`](../../docs/competition/PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md).

## Claim boundary

`visible_condition` records appearance only — `FRESH_APPEARING`,
`VISIBLY_DEGRADED`, `UNCERTAIN`. It is not a food-safety, edibility or
contamination judgement, and that vocabulary is deliberately absent from the
schema.
