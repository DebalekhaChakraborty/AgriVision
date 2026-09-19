# Licensed real-image corpus

Licence-verified photographs of apples, bananas and oranges, curated file by
file from Wikimedia Commons, used to calibrate the foreground-restricted
capture-quality policy.

**No image bytes are committed.** The committed artifacts are provenance
manifests: content hashes, creator, licence, attribution text and split
membership. Local paths never appear in them and are reconstructed from
`image_id` in code.

## What lives here

| Path | Committed | Purpose |
| --- | --- | --- |
| `manifests/candidate_pool.json` | yes | Every candidate that passed the licence gate, before curation |
| `manifests/licence_rejections.json` | yes | What was rejected and why — the gate's working |
| `manifests/curation.json` | yes | Per-file visual keep/drop decision, with the confirmed fruit and track |
| `manifests/licensed_corpus.json` | yes | The corpus: provenance and licence for every admitted image |
| `manifests/corpus_split.json` | yes | Deterministic source-group split with fingerprint |
| `manifests/derived_manifest.json` | yes | Controlled degradations: parameters, seeds, derived hashes |
| `manifests/synthetic_stress.json` | yes | Tier 3 stress scenes — provenance and claim boundary |
| `raw/` | **no** — gitignored | Downloaded image bytes |
| `cache/` | **no** — gitignored | Thumbnails and contact sheets used for curation |
| `derived/` | **no** — gitignored | Materialised degradations, only when explicitly requested |
| `synthetic/` | **no** — gitignored | Generated stress scenes |

## Why the bytes are not committed

Every licence here permits redistribution. Keeping the bytes local is a
judgement, not an obligation: the repository stays light, and the useful
artifact for review is the provenance record rather than a copy of someone
else's photograph. Committing a small attributed demo subset later is a
deliberate decision to take then, under the attribution and share-alike terms
each file carries.

## Three tiers, never mixed

| Provenance | What it is | What it may support |
| --- | --- | --- |
| `LICENSED_REAL` | a real photograph under a verified permissive licence | statements about behaviour on real imagery |
| `DERIVED_DEGRADED` | a controlled transformation of a licensed base | behaviour under a *characterised perturbation* |
| `SYNTHETIC_GENERATED` | a procedurally composed scene containing no fruit | exploratory stress probes only |

The schema enforces the separation: a `SYNTHETIC_GENERATED` record cannot carry
a real-image licence or sit in a real-image track, and `derive_from` refuses to
degrade anything that is not `LICENSED_REAL`.

## The rule that matters most

**Splits are assigned per source group, not per image.** A source group is the
creator identity, which is the conservative unit: two photographs by the same
contributor in the same fruit category are frequently two angles of the same
physical apple, and nothing in the metadata distinguishes that from two
unrelated shoots. Splitting on `image_id` would put one angle in calibration and
the other in held-out.

`corpus_split.json` is generated from provenance metadata alone, before any
quality metric is computed, and carries a fingerprint so a later edit is
detectable.

## Order of operations

```bash
# 1. screen metadata and apply the licence gate - downloads nothing
python -m competition.evaluation.build_licensed_corpus pool

# 2. cache thumbnails and render contact sheets for visual curation
python -m competition.evaluation.build_licensed_corpus thumbnails

# 3. after curation.json records the review: fetch the keepers at full size
python -m competition.evaluation.build_licensed_corpus harvest

# 4. split by source group, before any metric is examined
python -m competition.evaluation.split_licensed_corpus

# 5. build the controlled-degradation ladder
python -m competition.evaluation.degrade_licensed_corpus --split calibration

# 6. calibrate, then freeze
python -m competition.evaluation.calibrate_roi_policy

# 7. open held-out once
python -m competition.evaluation.degrade_licensed_corpus --split held_out
python -m competition.evaluation.evaluate_locked_policy
```

## Claim boundary

Thresholds here were calibrated on **licensed real fruit photographs with
controlled degradations**. They were not calibrated on phone-camera deployment
captures, and no result in this directory supports the phrase "real-world
validated". The optional micro-validation in
[`PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md`](../../../docs/competition/PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md)
exists to close that gap.
