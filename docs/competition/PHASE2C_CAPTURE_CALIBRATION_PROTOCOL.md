# Phase 2c — Deployment-Domain Capture Protocol and Calibration Methodology

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Phase 2b base commit | `f615689a4846696cd3eec1d56bb9ddf0a805854f` |
| **Status** | **PROTOCOL LOCKED — CAPTURE PENDING** |
| Photographs collected | **0** |
| Thresholds calibrated | **none** |
| Split fingerprint | `2feb9360e1c2bdf5` |
| Photographs required | **186** (144 required + 36 extended + 6 scene) |
| Tests | 413 passing |

> **Nothing here is calibrated.** This phase locks *how* the capture set will be
> built and *how* thresholds will later be chosen, before any image exists, so
> that the method cannot be shaped by the data it will be applied to. No
> threshold value appears anywhere in this document or in the tooling.

---

## 1. Why a new capture set is needed

Two phases have now ended at the same wall.

Phase 2 found the capture gate rejecting **96.7%** of research photographs.
Phase 2b diagnosed and removed one cause — background-driven clipping — and in
doing so exposed a second: **capture-quality thresholds are scope-specific**. ROI
Laplacian variance is roughly 5% of the whole-image value, so the threshold of
100 is a number from a different population and cannot transfer.

No further code fixes this. Choosing a correct ROI threshold requires
photographs from the domain the system will actually operate in, and the
research datasets are not that domain:

| Dataset | Licence | Usable for calibration? |
| --- | --- | --- |
| Kaggle frozen split | Unknown, no redistribution grant | **No** — wrong domain (studio product photography), and undistributable |
| Sultana 2022 | CC BY 4.0 | No — also curated product photography |
| FruitVision | CC BY-NC-ND 4.0 | No — ambiguous terms, and same domain problem |

Self-captured imagery solves the licensing problem *and* the domain problem at
once, and produces the demo and video material the submission needs anyway.

## 2. The rule that governs everything

**Splits are assigned per physical fruit, never per photograph.**

A dark photo of apple `APL-003` in the held-out set while a reference photo of
the same apple sits in calibration would leak the subject across the boundary.
The held-out result would then measure memorisation of that specific apple, not
generalisation. Every capture of one item inherits that item's split, and the
validator fails loudly if this is ever violated.

## 3. Collection targets

| Target | Items | Per fruit | Calibration | Held-out | Approx. captures |
| --- | --- | --- | --- | --- | --- |
| **TARGET** | **24** | 8 | **15** | **9** | **186** |
| MINIMUM | 18 | 6 | 12 | 6 | 150 |
| STRETCH | 30 | 10 | 18 | 12 | 240 |

TARGET breaks down as 24 x 6 required = **144**, plus 6 extended-condition
items x 6 sampled conditions = **36**, plus **6** scene captures = **186**.

A pragmatic solo-participant target, **not a statistical power calculation**.

**If MINIMUM is used**, say so explicitly in every result: with only two held-out
items per fruit, one atypical fruit moves the outcome materially, and no
threshold should be chosen on a narrow margin.

**STRETCH is not requested.** Do not buy 30 fruit on the strength of this
document.

## 4. The locked split

Generated **before any fruit was bought**, with seed `20260918`, fingerprint
`2feb9360e1c2bdf5`. Stratified 5 calibration / 3 held-out per fruit.

| Split | Items |
| --- | --- |
| **Held-out (9)** | `APL-003`, `APL-006`, `APL-008`, `BAN-003`, `BAN-004`, `BAN-008`, `ORG-004`, `ORG-005`, `ORG-006` |
| **Calibration (15)** | `APL-001`, `APL-002`, `APL-004`, `APL-005`, `APL-007`, `BAN-001`, `BAN-002`, `BAN-005`, `BAN-006`, `BAN-007`, `ORG-001`, `ORG-002`, `ORG-003`, `ORG-007`, `ORG-008` |

### The six extended-condition items, also preassigned

Fixed here, before any fruit is bought, rather than left as "pick any two". That
choice would otherwise be made once the fruit is on the table, and could be
influenced — even unconsciously — by how each piece looked, which is exactly the
selection effect the item-level split exists to prevent.

| Fruit | Calibration | Held-out |
| --- | --- | --- |
| Apple | **APL-002** | **APL-003** |
| Banana | **BAN-007** | **BAN-004** |
| Orange | **ORG-007** | **ORG-006** |

One calibration and one held-out item per fruit, so the extended conditions are
represented on both sides of the boundary. Selected by a salted hash of the item
id — independent of appearance, condition and every metric — and recorded in
`capture_split.json` with its own fingerprint. **The calibration/held-out
assignment is unchanged by this selection**, which is verifiable: the split
fingerprint `2feb9360e1c2bdf5` covers the assignment only and did not move.

Assignment is a hash of `(seed, item_id)` restricted to the declared catalogue,
so it does not depend on catalogue order and **extra fruit bought later joins
calibration without disturbing the held-out set**. The file carries a fingerprint
of its own assignment, so a later edit is detectable, and regeneration requires
`--force`.

## 5. Capture-condition matrix

Derived from the eleven conditions already in
[DEMO_DATA_PLAN.md](DEMO_DATA_PLAN.md), with three changes recorded here rather
than made silently:

* **"Visibly deteriorated produce" moved from condition to item attribute.** It
  describes the fruit, not how it was photographed, and is now
  `VisibleConditionAnnotation`.
* **"Mixed fruit in frame" and "Non-produce object" became scene cases.**
  Neither is a photograph of one catalogued item, so neither fits the per-item
  matrix.
* **Blur and darkness split into mild and severe.** The boundary between
  "remediable" and "recapture required" is precisely what calibration must
  locate, so both sides of it need examples.

### Required for every item (6 × 24 = **144** captures)

| # | Condition | Purpose | Procedure | Quality target | Remediation | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `REFERENCE` | Baseline the gate should accept | Ordinary indoor diffuse light, whole fruit visible, ~25–35 cm, plain everyday surface | ACCEPTABLE | no | both |
| 2 | `REFERENCE_REPEAT` | Capture-to-capture repeatability | Repeat after lowering and re-raising the camera; don't reproduce framing exactly | ACCEPTABLE | no | gate |
| 3 | `DIM` | Underexposure one gamma pass should recover | **Reduced** ambient light — main light off or away from the window. Fruit stays **clearly visible** to you. No flash. Not darkness; that is `DARK_SEVERE` | REMEDIABLE | **yes** | gate |
| 4 | `OVEREXPOSED` | Highlight clipping **on the subject** | Raise **overall** exposure: strong **diffuse** light, or exposure compensation up until highlights blow. **No concentrated hotspot** — that is `GLARE` | REMEDIABLE | **yes** | gate |
| 5 | `DEFOCUS_SEVERE` | Blur no tone operation recovers | Focus **away** from the subject enough to **destroy inspection detail** — tap-focus across the room, then shoot without refocusing. Surface texture unreadable | RECAPTURE_REQUIRED | no | gate |
| 6 | `CLUTTERED_BACKGROUND` | Foreground isolation on a realistic background | REFERENCE lighting and distance, everyday objects around and behind. **The subject itself stays well lit and in focus** — the difficulty is the surroundings | ADVISORY_ONLY | no | segmentation |

### Extended — the six preassigned items only (6 × 6 = **36** captures)

| # | Condition | Purpose | Procedure | Quality target | Tests |
| --- | --- | --- | --- | --- | --- |
| 7 | `DARK_SEVERE` | Unrecoverable shadow crush | **Near-dark**, intended to materially remove visual information. Curtains closed, lights off, no flash. Fruit barely discernible. Deliberately more extreme than `DIM` | RECAPTURE_REQUIRED | gate |
| 8 | `GLARE` | Specular highlight on the skin | **Directional** light — torch or bare bulb at an angle to produce a **specular highlight**. Overall exposure stays normal; the hotspot is the difficulty, not general brightness | ADVISORY_ONLY | both |
| 9 | `DEFOCUS_MILD` | **The boundary case calibration must locate** | **Slight** focus error; fruit stays clearly **recognisable**. Err toward too little blur rather than too much | REMEDIABLE | gate |
| 10 | `MOTION_BLUR` | Handheld motion, not tone-remediable | Move the **camera** steadily sideways as the shutter fires, fruit **nominally in frame**. Do not move the fruit or let it leave the frame | RECAPTURE_REQUIRED | gate |
| 11 | `PARTIAL_OCCLUSION` | Subject partly hidden | Cover roughly a quarter to a third with a hand or object, otherwise REFERENCE conditions. **Do not move, rotate or alter the fruit** — only occlude it | ADVISORY_ONLY | both |
| 12 | `SMALL_SUBJECT` | Exercises the foreground area guards | Increase **camera distance** to ~1 m, or place the fruit near a frame corner. **No cropping or digital resizing** — it must come from the capture | ADVISORY_ONLY | segmentation |

### Scene cases (not item-bound, 3 each = **6** captures)

`MIXED_FRUIT` — several fruits in frame; tests the single-label limitation.
`NON_PRODUCE` — an everyday object; tests out-of-taxonomy refusal.

**Total: exactly 186 photographs** — 144 + 36 + 6. At a few seconds each plus
setup between conditions, realistically two to three sittings.

## 6. Reference captures

Every item needs at least one `REFERENCE`, and `REFERENCE_REPEAT` is required
too so that capture-to-capture variation can be separated from real differences.

The reference should look like **a reasonable user photograph**, not a studio
shot. Perfect lighting would make the reference unrepresentative of deployment
and would bias the very thresholds being calibrated.

## 7. Device policy

Use **one primary device** for the whole calibration set. Record it as a
non-personal label (`PHONE_A`).

Note whether HDR or automatic scene enhancement is on. If it is, leave it on —
that is what a real user's photograph looks like — but record it, because it
changes the tone curve and therefore the metrics.

A second device may be added later as a **robustness set**. Do not mix devices
casually into the initial calibration set; a device difference would be
indistinguishable from a capture-condition difference.

## 8. Real degradations, not simulated ones

**Degrade physically.** Move the camera, dim the room, angle the light, cover
part of the fruit.

Do **not** digitally degrade a clean capture and file it as deployment-domain
evidence. Synthetic degradation remains valuable engineering evidence — Phases 1
and 1b rest on it — but it is a separate dataset answering a separate question.
The whole reason for this collection is that synthetic proxies have twice now
produced thresholds that failed on real photographs.

## 9. Quality-target rubric

Preregistered in the protocol, assigned by the photographer's intent and human
review, **never derived from the metrics being calibrated**. Deriving labels
from the metrics would make calibration circular — the thresholds would be
validated against their own output.

| Label | Meaning |
| --- | --- |
| `ACCEPTABLE` | Sufficient subject detail for automated inspection |
| `REMEDIABLE` | A recoverable capture issue; one bounded automated remediation is allowed |
| `RECAPTURE_REQUIRED` | Information materially missing or destroyed |
| `ADVISORY_ONLY` | Tests segmentation or background behaviour; should not necessarily block |

The schema enforces the mapping: a record cannot claim a quality target that
contradicts its condition's preregistered one. Changing a label means changing
the protocol deliberately, not editing a row.

### Visible condition — a separate, conservative annotation

`FRESH_APPEARING` · `VISIBLY_DEGRADED` · `UNCERTAIN`

Appearance only. `SAFE`, `UNSAFE`, `EDIBLE` and `CONTAMINATED` are absent from
the vocabulary by design, and a test asserts no food-safety term appears
anywhere in the protocol. `UNCERTAIN` exists so no one is forced into a
confident label.

**Do not deliberately cultivate spoilage.** Use naturally aged fruit if it is
already to hand. The goal is visible-condition diversity, not biological
experimentation.

## 10. Manifest schema

`CaptureRecord` carries: `image_id`, `item_id`, `fruit_type`,
`capture_condition`, `replicate_id`, `split`, `device_id`, `quality_target`,
`visible_condition`, `remediation_expected`, `content_sha256`, `width`,
`height`, `orientation`, `capture_notes`, `schema_version`.

**No filesystem path.** Paths live only in `capture_manifest.local.json`, which
is gitignored; a test asserts this.

Filenames: `<ITEM-ID>__<CONDITION>__<REPLICATE>.jpg`, e.g.
`APL-001__REFERENCE__1.jpg`, `SCENE__MIXED_FRUIT__1.jpg`.

## 11. Storage and privacy

`competition/data/self_captured/` is **gitignored**, verified by test before any
photograph exists.

Self-captured images are licensing-clean, but they are still kept local at this
stage: phone cameras routinely embed **GPS coordinates and device serial
numbers**, and nothing should reach version control before that is checked.

The validator scans for `GPSInfo`, `Artist`, `CameraOwnerName`,
`BodySerialNumber`, `ImageUniqueID` and related tags, and warns on each.

### When an EXIF warning is acceptable, and when it is not

**Acceptable on local raw captures.** `self_captured/raw/` is gitignored, so an
EXIF warning there is a note, not a defect. Do not strip metadata from raw
captures — device information is genuine provenance for the calibration set.

**Not acceptable on anything that leaves this machine.** Before an image is used
for any of the following, it must go through the sanitisation workflow:

- committed to the repository
- uploaded to Devpost
- served by the demo endpoint
- shown in the judge video or any submission material
- shared publicly in any other form

A warning that was fine locally becomes a privacy leak the moment the file is
distributed. Sanitised copies go in `self_captured/sanitised/` with metadata
stripped; **the canonical raw captures are never overwritten or modified**, and
raw-to-sanitised provenance is kept by content hash.

## 12. Calibration methodology — fixed now, executed later

Locked in `competition/evaluation/calibrate_quality_policy.py`, viewable with
`--show-method`. It contains **no threshold values**, and a test enforces that.

Per metric, it fixes the failure condition, the population, the candidate
analysis, and how errors are reported:

| Metric | Failure condition |
| --- | --- |
| ROI sharpness | Insufficient subject detail for reliable inspection |
| Mean luminance | Subject too dark or too bright |
| Contrast | Tonal range too compressed |
| Shadow clipping | Unrecoverable shadow crush on the subject |
| Highlight clipping | Unrecoverable blown highlights on the subject |
| Foreground validity | Subject not locatable, so no ROI metric is trustworthy |

### The sharpness rule

ROI sharpness is content-, exposure-, enhancement- and scale-dependent.
**It is not assumed that a single threshold exists.** Calibration will examine
the distribution by fruit type, by capture condition, and by subject scale.

Fruit-specific thresholds are avoided unless strongly justified: fruit identity
is not reliably known *before* the quality gate runs, so a fruit-specific
threshold would need information the gate does not have.

**If no stable threshold separates acceptable from unacceptable captures, that
result gets reported and the sharpness measurement gets redesigned** — for
instance normalised by local contrast, or measured at a fixed subject-relative
scale. It does not get tuned until a number appears to work.

### Objective

A **confusion matrix** against the rubric, not an accuracy number.

**FALSE ACCEPT** (an unusable capture is permitted) and **FALSE BLOCK** (an
acceptable capture is rejected) are reported separately at every candidate
threshold. For a safety-oriented gate false accepts matter more, but **no target
percentage is set in advance** — the trade-off curve is shown, rather than a
chosen point being justified afterwards.

A metric stays **advisory** when it separates poorly, when its failure mode is
recoverable, or when blocking on it would reject captures a human would accept.

## 13. Held-out discipline

1. Select thresholds on calibration items only.
2. Freeze the policy; record its fingerprint and the code revision.
3. **Then** evaluate once on the held-out items.

**If thresholds change after held-out results are seen, that set is no longer
confirmatory.** Any later use must be labelled exploratory, or a fresh untouched
set must be collected. This is not negotiable after the fact — it is the only
thing that makes the held-out number mean anything.

Enforced in code: `load_calibration_captures` returns calibration records only
and raises if a record claims the calibration split while its item is assigned
elsewhere. `calibrate_quality_policy` refuses to run below 40 captures, or
without examples at both ends of the rubric.

## 14. Optional mask audit (not required)

Phase 2b reports foreground **validity rate**, which measures guard passage, not
correctness. Real segmentation quality is still unmeasured.

Optionally, later: hand-draw polygon masks for 20–30 images spanning fruit types
and background conditions, using any external tool that exports polygons, and
compute real IoU. **This is optional and must not block Phase 2c.** No
annotation platform should be built for it.

## 15. Tooling

| Command | Purpose |
| --- | --- |
| `python -m competition.evaluation.split_capture_items` | Generate the item-level split (already done) |
| `python -m competition.evaluation.prepare_capture_manifest` | Build manifests from the raw directory |
| `python -m competition.evaluation.validate_capture_set` | Check the set before relying on it |
| `python -m competition.evaluation.calibrate_quality_policy --show-method` | Review the locked methodology |

The validator checks: item catalogue completeness, fruit classes, condition
coverage, missing required captures, duplicate content, **cross-split item
leakage**, unsupported extensions, unreadable images, implausible dimensions,
manifest/disk agreement, split-fingerprint agreement, and EXIF privacy.

---

# HUMAN CAPTURE CHECKLIST

Usable without reading any source code.

### Step 1 — Buy the fruit

**8 apples, 8 bananas, 8 oranges — 24 pieces.**

Pick visibly different individuals: different sizes, varieties and colours where
the shop has them. If a few are naturally bruised or over-ripe, **include them**
— genuinely degraded examples are valuable and hard to obtain later. Do not try
to create spoilage.

*If 24 is impractical, 6 per fruit (18 total) is acceptable. Tell me you used
the reduced set, because it weakens every held-out result.*

### Step 2 — Label each fruit

Put a small sticker or tape label on each piece:

```
APL-001 … APL-008      BAN-001 … BAN-008      ORG-001 … ORG-008
```

**The label must stay on the fruit for the whole session.** Every photo of
`APL-003` must be filed under `APL-003`.

### Step 3 — Which items get extra photos

Already decided, before you bought anything, so the choice cannot be swayed by
how a particular fruit looks. **These six items get the extended conditions:**

> ### **APL-002 · APL-003 · BAN-004 · BAN-007 · ORG-006 · ORG-007**

Everything else gets the six required photos only.

(The calibration/held-out split is also already fixed. You do not need to act on
it — photograph all 24 items identically, treat none of them differently.)

### Step 4 — Photograph each item

**For every one of the 24 items, take these 6 photos (144 total):**

| Photo | What to do |
| --- | --- |
| 1. `REFERENCE` | Normal room light, whole fruit in frame, ~30 cm away, plain table |
| 2. `REFERENCE_REPEAT` | Lower the phone, raise it again, take the same shot |
| 3. `DIM` | Main light off, or step away from the window. **Fruit must still look clearly visible to you.** No flash |
| 4. `OVEREXPOSED` | Make the whole photo too bright — strong **broad** light, or tap to brighten until highlights blow out. **Not a torch spot** |
| 5. `DEFOCUS_SEVERE` | Tap-focus on something across the room, then photograph the fruit. Should be **obviously** blurry |
| 6. `CLUTTERED_BACKGROUND` | Normal light, fruit still sharp and well lit, but everyday objects around and behind it |

**Then, for the six items named in Step 3 only, take these 6 extra photos each
(36 total):**

| Photo | What to do |
| --- | --- |
| 7. `DARK_SEVERE` | Near-dark room, lights off, no flash. Fruit barely visible |
| 8. `GLARE` | Normal brightness, but a torch angled to make a **bright spot** on the skin |
| 9. `DEFOCUS_MILD` | Focus just slightly off. Fruit still **clearly recognisable** — only a little soft |
| 10. `MOTION_BLUR` | Move the **phone** sideways as you press the shutter. Keep the fruit in frame |
| 11. `PARTIAL_OCCLUSION` | Cover about a third with your hand. **Don't move or turn the fruit** |
| 12. `SMALL_SUBJECT` | **Step back** about a metre so the fruit looks small. Do not crop afterwards |

**Finally, 6 scene photos** (no fruit label needed): 3 with several fruits
together, 3 of an everyday non-food object.

**186 photos in total: 144 + 36 + 6.**

### Step 5 — Name and store the files

```
APL-001__REFERENCE__1.jpg
APL-001__REFERENCE_REPEAT__1.jpg
APL-001__DIM__1.jpg
BAN-004__DEFOCUS_SEVERE__1.jpg
SCENE__MIXED_FRUIT__1.jpg
SCENE__NON_PRODUCE__1.jpg
```

Two underscores between parts. Condition names exactly as spelled above.

Put them all in:

```
competition/data/self_captured/raw/
```

This folder is already ignored by Git — nothing is published by copying files
there.

### Step 6 — Build the metadata

```bash
.venv-competition/bin/python -m competition.evaluation.prepare_capture_manifest
```

Reads the folder, hashes every image, fills in the split automatically.

### Step 7 — Validate

```bash
.venv-competition/bin/python -m competition.evaluation.validate_capture_set
```

Read the output. Fix anything marked `[ERROR]`. Re-run until errors reach zero.

**About the EXIF warnings:** they are **fine here**. `self_captured/raw/` is
gitignored, nothing leaves your machine, and the device metadata is genuine
provenance for the calibration set — do not strip it from the raw files.

They stop being fine the moment an image is **distributed**. Before any photo is
committed to the repository, uploaded to Devpost, served by the demo endpoint,
shown in the judge video, or shared publicly, it must go through the
sanitisation workflow into `self_captured/sanitised/`. The raw captures are
never overwritten.

### Step 8 — How you know you are finished

- [ ] 24 fruits labelled `APL/BAN/ORG-001..008`
- [ ] 6 photos for every item (**144**)
- [ ] 6 extra photos for APL-002, APL-003, BAN-004, BAN-007, ORG-006, ORG-007 (**36**)
- [ ] 6 scene photos (**6**)
- [ ] **186 files in total**
- [ ] All files in `competition/data/self_captured/raw/`
- [ ] `prepare_capture_manifest` runs without complaint
- [ ] `validate_capture_set` reports **0 errors**

Then tell me it is done, and calibration can begin.

**One thing to avoid:** do not delete and re-photograph an item after seeing any
metric output. That would make the split meaningless.
