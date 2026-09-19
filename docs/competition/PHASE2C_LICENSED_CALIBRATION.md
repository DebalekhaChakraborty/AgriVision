# Phase 2c-B — Licensed Real-Image Calibration and Controlled Degradation

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Base commit | `418934d5e05b68fcecac3aebf5260d824a6087fc` (Phase 2c-A) |
| **Status** | **CALIBRATED — policy frozen, held-out opened once** |
| Calibration corpus | 92 licence-verified real photographs, 86 source groups |
| Sources used | Wikimedia Commons only, 8 distinct permissive licences |
| Sources rejected | Kaggle research split, FruitVision, Fruits-360 |
| Controlled degradations | 20-rung ladder, 1,840 derived samples |
| Focus metric selected | `high_frequency_ratio` |
| Focus metric **rejected** | `laplacian_variance` — moves **88×** with exposure |
| Tests | **548 passing** (413 prior, unchanged; 135 added) |
| Consistency audit | contrast path traced end to end; §12 corrected |

> **What was calibrated, and on what.** Thresholds were calibrated on *licensed
> real fruit photographs with controlled degradations*. They were **not**
> calibrated on phone-camera deployment captures, and nothing here supports the
> phrase "real-world validated". A Gaussian blur is not a defocused lens and a
> multiplicative gain is not an underexposed sensor.

---

## 1. What changed, and why it was allowed to change

Phase 2c-A ended with the pipeline blocked on 186 photographs that did not
exist. The blocker was real but its premise was too narrow: it assumed the only
licence-safe imagery was imagery we took ourselves.

That is not true. Wikimedia Commons publishes the licence, the creator and the
attribution requirement **per file**, through an API. That makes a per-image
licence gate possible, which is the thing a packaged dataset can never offer —
a zip file carries one licence claim for everything inside it, and that claim
cannot be verified image by image.

Self-capture is not abandoned. It is
[reclassified](PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md) as an optional
real-device validation at one tenth the scale, because it remains the only thing
that can demonstrate camera-domain transfer.

---

## 2. The licence gate

`competition/data/licence_validation.py` is the only way an image enters the
corpus, and it runs on metadata **before any bytes are fetched**. Downloading
first and filtering later would mean holding bytes we had established no right
to hold.

Four rules, all default-deny:

1. **Unrecognised is rejected.** "I cannot identify this licence" and "this
   licence is forbidden" have the same effect. Absence of a notice is not
   permission.
2. **NonCommercial and NoDerivatives are rejected**, named separately so a
   rejection says which restriction applied. Every admitted licence permits both
   commercial use and adaptation, because the corpus is publicly submitted and
   every image is adapted.
3. **Fields must agree.** Short name, prose terms and deed URL are resolved
   independently. Disagreement is a fact about metadata quality and rejects.
4. **Subject-level restrictions override a permissive photo licence** —
   trademark, personality rights, and similar.

### One correction the gate needed

The first run rejected 64 files as `REJECTED_CONFLICTING`, all with the same
signature: short name `CC0`, deed URL the CC0 deed, prose "Creative Commons
Zero, Public Domain Dedication". Read field by field that resolves to two ids;
read as rights it is one grant stated twice.

Treating that as a conflict discarded the *most* permissively licensed images in
the pool for a disagreement that does not exist. The fix is a narrow equivalence
group, `{CC0-1.0, public-domain}`, which `_assert_equivalence_groups_are_sound`
verifies at import time grants identical commercial, derivative, attribution and
share-alike terms. Resolution picks the specific instrument so the recorded
licence stays checkable, and any disagreement crossing a group boundary still
rejects — `CC-BY-SA-4.0` against `CC-BY-4.0` is still refused, because
share-alike is a real difference.

This is a case where the honest move was to fix the gate rather than accept its
output. Nine CC0 images and one public-domain image are in the corpus as a
result.

---

## 3. Sources investigated

### Used

**Wikimedia Commons.** Per-file licence, creator and attribution through the
API; 16 curated categories; 8 distinct permissive licences in the final corpus.

### Rejected on licence

| Source | Licence | Why rejected |
| --- | --- | --- |
| Kaggle research split | Unknown | No redistribution grant and no identifiable rights holder. Excluded by the phase brief and by the gate. |
| FruitVision | CC BY-NC-ND 4.0 | NonCommercial *and* NoDerivatives. Every image here is adapted; both clauses are fatal. |
| Sultana 2022 | CC BY 4.0 | Licence is fine. Excluded on domain: curated studio product photography, the same population Phase 2 already showed does not transfer. |

### Rejected on evidence — Fruits-360

Fruits-360 is the obvious candidate and it was investigated properly. It was
**not** rejected on licence, and the reason it was rejected matters more than
the licence question.

**Licence, and a conflict.** The legacy repository `Horea94/Fruit-Images-Dataset`
declares **MIT** (2017–2020). The current upstream organisation `fruits-360`
declares **CC BY-SA 4.0** (2017–, Mihai Oltean) across every image repository,
at version 2026.9.03.0. Two grants from the same rights holder, and GitHub
reports the current ones as `NOASSERTION` because the LICENSE file is prose
rather than a recognised template. The conservative reading — the one this phase
would have applied — is the current upstream's CC BY-SA 4.0, the more
restrictive of the two.

**The disqualifying finding is structural.** The dataset's own published split
puts the same physical object on both sides of it:

```
Training/Apple 10   465 frames   (r0_0, r0_10, r0_100, r0_102, r0_104, …)
Test/Apple 10       231 frames   (r0_103, r0_107, r0_11, r0_111, r0_115, …)
```

"Apple 10" is one apple, filmed rotating on a motor for 20 seconds by one
webcam. Its frames are interleaved by index across Training, Validation and Test
by design (`k`, `k+1`, `k+2`, `k+3`). Any model evaluated on that split is
tested on near-duplicate frames of the apple it trained on. This is exactly the
leakage the phase brief's §5 warns about, and it is present in the dataset's
official partition — so the split cannot be reused, and reconstructing a correct
one is possible but leaves the remaining problems untouched.

**Which are, independently, fatal for this purpose:**

- The 100×100 branch has its background **algorithmically removed and replaced
  with white**. It is a pre-segmented cut-out, not a photograph of a scene.
  Calibrating a capture-quality gate on it — when the gate's entire Phase 2
  failure was background-driven highlight clipping — would be circular.
- One Logitech C920, one white paper backdrop, one lighting setup, one
  operator. Calibrating exposure thresholds on that would encode one webcam's
  response curve.
- 100×100 px, against a corpus that needs resolution variation to detect the
  resolution confound Phase 1 identified.

Fruits-360 is a good dataset for the classification task it was built for. It is
the wrong instrument for calibrating a capture-quality gate, and padding the
corpus with it would have bought a bigger number at the cost of a worse answer.

---

## 4. Licence provenance of the corpus actually used

Every image carries its own licence, creator and attribution string. Nothing
inherits a dataset-level claim.

| Licence | Images | Source groups | Commercial | Derivatives | Attribution | Share-alike |
| --- | ---: | ---: | --- | --- | --- | --- |
| CC-BY-SA-4.0 | 37 | 35 | yes | yes | **required** | **required** |
| CC-BY-SA-3.0 | 20 | 18 | yes | yes | **required** | **required** |
| CC-BY-2.0 | 15 | 14 | yes | yes | **required** | no |
| CC0-1.0 | 9 | 8 | yes | yes | no | no |
| CC-BY-SA-2.0 | 5 | 5 | yes | yes | **required** | **required** |
| CC-BY-3.0 | 4 | 4 | yes | yes | **required** | no |
| CC-BY-4.0 | 1 | 1 | yes | yes | **required** | no |
| public-domain | 1 | 1 | yes | yes | no | no |
| **Total** | **92** | **86** | | | | |

Source: Wikimedia Commons, verified per file through the MediaWiki
`imageinfo`/`extmetadata` API on 2026-09-18. Evidence for each image — deed URL,
creator, attribution text, content hash — is in
`competition/data/licensed_real/manifests/licensed_corpus.json`.

**62 of 92 images carry share-alike.** No adapted image is redistributed in this
phase: derived samples are regenerated from a base, a transformation and a seed
rather than stored, and no image bytes are committed. Any later public use of an
adapted image inherits CC BY-SA and must say so.

---

## 5. Curation — why category membership was not enough

401 candidates passed the licence gate. Commons categories organise by subject
matter, not photographic composition, so the pools contained a **wax model of an
apple**, a **car parade**, an apricot filed under a fruit market, pumpkins and
physalis under "Orange fruit", and a great many apple *trees*.

Every one of the 401 was rendered as a numbered thumbnail across 12 contact
sheets and reviewed by eye. Both the fruit and the track recorded in
`manifests/curation.json` are that review's judgement and override the
category's proposal.

**Rejected:** fruit on the tree, blossom, orchards; cut, peeled or cooked;
a different species; illustrations, paintings, renders, postcards, diagrams,
packaging; monochrome, which colour-based foreground isolation cannot fairly be
judged on; people or objects dominant with the fruit incidental.

Track A additionally required the image to be **in focus and legibly exposed**,
because undegraded Track A images carry a scored ACCEPTABLE label. Track B was
curated for subject identifiability only and carries no scored label — those
images are deliberately hard.

---

## 6. Corpus composition

| | Apple | Banana | Orange | Total |
| --- | ---: | ---: | ---: | ---: |
| **Track A** — clean base | 20 | 20 | **10** | **50** |
| **Track B** — natural scene | 14 | 14 | 14 | **42** |
| Total | 34 | 34 | 24 | **92** |

**Orange Track A fell short of 20 and was not padded.** The permissively
licensed Commons orange categories are dominated by trees, blossom, cut
cross-sections, and other citrus filed as oranges — mandarins, tangerines,
kumquats, limes. Ten images survived a criterion that required a whole sweet
orange, in focus, as the dominant subject. Relabelling mandarins as oranges
would have reached the target by corrupting the label.

- 86 distinct creators for 92 images — at most 2 images per creator, by design.
- Short edge 640–1,920 px, median 1,360.
- Retrieved as width-bounded renditions (Commons serves a fixed ladder; 1920,
  1280 and 500 are permitted, arbitrary widths return HTTP 400). This is lighter
  on Commons than fetching originals and bounds the resolution confound: the
  corpus spans 640–1,920 px, and resolution effects are measured over that range
  and no wider.

---

## 7. Source-group split

| | Groups | Images | Share |
| --- | ---: | ---: | ---: |
| Calibration | — | 65 | 70.7% |
| Held-out | — | 27 | 29.3% |
| Total | 86 | 92 | |

Seed `20260918`, fingerprint **`77a995ceff0d33f7`**, stratified across all six
fruit × track strata.

**The unit is the source group, never the image.** A source group is the creator
identity — the conservative choice, because two photographs by the same
contributor in the same category are frequently two angles of the same physical
apple and no metadata field distinguishes that from two unrelated shoots.

The split was generated from provenance metadata alone, before any quality
metric was computed on the corpus, and it carries a fingerprint so a later edit
is detectable.

**Access control, not convention.** `CorpusView` raises if a calibration view is
handed a held-out record, and refuses to construct a held-out view without a
frozen policy fingerprint. A comment saying "do not look" is not a control.

---

## 8. Controlled degradation

A 20-rung ladder over seven families, applied to every base image: Gaussian
blur, underexposure, overexposure, contrast reduction, glare, occlusion and
subject-scale reduction. 1,300 calibration samples and 540 held-out.

Expected verdicts were fixed from **transformation severity alone**, before any
metric was computed:

| Verdict | Rungs | Scored? |
| --- | --- | --- |
| `UNUSABLE` | blur σ≥4, gain ≤0.12 or ≥2.6, contrast ≤0.2, glare 0.9, occlusion 0.5 | yes |
| `ACCEPTABLE` | undegraded Track A | yes |
| `BORDERLINE` | blur σ 1–2, gain 0.25–0.6 and 1.6, contrast 0.5, glare 0.5, occlusion 0.2 | **reported, never scored** |
| `UNLABELLED` | subject-scale rungs; all undegraded Track B | no |

Rungs where a defensible person would disagree are `BORDERLINE` and excluded
from the confusion matrix rather than forced to one side to tidy the numbers. A
rubric adjusted after seeing the metrics is not a rubric.

Subject scale is deliberately `UNLABELLED`: a small subject is not a defective
capture, it is a harder one, and labelling it either way would smuggle a
judgement into the rubric. It exists to measure the confound, not to be scored.

Derived bytes are **not stored**. 1,840 losslessly encoded variants of 1920 px
photographs is several gigabytes fully determined by a base image, a
transformation, a level and a seed. Each derived sample is regenerated on
demand, and the manifest pins its `derived_sha256` so a regenerated sample is
checked rather than trusted — a mismatch raises instead of measuring silently.

---

## 9. Foreground re-evaluation — a resolution-dependent bug in Phase 2b

Running the Phase 2b foreground method over full-resolution photographs
immediately produced a wall: **966 of 1,365 samples had no usable mask**, and
the dominant reason was `MASK_FRAGMENTED`, with median component counts of 19 on
plain-background images and 69 on scenes, against a guard of 12.

The cause was a fixed constant. `_clean_mask` used a **7-pixel** morphological
kernel regardless of image size. That is 2.7% of a 256 px fixture — the
population Phase 2b was developed against — and 0.36% of a 1920 px photograph.
At full resolution the opening step removed almost nothing, and thresholding
speckle survived as hundreds of tiny components. The fragmentation guard was
functioning as a resolution detector.

This is the same class of error as Phase 1's exposure coupling and Phase 2b's
scope mismatch: **a constant calibrated on one population applied to another.**

Two scale-relative corrections:

- `CLEANUP_KERNEL_FRACTION = 7/256`, so the kernel is 7 px at 256 px — Phase 2b's
  own value expressed relatively, not a new number — and 39 px at 1920 px.
- `min_component_area_fraction = 0.001`: components below 0.1% of the frame are
  thresholding speckle, not pieces of a shattered subject. They are excluded from
  the *count* but not from the foreground area, so "fragmented" keeps meaning
  "the subject broke apart".

All 44 Phase 2b foreground tests pass unchanged.

| Track | Valid masks before | Valid masks after |
| --- | ---: | ---: |
| Clean base (n=35) | 42.9% | **85.7%** |
| Natural scene (n=30) | 6.7% | **30.0%** |
| `MASK_FRAGMENTED` (all) | 44 | **2** |

**The residual failure on Track B is correct behaviour, not a remaining bug.**
The reasons are now `EXCESSIVE_BORDER_CONTACT` (18) and
`FOREGROUND_NEAR_FULL_FRAME` (11), with median foreground fraction 0.82 and
median border contact 0.69. Those are photographs of market stalls and fruit
bowls — there *is* no single subject to isolate. A method that returned a
confident mask for a crate of oranges would be worse, not better. The honest
reading is that single-subject isolation works on single-subject captures and
correctly declines scenes, and that multi-object handling is a separate
capability that does not exist yet.

---

## 10. Sharpness — the question this phase existed to answer

Phase 1 and Phase 2b both left variance-of-Laplacian under suspicion. Four
candidate focus metrics were measured over the same pixel selection, against a
rule fixed in advance:

- separation AUC ≥ 0.95
- a threshold holding false accepts ≤ 0.10, minimising false blocks
- median value moving < **1.5×** across the exposure ladder on the same images
- median value moving < **3.0×** across the subject-scale ladder

The exposure criterion encodes the Phase 1 finding as a *disqualifying property*.
It is allowed to reject a metric that separates the classes perfectly.

| Metric | AUC | Threshold | FA | FB | Underexpose | Overexpose | Scale | Admitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `laplacian_variance` | **1.000** | 2.274 | 0.097 | 0.000 | **88.3×** | 3.42× | 2.92× | **no** |
| `tenengrad` | 0.955 | 494.9 | 0.097 | 0.067 | **212.4×** | 2.84× | 3.86× | **no** |
| `normalised_gradient_energy` | 0.969 | 0.299 | 0.097 | 0.033 | 1.52× | 2.43× | 2.55× | **no** |
| `high_frequency_ratio` | 0.999 | **0.318** | **0.097** | **0.000** | **1.13×** | **1.10×** | **1.03×** | **yes** |

### What this says

**Variance of the Laplacian sorts blurred from sharp perfectly — AUC 1.000 — and
is still unusable as a global threshold.** Its median moves **88-fold** across
the underexposure ladder on the *same photographs*. A threshold set on
well-exposed images would reject sharp photographs taken in a dim kitchen, which
is precisely the failure Phase 2 observed and Phase 2b could not fix by changing
the number. It is an excellent *relative* focus measure within one image and a
meaningless *absolute* one across images. **Stop tuning it as a global
threshold.** It is explicitly retained for **within-image comparison** — the
Phase 1 degradation sweeps, and any future before/after check on a single
capture — where the exposure and scale terms are constant and therefore cancel.

Had the rule been "maximise accuracy", Laplacian variance would have won
outright. That is why the invariance criterion was written down first.

**Tenengrad is worse** — 212× — which is expected: it is raw squared gradient
magnitude, so it scales as g² with no normalisation at all.

**Normalised gradient energy did most of what it was designed to do.** Dividing
gradient energy by intensity variance cancels a multiplicative exposure change
algebraically, and on synthetic fixtures it holds to 0.1%. On real imagery it
moves 1.52× and 2.43× — just outside the limit. The residual is explained, and
it is not a failure of the construction: `convertScaleAbs` **saturates**, so the
overexposure rungs clip highlights and destroy gradient energy and intensity
variance at different rates, and the deepest underexposure rungs quantise to few
levels. The invariance is exact for pure scaling and approximate for a transform
that also clips. Worth revisiting on a ladder that models exposure without
clipping.

**The high-frequency ratio wins on every criterion**, and by construction rather
than by luck: it is a ratio of spectral energy above a radial cutoff to total
spectral energy, so a uniform gain cancels in numerator and denominator, and the
cutoff is defined on a *normalised* frequency grid, so it does not move with
resolution. 1.13× / 1.10× / 1.03× is invariance, not tuning.

Its honest caveat is in the implementation: a Fourier transform needs a
contiguous rectangle, so it is computed on the mask's bounding box, Hann-windowed
in both axes to stop the rectangle's own edges leaking a cross of spurious
high-frequency energy. Whenever the subject is not box-shaped it sees some
background. That is recorded, not hidden.

**So a defensible global focus threshold does exist — just not on the metric
three phases had been trying to fix.**

---

## 11. The locked policy

`competition/evaluation/results/phase2c_locked_policy.json`

| Threshold | Value | Status |
| --- | --- | --- |
| `focus_metric` | `high_frequency_ratio` | **calibrated** |
| `focus_floor` | 0.3182 | **calibrated** |
| `underexposed_mean_luminance` | 0.0527 | **calibrated** (AUC 1.000, FB 0.000) |
| `low_contrast_limit` | 0.1314 | **calibrated** (AUC 1.000, FB 0.000) |
| `overexposed_mean_luminance` | 0.80 | *provisional* — AUC 0.943 < 0.95 |
| `shadow_clip_fraction_limit` | 0.05 | *provisional* — AUC 0.924, best FB 0.133 |
| `highlight_clip_fraction_limit` | 0.05 | *provisional* — AUC 0.935, best FB 0.200 |

**Status: `PARTIALLY_CALIBRATED_SOME_THRESHOLDS_PROVISIONAL`.** Four of seven
thresholds are evidence-backed; three retain their Phase 1 values because no
candidate threshold met the rule. Calling the whole policy "calibrated" would be
the whole-image substitution Phase 2b exists to warn against, wearing a better
label. A threshold nobody could justify is better left visibly unjustified than
given a figure with a decimal point on it.

### Why the clipping thresholds could not be calibrated — and what that confirms

| Statistic | Whole image | ROI | Ratio |
| --- | ---: | ---: | ---: |
| Median highlight-clipped fraction | 3.34 × 10⁻⁴ | **1.51 × 10⁻⁵** | 22× lower |
| Median shadow-clipped fraction | 8.02 × 10⁻⁴ | 1.32 × 10⁻³ | — |
| Median Laplacian variance | 179.8 | 175.0 | ~equal |

Once the background is excluded, **highlight clipping on the subject is 22×
rarer**, and at the Phase 1 limit of 0.05 the flag effectively never fires. This
is a direct, quantified confirmation of the Phase 2b diagnosis: in Phase 2,
highlight clipping fired on 80% of research photographs because the measurement
was reading the backdrop. The metric cannot separate the classes here because,
restricted to the fruit, there is almost nothing to separate.

The median Laplacian figures are close because this corpus is natural
photography with textured backgrounds, not the bright-backdrop product
photography where Phase 2b measured ROI variance at ~5% of whole-image. The 5%
figure was a property of that population, and this is what it looks like on
another — which is the same lesson again.

---

## 12. Held-out evaluation — opened once, ledgered

Policy threshold fingerprint **`78b1e2151773787a`**, lock fingerprint
`15b7fb0908d6d622`, status **CONFIRMATORY**. 27 images across 25 source groups,
plus 540 controlled degradations of them.

The held-out view cannot be constructed without a frozen policy fingerprint, and
every run is appended to `phase2c_heldout_ledger.json`. A run under *different*
thresholds after held-out has been opened is automatically labelled EXPLORATORY
and cannot be reported as a first look. The thresholds never changed, so both
recorded runs carry the same fingerprint and the same status.

### Undegraded real photographs

| | Value |
| --- | ---: |
| Scored acceptable (Track A, assessable) | 10 |
| True accepts | **10** |
| **False blocks** | **0** (0.0%) |
| Unassessable — no valid mask | 10 |
| Track B accepted (unlabelled) | 7 |

**No competent photograph in the held-out set was rejected.** That is the number
Phase 2 could not achieve: the gate rejected 96.7% of research photographs, and
Phase 2b improved it only to 83.3%.

### Controlled degradations

| | Value |
| --- | ---: |
| Scored unusable | 180 |
| True blocks | 141 |
| **False accepts** | **39 (21.7%)** |
| Unassessable | 174 |

### Where the false accepts come from — three different mechanisms

| Family | Eligible | Escalated | Remediation | No mask |
| --- | ---: | ---: | ---: | ---: |
| `gaussian_blur` | 15 | **59** | 0 | 34 |
| `underexpose` | 26 | **39** | 2 | 41 |
| `overexpose` | 14 | **36** | 1 | 30 |
| `glare` | **37** | **0** | 0 | 17 |
| `reduce_contrast` | **63** | **0** | 0 | 18 |
| `occlude` | **24** | 1 | 0 | 29 |
| `shrink_subject` | 47 | 2 | 0 | 5 |

Blur and exposure are caught. Glare, contrast and occlusion are not — but **not
for the same reason**, and an earlier draft of this note wrongly described all
three as "missing detectors". The distinction was resolved by tracing each
family from transform to gate outcome on **calibration** groups
(`competition/evaluation/audit_contrast_path.py`; the held-out artifacts are
confirmatory and frozen, so no mechanism was re-derived from them):

| Family (UNUSABLE rungs) | Scored | Raised a flag | **Blocked** | Mechanism |
| --- | ---: | ---: | ---: | --- |
| `reduce_contrast` | 58 | **53 (91%)** | **0 (0%)** | detected, deliberately not blocking |
| `glare` | 30 | 3 | 3 (10%) | **no detector** |
| `occlude` | 20 | 6 | 5 (25%) | **no detector**; blocked only incidentally |

#### Contrast — the detector works; the policy chooses not to block

The contrast metric responds cleanly and the calibrated threshold of 0.1314 is
correct:

| `reduce_contrast` factor | Median ROI contrast | `LOW_CONTRAST` raised | Blocked |
| ---: | ---: | ---: | ---: |
| 1.0 (undegraded) | 0.5608 | 0 / 12 | 0 |
| 0.5 *(borderline)* | 0.2431 | 0 / 10 | 0 |
| 0.2 *(unusable)* | 0.0941 | **9 / 11** | 0 |
| 0.08 *(unusable)* | 0.0392 | **9 / 9** | 0 |

The threshold separates the classes exactly as calibrated — no false positives on
undegraded images, and it fires on the unusable rungs. It produces **zero
escalations** because `LOW_CONTRAST` sits in `ADVISORY_FLAGS` rather than
`DEFAULT_BLOCKING_FLAGS`, a Phase 1b decision taken on the stated grounds that
low contrast "degrades the signal without removing it" and is the most common
residual flag after tone remediation.

So the gate **knew** these captures were low-contrast and passed them anyway, by
design. That is a defensible policy position and it was not a defect — but
counting those samples as "false accepts" alongside genuinely undetected glare
conflates a policy choice with a blind spot. Whether the advisory status is
still justified now that a calibrated ROI threshold exists is a **policy
question for Phase 2d**, not a threshold to move here, and it was deliberately
not changed after held-out results were known.

#### Glare and occlusion — genuinely undetected

- **Glare.** A specular hotspot is a *local* highlight. ROI clipping is a
  *global* fraction over the subject region, and a patch covering a few percent
  of the fruit never approaches any sensible fraction limit. The 10% that were
  blocked reached the fraction limit by accident of size, not by detection.
  Detecting glare needs a local statistic, which does not exist.
- **Occlusion.** Nothing in the pipeline looks for an occluder. The escalations
  came from the occluder happening to break the mask geometry.

Moving any existing threshold far enough to catch glare would reject correctly
exposed photographs, which is why the preregistered rule refused those
thresholds in the first place.

### Gate behaviour

| Measure | Value |
| --- | ---: |
| Assessable samples | 383 of 567 |
| Eligibility rate | **63.4%** |
| Escalation rate | 35.8% |
| Segmentation unavailable | 184 (32.5%) |
| Remediation attempt rate | 0.8% (3 samples) |
| Remediation accepted | 2 of 3 |
| Max automated attempts | 1 |

The remediation rate is low **by design**, not by failure. Only flags a single
tone correction can genuinely clear — `UNDEREXPOSED` and `OVEREXPOSED` — route
to remediation. Blur, clipping and insufficient resolution escalate, because the
detail is gone and no enhancement recovers it.

That distinction was worth enforcing. During development a deeply underexposed
sample (ROI mean luminance 0.027, flagged `UNDEREXPOSED`, `LOW_CONTRAST` **and**
`SHADOW_CLIPPING`) had every flag cleared by a single gamma of 0.25, which lifted
mean luminance to 0.419. **Gamma did not un-clip anything** — it moved already
crushed pixels above the L\* threshold the flag is defined on. Had clipping been
treated as remediable, the system would have "fixed" an unrecoverable capture
cosmetically and passed it to the model. Clipping escalates for this reason.

---

## 13. ROI versus whole-image gating — an uncomfortable result, reported as measured

| | ROI | Whole image |
| --- | ---: | ---: |
| Assessable | 383 | 567 |
| Scored unusable | 180 | 270 |
| **False accepts** | 39 (**21.7%**) | 20 (**7.4%**) |
| **False blocks** | 0 (**0.0%**) | 1 (**6.7%**) |

**On this corpus, whole-image gating has the lower false-accept rate.** That is
the opposite of what Phase 2b's motivation would predict, and it is reported
because it was measured.

The explanation is in how the two corpora differ, and it does not overturn Phase
2b:

1. **The degradations were applied globally.** Glare, occlusion and contrast
   collapse were applied to the whole frame, so whole-image statistics respond to
   them. Restricting measurement to the fruit deliberately discards the
   background — including the part of the perturbation that lives there. A
   deployment capture would have glare on the *subject*, not uniformly across the
   frame.
2. **This corpus has no blown backdrops.** Phase 2b's finding came from bright
   product photography where whole-image clipping fired on 80% of images for the
   wrong reason. Commons photographs have natural, textured backgrounds — so
   whole-image gating is not being sabotaged here, and its false-block rate of
   6.7% against ROI's 0.0% is the residue of that effect, not its absence.
3. **ROI gating declines 32.5% of samples** rather than answering wrongly. Those
   are not silent passes; they escalate.

The honest summary: **ROI gating is more conservative and more precise about
what it will judge; whole-image gating answers more often and is more sensitive
to whole-frame perturbations.** Which is preferable depends on whether the
degradation lives on the subject or in the frame, and that question is settled
by deployment captures, not by this corpus. Neither scope is declared the winner
here.

---

## 14. Condition model on eligible real images

The condition model predicts a joint (fruit, condition) label, and exactly one
half of that has ground truth on this corpus.

| | Value |
| --- | ---: |
| Eligible (passed the gate) | 17 |
| Blocked by the gate | 10 |
| Model failures | 0 |
| **Fruit-type accuracy** | **88.2%** (15/17) |
| Visible-condition accuracy | **not computed** |
| Confidence — median | 0.416 |
| Confidence — p10 / p90 | 0.275 / 0.648 |

**Fruit type is scored** because every image was identified by eye during
curation, before any model ran; that identification is why it is in the corpus.
Two errors: one apple read as an orange, one banana read as an apple.

**Visible condition is not scored, and no figure is offered.** Commons
contributors do not record whether the fruit they photographed was fresh or
deteriorating. Labelling them by eye now, or assuming market produce is fresh,
would convert an unlabelled set into a fabricated benchmark. The distribution
(10 `fresh`, 7 `deteriorated`) is reported as output, not as performance.

**The confidence distribution is the interesting number.** A median of 0.416
across six classes is low for a model that reported high confidence on its own
research test set. These are natural photographs on varied backgrounds, and the
model was trained on curated studio product photography. This is a **domain-shift
signal**, and it is worth stating plainly: the gate can pass an image that the
condition model is nonetheless poorly equipped to judge. Capture quality and
inference reliability are separate problems, and solving the first does not
solve the second.

---

## 15. Synthetic stress tier — exploratory only

12 procedurally composed scenes, drawn with OpenCV primitives. No
image-generation model was used anywhere in this phase, and these scenes contain
no fruit. They are stored under a separate provenance type, evaluated into a
separate results file, and excluded from threshold calibration and held-out
evaluation.

| Stress case | Mask valid | Gate outcome |
| --- | :---: | --- |
| `plain_background` | yes | ELIGIBLE |
| `busy_background` | yes | ELIGIBLE |
| `gradient_background` | yes | ELIGIBLE |
| `low_contrast_scene` | yes | ELIGIBLE |
| `subject_touching_borders` | yes | ELIGIBLE |
| `directional_hotspot` | yes | **ESCALATED** — `HIGHLIGHT_CLIPPING` |
| `tiny_subject` | no | refused — foreground 0.3% |
| `subject_fills_frame` | no | refused — foreground 100% |
| `fragmented_subject` | no | refused |
| `heavy_occlusion` | no | refused — occluder split the mask |
| `three_equal_subjects` | **yes** | **ELIGIBLE** |
| `same_colour_distractor` | **yes** | **ELIGIBLE** |

**The last two rows are a real limitation, found by probing for it.** Scenes with
three equally sized subjects and with a same-coloured distractor produced masks
the geometric guards accepted. On a cluttered background, saturation-Otsu
segments coloured background regions too, and the dominance guard is then
satisfied by a large background blob rather than by a subject.

This is the boundary of what the guards can do: they detect **geometric**
failure — empty, full-frame, fragmented, border-hugging — and they cannot detect
a mask that is well-formed and *wrong*, because there is no ground truth to
compare against. `ForegroundEvidence.valid` means "this mask is not obviously
broken", never "this mask is correct". Nothing in this repository should be read
as claiming otherwise.

Note also that `directional_hotspot` was caught here while real glare was not
(§12). The synthetic hotspot is large and blown enough to move a global fraction;
a real specular highlight is smaller. That difference is exactly why synthetic
results are kept separate: they can make a detector look better than it is.

---

## 16. Claim boundary

**Supportable:**

- "Capture-quality thresholds were calibrated on licensed real fruit imagery
  with controlled degradations."
- "Foreground behaviour was evaluated on licensed natural-background images."
- "Generated images were used only as supplemental stress tests."
- "On held-out licensed photographs, the gate rejected none of the competent
  captures it could assess."
- "Fruit type was identified correctly on 88.2% of eligible held-out images."

**Not supportable, and not claimed anywhere:**

- "Calibrated on real phone-camera deployment data." It was not.
- "Real-world validated." No deployment-domain imagery exists yet.
- Any visible-condition accuracy figure on this corpus.
- Any claim that the system detects glare, occlusion, pathogens, toxins,
  microbiological contamination, internal spoilage, food safety or edibility.
- Any claim that a valid foreground mask is a correct one.

---

## 17. Demo image shortlist — metadata only

`competition/data/licensed_real/manifests/demo_shortlist.json` records 12
candidates, two per fruit-and-track bucket, **ranked by how few obligations the
licence carries**: public-domain and CC0 first, then attribution-only, then
share-alike last. Eight of the twelve are CC0 or public domain and carry no
obligation at all.

For each entry the shortlist records the source, creator, licence, attribution
string, whether modification is permitted, whether public display is permitted,
and why the image is useful for a demo.

**No demo has been built and no image has been used.** Two cautions are recorded
with the list:

- Showing a *degraded* variant of a share-alike image in a video is an
  adaptation, and obliges that adaptation to be offered under CC BY-SA. That is
  a commitment to make deliberately, not by reaching for a convenient
  photograph.
- Held-out images may be shown in a demo, but must never be used to justify a
  threshold.

Self-captured imagery remains preferable for the final live demo, for the reason
[DEMO_DATA_PLAN.md](DEMO_DATA_PLAN.md) was written: it is the deployment domain
and it carries no attribution slate.

---

## 18. What is still open

1. **Glare and occlusion have no detector.** Needs local highlight statistics
   and a visibility measure, not a threshold change.
2. **`LOW_CONTRAST` is advisory and never blocks**, even though its ROI
   threshold is now calibrated and fires on 91% of unusable contrast rungs.
   That advisory status was chosen in Phase 1b before a calibrated ROI threshold
   existed; whether it is still the right call is a policy question, deliberately
   left open rather than settled after held-out results were seen.
3. **Three ROI thresholds remain provisional.** Overexposure, shadow clipping
   and highlight clipping could not be calibrated on this corpus because,
   restricted to the subject, the classes barely separate.
4. **Segmentation declines a third of samples.** Correct behaviour for
   multi-subject scenes, but multi-object handling is a capability that does not
   exist.
5. **Normalised gradient energy deserves a second look** on an exposure ladder
   that does not clip, where its algebraic invariance should hold.
6. **Camera-domain transfer is unmeasured.** The optional 3–6 fruit validation
   in [PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md](PHASE2C_CAPTURE_CALIBRATION_PROTOCOL.md)
   is the only thing that closes it.
7. **The condition model shows domain shift** — median confidence 0.416. Gate
   quality and inference reliability are separate problems.
