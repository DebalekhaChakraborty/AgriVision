# Phase 2d — Local Capture-Artefact Evidence

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Base commit | `6b62bb544b9fdd78ad2d8ab2aac43936f238e335` (Phase 2c-B) |
| **Status** | **UNCOMMITTED** |
| Artefacts addressed | local glare · contrast loss · subject visibility |
| Validation pool | 29 licensed real images, 29 unseen source groups, fingerprint `e9e7ace82c40783e` |

> **Three different problems.** Glare is a *local photometric* artefact,
> contrast loss is a *global tonal* one, and visibility insufficiency is a
> *coverage* problem. They are measured separately, calibrated separately and
> reported separately. There is no combined "artefact accuracy" figure, because
> averaging a detector that works with one that does not would hide both.

---

## 1. Why this phase exists

Phase 2c-B's held-out run showed a 21.7% false-accept rate on controlled
degradations, and the consistency audit that opened this phase found it was not
one phenomenon:

| Family (unusable rungs) | Scored | Flagged | Blocked | Mechanism |
| --- | ---: | ---: | ---: | --- |
| `reduce_contrast` | 58 | 53 (91%) | **0 (0%)** | detected, advisory by design |
| `glare` | 30 | 3 | 3 (10%) | **no detector** |
| `occlude` | 20 | 6 | 5 (25%) | **no detector**, blocked incidentally |

So Phase 2d has two jobs of different kinds: build detectors that genuinely do
not exist (glare, visibility), and decide a *policy* question that was already
answerable (contrast).

---

## 2. The contrast consistency audit, and what it changed

**Root cause: the detector works and the policy chooses not to block.** Traced
end to end on calibration groups
(`competition/evaluation/audit_contrast_path.py`):

| `reduce_contrast` factor | Median ROI contrast | `LOW_CONTRAST` raised | Blocked |
| ---: | ---: | ---: | ---: |
| 1.0 | 0.5608 | 0 / 12 | 0 |
| 0.5 *(borderline)* | 0.2431 | 0 / 10 | 0 |
| 0.2 *(unusable)* | 0.0941 | **9 / 11** | 0 |
| 0.08 *(unusable)* | 0.0392 | **9 / 9** | 0 |

The metric responds, the calibrated threshold of 0.1314 fires with no false
positives on undegraded images, and nothing escalates because `LOW_CONTRAST` is
in `ADVISORY_FLAGS`, not `DEFAULT_BLOCKING_FLAGS`. That was a Phase 1b decision
taken on stated grounds — low contrast "degrades the signal without removing
it", and it is the most common residual flag after tone remediation.

**Which of the five candidate explanations is true:** the flag is advisory-only
(1), and the Phase 2c-B report wording was inaccurate (5). Not a wiring fault
(2) — the threshold reaches the flag correctly. Not an unresponsive metric (3),
except at the 0.5 rung where not firing is the correct answer. No other defect
(4).

**What changed as a result:** the Phase 2c-B documentation wording, and a
two-tier contrast ladder here. Advisory was right for *mild* loss and wrong for
*collapse*; a single threshold could not express both. No Phase 2c-B threshold
was altered, and the confirmatory held-out artifacts were not regenerated.

---

## 3. The independent validation pool

The Phase 2c-B held-out groups are spent. They were opened confirmatorily under
policy `78b1e2151773787a`, and Phase 2d changes the detector set, so any re-run
against them is exploratory by construction.

`competition/evaluation/build_validation_pool.py` builds a separate pool with
three independence constraints, all enforced in code:

1. **No shared source group.** Every creator in the Phase 2c-B corpus is
   excluded, so no contributor appears on both sides.
2. **No previously screened file.** The Phase 2c-B *candidate pool* was 401
   files, far larger than the 92-image corpus it produced. A file rejected then
   is still licence-clean now, but its appearance already informed a curation
   judgement, so all 401 are excluded by id.
3. **Disjoint categories.** Drawn from Commons categories not used in Phase
   2c-B — `Malus domestica`, `Apple cultivars`, `Musa acuminata`, `Fruit
   vendors`, `Farmers' markets` and others.

The first attempt satisfied only constraint 1 and produced a pool of which 152
of 168 files had already been screened. That was discarded rather than reported
as independent.

**Verified: 0 file overlap, 0 shared source groups, 114 distinct creators in the
pool.** 29 images were curated from it by eye, frozen with fingerprint
`e9e7ace82c40783e` before any detector parameter was chosen.

**Limitation, stated rather than padded around.** The categories that satisfy
the independence constraint are botanical and market-oriented, so clean
single-subject photographs are scarce: the pool is 9 clean-base and 20
natural-scene images against a target that wanted more of the former. It was not
topped up with off-subject images to reach a rounder number.

---

## 4. Glare — local highlight evidence

### Why whole-image highlight clipping is not a glare detector

A specular highlight covering a few percent of a fruit never approaches a global
clipped-pixel fraction limit, and raising that limit until it does would reject
correctly exposed photographs. Phase 2c-B measured the consequence: 10%
detection, and those blocks came from patch size, not detection.

### What separates a highlight from a bright fruit

"Bright pixels" is not the signal — a yellow banana is bright, an orange is
bright and vividly coloured. Two properties are required together:

1. **Local excess luminance**, measured at several scales. Subtracting a blurred
   copy of L\* leaves a smooth illumination gradient near zero and a hotspot
   standing above it.
2. **Desaturation relative to the subject's own chroma.** Specular reflection
   returns the illuminant's colour, so a highlight washes toward white while the
   fruit around it keeps its saturation. Compared against the ROI's own median,
   so a pale fruit is not penalised for being pale.

Plus a direct admission for **blown** pixels: a severe highlight clips to flat
white and therefore has no local excess *anywhere in its interior*.

### Two measurement defects found and fixed

**Single-scale blindness.** A background estimate at one scale can only see
highlights smaller than itself. With a single 0.25-radius estimator, a highlight
covering 12% of the subject was invisible — local excess inside it was near
zero, with a response only at its rim. The excess is now the elementwise maximum
across sigmas of 0.10, 0.30 and 0.90 of the subject radius.

**Pathological cost at the widest scale.** A direct Gaussian builds a kernel
about six sigma wide, so the 0.90 scale on a 1920px photograph needed a ~3000px
kernel — minutes per image. Above a cap of 24px the blur is computed on a
correspondingly downscaled copy and resized back, which is the standard
equivalence and is accurate well within the tolerance of a threshold expressed
in whole L\* units. Measurement cost fell from minutes to ~340ms.

### A ground-truth defect that mattered more

The Phase 1 `add_glare` places its highlight at a seeded point **anywhere in the
frame**. Measured on this corpus, the median injected highlight had only **25%**
of its area inside the subject region. Most "glare positive" samples were
photographs whose fruit had no glare on it, and scoring a subject-region
detector against that labelling measures the placement, not the detector.

`add_glare_at` and `occlude_at` take an explicit centre and **return the
affected region as a mask**, so ground truth is known rather than assumed. The
Phase 1 transforms are unchanged; these are additions.

---

## 5. Visibility — what it measures, and what it refuses to claim

**This is not occlusion detection, and it is not named as one.** Classical
single-image OpenCV cannot establish that a hand, a leaf or a price tag is
covering a fruit: there is no model of the fruit's complete outline, and a
subject that genuinely is that shape is indistinguishable from one partly
hidden.

What can be measured is whether the visible region is *shaped like a complete
subject*: convex, unbroken, holeless, inside the frame. The evidence is named
`SUBJECT_VISIBILITY_INSUFFICIENT` and phrased as a statement about the region.
A test asserts that the words "hand", "occluder", "occlusion detected" and
"covered by" never appear in the emitted evidence.

Signals: hull fill, solidity, border truncation (normalised by the *subject's*
outline, not the frame's, so a small fruit running off the edge is not scored
low for being small), internal hole fraction, maximum convexity-defect depth,
and fragmentation.

### The hole-filling finding

Phase 2b's `_clean_mask` fills interior holes, which is right for measuring a
subject — a mask of a fruit should not be porous because a highlight desaturated
a patch. But filling also closes the one place an occluder *inside* the
silhouette shows up. A black rectangle covering three quarters of a fruit left a
filled mask that still looked like an intact subject, and
`internal_hole_fraction` was structurally zero.

`foreground.segment_without_fill` exposes the unfilled mask, and visibility
measures interior evidence on it. This is not a synthetic-rectangle trick: a
hand, label or leaf differs from the fruit in saturation or lightness, so it is
excluded by the same thresholding and leaves a genuine hole there too.

### Central occlusion mostly never reaches this detector

Measured on calibration images, an occluder covering 20% of the subject area and
centred on it caused **every** test image to fail the Phase 2b foreground
validity guards outright. Those captures escalate as
`ARTIFACT_EVIDENCE_UNAVAILABLE` — the correct outcome, but reached through
segmentation rather than through visibility evidence. The controlled evaluation
reports the two paths separately, so the visibility detector is not credited
with refusals the foreground guards made.

---

## 6. Actions

One closed vocabulary, extended rather than duplicated: `RemediationAction`
gains `REQUEST_REPOSITION_LIGHT`, and `ReasonCode` gains five Phase 2d entries.

| Finding | Action | Reason code |
| --- | --- | --- |
| Segmentation invalid | `REQUEST_HUMAN_REVIEW` | `ARTIFACT_EVIDENCE_UNAVAILABLE` |
| Subject visibility insufficient | `REQUEST_RECAPTURE` | `SUBJECT_VISIBILITY_INSUFFICIENT` |
| Local glare | `REQUEST_REPOSITION_LIGHT` | `GLARE_LOCAL_HIGHLIGHT` |
| Severe contrast loss | `REQUEST_RECAPTURE` | `SEVERE_CONTRAST_LOSS` |
| Moderate contrast loss | `APPLY_CLAHE` | `MODERATE_CONTRAST_LOSS_RECOVERABLE` |
| Nothing found | `NONE` | `CAPTURE_ACCEPTABLE` |

**Precedence: coverage, then photometry, then tone.** If the subject is not
fully in frame, its brightness is beside the point — reporting a glare finding
about a fruit half outside the picture would be precise about the wrong thing.

**Glare never routes to an enhancement**, and this is enforced in the policy
rather than left to a caller. No tone curve recovers the surface under a blown
highlight; offering one would produce an image that looks treated and carries no
more information, and the pipeline would then infer a condition from it. A test
asserts it.

`REQUEST_REPOSITION_LIGHT` is a human action: `requires_human` is true and
`is_automated` is false.

---

## 7. Trace

    isolate_foreground
        -> measure_local_highlights / measure_visibility / ROI contrast
        -> GLARE_RISK | SUBJECT_VISIBILITY_INSUFFICIENT | SEVERE_LOW_CONTRAST
        -> REQUEST_REPOSITION_LIGHT | REQUEST_RECAPTURE | APPLY_CLAHE

Every step records its inputs and outputs, so an action is attributable to a
named OpenCV measurement rather than asserted. **Nothing runs on an untrusted
mask**: if foreground isolation fails its guards, no local evidence is produced
at all and the analysis escalates, because a highlight measured on a backdrop is
not a fact about the fruit.

The interactive loop — dispatching the action, re-capturing, re-assessing —
remains Phase 3. Phase 2d builds the evidence and the decision.

---

## 8. Reproduction

```bash
# independent validation pool (screen -> curate on contact sheets -> fetch)
python -m competition.evaluation.build_validation_pool pool
python -m competition.evaluation.build_validation_pool thumbnails
python -m competition.evaluation.build_validation_pool harvest

# the Phase 2c-B contrast audit
python -m competition.evaluation.audit_contrast_path

# calibrate detectors on calibration groups, then freeze
python -m competition.evaluation.calibrate_artifact_policy

# controlled ground truth, then one look at the new pool
python -m competition.evaluation.phase2d_artifacts controlled
python -m competition.evaluation.phase2d_artifacts validation
python -m competition.evaluation.phase2d_artifacts latency
```

---

## 9. Calibration — the preregistered rule, and what it rejected

On calibration groups only, over an explicit parameter grid: reject any setting
whose false-positive rate on undegraded images exceeds **0.10**; among the rest
take the highest detection rate on the severe rung; declare the detector
**not usable for gating** if that rate is below **0.70**.

| Detector | Best detection | False positive | Cleared 0.70? |
| --- | ---: | ---: | :---: |
| Severe contrast | **0.870** | **0.000** | **yes** |
| Glare | 0.655 | 0.033 | **no** |
| Visibility | 0.400 | 0.033 | **no** |

**The floor was not lowered after seeing 0.655.** It governed *blocking*, and a
detector below it still produces evidence worth recording in a trace and showing
to a person. What it may not do is send a photograph back on its own authority.
So `ArtifactPolicy.glare_blocks` and `visibility_blocks` default to **False**:
the findings appear in `flags` and in the explanation, and the action stays
`NONE`. Severe contrast is the one artefact permitted to gate.

Frozen fingerprints — highlight `10757655e4c34533`, visibility `213bac848e52951f`,
artifact `b3cd5db5ca193027` — recorded before the validation pool was opened.

---

## 10. Controlled results

### Glare — a clean graded response, below the bar

| Injected severity | n | Flag rate | Median detected ROI fraction | Median injected ROI fraction | Median IoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 30 | **0.033** | 0.064 | — | — |
| 0.3 | 30 | 0.167 | 0.083 | 0.088 | 0.28 |
| 0.6 | 29 | 0.379 | 0.106 | 0.088 | 0.51 |
| 0.9 | 29 | **0.655** | 0.122 | 0.088 | **0.58** |

The response is monotonic in severity, the detected area tracks the injected
area closely (0.122 against 0.088 at full severity — mild over-detection at the
soft rim), and localisation reaches IoU 0.58. This is a detector that works and
misses a third of severe cases.

### Visibility — the detector never fires; segmentation does the work

| Subject coverage | n | Blocked | `ARTIFACT_EVIDENCE_UNAVAILABLE` | `SUBJECT_VISIBILITY_INSUFFICIENT` |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 30 | 0 | 0 | 0 |
| 0.10 | 30 | 1 | 1 | **0** |
| 0.25 | 30 | 5 | 5 | **0** |
| 0.45 | 30 | 14 | 14 | **0** |

**Every refusal came from the foreground validity guards, not from visibility
evidence.** An occluder large enough to matter destroys the mask before the
visibility measurement runs; one small enough to leave a valid mask leaves a
mask that still looks like a complete subject. The escalation behaviour is
correct — coverage 0.45 refuses 47% of captures — but it is not attributable to
anything Phase 2d built, and the report does not credit it as such.

### Contrast — the one that works

| `reduce_contrast` factor | n | Median ROI contrast | Severe | Moderate | Acceptable | No mask |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 35 | 0.600 | 0 | 0 | 30 | 5 |
| 0.50 *(borderline)* | 35 | 0.247 | 0 | 0 | 28 | 7 |
| 0.20 *(unusable)* | 35 | 0.094 | **22** | 3 | 5 | 5 |
| 0.08 *(unusable)* | 35 | 0.039 | **28** | 0 | 0 | 7 |

Zero false blocks on undegraded captures, zero on the borderline rung, and the
unusable rungs now escalate instead of passing silently. That is the Phase 2c-B
false-accept family closed.

The calibrated severe limit of **0.12** sits just under the Phase 2c-B advisory
limit of 0.1314, so the "moderate" band between them is narrow and only 3 of 105
samples landed in it. The contrast distribution on this corpus is effectively
bimodal. The limit is also at the top of its grid, which was capped at 0.1314 by
construction — a severe limit above the advisory one would invert the ladder —
so the ceiling is principled rather than an artefact of where the sweep stopped.

---

## 11. Validation — the independent pool, opened once

29 images, 29 distinct source groups, none shared with Phase 2c-B, frozen at
fingerprint `e9e7ace82c40783e`. Status **CONFIRMATORY**.

| Measure | Value |
| --- | ---: |
| Images | 29 |
| **Segmentation valid** | **12 (41%)** |
| Escalated as `ARTIFACT_EVIDENCE_UNAVAILABLE` | 17 |
| Outcome on all 12 assessable images | `CAPTURE_ACCEPTABLE` |
| **Glare false-positive rate** | **0.167** (2 / 12) |
| Visibility false-positive rate | 0.083 (1 / 12) |
| **Severe contrast false-positive rate** | **0.000** (0 / 12) |
| Injected-glare detection on unseen groups | **0.50** (6 / 12) |

**The glare detector transfers worse than its calibration figures suggest.**
False positives rose from 0.033 to 0.167 and detection fell from 0.655 to 0.500
on photographs by contributors the calibration never saw. A 17% false-positive
rate would send roughly one good capture in six back to the user for no reason.
This is the strongest evidence that keeping it non-gating was right, and it is
the number that would have been hidden by reusing the spent held-out set.

**Severe contrast held at zero false positives** on unseen data, matching its
calibration behaviour.

Only 41% of the pool produced a usable mask, which caps how much any of these
figures can say. The pool is scene-heavy because the independence constraint
forced botanical and market categories, and market scenes have no single
subject. The numbers above rest on 12 images.

---

## 12. Latency

Local CPU, OpenCV 5.0.0, median image 1848 × 1359. **Not an AWS measurement and
not a deployment figure.**

| Stage | n | Median | p95 |
| --- | ---: | ---: | ---: |
| Foreground isolation | 12 | 89 ms | 114 ms |
| Glare evidence | 10 | **166 ms** | 191 ms |
| Visibility evidence | 10 | 18 ms | 21 ms |
| Contrast evidence | 10 | 20 ms | 33 ms |
| **Combined analysis** | 10 | **296 ms** | 348 ms |

Glare dominates because of the three-scale background estimate. Before the
downscaled-blur fix the widest scale alone took minutes per image.

---

## 13. Claim boundaries

**Supportable:**

- "Severe contrast loss is detected and blocked, with zero false positives on
  both calibration and independent validation imagery."
- "Local highlight evidence responds monotonically to injected glare severity
  and localises it at IoU 0.58 on calibration images."
- "Subject-visibility evidence was built, measured, and did not meet the
  preregistered bar; occlusion is currently caught by foreground validity
  guards instead."

**Not supportable, and not claimed:**

- "Glare detection works." It reaches 0.655 on controlled injections and 0.500
  on unseen data, and it is not permitted to block.
- "Occlusion is detected." Nothing here detects an occluder. The evidence
  measures the *shape of the visible region*, and the words "hand", "occluder"
  and "covered by" are absent from the output by test.
- Any transfer claim from injected highlights to real specular reflections, or
  from a black rectangle to a hand.
- Anything about produce condition. A bright region is not damage.

---

## 14. Known limitations

1. **Glare is evidence-only.** 0.655 controlled, 0.500 on validation, 0.167
   false positives on validation.
2. **Visibility insufficiency has no working detector.** 0.40 detection; every
   controlled refusal came from segmentation failure.
3. **Segmentation is the binding constraint everywhere.** 41% of the validation
   pool and 14–20% of undegraded calibration images produce no usable mask, and
   no local artefact evidence can be produced without one.
4. **The validation pool is small and scene-heavy** — 29 images, 12 assessable,
   9 clean-base.
5. **Injections are characterised perturbations, not real artefacts.** A smooth
   additive disc is not a specular reflection; a black square is not a hand.
6. **The moderate-contrast band is nearly empty**, so `APPLY_CLAHE` is reachable
   in principle and almost never reached in practice on this data.
