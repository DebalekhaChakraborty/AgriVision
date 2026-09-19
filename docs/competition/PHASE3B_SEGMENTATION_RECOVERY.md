# Phase 3b — Bounded foreground segmentation recovery

**Status: COMPLETE — UNCOMMITTED.** 761 tests pass (730 prior, unchanged).

Phase 2d measured the problem honestly: 12 of 29 independent validation
photographs produced a foreground mask that passed its guards. Phase 3 made that
failure *safe* by refusing to measure a region it could not attribute. 41%
assessability is still a poor foundation, and this phase asks what can be
recovered without making the system more confidently wrong.

The answer turned out to depend entirely on a distinction the headline number
hides.

---

## 0. Deployment input contract

**One primary produce item per inspection capture.**

This is the scope the system is built and measured for, and it is now stated as
a contract rather than left implicit in the corpus tracks.

The system does **not** perform semantic extraction of one fruit from:

- market stalls
- fruit piles or crates
- trees carrying multiple fruits
- arbitrary multi-object scenes

Such an image is expected to produce `INSUFFICIENT_VISUAL_EVIDENCE` and a
terminal `REQUEST_RECAPTURE` or `REQUEST_HUMAN_REVIEW`. That is correct
behaviour under this contract, not a defect: §4 measured what happens when the
fallback is pushed past it, and 18 of 19 recovered scene masks were an arbitrary
block carved out of a pile.

**Multi-object produce segmentation is not claimed anywhere in this project.**

This is a scope boundary. It is not a food-safety statement, and nothing here
speaks to whether any produce is safe, edible or contaminated.

---

## 1. The headline number was measuring two different things

Diagnosing the 65-image development split separates cleanly by corpus track:

| Track | What it contains | Primary valid |
| --- | --- | ---: |
| `CLEAN_BASE` | one fruit, photographed as a product | **30/35 — 85.7%** |
| `NATURAL_SCENE` | markets, crates, trees, piles | **9/30 — 30.0%** |

The deployment contract is a person photographing **one fruit** for inspection.
That is the `CLEAN_BASE` case, and the primary segmenter already handles 85.7%
of it. The 41% figure is dominated by scene photography, where there is often no
single subject to find and **refusing is the correct behaviour**.

This does not make the limitation disappear. It relocates it.

### The dominant failure shape

20 of 26 failures share one signature: foreground fraction ≥ 0.68 with a single
component covering essentially the whole frame. Otsu found no saturation
bimodality because the entire scene is colourful, so the threshold landed
somewhere that put nearly everything on the foreground side.

| Reason | Count |
| --- | ---: |
| `EXCESSIVE_BORDER_CONTACT` | 20 |
| `FOREGROUND_NEAR_FULL_FRAME` | 11 |
| `NO_DOMINANT_COMPONENT` | 4 |
| `MASK_FRAGMENTED` | 2 |

---

## 2. What was built

The primary is untouched. `saturation_otsu` remains the default and its code
path is byte-identical, because every Phase 2b, 2c-B and 2d result describes
those exact masks — changing them would invalidate a calibration rather than
improve one. A test asserts the primary mask is unchanged when it succeeds.

The ladder is consulted **only** when the primary fails, in cost order:

| Candidate | Why it is there |
| --- | --- |
| `chroma_distance` *(new)* | Border-background distance in a\*/b\* only. `border_lab_distance` includes L\*, which partly cancels a subject that differs in colour but not brightness. Agrees with the primary at median IoU **0.821** — the closest of any alternate. |
| `border_lab_distance` | The existing method, at working resolution. |
| `grabcut_scaled` *(new)* | GrabCut at 480 px, **one** iteration. |

### GrabCut was rejected on cost, and cost is fixable

Phase 2b recorded GrabCut as the only method to pass the guards on every
research photograph, then rejected it at 649 ms median and 3.3 s p95. Downscaling
addresses the objection without touching what made it attractive.

Two measurements shaped it. **One iteration beat three** — on the 26 failures,
one recovered 24 and three recovered 22, at 909 ms against 1263 ms; more
refinement was slower *and* slightly worse. And most of the residual cost was
not GrabCut at all but the shared cleanup: `_clean_mask` at full resolution
costs 31 ms of a 44 ms segmentation, against **2.9 ms** at working resolution for
an IoU of **0.974**. Fallbacks clean at working resolution; the primary still
does not, so its masks are unchanged.

`lightness_otsu` was built, measured and dropped: median IoU **0.343** against
the primary on images where the primary succeeds, meaning it reliably finds
something else.

---

## 3. What it recovers

| Track | Before | After | Recovered |
| --- | ---: | ---: | ---: |
| `CLEAN_BASE` | 85.7% | **100.0%** | +5 |
| `NATURAL_SCENE` | 30.0% | **93.3%** | +19 |
| Overall | 60.0% | 96.9% | +24 |

Accepted methods: `saturation_otsu` 39, `chroma_distance` 12, `grabcut_scaled` 9,
`border_lab_distance` 3.

**These are validity rates, not correctness rates.** "Recovered" means a mask
passed the same unrelaxed guards. Read alone, the 96.9% would be a misleading
number, which is why the next section exists.

---

## 4. Whether the recovered masks are right

The guards test geometry. They cannot distinguish one apple from a compact block
carved out of a heap of apples, because the two have the same geometry. So the
24 recovered masks were rendered as overlays and adjudicated by eye.

> **This is single-rater visual adjudication, not ground truth.** The rater is
> the implementer. No pixel-accurate reference masks exist for these
> photographs, no second rater scored them, and no inter-rater agreement is
> available. It is recorded because it is the only evidence in this phase that
> speaks to correctness, and leaving it as an unwritten impression would have
> been worse than stating its limits.

| Track | n | Usable | Wrong | Usable rate |
| --- | ---: | ---: | ---: | ---: |
| `CLEAN_BASE` | 5 | 5 | 0 | **100%** |
| `NATURAL_SCENE` | 19 | 1 | 18 | **5.3%** |

**The ladder is reliable exactly where the product contract applies and
unreliable outside it.** Every single-subject recovery was usable. On scenes, 18
of 19 were an arbitrary block cut out of a pile — compact, dominant, not touching
the border, and therefore indistinguishable from a real subject to any guard.
One scene recovery was genuinely correct: a single green apple isolated from
surrounding red ones.

Per-image verdicts with notes: `results/phase3b/adjudication.json`.

---

## 5. Two discriminators were built and both failed

Recorded so a later reader does not spend the same effort twice.

**Boundary–edge support** — the fraction of the mask outline lying on a strong
image gradient, on the theory that a real object boundary sits on a gradient
ridge while a threshold artifact cut through a pile does not. It measured the
**wrong way round**: 0.760 median for the arbitrary crops against 0.573 for the
correct clean-base masks. In a dense pile there is texture everywhere, so any cut
lands on an edge.

**Subject multiplicity** — a distance-transform disc ratio, on the theory that
one fruit gives one inscribed disc and a pile gives many. It ranks the clearest
single subjects first (1.09, 1.10, 1.10) and the sprawling market crops last
(3.03, 3.17, 3.59), but it is confounded by shape: a bunch of bananas scores 1.88
and a legitimate three-apple product shot scores worse than some piles. It
measures elongation as much as multiplicity.

Neither is used and no threshold was fitted to make either look better than it
is. **There is currently no reliable classical test for "this photograph
contains a single subject."**

---

## 6. How silent acceptance is prevented

The objective was to recover masks *without increasing silent wrong-mask
acceptance*. Three things deliver the "not silent" half:

1. **Guards are not relaxed.** A recovered mask passes exactly the checks the
   primary would have had to pass. This widens which images yield a mask without
   widening what counts as one.
2. **Provenance is recorded.** Every mask carries `PRIMARY`, `FALLBACK` or
   `NONE` and the method that produced it, plus every candidate attempted.
3. **Maturity is lowered.** `foreground.recovered` is registered `PROVISIONAL`,
   never `CALIBRATED`. The Phase 3 trace shows the provenance, the maturity and
   the adjudication caveat on the segmentation step itself.

What this does **not** do is make recovered masks correct. On a multi-subject
scene the acceptance is wrong and visibly labelled, rather than wrong and
invisible. That is an improvement in auditability, not in perception.

**The fallback is OFF by default.** Every committed Phase 2c-B, 2d and 3 result
was produced with the primary alone, and a default that silently changed which
masks those numbers describe would invalidate them. Enabling it is a deployment
decision with documented consequences.

---

## 7. Latency

Local CPU, full-resolution licensed photographs.

| Path | n | Median | p95 |
| --- | ---: | ---: | ---: |
| Primary accepted (ladder not consulted) | 117 | 91.8 ms | 211.4 ms |
| Fallback invoked | 78 | 293.4 ms | 1670.2 ms |

The ladder costs nothing when the primary succeeds. The fallback figure is the
price of an image that would otherwise have been refused outright. The p95 is
dominated by GrabCut on large images.

> Local CPU only. **Not an AWS figure.**

---

## 8. Limitations

**No ground-truth masks exist.** Every correctness statement here rests on
single-rater visual adjudication of 24 images.

**No single-subject detector exists.** Two were built and both failed (§5). Until
one exists, the ladder cannot be told when not to fire.

**Scene recovery is actively wrong 18 times in 19.** Enabling the fallback on
scene photography produces confidently-shaped masks of nothing in particular.

**Development split only.** Everything here was measured on the 65 Phase 2c-B
calibration images. The held-out groups and the Phase 2d validation set were each
opened once and were deliberately not touched, so **no confirmatory Phase 3b
number exists yet**. A fresh untouched pool is required before any of these rates
can be called confirmatory.

**The 29-image Phase 2d validation figure is not re-measured here** for the same
reason.

---

## 9. Claim boundary

Supported:

> On single-subject captures in the development split, a bounded classical
> fallback ladder raised foreground validity from 85.7% to 100%, and all five
> recovered masks were adjudicated usable by one rater.

**Not** claimed:

- That overall assessability is 96.9% in any useful sense — that number pools a
  reliable case with an unreliable one.
- Any confirmatory rate. Nothing here was measured on untouched data.
- That recovered masks are as trustworthy as primary masks.
- That the system can tell a single fruit from a pile. It cannot.

---

## 10. Reproduction

```bash
python -m competition.evaluation.phase3b_segmentation diagnose
python -m competition.evaluation.phase3b_segmentation recover
python -m competition.evaluation.phase3b_segmentation adjudicate
python -m competition.evaluation.phase3b_segmentation latency
python -m competition.evaluation.phase3b_segmentation all

python -m pytest tests/competition/test_foreground_fallback.py -q
```

Requires the local licensed corpus (gitignored). Without it the evaluation
raises rather than inventing a population.
