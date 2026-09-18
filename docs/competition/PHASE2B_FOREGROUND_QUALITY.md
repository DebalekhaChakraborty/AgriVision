# Phase 2b — Foreground Isolation and ROI-Restricted Capture Quality

| Field | Value |
| --- | --- |
| Branch | `competition/opencv-aws-2026` |
| Phase 2 base commit | `1cd167e3b2e6b58626bd589b5752e5d8c480049d` |
| OpenCV version | **5.0.0** (`opencv-python==5.0.0.93`) |
| Default method | **`saturation_otsu`** |
| ROI gating | **implemented, off by default** — see §8 |
| Tests | 312 passing |

> **Scope boundary.** This phase produces a **subject region** — where the
> produce is in the frame. It does **not** produce a bruise mask, a rot mask, a
> lesion mask or any surface-deterioration map. Anomaly localisation does not
> exist, and nothing here may be described as if it does.

**What may be claimed from this phase:** foreground-restricted quality
assessment eliminated background-induced clipping flags on validly segmented
product photographs. **What may not:** that AgriVision detects, localises or
segments spoilage, defects or rotten regions, or that the capture gate is now
fixed.

---

## 1. Measured motivation

Phase 2 found the capture gate rejected **96.7%** of real research photographs.
The diagnosis was that these are product photographs with bright backgrounds and
dark surrounds, so whole-image clipping statistics measure the **backdrop**, not
the fruit. Threshold relaxation would have hidden that rather than fixed it.

## 2. Methods compared

Three classical OpenCV approaches. No segmentation network and no foundation
model — a learned segmenter would reintroduce exactly the dependency weight
Phase 2 removed.

| Method | Approach | Synthetic IoU | Real validity | Median latency | p95 |
| --- | --- | --- | --- | --- | --- |
| **`saturation_otsu`** | Otsu on HSV saturation + morphology | **0.905** | **0.985** | **8.0 ms** | 12.2 ms |
| `border_lab_distance` | Border ring models the background; Otsu on CIELAB distance | 0.905 | 0.859 | 15.5 ms | 25.2 ms |
| `grabcut_rect` | GrabCut seeded with an inset rectangle | 0.892 | 1.000 | **649 ms** | **3292 ms** |

**Selected: `saturation_otsu`** — equal best synthetic IoU, highest validity
among the fast methods, and the lowest latency.

**Rejected `grabcut_rect` on cost.** Its 100% validity is not the advantage it
appears: GrabCut seeded with a rectangle tends to return *something*
rectangle-shaped that passes geometric guards, so validity rate flatters it. At
649 ms median and 3.3 s p95 it is 53–270× the selected method for no measured
quality gain.

**`border_lab_distance` retained as the more general fallback.** Saturation
thresholding assumes a coloured subject on a neutral background and will fail on
a saturated backdrop or desaturated produce; the border method models whatever
the frame edge actually contains.

**Watershed was not used.** It was considered and deliberately excluded: no
defensible reproducible marker-generation rule presented itself for single-object
product photography, and the other methods already solve the problem. Including
it merely because OpenCV offers it would have added fragmentation risk for
nothing.

**Honest caveat on "validity rate":** it measures how often the geometric guards
*pass*, not whether the mask is *correct*. No ground-truth masks exist for real
photographs, so real-image segmentation accuracy is unmeasured and no IoU is
reported for real data.

## 3. Validity guards

A mask is never trusted implicitly. `ForegroundEvidence.valid` is explicit, and
every failing check is reported, not just the first:

| Guard | Rejects |
| --- | --- |
| `min_foreground_fraction` 0.03 | subject too small to measure |
| `max_foreground_fraction` 0.92 | grabbed the whole frame |
| `max_border_contact_fraction` 0.60 | grabbed the background instead |
| `min_largest_component_dominance` 0.65 | no dominant subject |
| `max_component_count` 12 | shattered mask |
| `min_bounding_box_side` 24 px | implausible geometry |

Real-image validity was 266/270 (98.5%); 257 had enough subject left to measure
after erosion. On synthetic fixtures the guards correctly rejected the
`multiple_regions` fixture (`NO_DOMINANT_COMPONENT`) and `small_subject`
(`FOREGROUND_TOO_SMALL`, `BOUNDING_BOX_TOO_SMALL`).

## 4. ROI-restricted metrics

There is **one** implementation. `assess_capture_quality(image, policy, mask)`
restricts every metric to the mask and stamps the record `FOREGROUND_MASKED`;
without a mask it measures the frame and stamps `WHOLE_IMAGE`. No second code
path exists, so the two scopes cannot drift. Whole-image evidence is always
retained alongside ROI evidence for trace and debugging.

### Mask-edge sharpness — the critical detail

A binary mask boundary is a step discontinuity. Zeroing the background and then
differentiating manufactures an enormous gradient exactly where the mask ends,
so the more aggressively an image is masked, the *sharper* it appears.

The implementation avoids this: the Laplacian is computed on the **unmodified**
grayscale image, the mask only selects which responses are aggregated, and the
mask is **eroded by 5 px** first so genuine subject/background edges are
excluded too.

Measured on synthetic fixtures with known masks:

| Fixture | Whole image | Naive background-zeroing | Erode 0 px | Erode 5 px | Erode 11 px |
| --- | --- | --- | --- | --- | --- |
| `dark_on_bright` | 1408.2 | 387.0 | **3030.5** | 1037.4 | 1032.8 |
| `bright_on_dark` | 1325.7 | 1393.7 | **2889.9** | 1036.4 | 1031.5 |
| `low_contrast` | 35.7 | **732.1** | 128.4 | 127.7 | 128.1 |

Two failure modes, both real:

* **Un-eroded masking inflates sharpness ~2.9×** on high-contrast subjects
  (3030.5 against 1037.4). Median inflation across all fixtures was **1.506×**.
* **Naive background-zeroing is erratic**, not merely biased: 5.7× too high on
  `low_contrast` (732.1 against 127.7), and *lower* than whole-image on
  `dark_on_bright` because zeroing flattened the background.

Erosion of 5 px and 11 px agree to within 0.5%, so the boundary effect is fully
suppressed at the default and further erosion buys nothing. Tests pin both the
inflation bound and the stability.

## 5. Synthetic segmentation

IoU and Dice are legitimate here only because the masks are **constructed**.
Eight fixtures: dark-on-bright, bright-on-dark, textured-on-neutral,
border-touching, multiple regions, small subject, subject-with-distractor,
low-contrast.

`saturation_otsu`: mean IoU **0.905**, mean Dice 0.924, 6/8 valid. The two
"failures" are the guard cases behaving correctly.

**These scores do not represent real produce segmentation quality.** The
fixtures are synthetic discs on flat backgrounds and all three methods scored
within 0.013 IoU of each other, which is why the real-image comparison decided
the default.

## 6. The critical experiment — background-driven clipping

270 real validation photographs, whole-image against foreground-restricted:

| Metric | Whole image | Foreground ROI |
| --- | --- | --- |
| `highlight_clip_fraction` median | 0.2821 | **0.0000** |
| `highlight_clip_fraction` p95 | 0.6044 | **0.0031** |
| `shadow_clip_fraction` median | 0.1323 | **0.0000** |
| `shadow_clip_fraction` p95 | 0.2853 | **0.0119** |
| `HIGHLIGHT_CLIPPING` flag rate | 80.0% | **0%** |
| `SHADOW_CLIPPING` flag rate | 67.8% | **3.5%** |

**The hypothesis is confirmed.** The clipping that blocked four-fifths of real
photographs was in the background, not on the produce. Restricting to the
subject removes it essentially completely.

## 7. The gate is not fixed — and that matters more

| Measure | Whole image | ROI restricted |
| --- | --- | --- |
| Blocked rate | 96.7% | **83.3%** |

An improvement, but far short of a fix, and the reason is important:

| Flag | Whole image | ROI restricted |
| --- | --- | --- |
| `HIGHLIGHT_CLIPPING` | 80.0% | 0% |
| `SHADOW_CLIPPING` | 67.8% | 3.5% |
| **`BLUR_RISK`** | **19.3%** | **82.5%** |
| `OVEREXPOSED` | 13.3% | 9.3% |

**`BLUR_RISK` now fires far more often**, and it is the sole remaining reason
the gate blocks. Measured on 100 paired images:

```
Laplacian variance   whole image   median 430.6
Laplacian variance   foreground    median  19.9
ratio ROI/whole                    median  0.053
below the blur threshold of 100:   whole 25%   ROI 90%
```

### How this result should and should not be read

**It must not be read as "ROI made sharpness worse."** Nothing about the images
changed, and nothing about the sharpness computation became less accurate.

**Foreground restriction changed the measurement domain and thereby invalidated
the whole-image sharpness threshold.** The whole-image distribution (median
430.6) was strongly influenced by subject-outline and background edges — a
single high-contrast boundary contributing far more gradient energy than the
produce surface itself. ROI measurement excludes that boundary by design, so the
remaining signal is genuine surface texture, and its distribution sits an order
of magnitude lower (median 19.9, a ratio of 0.053).

The threshold of **100 was derived against the whole-image distribution and
cannot be transferred to the ROI scope.** It is not "too strict" or "too
lenient" — it is a number from a different population. A correctly calibrated
ROI threshold would land somewhere in the region of the ROI distribution, and
where exactly is an empirical question that cannot be answered from these
photographs (see §8).

**The general finding: capture-quality thresholds are scope-specific and do not
transfer between whole-image and foreground-restricted measurement.** Any
threshold set must record the scope it was calibrated against. This is the same
class of error as the Phase 1 exposure/sharpness coupling — a threshold silently
carrying an assumption about the population it was measured over.

## 8. Gate integration, and why ROI gating is off by default

`InspectionPolicy.use_foreground_roi` is **`False`**.

That is an evidence-based decision, not caution. Enabling ROI gating today would
swap one miscalibration for another: it removes a real 80% false-block from
clipping and introduces an 82.5% false-block from blur. The capability is
implemented, measured and opt-in; correcting it requires a **calibrated ROI
threshold set**, which is Phase 6 work on imagery from the deployment domain.

No threshold was tuned to improve the block rate. **The goal was correctness, not
a target pass percentage.**

When enabled, the logic is conservative:

```
segmentation valid  -> ROI evidence drives the gate; whole-image kept for trace
segmentation invalid-> whole-image evidence retained; nothing fabricated
ROI too small       -> explicit fallback, recorded, never silent
```

Every outcome is traced via an `isolate_foreground` step carrying validity,
reasons and the resulting `gate_scope`.

## 9. Pipeline ordering and mask stability

Segmentation runs on the **canonical image after remediation**, not before.

Mask agreement before versus after remediation, 51 real images:

| Remediation | Median IoU | p05 IoU | Fraction below 0.90 |
| --- | --- | --- | --- |
| `gamma_0.6` | 0.9917 | 0.9337 | 0.000 |
| `gamma_1.6` | 0.9878 | 0.9523 | 0.000 |
| `clahe` | 0.9904 | 0.9479 | 0.000 |

With the selected method masks are **stable** — nothing fell below 0.90 IoU.
Notably this is method-dependent: `border_lab_distance` was *not* stable under
the same test (p05 0.21–0.61, a third of cases below 0.90 IoU), which is an
additional point in favour of the selected default.

The mask is recomputed anyway. Reuse would be a performance optimisation worth
about 12 ms, and B13's instruction is explicit that masks must not be reused
solely for convenience; guaranteeing the mask describes the image actually being
gated is worth more.

## 10. Latency

Local CPU, 270 real photographs, 60 samples after 5 warm-ups. **Not AWS.**

| Stage | Median | p95 |
| --- | --- | --- |
| Foreground segmentation | 12.25 ms | 15.27 ms |
| ROI quality assessment | 9.15 ms | 12.95 ms |
| **Combined perception** | **21.12 ms** | 26.42 ms |

Separate from condition-model inference (17.52 ms median via `cv2.dnn`,
measured in Phase 2).

## 11. Limitations

- **The gate is still not usable on real photography** (83.3% blocked). Phase 2b
  removed one cause and exposed another; it did not deliver a working gate.
- ROI thresholds are uncalibrated, so ROI gating stays off.
- **Real-image segmentation accuracy is unmeasured.** No ground-truth masks
  exist; validity rate measures guard passage, not correctness.
- Synthetic IoU is measured on discs against flat backgrounds and is far easier
  than real photography.
- `saturation_otsu` assumes a coloured subject on a neutral background and will
  fail on a saturated backdrop or desaturated produce.
- Evaluation is one dataset, one background style — product photography. Nothing
  is known about field or packhouse conditions.
- ROI `highlight_clip_fraction` p95 is 0.0031, so a small number of subjects do
  have genuine specular clipping; foreground restriction does not make that
  disappear, nor should it.
- No anomaly, defect or spoilage localisation exists.

## 12. Reproduction

```bash
.venv-competition/bin/python -m pytest tests/competition -q
.venv-competition/bin/python -m competition.evaluation.phase2b_foreground
```

Artifacts in `competition/evaluation/results/phase2b/`:
`synthetic_segmentation_metrics.json`, `foreground_method_comparison.json`,
`real_foreground_statistics.json`, `gate_comparison.json`, `mask_stability.json`,
`mask_edge_sharpness.json`, `latency.json`, and three plots. Real photographs are
read in place; **no image bytes are stored or committed**.

## 13. Not done in this phase

Anomaly/defect localisation, ROI threshold calibration, self-captured imagery,
AWS deployment, Bedrock, and UI.
