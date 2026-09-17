# V2 Phase 3C — Label-Efficiency Analysis

## Scope and evidence boundary

Phase 3C asks how performance changes between zero labels and the frozen full-data Phase 3A endpoints. It trains linear probes on the unchanged, L2-normalized DINOv2, original CLIP, and SigLIP2 embeddings. The encoders, their revisions, preprocessing, dataset split, test set, and Phase 3A/3B results were not changed or rerun.

The supervised estimates below are means across exactly five pre-registered nested subset draws (seeds 42–46). Each interval is a two-sided t interval for the mean with four degrees of freedom and is labelled **95% interval across pre-registered subset draws**. It describes variation among these five draws; it is not complete predictive uncertainty.

CLIP and SigLIP2's zero-label points are the frozen strict-P1 Phase 3B results. Those points use text prototypes. Every point at one or more labels uses a trained linear probe. Consequently, the 0-to-1 transition changes both supervision amount and downstream classifier and cannot support a pure causal label-count claim. DINOv2 has no applicable zero-label semantic classifier.

## Aggregate results

All values are percentages. Standard deviation and interval are calculated across the five subset draws; the weighted F1 and rotten-apple recall columns are five-draw means.

| Model | Labels/class | Mean accuracy | Std | 95% interval across pre-registered subset draws | Weighted F1 | Rotten-apple recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 | 1 | 81.14 | 5.67 | [74.10, 88.18] | 80.63 | 63.83 |
| DINOv2 | 5 | 92.97 | 0.70 | [92.11, 93.84] | 92.99 | 87.42 |
| DINOv2 | 10 | 93.40 | 0.92 | [92.26, 94.53] | 93.42 | 86.56 |
| DINOv2 | 25 | 94.20 | 1.59 | [92.22, 96.17] | 94.20 | 85.22 |
| DINOv2 | 50 | 92.22 | 2.27 | [89.40, 95.04] | 92.19 | 74.31 |
| DINOv2 | 100 | 92.59 | 1.02 | [91.32, 93.86] | 92.56 | 76.04 |
| Original CLIP | 1 | 74.28 | 10.21 | [61.60, 86.97] | 72.24 | 74.64 |
| Original CLIP | 5 | 89.85 | 2.38 | [86.89, 92.81] | 89.84 | 84.19 |
| Original CLIP | 10 | 91.02 | 1.82 | [88.77, 93.28] | 91.02 | 85.09 |
| Original CLIP | 25 | 94.37 | 0.68 | [93.52, 95.22] | 94.38 | 89.38 |
| Original CLIP | 50 | 94.63 | 0.47 | [94.05, 95.21] | 94.63 | 87.55 |
| Original CLIP | 100 | 95.30 | 0.17 | [95.09, 95.51] | 95.29 | 88.22 |
| SigLIP2 | 1 | 87.52 | 5.67 | [80.48, 94.56] | 87.36 | 77.64 |
| SigLIP2 | 5 | 97.21 | 0.75 | [96.29, 98.14] | 97.21 | 93.18 |
| SigLIP2 | 10 | 97.66 | 0.44 | [97.11, 98.21] | 97.66 | 94.11 |
| SigLIP2 | 25 | 98.12 | 0.49 | [97.52, 98.73] | 98.12 | 94.34 |
| SigLIP2 | 50 | 98.37 | 0.19 | [98.13, 98.60] | 98.37 | 94.74 |
| SigLIP2 | 100 | 98.50 | 0.17 | [98.29, 98.72] | 98.50 | 94.71 |

The machine-readable [combined summary](../v2/results/label_efficiency_summary.json) also records the mean, sample standard deviation, minimum, maximum, t interval, full-performance gap, performance retention, and class-level precision/recall/F1 for every model and budget.

## Endpoint and threshold summary

| Model | Strict zero-shot accuracy | 1-shot | 5-shot | 10-shot | 25-shot | 50-shot | 100-shot | Frozen full-data accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 | N/A | 81.14 | 92.97 | 93.40 | 94.20 | 92.22 | 92.59 | 89.07 |
| Original CLIP | 89.96 | 74.28 | 89.85 | 91.02 | 94.37 | 94.63 | 95.30 | 95.85 |
| SigLIP2 | 97.89 | 87.52 | 97.21 | 97.66 | 98.12 | 98.37 | 98.50 | 98.63 |

| Model | First tested budget at ≥95% of full | First tested budget within 1 point of full |
| --- | ---: | ---: |
| DINOv2 | 5/class | 5/class |
| Original CLIP | 25/class | 100/class |
| SigLIP2 | 5/class | 10/class |

No interpolation was used. DINOv2's thresholds need special care: its 5-shot mean exceeds its frozen Phase 3A full-data accuracy. That counterintuitive result is genuine under this protocol, but it does not mean that discarding labels intrinsically improves DINOv2. The full-data run selected epoch 2 after reaching perfect validation accuracy, whereas the low-shot families often selected later epochs; the fixed 50-epoch validation-selection procedure, the small validation set, and dataset shift can all affect the endpoint comparison.

At 100 labels/class, the full-performance gaps (full minus mean few-shot) were 0.55 points for CLIP, 0.13 for SigLIP2, and -3.52 for DINOv2. Corresponding performance retention was 99.43%, 99.87%, and 103.95%. A retention above 100% is a descriptive consequence of the frozen lower full-data endpoint, not evidence of negative sample complexity.

## Model ranking and adaptation observations

Mean-accuracy rankings were:

- 1, 5, and 10 labels/class: SigLIP2 > DINOv2 > CLIP.
- 25, 50, and 100 labels/class: SigLIP2 > CLIP > DINOv2.

The ranking therefore stabilizes at 25 labels/class among the tested budgets. SigLIP2 is strongest at both 1-shot and 5-shot.

CLIP's strict zero-shot accuracy was 89.96%. Its supervised mean was lower at 1-shot, essentially equal at 5-shot (-0.10 point), then higher by 1.07, 4.42, 4.68, and 5.34 points at 10, 25, 50, and 100 labels/class. Its frozen full-data probe was 5.89 points above strict zero-shot. Under this setup, CLIP benefits substantially more from supervised linear adaptation than SigLIP2 once enough labels are available.

SigLIP2's strict zero-shot accuracy was already 97.89%. Its 1-, 5-, and 10-shot means were lower by 10.37, 0.67, and 0.23 points; its 25-, 50-, and 100-shot means were higher by 0.24, 0.48, and 0.61 point. The frozen full-data advantage was 0.74 point. Thus supervised adaptation provides a modest high-budget gain rather than a large one. With five draws, this study does not establish a universal minimum label requirement or a causal advantage at the 0-to-1 boundary.

DINOv2 rapidly rose from 81.14% at 1-shot to 92.97% at 5-shot and peaked at 94.20% at 25-shot, temporarily exceeding CLIP at 5 and 10 labels/class. It never caught SigLIP2, CLIP retook second place at 25-shot, and DINOv2 declined at 50-shot before a small 100-shot recovery. DINOv2 therefore closes its Phase 3A deficit only in the narrow sense of exceeding its own frozen full-data result; it does not close the high-budget gap to the language-aligned representations.

## Stability across subset draws

At 1-shot, SigLIP2 had the smallest accuracy standard deviation (5.669 points), effectively tied with DINOv2 (5.670) and substantially below CLIP (10.213). At 5-shot, DINOv2 was most stable (0.697), followed by SigLIP2 (0.746) and CLIP (2.384).

Variance generally fell as labels increased for CLIP and SigLIP2. DINOv2 was not monotonic: its accuracy standard deviation fell sharply at 5-shot, then rose to 2.272 points at 50-shot before falling to 1.023 at 100-shot. Five draws reveal subset sensitivity but are too few to characterize every source of predictive uncertainty.

At 1-shot, the most subset-sensitive recalls were CLIP fresh apple (36.18-point standard deviation), CLIP rotten apple (29.58), CLIP rotten banana (26.84), DINOv2 rotten apple (22.63), and SigLIP2 rotten apple (16.02). This confirms that aggregate stability can conceal class-specific instability.

## Class-level recall curves

### DINOv2

| Class | 1 | 5 | 10 | 25 | 50 | 100 | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh apple | 94.78 | 97.01 | 99.90 | 100.00 | 100.00 | 100.00 | 100.00 |
| Rotten apple | 63.83 | 87.42 | 86.56 | 85.22 | 74.31 | 76.04 | 66.06 |
| Fresh banana | 92.60 | 99.42 | 99.00 | 99.48 | 99.37 | 99.42 | 96.85 |
| Rotten banana | 72.26 | 85.66 | 86.75 | 91.70 | 93.58 | 93.77 | 100.00 |
| Fresh orange | 89.64 | 97.27 | 97.89 | 98.51 | 98.51 | 98.45 | 100.00 |
| Rotten orange | 86.25 | 96.67 | 96.33 | 96.03 | 96.67 | 96.33 | 80.40 |

### Original CLIP

| Class | 1 | 5 | 10 | 25 | 50 | 100 | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh apple | 58.99 | 92.15 | 96.91 | 96.51 | 97.67 | 98.53 | 99.75 |
| Rotten apple | 74.64 | 84.19 | 85.09 | 89.38 | 87.55 | 88.22 | 87.52 |
| Fresh banana | 92.97 | 99.42 | 99.69 | 99.42 | 99.53 | 99.63 | 99.74 |
| Rotten banana | 60.91 | 82.91 | 81.70 | 92.68 | 95.51 | 97.32 | 99.25 |
| Fresh orange | 87.68 | 97.16 | 97.32 | 96.75 | 97.78 | 97.63 | 98.45 |
| Rotten orange | 75.78 | 89.08 | 92.11 | 94.89 | 93.40 | 93.70 | 93.80 |

### SigLIP2

| Class | 1 | 5 | 10 | 25 | 50 | 100 | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh apple | 93.87 | 99.80 | 99.70 | 99.95 | 100.00 | 100.00 | 100.00 |
| Rotten apple | 77.64 | 93.18 | 94.11 | 94.34 | 94.74 | 94.71 | 94.84 |
| Fresh banana | 95.01 | 99.58 | 99.79 | 99.84 | 99.74 | 99.74 | 99.74 |
| Rotten banana | 77.40 | 96.64 | 96.75 | 98.34 | 99.17 | 99.55 | 99.81 |
| Fresh orange | 95.10 | 98.97 | 99.38 | 99.54 | 99.43 | 99.48 | 99.74 |
| Rotten orange | 94.94 | 97.52 | 98.46 | 98.71 | 98.81 | 99.21 | 99.26 |

Small adjacent decreases occurred in several class curves. Most were below two points, but DINOv2 rotten-apple recall fell 10.92 points from 25 to 50 labels/class. CLIP rotten-apple recall fell 1.83 points over the same step, and SigLIP2 rotten-apple recall was effectively flat from 50 to 100 (-0.03 point). These observations show that validation-selected finite-sample curves need not be monotonic; they do not prove that the additional labels caused degradation.

## Rotten-apple finding

Rotten apple remained the clearest difficult class. At 1-shot, mean recall was 63.83% for DINOv2, 74.64% for CLIP, and 77.64% for SigLIP2, with high seed sensitivity. SigLIP2 improved to 93.18% by 5-shot and stayed near 94–95% thereafter. CLIP rose more gradually and peaked at 89.38% at 25-shot before settling at 88.22% at 100-shot. DINOv2 peaked at 87.42% at 5-shot, declined thereafter, and ended at 76.04% at 100-shot. Under this frozen dataset and protocol, SigLIP2 provides the strongest and most stable high-budget rotten-apple separation.

## Answers to the registered questions

1. **Strongest at 1-shot:** SigLIP2, 87.52% mean accuracy.
2. **Strongest at 5-shot:** SigLIP2, 97.21%.
3. **Ranking stabilization:** 25 labels/class; the ranking remains SigLIP2 > CLIP > DINOv2 through 100.
4. **Fewest labels to 95% of full:** DINOv2 and SigLIP2 tie at 5/class; CLIP requires 25/class.
5. **Within one point of full first:** DINOv2 at 5/class, with the endpoint caveat above; SigLIP2 at 10 and CLIP at 100.
6. **SigLIP2 supervised gain:** modest, not large—100-shot is +0.61 point and full-data is +0.74 point versus strict zero-shot.
7. **CLIP versus SigLIP2 adaptation:** CLIP benefits more in absolute accuracy once ≥10 labels/class are available.
8. **DINOv2 deficit:** it exceeds its own frozen full-data endpoint from 5-shot onward but never catches SigLIP2 and trails CLIP from 25-shot onward.
9. **Lowest low-shot variance:** SigLIP2 by a negligible margin at 1-shot; DINOv2 at 5-shot.
10. **Rotten-apple evolution:** strongest and nearly saturated by 5-shot for SigLIP2; gradual but non-monotonic for CLIP; sharply non-monotonic for DINOv2.
11. **Unexpected decreases:** yes, especially DINOv2 overall and rotten-apple recall from 25 to 50; smaller adjacent class declines occur for all models.
12. **Supported boundary:** five draws support dataset- and protocol-specific mean/dispersion descriptions. They do not establish universal food-freshness sample complexity, full predictive uncertainty, causal superiority of language alignment, or safe-food assessment capability.

## Limitations

- Only one frozen dataset, one leakage-aware split, and six classes from three fruit identities are evaluated.
- Five subset draws provide limited uncertainty resolution; the intervals do not include dataset, checkpoint, optimization-seed, or deployment-domain uncertainty.
- Probe initialization and training seed are deliberately fixed, so measured variability mainly reflects selected labelled examples.
- Validation has 270 images and influences every checkpoint choice; full-data and low-shot selected epochs differ.
- Zero-shot and supervised endpoints use different downstream decision mechanisms.
- No external-domain, temporal-shift, calibration, quality-scoring, regression, explanation, or food-safety evaluation is included.
- Visible appearance labels cannot establish microbiological safety.

## Figures

- [Accuracy curve](../v2/results/label_efficiency_accuracy.png)
- [Weighted-F1 curve](../v2/results/label_efficiency_weighted_f1.png)
- [Rotten-apple recall curve](../v2/results/label_efficiency_rotten_apple_recall.png)
- [Subset-draw variance](../v2/results/label_efficiency_variance.png)
- [All class recall curves](../v2/results/label_efficiency_class_recall.png)
