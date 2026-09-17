# Cross-Domain Generalization Analysis — Experiment 013

## Primary result

All systems were frozen before seeing the external benchmark. The 1,200 evaluation images are official original photographs only, balanced at 200 per canonical class. No target-domain training, validation, calibration, prompt selection, or adaptation occurred.

| System | Source Test Acc | External Acc | Domain Gap | Retention | External Macro F1 | External Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 74.09% | 34.58% | −39.51 pp | 46.68% | 27.72% | 27.72% |
| ResNet50 | 94.18% | 69.33% | −24.85 pp | 73.62% | 65.10% | 65.10% |
| EfficientNet-B0 | 94.55% | 68.75% | −25.80 pp | 72.71% | 67.38% | 67.38% |
| MobileNetV3-Large | 93.03% | 70.75% | −22.28 pp | 76.05% | 65.92% | 65.92% |
| DINOv2 + linear probe | 89.07% | 68.17% | −20.90 pp | 76.54% | 63.27% | 63.27% |
| CLIP + linear probe | 95.85% | 61.00% | −34.85 pp | 63.64% | 54.96% | 54.96% |
| SigLIP2 + linear probe | 98.63% | 82.75% | −15.88 pp | 83.90% | 81.24% | 81.24% |
| CLIP strict P1 zero-shot | 89.96% | 80.75% | −9.21 pp | 89.77% | 80.03% | 80.03% |
| SigLIP2 strict P1 zero-shot | 97.89% | **90.75%** | **−7.14 pp** | **92.71%** | **90.70%** | **90.70%** |

Gaps are external minus source; negative values are degradation. Weighted and macro results coincide externally because all six classes have equal support, though their constituent class performance differs.

## External metrics and uncertainty

Intervals are explicitly **95% class-stratified bootstrap intervals over the external evaluation sample**, based on 5,000 resamples with seed 42. They describe sample-level uncertainty, not full model uncertainty.

| System | Accuracy (95% interval) | Macro F1 (95% interval) |
|---|---:|---:|
| Custom CNN | 34.58% [32.75, 36.42] | 27.72% [25.75, 29.69] |
| ResNet50 | 69.33% [67.42, 71.25] | 65.10% [62.74, 67.37] |
| EfficientNet-B0 | 68.75% [66.50, 71.17] | 67.38% [64.92, 69.90] |
| MobileNetV3-Large | 70.75% [68.83, 72.58] | 65.92% [63.94, 67.91] |
| DINOv2 + linear probe | 68.17% [66.42, 69.92] | 63.27% [60.85, 65.66] |
| CLIP + linear probe | 61.00% [59.08, 62.83] | 54.96% [52.59, 57.16] |
| SigLIP2 + linear probe | 82.75% [81.08, 84.42] | 81.24% [79.24, 83.29] |
| CLIP strict P1 zero-shot | 80.75% [78.83, 82.75] | 80.03% [77.90, 82.21] |
| SigLIP2 strict P1 zero-shot | 90.75% [89.25, 92.33] | 90.70% [89.16, 92.29] |

## Pre-registered paired comparisons

Positive differences favor the left-hand system. Intervals use the same paired, within-class resamples.

| Comparison | Observed accuracy difference | Mean bootstrap difference | 95% paired interval |
|---|---:|---:|---:|
| SigLIP2 probe − EfficientNet-B0 | +14.00 pp | +14.00 pp | [+11.42, +16.42] pp |
| CLIP probe − EfficientNet-B0 | −7.75 pp | −7.77 pp | [−10.33, −5.25] pp |
| SigLIP2 strict zero-shot − SigLIP2 probe | +8.00 pp | +8.00 pp | [+6.25, +9.75] pp |
| CLIP strict zero-shot − CLIP probe | +19.75 pp | +19.76 pp | [+17.41, +22.25] pp |

No all-pairs significance search was performed.

## Ranking change

Source ranking by accuracy was: SigLIP2 probe, SigLIP2 zero-shot, CLIP probe, EfficientNet-B0, ResNet50, MobileNetV3, CLIP zero-shot, DINOv2 probe, CNN.

External ranking was: SigLIP2 zero-shot, SigLIP2 probe, CLIP zero-shot, MobileNetV3, ResNet50, EfficientNet-B0, DINOv2 probe, CLIP probe, CNN.

The largest reversal is semantic versus source-trained CLIP: CLIP's probe fell from third to eighth, while strict CLIP zero-shot rose from seventh to third. MobileNetV3 also became the strongest conventional transfer model. DINOv2 stayed below the three ImageNet transfer systems, but its deficit to EfficientNet narrowed from 5.49 to 0.58 points, so it became relatively stronger despite absolute degradation.

## Class-level recall shift

Each cell is `external recall (external − source)`.

| System | Fresh apple | Fresh banana | Fresh orange | Rotten apple | Rotten banana | Rotten orange |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 60.5% (−33.9) | 3.0% (−75.0) | 15.5% (−59.0) | 27.5% (−23.9) | 98.5% (+5.1) | 2.5% (−56.1) |
| ResNet50 | 94.5% (−5.2) | 78.5% (−21.2) | 94.0% (−3.4) | 40.5% (−50.7) | 95.5% (−0.3) | 13.0% (−69.6) |
| EfficientNet-B0 | 67.5% (−32.0) | 73.5% (−26.5) | 85.0% (−14.0) | 62.0% (−27.5) | 97.0% (+1.7) | 27.5% (−59.3) |
| MobileNetV3 | 85.5% (−12.5) | 82.5% (−17.5) | 96.5% (−1.4) | 66.0% (−19.7) | 89.0% (−8.4) | 5.0% (−77.1) |
| DINOv2 probe | 100.0% (+0.0) | 45.5% (−51.4) | 100.0% (+0.0) | 52.0% (−14.1) | 100.0% (+0.0) | 11.5% (−68.9) |
| CLIP probe | 37.0% (−62.7) | 3.0% (−96.7) | 89.5% (−9.0) | 94.0% (+6.5) | 100.0% (+0.8) | 42.5% (−51.3) |
| SigLIP2 probe | 99.5% (−0.5) | 32.0% (−67.7) | 95.5% (−4.2) | 89.0% (−5.8) | 100.0% (+0.2) | 80.5% (−18.8) |
| CLIP zero-shot | 94.0% (+2.4) | 55.0% (−43.7) | 50.0% (−20.4) | 86.0% (−1.7) | 99.5% (+1.6) | 100.0% (+8.2) |
| SigLIP2 zero-shot | 100.0% (+0.0) | 76.5% (−23.2) | 88.5% (−10.7) | 89.0% (−5.8) | 93.5% (−5.6) | 97.0% (+1.2) |

Across systems, fresh banana had the largest mean recall collapse (−46.99 points); rotten orange was close (−43.52) and had the lowest mean external recall (42.17%) and mean external F1. Rotten apple remained challenging for several systems, especially CNN and ResNet50, but it was not the most consistently difficult external class. Rotten banana was unusually stable (−0.55 mean points; 97.0% mean recall).

## Semantic error taxonomy

| System | Errors | Condition | Fruit identity | Combined |
|---|---:|---:|---:|---:|
| CNN | 785 | 201 (25.6%) | 338 (43.1%) | 246 (31.3%) |
| ResNet50 | 368 | 315 (85.6%) | 49 (13.3%) | 4 (1.1%) |
| EfficientNet-B0 | 375 | 290 (77.3%) | 69 (18.4%) | 16 (4.3%) |
| MobileNetV3 | 351 | 236 (67.2%) | 111 (31.6%) | 4 (1.1%) |
| DINOv2 probe | 382 | 377 (98.7%) | 5 (1.3%) | 0 |
| CLIP probe | 468 | 391 (83.5%) | 68 (14.5%) | 9 (1.9%) |
| SigLIP2 probe | 207 | 202 (97.6%) | 5 (2.4%) | 0 |
| CLIP zero-shot | 231 | 212 (91.8%) | 19 (8.2%) | 0 |
| SigLIP2 zero-shot | 111 | 106 (95.5%) | 5 (4.5%) | 0 |

Condition errors dominate for every system except the custom CNN. The pattern suggests that most pretrained representations preserve fruit identity better than the dataset-specific fresh/rotten boundary under shift.

## Efficiency/generalization context

| System | External accuracy | Accuracy retention | External model-forward ms/image | Prior source model-forward ms/image |
|---|---:|---:|---:|---:|
| ResNet50 | 69.33% | 73.62% | 31.46 | 31.28 |
| EfficientNet-B0 | 68.75% | 72.71% | 11.77 | 12.55 |
| MobileNetV3 | 70.75% | 76.05% | **5.74** | **6.44** |
| DINOv2 probe | 68.17% | 76.54% | 92.36 | 93.93 |
| CLIP probe | 61.00% | 63.64% | 106.89 | 79.02 |
| SigLIP2 probe | 82.75% | 83.90% | 97.78 | 82.34 |
| CLIP zero-shot | 80.75% | 89.77% | 76.73 | 79.27 |
| SigLIP2 zero-shot | 90.75% | 92.71% | 105.84 | 80.68 |

External timings use the same model-forward-only scope after deterministic preprocessing; decoding and transforms are excluded. The CNN external forward time was 4.62 ms/image, but no directly comparable prior source timing was recorded, so it is omitted from the paired timing table. Runtime variation remains possible on a shared CPU host. MobileNetV3 gives the best speed among pretrained systems and slightly higher external accuracy than ResNet50/EfficientNet; SigLIP2 zero-shot trades much higher compute for the strongest generalization result.

## Required research answers

1. **Highest external accuracy:** SigLIP2 strict P1 zero-shot, 90.75%.
2. **Smallest absolute degradation:** SigLIP2 strict P1 zero-shot, −7.14 percentage points.
3. **Highest source-performance retention:** SigLIP2 strict P1 zero-shot, 92.71% accuracy retention.
4. **SigLIP2 advantage:** Yes within this benchmark. Its probe remains the best source-trained head, while its strict zero-shot system becomes best overall externally. This is one-domain evidence, not universal superiority.
5. **EfficientNet competitiveness:** It remains competitive with ResNet50 and DINOv2 but is slightly behind MobileNetV3; its key benefit is favorable compute, not the highest external accuracy.
6. **DINOv2 relative movement:** Relatively stronger against EfficientNet (deficit shrinks to 0.58 points), but still below all three ImageNet transfer systems in external accuracy.
7. **CLIP probe versus zero-shot:** No. CLIP zero-shot is 19.75 points higher externally; paired 95% interval [+17.41, +22.25].
8. **SigLIP2 probe versus zero-shot:** No. SigLIP2 zero-shot is 8.00 points higher externally; paired 95% interval [+6.25, +9.75].
9. **Source-head specialization:** For both pre-registered vision-language comparisons, the text-based strict classifier is more robust externally. This supports domain specialization of these fitted heads in this setting, not a general causal law.
10. **Largest class collapses:** Fresh banana has the largest average recall loss; rotten orange is nearly as severe and has the lowest average external recall.
11. **Condition versus identity errors:** Condition errors dominate eight of nine systems; only the custom CNN has more identity than condition errors.
12. **Rotten apple:** It is difficult for some models but is not the hardest class overall; rotten orange and fresh banana are more consistently impaired.
13. **Rankings:** Yes. Zero-shot systems rise substantially, CLIP's source-trained probe falls sharply, and MobileNetV3 leads the conventional transfer group externally.
14. **Representation quality versus separability:** Near-ceiling source separability does not guarantee domain-stable decision boundaries. SigLIP2 representations remain strong, but its strict semantic boundary transfers better than its fitted source head; CLIP shows the same effect more strongly.

## Conservative conclusion and limitations

The benchmark demonstrates external-domain generalization on one independently collected dataset. It does not prove real-world generalization. The external sample is balanced and finite, contains seven official within-class duplicate pairs, and may differ from the source in many confounded ways. Bootstrap intervals quantify resampling variability of these images only, not retraining, model, dataset, site, or deployment uncertainty.

Visible fresh/rotten labels are not food-safety or edibility assessments. No claim is made about microbial or chemical spoilage, shelf life, or unseen fruit categories. Further domains would be required to test whether the observed rank reversals reproduce elsewhere; any target adaptation must be a separately registered future experiment.

