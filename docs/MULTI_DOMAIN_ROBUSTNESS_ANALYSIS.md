# Multi-Domain Robustness Analysis — Experiment 014

## Design and frozen ranking rule

Experiment 014 asks whether conclusions from the Sultana benchmark persist on FruitVision. It reuses frozen source-test and Experiment 013 results without inference, and evaluates the same nine systems exactly once on all 4,185 eligible FruitVision originals. FruitVision was never used for training, validation, calibration, prompt selection, or adaptation.

Before FruitVision results were inspected, robustness ranking was fixed as: (1) highest worst external accuracy, then (2) highest mean external accuracy. Source accuracy is displayed but is not averaged into an overall score.

## Three-domain master table

| System | Source Acc | Sultana Acc | FruitVision Acc | Mean External Acc | Worst External Acc | Source weighted F1 | Sultana weighted F1 | FruitVision weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 74.09% | 34.58% | 38.95% | 36.77% | 34.58% | 74.03% | 27.72% | 35.25% |
| ResNet50 | 94.18% | 69.33% | 76.20% | 72.77% | 69.33% | 94.13% | 65.10% | 74.19% |
| EfficientNet-B0 | 94.55% | 68.75% | 70.30% | 69.52% | 68.75% | 94.52% | 67.38% | 68.77% |
| MobileNetV3-Large | 93.03% | 70.75% | 70.37% | 70.56% | 70.37% | 92.98% | 65.92% | 67.46% |
| DINOv2 + linear probe | 89.07% | 68.17% | 66.38% | 67.27% | 66.38% | 89.01% | 63.27% | 61.75% |
| CLIP + linear probe | 95.85% | 61.00% | 55.24% | 58.12% | 55.24% | 95.84% | 54.96% | 44.81% |
| SigLIP2 + linear probe | 98.63% | 82.75% | 73.52% | 78.14% | 73.52% | 98.62% | 81.24% | 68.78% |
| CLIP strict P1 zero-shot | 89.96% | 80.75% | 67.43% | 74.09% | 67.43% | 90.09% | 80.03% | 64.47% |
| SigLIP2 strict P1 zero-shot | 97.89% | **90.75%** | **86.31%** | **88.53%** | **86.31%** | 97.89% | **90.70%** | **85.95%** |

SigLIP2 strict zero-shot is first by FruitVision accuracy, mean external accuracy, worst external accuracy, and both worst external macro F1 (85.59%) and weighted F1 (85.95%). The full pre-registered robustness order is SigLIP2 zero-shot, SigLIP2 probe, MobileNetV3, ResNet50, EfficientNet, CLIP zero-shot, DINOv2, CLIP probe, and Custom CNN.

## Retention and worst-case behavior

| System | Sultana retention | FruitVision retention | Mean retention | Worst retention | Worst external class recall |
|---|---:|---:|---:|---:|---:|
| Custom CNN | 46.68% | 52.57% | 49.62% | 46.68% | 2.50% |
| ResNet50 | 73.62% | 80.91% | 77.26% | 73.62% | 13.00% |
| EfficientNet-B0 | 72.71% | 74.35% | 73.53% | 72.71% | 27.50% |
| MobileNetV3-Large | 76.05% | 75.64% | 75.85% | 75.64% | 5.00% |
| DINOv2 + linear probe | 76.54% | 74.53% | 75.53% | 74.53% | 11.50% |
| CLIP + linear probe | 63.64% | 57.64% | 60.64% | 57.64% | 1.87% |
| SigLIP2 + linear probe | 83.90% | 74.55% | 79.22% | 74.55% | 6.28% |
| CLIP strict P1 zero-shot | 89.77% | 74.96% | 82.36% | 74.96% | 27.94% |
| SigLIP2 strict P1 zero-shot | **92.71%** | **88.17%** | **90.44%** | **88.17%** | **61.27%** |

Retention reinforces the absolute ranking: SigLIP2 zero-shot has the best mean and worst retention. High source accuracy alone is not predictive of robustness: CLIP probe ranks third on source accuracy but eighth on each external domain, while MobileNetV3 ranks sixth on source and fourth on both external domains.

## FruitVision uncertainty

The interval label is: **95% class-stratified bootstrap interval over the FruitVision evaluation sample.** Seed 42 and 5,000 resamples were used.

| System | Accuracy | 95% interval | Macro F1 | 95% interval |
|---|---:|---:|---:|---:|
| Custom CNN | 38.95% | 37.85–40.05% | 34.79% | 33.64–35.93% |
| ResNet50 | 76.20% | 75.20–77.18% | 73.85% | 72.68–75.02% |
| EfficientNet-B0 | 70.30% | 69.22–71.37% | 68.53% | 67.30–69.77% |
| MobileNetV3-Large | 70.37% | 69.41–71.33% | 67.32% | 66.15–68.46% |
| DINOv2 + linear probe | 66.38% | 65.40–67.38% | 61.27% | 59.81–62.72% |
| CLIP + linear probe | 55.24% | 54.48–56.03% | 46.07% | 45.05–47.10% |
| SigLIP2 + linear probe | 73.52% | 72.66–74.38% | 69.00% | 67.91–70.05% |
| CLIP strict P1 zero-shot | 67.43% | 66.33–68.58% | 63.79% | 62.43–65.17% |
| SigLIP2 strict P1 zero-shot | **86.31%** | **85.38–87.31%** | **85.59%** | **84.56–86.68%** |

Exactly four pre-registered paired comparisons were run. Paired-bootstrap mean FruitVision accuracy differences (left minus right) and 95% intervals were: SigLIP2 zero-shot vs probe +12.79 points [11.88, 13.74]; CLIP zero-shot vs probe +12.18 [10.99, 13.41]; SigLIP2 zero-shot vs MobileNetV3 +15.94 [14.65, 17.23]; and SigLIP2 zero-shot vs CLIP zero-shot +18.89 [17.63, 20.12]. The corresponding observed differences are +12.78, +12.19, +15.94, and +18.88 points. All four intervals exclude zero. These comparisons are confined to this evaluation sample.

## Ranking stability

| System | Source rank | Sultana rank | FruitVision rank | Best | Worst | Range |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 9 | 9 | 9 | 9 | 9 | **0** |
| ResNet50 | 5 | 5 | 2 | 2 | 5 | 3 |
| EfficientNet-B0 | 4 | 6 | 5 | 4 | 6 | 2 |
| MobileNetV3-Large | 6 | 4 | 4 | 4 | 6 | 2 |
| DINOv2 + linear probe | 8 | 7 | 7 | 7 | 8 | **1** |
| CLIP + linear probe | 3 | 8 | 8 | 3 | 8 | 5 |
| SigLIP2 + linear probe | 1 | 2 | 3 | 1 | 3 | 2 |
| CLIP strict P1 zero-shot | 7 | 3 | 6 | 3 | 7 | 4 |
| SigLIP2 strict P1 zero-shot | 2 | 1 | 1 | 1 | 2 | **1** |

The Custom CNN has the numerically lowest volatility only because it is consistently ninth; rank stability is not quality. Among competitive systems, SigLIP2 zero-shot is most stable (range 1). DINOv2 also has range 1, consistently in seventh/eighth place; its relative rank improves from eighth on source to seventh on both external domains.

Pairwise rank association is moderate for source–Sultana (Spearman 0.567; Kendall tau-b 0.333), moderate for source–FruitVision (0.617; 0.500), and stronger for Sultana–FruitVision (0.833; 0.722). Average ranks and Kendall tau-b are registered for ties, although no accuracy tie occurred. P-values are descriptive and were not used for selection.

## Zero-shot versus source-trained probes

CLIP strict zero-shot exceeds the CLIP probe by 19.75 points on Sultana and 12.19 on FruitVision (mean +15.97). SigLIP2 strict zero-shot exceeds its probe by 8.00 and 12.78 points (mean +10.39). The external zero-shot advantage therefore replicates for both families on both external datasets, even though the probes are stronger in-distribution. This is evidence for robustness under these two shifts, not a universal superiority claim.

## Class and semantic-error findings

Across systems and the two external domains, rotten banana is easiest and most stable (mean recall 97.46%, mean domain range 2.30 points). Rotten orange is most consistently difficult by mean recall (43.73%), while rotten apple is least stable by mean external range (22.71 points; maximum 58.06). Fresh banana is also difficult (53.19% mean recall) and has a comparable 21.79-point mean range. Thus the single-domain observation does not reduce to one universal difficult class.

Each cell below gives source/Sultana/FruitVision recall, preserving every model/class result in compact form.

| System | Fresh apple | Fresh banana | Fresh orange | Rotten apple | Rotten banana | Rotten orange |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 94.4/60.5/30.6 | 78.0/3.0/69.3 | 74.5/15.5/9.4 | 51.4/27.5/4.4 | 93.4/98.5/97.8 | 58.6/2.5/24.4 |
| ResNet50 | 99.7/94.5/63.4 | 99.7/78.5/97.7 | 97.4/94.0/100.0 | 91.2/40.5/77.9 | 95.8/95.5/90.0 | 82.6/13.0/24.2 |
| EfficientNet-B0 | 99.5/67.5/40.9 | 100.0/73.5/95.2 | 99.0/85.0/100.0 | 89.5/62.0/56.3 | 95.3/97.0/97.3 | 86.8/27.5/29.4 |
| MobileNetV3-Large | 98.0/85.5/31.1 | 100.0/82.5/99.3 | 97.9/96.5/100.0 | 85.7/66.0/74.9 | 97.4/89.0/96.5 | 82.1/5.0/19.5 |
| DINOv2 probe | 100.0/100.0/99.5 | 96.9/45.5/29.0 | 100.0/100.0/100.0 | 66.1/52.0/35.7 | 100.0/100.0/100.0 | 80.4/11.5/29.0 |
| CLIP probe | 99.7/37.0/4.3 | 99.7/3.0/1.9 | 98.5/89.5/99.5 | 87.5/94.0/90.3 | 99.2/100.0/100.0 | 93.8/42.5/48.0 |
| SigLIP2 probe | 100.0/99.5/93.5 | 99.7/32.0/6.3 | 99.7/95.5/99.6 | 94.8/89.0/65.4 | 99.8/100.0/100.0 | 99.3/80.5/79.4 |
| CLIP strict zero-shot | 91.6/94.0/85.5 | 98.7/55.0/29.5 | 70.4/50.0/92.7 | 87.7/86.0/27.9 | 97.9/99.5/100.0 | 91.8/100.0/67.2 |
| SigLIP2 strict zero-shot | 100.0/100.0/95.3 | 99.7/76.5/79.7 | 99.2/88.5/93.4 | 94.8/89.0/61.3 | 99.1/93.5/99.7 | 95.8/97.0/86.4 |

Condition errors dominate both external domains for all eight pretrained systems. The Custom CNN is the exception in both domains: fruit-identity errors are its largest category (43.06% on Sultana, 61.41% on FruitVision). For SigLIP2 zero-shot, condition errors are 95.50% of Sultana errors and 77.14% of FruitVision errors; increased FruitVision identity errors show that error composition still shifts by domain.

## Accuracy, compute, and robustness

| System | CPU forward ms/image | Source Acc | Mean External Acc | Worst External Acc |
|---|---:|---:|---:|---:|
| Custom CNN | 3.99 | 74.09% | 36.77% | 34.58% |
| MobileNetV3-Large | 7.26 | 93.03% | 70.56% | 70.37% |
| EfficientNet-B0 | 12.27 | 94.55% | 69.52% | 68.75% |
| ResNet50 | 29.78 | 94.18% | 72.77% | 69.33% |
| CLIP probe / zero-shot | 68.60 | 95.85% / 89.96% | 58.12% / 74.09% | 55.24% / 67.43% |
| SigLIP2 probe / zero-shot | 77.49 | 98.63% / 97.89% | 78.14% / **88.53%** | 73.52% / **86.31%** |
| DINOv2 + linear probe | 95.09 | 89.07% | 67.27% | 66.38% |

SigLIP2 zero-shot gives the strongest measured accuracy/robustness outcome, but MobileNetV3 is the clearest efficiency/robustness compromise: it is about 10.7× faster by the registered CPU forward measurement while retaining a 70.37% worst external accuracy. Neither is universally superior because compute constraints and error costs differ.

## Required research-question answers

1. **Highest FruitVision accuracy:** SigLIP2 strict P1 zero-shot, 86.31%.
2. **Highest mean external accuracy:** SigLIP2 strict P1 zero-shot, 88.53%.
3. **Highest worst external accuracy:** SigLIP2 strict P1 zero-shot, 86.31%.
4. **Does SigLIP2 zero-shot remain first?** Yes, on both external datasets and under the frozen robustness rule.
5. **Does CLIP zero-shot beat its probe on both external datasets?** Yes: +19.75 points on Sultana and +12.19 on FruitVision.
6. **Does SigLIP2 zero-shot beat its probe on both?** Yes: +8.00 and +12.78 points.
7. **Is the Phase 3D zero-shot robustness finding replicated?** Yes for both model families on FruitVision, within the stated two-external-domain boundary.
8. **Most robust conventional ImageNet model:** MobileNetV3 has the highest worst external accuracy (70.37%) and mean external accuracy (70.56%) among the three ImageNet transfer models; ResNet50 has the highest FruitVision score alone.
9. **DINOv2 relative rank:** It improves from eighth on source to seventh on Sultana and FruitVision, but remains below all three conventional transfer models in the pre-registered robustness order.
10. **Lowest rank volatility:** Custom CNN has range 0 because it is always ninth. Among competitive systems, SigLIP2 zero-shot has range 1; DINOv2 also has range 1 at ranks 7–8.
11. **Ranking correlation:** Source–Sultana is 0.567 Spearman/0.333 Kendall tau-b; source–FruitVision 0.617/0.500; Sultana–FruitVision 0.833/0.722.
12. **Most consistently difficult class:** Rotten orange has the lowest mean recall across systems/external domains (43.73%); rotten apple is least stable by mean range.
13. **Are condition errors dominant in both?** Yes for all eight pretrained systems, but not for the Custom CNN.
14. **Source/external trade-off:** Yes. CLIP and SigLIP2 probes win their family comparisons on source, while their zero-shot counterparts win on both external domains. CLIP probe's third-place source rank falls to eighth externally.
15. **Does high source accuracy predict robustness?** Not reliably in this nine-system cohort; the rank correlations are only moderate and source rank reversals are material.
16. **Best accuracy/compute/robustness compromise:** MobileNetV3 is the strongest lightweight compromise; SigLIP2 zero-shot is the strongest absolute robustness result at substantially greater CPU cost.

## Conclusions and boundaries

SigLIP2 strict P1 zero-shot remains first on both external domains and under the pre-registered worst-domain rule. The zero-shot advantage over source-trained heads replicates for both CLIP and SigLIP2. External-domain rankings correlate more strongly with each other than either does with source, and several source ranks change materially. These results support **robustness across two independently collected external fruit-image datasets**, not general real-world robustness.

Limitations include only two external datasets; six visible appearance labels; unequal domain sizes and class counts; retained official duplicate records; possible dependence among repeated capture sessions; pHash false positives under standardized composition; aggregate rather than causal domain profiling; CPU timing specific to this host; and no assessment of calibration, food safety, chemical contamination, microbiological spoilage, or formalin. FruitVision formalin-mixed images were excluded completely.
