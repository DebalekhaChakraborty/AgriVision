# Experiment 010 — DINOv2 ViT-B/14 Label Efficiency

## Objective

Measure the dataset-specific label efficiency and subset sensitivity of the frozen Experiment 005 DINOv2 representation between 1 and 100 labelled images per class.

## Encoder provenance

The unchanged `facebook/dinov2-base` checkpoint at revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` was reused through its Phase 3A embedding caches. The checkpoint file SHA-256 is `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`. The 768-dimensional embeddings retain the Phase 3A preprocessing identity and L2 normalization. The encoder remained frozen and was not rerun.

## Label budgets, subsets, and probe protocol

- Budgets: 1, 5, 10, 25, 50, and 100 images/class (6–600 total).
- Subset seeds: 42, 43, 44, 45, and 46; nested, stratified, balanced, sampled without replacement.
- Optimization seed: 42 for every run.
- Probe: `nn.Linear(768, 6)`, CrossEntropyLoss, Adam, learning rate 0.001, batch size 64, 50 epochs, no augmentation.
- Selection: highest validation accuracy, then lower validation loss.
- Test: every selected checkpoint evaluated once only after all 90 Phase 3C checkpoints and metadata were frozen.

Exact file selections and hashes are in `label_efficiency_manifests/`. No test labels informed training, subset construction, or selection.

## Aggregate results

| Labels/class | Accuracy mean ± std | 95% t interval | Weighted F1 | Rotten-apple recall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 81.14 ± 5.67 | [74.10, 88.18] | 80.63 | 63.83 |
| 5 | 92.97 ± 0.70 | [92.11, 93.84] | 92.99 | 87.42 |
| 10 | 93.40 ± 0.92 | [92.26, 94.53] | 93.42 | 86.56 |
| 25 | 94.20 ± 1.59 | [92.22, 96.17] | 94.20 | 85.22 |
| 50 | 92.22 ± 2.27 | [89.40, 95.04] | 92.19 | 74.31 |
| 100 | 92.59 ± 1.02 | [91.32, 93.86] | 92.56 | 76.04 |

The interval is the 95% interval across pre-registered subset draws, not complete predictive uncertainty. The frozen Experiment 005 full-data endpoint—reused, not rerun—is 89.07% accuracy, 89.01% weighted F1, and 66.06% rotten-apple recall.

## Class-level and variability findings

DINOv2 improved sharply from 1- to 5-shot and peaked at 25-shot, but the curve was not monotonic. Rotten-apple recall was especially sensitive and fell from 85.22% at 25-shot to 74.31% at 50-shot. The 5-shot mean already exceeds the frozen full-data endpoint; this reflects the fixed validation-selected training protocol and differing selected epochs, not evidence that fewer labels are inherently better. DINOv2 had the lowest 5-shot accuracy variance, but variability rose again at 25 and 50 shots.

## Limitations

These results apply only to this dataset, split, representation, and linear-probe protocol. Five draws do not capture complete uncertainty. The experiment does not establish general freshness sample complexity, causal effects of self-supervision, external-domain robustness, or food safety.
