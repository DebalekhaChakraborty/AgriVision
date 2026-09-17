# Experiment 011 — Original CLIP ViT-B/16 Label Efficiency

## Objective

Measure the dataset-specific label efficiency and subset sensitivity of the frozen Experiment 006 original CLIP image representation, and compare its supervised curve with the frozen strict-P1 zero-shot endpoint.

## Encoder provenance

The unchanged `timm/vit_base_patch16_clip_224.openai` checkpoint at revision `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` was reused through its Phase 3A embedding caches. The checkpoint file SHA-256 is `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`. The 512-dimensional embeddings retain the Phase 3A OpenCLIP preprocessing identity and L2 normalization. The encoder remained frozen and was not rerun.

## Label budgets, subsets, and probe protocol

- Budgets: 1, 5, 10, 25, 50, and 100 images/class (6–600 total).
- Subset seeds: 42, 43, 44, 45, and 46; nested, stratified, balanced, sampled without replacement.
- Optimization seed: 42 for every run.
- Probe: `nn.Linear(512, 6)`, CrossEntropyLoss, Adam, learning rate 0.001, batch size 64, 50 epochs, no augmentation.
- Selection: highest validation accuracy, then lower validation loss.
- Test: every selected checkpoint evaluated once only after all 90 Phase 3C checkpoints and metadata were frozen.

Exact file selections and hashes are in `label_efficiency_manifests/`. No test labels informed training, subset construction, or selection.

## Aggregate results

| Labels/class | Accuracy mean ± std | 95% t interval | Weighted F1 | Rotten-apple recall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 74.28 ± 10.21 | [61.60, 86.97] | 72.24 | 74.64 |
| 5 | 89.85 ± 2.38 | [86.89, 92.81] | 89.84 | 84.19 |
| 10 | 91.02 ± 1.82 | [88.77, 93.28] | 91.02 | 85.09 |
| 25 | 94.37 ± 0.68 | [93.52, 95.22] | 94.38 | 89.38 |
| 50 | 94.63 ± 0.47 | [94.05, 95.21] | 94.63 | 87.55 |
| 100 | 95.30 ± 0.17 | [95.09, 95.51] | 95.29 | 88.22 |

The interval is the 95% interval across pre-registered subset draws, not complete predictive uncertainty. The frozen strict-P1 zero-shot endpoint is 89.96% accuracy; the frozen Experiment 006 full-data endpoint is 95.85% accuracy and 95.84% weighted F1. Both were reused, not rerun.

## Class-level and variability findings

CLIP was highly subset-sensitive at 1-shot, especially for fresh-apple, rotten-apple, and rotten-banana recall. Accuracy variance fell consistently as labels increased. Five-shot performance was essentially equal to strict zero-shot, while 10-shot and higher linear probes increasingly exceeded it; the 100-shot mean was 5.34 points above strict zero-shot and 0.55 point below full-data.

## Limitations

The zero-shot text-prototype classifier and supervised linear probes are different decision mechanisms, so the 0-to-1 transition cannot be interpreted as label count alone. Results apply only to this dataset, split, representation, and protocol. Five draws do not capture complete uncertainty, external-domain robustness, or food safety.
