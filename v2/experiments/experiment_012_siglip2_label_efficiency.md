# Experiment 012 — SigLIP2 Base Label Efficiency

## Objective

Measure the dataset-specific label efficiency and subset sensitivity of the frozen Experiment 007 SigLIP2 representation, and determine whether supervised linear adaptation adds to its strong strict-P1 zero-shot endpoint.

## Encoder provenance

The unchanged `google/siglip2-base-patch16-224` checkpoint at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` was reused through its Phase 3A embedding caches. The checkpoint file SHA-256 is `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`. The 768-dimensional embeddings retain the Phase 3A preprocessing identity and L2 normalization. The encoder remained frozen and was not rerun.

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
| 1 | 87.52 ± 5.67 | [80.48, 94.56] | 87.36 | 77.64 |
| 5 | 97.21 ± 0.75 | [96.29, 98.14] | 97.21 | 93.18 |
| 10 | 97.66 ± 0.44 | [97.11, 98.21] | 97.66 | 94.11 |
| 25 | 98.12 ± 0.49 | [97.52, 98.73] | 98.12 | 94.34 |
| 50 | 98.37 ± 0.19 | [98.13, 98.60] | 98.37 | 94.74 |
| 100 | 98.50 ± 0.17 | [98.29, 98.72] | 98.50 | 94.71 |

The interval is the 95% interval across pre-registered subset draws, not complete predictive uncertainty. The frozen strict-P1 zero-shot endpoint is 97.89% accuracy; the frozen Experiment 007 full-data endpoint is 98.63% accuracy and 98.62% weighted F1. Both were reused, not rerun.

## Class-level and variability findings

SigLIP2 led every supervised budget. It was effectively tied with DINOv2 for the lowest 1-shot variance and approached saturation by 5–10 labels/class. One-, five-, and ten-shot means did not exceed strict zero-shot; the 25-, 50-, and 100-shot means improved on it by 0.24, 0.48, and 0.61 point. Supervision therefore produced a modest high-budget gain, not a large one. Rotten-apple recall rose from 77.64% at 1-shot to 93.18% at 5-shot and remained near 94–95% thereafter.

## Limitations

The zero-shot text-prototype classifier and supervised linear probes are different decision mechanisms, so the 0-to-1 transition cannot be interpreted as label count alone. Results apply only to this dataset, split, representation, and protocol. Five draws do not capture complete uncertainty, external-domain robustness, or food safety.
