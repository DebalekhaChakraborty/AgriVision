# Experiment 005 — DINOv2 ViT-B/14 Linear Probe

## Objective

Measure the visible fruit-freshness information encoded by the frozen, self-supervised DINOv2 ViT-B/14 representation.

## Registered Protocol

- Exact checkpoint and revision: `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`
- Deterministic model-native evaluation preprocessing; no augmentation
- One 768-dimensional L2-normalized embedding per image
- Frozen encoder; `nn.Linear(768, 6)` probe only
- Adam, learning rate 0.001, batch size 64, 50 epochs, seed 42
- Frozen V1 split; validation-only checkpoint selection; one test extraction/evaluation

## Results

Exactly one registered run was executed. The probe completed all 50 epochs and validation selection retained epoch 2 (99.26% validation accuracy, 1.2906 validation loss). The selected checkpoint was then evaluated once on all 2,698 held-out images.

| Metric | Value |
| --- | ---: |
| Test loss | 1.3254 |
| Accuracy | 89.07% |
| Macro precision / recall / F1 | 91.00% / 90.55% / 89.34% |
| Weighted precision / recall / F1 | 92.02% / 89.07% / 89.01% |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| fresh apple | 66.50% | 100.00% | 79.88% | 395 |
| fresh banana | 100.00% | 96.85% | 98.40% | 381 |
| fresh orange | 82.91% | 100.00% | 90.65% | 388 |
| rotten apple | 100.00% | 66.06% | 79.56% | 601 |
| rotten banana | 97.79% | 100.00% | 98.88% | 530 |
| rotten orange | 98.78% | 80.40% | 88.65% | 403 |

The largest errors were rotten apple predicted as fresh apple (198 images) and rotten orange predicted as fresh orange (78). The large validation/test gap cautions against treating the 270-image validation score as a generalization estimate.

## Reproducibility and compute

- Dataset: 1,539 train / 270 validation / 2,698 test; split-report SHA-256 `d6a532e2360384a1e23967f9a6a1d26204f983e57ad67d7cc901291132539bea`
- Encoder: 86,580,480 frozen parameters; 768-dimensional embedding
- Probe: 4,614 trainable parameters; 2.35 seconds training time
- Checkpoint: 346,345,912 bytes; SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`
- Test encoder forward: 93.90 ms/image; encoder + L2 normalization + probe: 93.93 ms/image
- Test-loop wall time: 270.98 seconds; observed peak process RSS: 1,406.56 MB
- Runtime: Python 3.11.2, PyTorch 2.13.0+cpu, Transformers 5.16.1, CPU-only 20-vCPU host

`history.json`, `test_metrics.json`, `classification_report.json`, `confusion_matrix.png`, learning curves, and the complete environment/config/cache audit are in `v2/results/experiment_005/`. The ignored embedding caches and probe checkpoint are identified by SHA-256 in `metadata.json`.

These results describe one seed, one fixed split, and one linear-probe protocol. They do not estimate uncertainty or isolate a causal effect of self-supervision, and visible appearance is not a food-safety measurement.
