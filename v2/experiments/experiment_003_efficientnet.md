# Experiment 003 — EfficientNet-B0 Transfer Learning

## Objective

Measure a frozen ImageNet-pretrained EfficientNet-B0 representation against the completed from-scratch Experiment 001 control.

## Registered Protocol

- Frozen V1 leakage-safe split: 1,539 train, 270 validation, 2,698 test
- Pretrained weights: `EfficientNet_B0_Weights.IMAGENET1K_V1`
- Frozen backbone; one randomly initialized six-class linear head
- 224 × 224 input; mild registered train augmentation; ImageNet normalization
- Adam, learning rate 0.001, batch size 32, CrossEntropyLoss, seed 42, 20 epochs
- Validation-only checkpoint selection; one test pass after selection

## Results

One registered 20-epoch run completed on 2026-09-01. Validation selected epoch 8, after which the frozen test split was evaluated exactly once.

| Measure | Result |
| --- | ---: |
| Selected epoch | 8 |
| Best validation accuracy | 96.67% |
| Best validation loss | 0.1351 |
| Test loss | 0.1525 |
| Test accuracy | 94.55% |
| Macro precision | 94.33% |
| Macro recall | 95.02% |
| Macro F1 | 94.52% |
| Weighted F1 | 94.52% |
| Rotten-apple recall | 89.52% |

## Computational Profile

| Measure | Result |
| --- | ---: |
| Total parameters | 4,015,234 |
| Trainable parameters | 7,686 |
| Training time | 880.92 s (14.68 min) |
| CPU model-forward time | 12.555 ms/image |

Training time covers all 20 train/validation epochs. Per-image inference time is the accumulated model forward-pass time divided by 2,698 test images; decoding and transforms are excluded.

## Observations

- This experiment had the highest test accuracy and weighted F1 in Phase 2, exceeding Experiment 001 by 20.46 and 20.49 percentage points, respectively.
- Rotten-apple recall increased from 51.41% to 89.52%; 44 rotten apples were predicted as fresh apple.
- Fresh-banana recall was 100%, and fresh-orange recall was 98.97%.
- Rotten-orange recall remained the weakest class result at 86.85%, though it was the strongest rotten-orange result in Phase 2.
- Compared with ResNet50, EfficientNet-B0 was 0.37 percentage points more accurate in this run, used about one-sixth as many total parameters, and measured about 2.49× faster per image on the same CPU.

## Artifacts

- History: `v2/results/experiment_003/history.json`
- Metrics and classification report: `v2/results/experiment_003/metrics.json`
- Learning curves and confusion matrix: `v2/results/experiment_003/`
- Metadata: `v2/experiments/experiment_003_metadata.json`
- Checkpoint: `v2/checkpoints/best_efficientnet.pt` (local and Git-ignored)

## Limitations

- This is one fixed frozen-backbone run, not a fine-tuning or hyperparameter study.
- Relative to Experiment 001, architecture, input resolution, augmentation, and normalization changed alongside pretraining; the observed difference is therefore not a causal estimate of pretraining alone.
- CPU timings are environment-specific and exclude data loading from the per-image inference measure.
- The result concerns visible condition on this dataset, not microbiological food safety or unrestricted deployment imagery.
