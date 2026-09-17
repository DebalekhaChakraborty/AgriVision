# Experiment 002 — ResNet50 Transfer Learning

## Objective

Measure a frozen ImageNet-pretrained ResNet50 representation against the completed from-scratch Experiment 001 control.

## Registered Protocol

- Frozen V1 leakage-safe split: 1,539 train, 270 validation, 2,698 test
- Pretrained weights: `ResNet50_Weights.IMAGENET1K_V2`
- Frozen backbone; one randomly initialized six-class linear head
- 224 × 224 input; mild registered train augmentation; ImageNet normalization
- Adam, learning rate 0.001, batch size 32, CrossEntropyLoss, seed 42, 20 epochs
- Validation-only checkpoint selection; one test pass after selection

## Results

One registered 20-epoch run completed on 2026-09-01. Validation selected epoch 20, after which the frozen test split was evaluated exactly once.

| Measure | Result |
| --- | ---: |
| Selected epoch | 20 |
| Best validation accuracy | 95.56% |
| Best validation loss | 0.1400 |
| Test loss | 0.1595 |
| Test accuracy | 94.18% |
| Macro precision | 94.15% |
| Macro recall | 94.43% |
| Macro F1 | 94.10% |
| Weighted F1 | 94.13% |
| Rotten-apple recall | 91.18% |

## Computational Profile

| Measure | Result |
| --- | ---: |
| Total parameters | 23,520,326 |
| Trainable parameters | 12,294 |
| Training time | 1,494.16 s (24.90 min) |
| CPU model-forward time | 31.282 ms/image |

Training time covers all 20 train/validation epochs. Per-image inference time is the accumulated model forward-pass time divided by 2,698 test images; decoding and transforms are excluded.

## Observations

- Held-out accuracy was 20.09 percentage points above Experiment 001 in this registered comparison.
- Rotten-apple recall increased from 51.41% to 91.18%; 49 rotten apples were still predicted as fresh apple.
- Fresh-apple and fresh-banana recall were 99.75% and 99.74%, respectively.
- Rotten-orange recall remained the weakest class result at 82.63%.

## Artifacts

- History: `v2/results/experiment_002/history.json`
- Metrics and classification report: `v2/results/experiment_002/metrics.json`
- Learning curves and confusion matrix: `v2/results/experiment_002/`
- Metadata: `v2/experiments/experiment_002_metadata.json`
- Checkpoint: `v2/checkpoints/best_resnet50.pt` (local and Git-ignored)

## Limitations

- This is one fixed frozen-backbone run, not a fine-tuning or hyperparameter study.
- Relative to Experiment 001, architecture, input resolution, augmentation, and normalization changed alongside pretraining; the observed difference is therefore not a causal estimate of pretraining alone.
- CPU timings are environment-specific and exclude data loading from the per-image inference measure.
- The result concerns visible condition on this dataset, not microbiological food safety or unrestricted deployment imagery.
