# Experiment 004 — MobileNetV3-Large Transfer Learning

## Objective

Measure a frozen ImageNet-pretrained MobileNetV3-Large representation against the completed from-scratch Experiment 001 control.

## Registered Protocol

- Frozen V1 leakage-safe split: 1,539 train, 270 validation, 2,698 test
- Pretrained weights: `MobileNet_V3_Large_Weights.IMAGENET1K_V2`
- Frozen backbone; one randomly initialized six-class linear head
- 224 × 224 input; mild registered train augmentation; ImageNet normalization
- Adam, learning rate 0.001, batch size 32, CrossEntropyLoss, seed 42, 20 epochs
- Validation-only checkpoint selection; one test pass after selection

## Results

One registered 20-epoch run completed on 2026-09-01. Validation selected epoch 1, after which the frozen test split was evaluated exactly once.

| Measure | Result |
| --- | ---: |
| Selected epoch | 1 |
| Best validation accuracy | 96.67% |
| Best validation loss | 0.4372 |
| Test loss | 0.4985 |
| Test accuracy | 93.03% |
| Macro precision | 92.95% |
| Macro recall | 93.52% |
| Macro F1 | 92.97% |
| Weighted F1 | 92.98% |
| Rotten-apple recall | 85.69% |

## Computational Profile

| Measure | Result |
| --- | ---: |
| Total parameters | 2,977,718 |
| Trainable parameters | 5,766 |
| Training time | 640.22 s (10.67 min) |
| CPU model-forward time | 6.441 ms/image |

Training time covers all 20 train/validation epochs. Per-image inference time is the accumulated model forward-pass time divided by 2,698 test images; decoding and transforms are excluded.

## Observations

- Held-out accuracy and weighted F1 exceeded Experiment 001 by 18.94 and 18.95 percentage points, respectively.
- Rotten-apple recall increased from 51.41% to 85.69%; 63 rotten apples were predicted as fresh apple.
- Fresh-banana recall was 100%, and rotten-banana recall was 97.36%.
- This was the smallest and fastest Phase 2 model. It was 1.52 percentage points less accurate than EfficientNet-B0 but measured about 1.95× faster per image.
- Epoch 1 remained selected because no later epoch matched its 96.67% validation accuracy; its 0.4372 validation loss is reported without overriding the primary metric.

## Artifacts

- History: `v2/results/experiment_004/history.json`
- Metrics and classification report: `v2/results/experiment_004/metrics.json`
- Learning curves and confusion matrix: `v2/results/experiment_004/`
- Metadata: `v2/experiments/experiment_004_metadata.json`
- Checkpoint: `v2/checkpoints/best_mobilenetv3.pt` (local and Git-ignored)

## Limitations

- This is one fixed frozen-backbone run, not a fine-tuning or hyperparameter study.
- Relative to Experiment 001, architecture, input resolution, augmentation, and normalization changed alongside pretraining; the observed difference is therefore not a causal estimate of pretraining alone.
- CPU timings are environment-specific and exclude data loading from the per-image inference measure.
- The result concerns visible condition on this dataset, not microbiological food safety or unrestricted deployment imagery.
