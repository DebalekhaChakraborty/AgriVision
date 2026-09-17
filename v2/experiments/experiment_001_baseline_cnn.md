# Experiment 001 — Modern CNN Baseline Reproduction

## Objective

Modern reproduction of the V1 from-scratch CNN experiment using PyTorch while preserving the scientific question, frozen data split, input size, architecture concept, optimizer family, batch size, and epoch count.

## Environment

- Python: 3.11.2
- PyTorch: 2.13.0+cpu
- torchvision: 0.28.0+cpu
- Hardware/device: Intel Xeon 2.20 GHz CPU, 20 logical CPUs; CUDA unavailable
- Platform: Linux x86_64
- Started: 2026-09-01 10:57:15 UTC
- Training completed: 2026-09-01 11:05:39 UTC
- Test evaluated: 2026-09-01 11:06:30 UTC

## Dataset

- Dataset: Kalluri, *Fruits fresh and rotten for classification*, Kaggle version 1
- Classes: six frozen V1 classes
- Split: frozen leakage-safe V1 train/validation/test manifests; no resplitting
- Counts: 1,539 train, 270 validation, 2,698 test
- Raw images: local only and excluded from Git

## Architecture

Three unpadded 3×3 convolution blocks with 32, 64, and 128 channels, each followed by ReLU and 2×2 max pooling; flatten; dense 128 with ReLU; six-class logits. No dropout, batch normalization, attention, residual connection, pretrained weight, or modern regularizer.

## Hyperparameters

| Parameter | Registered value |
| --- | --- |
| Image size | 150 × 150 RGB |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Seed | 42 |

## Results

One registered 20-epoch run was completed. The checkpoint was selected only from validation performance. The frozen test split was opened once after selection and evaluated in one inference pass.

| Measure | Observed result |
| --- | ---: |
| Selected epoch | 2 |
| Training accuracy at selected epoch | 81.61% |
| Best validation accuracy | 83.33% |
| Best validation loss | 0.6173 |
| Test loss | 0.7288 |
| Test accuracy | 74.09% |
| Test macro precision | 77.53% |
| Test macro recall | 75.04% |
| Test macro F1 | 74.57% |
| Test weighted precision | 77.20% |
| Test weighted recall | 74.09% |
| Test weighted F1 | 74.03% |

The selected checkpoint SHA-256 is `37e582bb8ec034b04d26ab6f2a6168fc251d9d154ae4bf711b1a8517be33e96d`. Exact histories, per-class metrics, the full classification report, and the confusion-matrix values are stored in the generated JSON artifacts.

## Observations

- Training accuracy reached 100% by epoch 13 and remained there, while validation accuracy did not exceed the epoch-2 maximum. Validation loss increased from 0.6173 at selection to 1.5652 at epoch 20. This is strong evidence of overfitting in this fixed run.
- Rotten banana had the highest recall (93.40%) and F1 (89.27%).
- Rotten apple had the lowest recall (51.41%); 215 of 601 rotten-apple images were predicted as fresh apple.
- Fresh apple had high recall (94.43%) but low precision (51.66%), consistent with the model over-predicting that class.
- These results establish the controlled PyTorch baseline. They do not establish an accuracy improvement over V1: the V1 selected dropout model used a materially different architecture, while the closer V1 no-dropout baseline was produced under different framework defaults and numerical behavior.

## Limitations

- This is one fixed run, not a hyperparameter search.
- Framework and default initialization differences prevent bitwise equivalence with V1.
- The comparison isolates a modern implementation of the same CNN concept; it does not isolate every low-level numerical difference.
- The training set is small after conservative transformation-family leakage removal, and the unregularized network has 4,829,126 trainable parameters.
- The original test partition may differ from deployment imagery in lighting, background, camera, viewpoint, and fruit varieties.
- The dataset license is unknown, so the raw images and trained checkpoint remain local and are not redistributed through Git.
- The dataset remains a visible-condition benchmark, not a food-safety dataset.
