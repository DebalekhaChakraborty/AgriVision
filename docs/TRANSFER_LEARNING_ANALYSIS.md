# V2 Phase 2 Transfer Learning Analysis

## Protocol Boundary

Experiments 002–004 use the frozen V1 train, validation, and test partitions, deterministic class mapping, seed 42, validation-only checkpoint selection, and exactly one held-out test pass per selected checkpoint. Their common preprocessing differs from Experiment 001: 224 × 224 ImageNet-normalized inputs, mild training augmentation, and deterministic resize plus center crop for validation and test.

## Comparison

| Experiment | Model | Training Strategy | Test Accuracy | Weighted F1 | Parameters | Observation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 001 | CNN | From scratch | 74.09% | 74.03% | 4,829,126 total/trainable | Control; substantial overfitting |
| 002 | ResNet50 | Frozen ImageNet transfer | 94.18% | 94.13% | 23,520,326 total; 12,294 trainable | Strong recall; largest and slowest transfer model |
| 003 | EfficientNet-B0 | Frozen ImageNet transfer | **94.55%** | **94.52%** | 4,015,234 total; 7,686 trainable | Best accuracy/F1 and strongest balance in this study |
| 004 | MobileNetV3-Large | Frozen ImageNet transfer | 93.03% | 92.98% | 2,977,718 total; 5,766 trainable | Smallest and fastest transfer model |

## Validation and Computational Results

| Experiment | Selected epoch | Validation accuracy | Validation loss | Training time | CPU forward time/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| 001 | 2 | 83.33% | 0.6173 | Not instrumented comparably | Not measured; test not reopened |
| 002 | 20 | 95.56% | 0.1400 | 24.90 min | 31.282 ms |
| 003 | 8 | 96.67% | 0.1351 | 14.68 min | 12.555 ms |
| 004 | 1 | 96.67% | 0.4372 | 10.67 min | 6.441 ms |

Phase 2 training time covers all 20 train/validation epochs. Inference timing measures only model forward passes on the same CPU; image decoding and transforms are excluded. Experiment 001 was frozen before this timing protocol, so its test set was not reopened merely to obtain a runtime number.

## Class-Level Recall

| Class | Exp. 001 CNN | Exp. 002 ResNet50 | Exp. 003 EfficientNet-B0 | Exp. 004 MobileNetV3-Large |
| --- | ---: | ---: | ---: | ---: |
| Fresh apple | 94.43% | 99.75% | 99.49% | 97.97% |
| Fresh banana | 77.95% | 99.74% | 100.00% | 100.00% |
| Fresh orange | 74.48% | 97.42% | 98.97% | 97.94% |
| Rotten apple | 51.41% | **91.18%** | 89.52% | 85.69% |
| Rotten banana | 93.40% | 95.85% | 95.28% | **97.36%** |
| Rotten orange | 58.56% | 82.63% | **86.85%** | 82.13% |

## Research Questions

### 1. Did pretrained features improve over the CNN trained from scratch?

Within this fixed dataset and protocol, all three transfer pipelines measured substantially higher held-out accuracy than Experiment 001: +20.09 points for ResNet50, +20.46 for EfficientNet-B0, and +18.94 for MobileNetV3-Large. Weighted-F1 changes were nearly identical. This supports the practical value of the registered pretrained representations here, but it is not a causal estimate of pretraining alone because model architecture and preprocessing changed too.

### 2. Which architecture provided the best accuracy?

EfficientNet-B0 had the highest test accuracy (94.55%), macro F1 (94.52%), and weighted F1 (94.52%). Its lead over ResNet50 was only 0.37 accuracy points, so one run does not establish a universal ranking.

### 3. Was the measured improvement worth the additional computation?

EfficientNet-B0 provided the strongest observed balance: it slightly exceeded ResNet50 while using about 5.86× fewer total parameters and measuring about 2.49× faster per image. MobileNetV3-Large traded 1.52 accuracy points relative to EfficientNet for approximately 1.95× faster forward passes and 1.35× fewer parameters. A direct Phase 1 runtime comparison is unavailable because Experiment 001 was frozen before the Phase 2 timing method was registered.

### 4. Which classes improved most?

The largest gains occurred in the classes that were weakest for Experiment 001. All transfer models substantially improved rotten apple, rotten orange, fresh orange, and fresh banana recall. Rotten banana already had 93.40% baseline recall and therefore showed smaller gains.

### 5. Did rotten-apple recall improve?

Yes. Recall increased from 51.41% in Experiment 001 to 91.18% with ResNet50, 89.52% with EfficientNet-B0, and 85.69% with MobileNetV3-Large. Confusion with fresh apple fell from 215 cases to 49, 44, and 63, respectively.

## Interpretation Boundary and Limitations

- These are three single-seed, frozen-backbone runs without fine-tuning or hyperparameter search.
- Phase 2 changes architecture, resolution, augmentation, and ImageNet normalization in addition to pretrained initialization. The comparison evaluates complete registered pipelines, not pretraining as an isolated causal variable.
- ImageNet contains fruit categories, so broad semantic overlap is expected; no claim is made that the representations are independent of all fruit imagery.
- CPU timing is local-environment evidence, not a portable latency guarantee. Data loading is excluded from per-image inference time.
- Repeated use of one benchmark test split across research phases supports comparison but should not become iterative model-selection feedback. No Phase 2 configuration was changed after test evaluation.
- These models classify visible appearance only and cannot establish food safety.
