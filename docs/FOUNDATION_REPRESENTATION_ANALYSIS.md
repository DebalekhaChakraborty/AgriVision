# V2 Phase 3A Foundation Representation Analysis

## Controlled question and protocol

Phase 3A asks how much fruit-freshness information is available in three frozen foundation image representations. Each physical image was deterministically processed once with its model-native evaluation transform, encoded once, L2-normalized, and cached. The same `nn.Linear(embedding_dim, 6)` probe was then trained with Adam, learning rate 0.001, batch size 64, 50 epochs, and seed 42. Checkpoint selection used validation accuracy with lower validation loss as the tie-break. The held-out test set was opened once per selected probe.

No encoder parameter was trained. No augmentation, prompt, text input, zero-shot prediction, nonlinear probe, model-specific tuning, or repeated seed was used.

## Complete research ladder

| Exp | Model | Representation regime | Trainable params | Test accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: | ---: |
| 001 | Custom CNN | From scratch | 4,829,126 | 74.09% | 74.03% |
| 002 | ResNet50 | ImageNet transfer | 12,294 | 94.18% | 94.13% |
| 003 | EfficientNet-B0 | ImageNet transfer | 7,686 | 94.55% | 94.52% |
| 004 | MobileNetV3-Large | ImageNet transfer | 5,766 | 93.03% | 92.98% |
| 005 | DINOv2 ViT-B/14 | Self-supervised foundation | 4,614 | 89.07% | 89.01% |
| 006 | CLIP ViT-B/16 | Vision-language contrastive | 3,078 | 95.85% | 95.84% |
| 007 | SigLIP2 Base | Modern vision-language foundation | 4,614 | **98.63%** | **98.62%** |

The first four rows are frozen Phase 1/2 evidence and were neither changed nor rerun. Phase 2 and Phase 3A are informative but not perfectly controlled against each other: Phase 2 uses ImageNet backbones with stochastic train-time image augmentation, while Phase 3A uses one deterministic cached embedding per image. Within Phase 3A, however, the downstream probe and selection protocol are identical.

## Foundation-model test results

| Exp | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 005 DINOv2 | 89.07% | 91.00% | 90.55% | 89.34% | 92.02% | 89.07% | 89.01% |
| 006 CLIP | 95.85% | 95.67% | 96.42% | 95.90% | 96.14% | 95.85% | 95.84% |
| 007 SigLIP2 | **98.63%** | **98.51%** | **98.90%** | **98.69%** | **98.67%** | **98.63%** | **98.62%** |

## Per-class recall

| Class | Exp 001 | Best Phase 2 | DINOv2 | CLIP | SigLIP2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh apple | 94.43% | 99.75% | **100.00%** | 99.75% | **100.00%** |
| fresh banana | 77.95% | **100.00%** | 96.85% | 99.74% | 99.74% |
| fresh orange | 74.48% | 98.97% | **100.00%** | 98.45% | 99.74% |
| rotten apple | 51.41% | 91.18% | 66.06% | 87.52% | **94.84%** |
| rotten banana | 93.40% | 97.36% | **100.00%** | 99.25% | 99.81% |
| rotten orange | 58.56% | 86.85% | 80.40% | 93.80% | **99.26%** |

“Best Phase 2” is the highest recall among Experiments 002–004 for that class, not one selected Phase 2 model. DINOv2's main errors were rotten apple predicted as fresh apple (198/601) and rotten orange predicted as fresh orange (78/403). CLIP reduced those confusions to 59 and 23 respectively. SigLIP2 reduced them to 15 and 3; its other rotten-apple errors were primarily rotten orange (15/601).

Relative to the best Phase 2 class recall, DINOv2 regressed on fresh banana, rotten apple, and rotten orange. CLIP regressed slightly on fresh banana and fresh orange and more materially on rotten apple, while improving rotten banana and rotten orange. SigLIP2 exceeded the best Phase 2 recall for five classes and was 0.26 percentage points below the 100% Phase 2 fresh-banana recall.

## Computational comparison

All timings below were measured on the same 20-vCPU, CPU-only host. “Forward” is model forward time after preprocessing; the full Phase 3A path includes encoder, L2 normalization, and the linear probe. Wall-clock input loading/preprocessing is excluded from the forward figure.

| Exp | Encoder params | Embedding | Checkpoint file | Selected epoch | Probe time | Test forward | Observed peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 005 DINOv2 | 86,580,480 | 768 | 346.35 MB | 2 | 2.35 s | 93.93 ms/image | 1,406.56 MB |
| 006 CLIP | 86,192,640 | 512 | 598.52 MB | 50 | 2.15 s | **79.02 ms/image** | 2,457.46 MB |
| 007 SigLIP2 | 92,884,224 | 768 | 1,500.80 MB | 50 | 4.20 s | 82.34 ms/image | 1,506.88 MB |

The checkpoint sizes are not equivalent encoder-size measures: CLIP and SigLIP2 files are combined image-text checkpoints, whereas only their vision path participates in this benchmark. Peak RSS includes framework and model-loading behavior and should be treated as an observed process maximum, not an architecture-only memory requirement.

EfficientNet-B0 remains highly competitive: its 94.55% accuracy is 1.30 points below CLIP and 4.08 points below SigLIP2, but its recorded 12.56 ms/image forward time is about 6.3 times faster than CLIP and 6.6 times faster than SigLIP2. Its 4,015,234 total parameters are also roughly 21–23 times fewer than the foundation vision encoders. DINOv2 was both slower and 5.49 accuracy points below EfficientNet-B0 in these runs.

## Answers to the research questions

1. **Do frozen foundation representations outperform the from-scratch CNN?** Yes in all three observed runs. Accuracy improved by 14.97 points for DINOv2, 21.76 for CLIP, and 24.54 for SigLIP2 over Experiment 001.

2. **Do they outperform frozen ImageNet-transfer representations?** Not uniformly. DINOv2 trailed every Phase 2 model. CLIP exceeded the strongest Phase 2 accuracy by 1.30 points, and SigLIP2 exceeded it by 4.08 points.

3. **Which foundation representation performs best?** SigLIP2, at 98.63% accuracy and 98.62% weighted F1.

4. **Which performs best on rotten apple?** SigLIP2, with 94.84% recall and 97.35% F1. It also exceeded the best Phase 2 rotten-apple recall of 91.18%.

5. **Does semantic vision-language pretraining appear beneficial?** CLIP and SigLIP2 were stronger than DINOv2 here, including on visible degradation. That is a useful association, not causal evidence that language supervision produced the gain.

6. **How does self-supervised DINOv2 compare with CLIP/SigLIP2?** DINOv2 was 6.78 accuracy points below CLIP and 9.56 below SigLIP2. Its fresh/rotten boundary for apples and oranges transferred poorly relative to the two image-text systems despite near-perfect validation accuracy.

7. **How do size, cost, and performance relate?** Parameter count alone did not order the models: similarly sized DINOv2 and CLIP differed by 6.78 accuracy points, while CLIP was fastest among the three. SigLIP2 was largest and most accurate, but the single cohort is too small to infer a general scaling law.

8. **Is EfficientNet-B0 still competitive?** Yes. It nearly matched CLIP while using far fewer parameters and substantially less CPU forward time, and it beat DINOv2.

9. **Did foundation models regress on any classes?** Yes. DINOv2 regressed materially on several Phase 2 class recalls. CLIP remained below the best Phase 2 rotten-apple recall and had small fresh-banana/fresh-orange regressions. SigLIP2's only class-level recall regression against the per-class Phase 2 maximum was a 0.26-point fresh-banana difference.

10. **What can one single-seed linear-probe benchmark establish?** It can rank these exact frozen representation systems under this registered protocol and split. It cannot estimate seed variance, confidence intervals, label efficiency, zero-shot ability, fine-tuning potential, calibration, cross-dataset robustness, or real-world food-safety performance.

## Causal and external-validity limitations

The models differ simultaneously in architecture, parameter count, pretraining objective, pretraining data, preprocessing, and embedding dimension. The experiment therefore compares complete pretrained representation systems; it does not isolate an effect caused solely by self-supervision or vision-language training. Phase 2 also differs in augmentation and backbone family.

The validation set contains only 270 images and the held-out source-test distribution is not interchangeable with it; DINOv2's 99.26% validation accuracy but 89.07% test accuracy illustrates that gap. The dataset contains only six labels and includes stock-like photographs and transformation families. Results are from one seed, one split, one CPU host, and one fixed probe protocol. Visible freshness labels do not measure pathogens, toxins, internal spoilage, smell, or microbiological food safety.

Exact machine-readable metrics, class reports, confusion matrices, histories, timings, cache identities, and selected-checkpoint hashes are stored under `v2/results/experiment_005/`, `experiment_006/`, and `experiment_007/`.
