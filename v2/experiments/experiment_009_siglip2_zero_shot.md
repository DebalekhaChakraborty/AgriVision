# Experiment 009 — SigLIP2 Base Zero-Shot Classification

## Objective

Measure whether the SigLIP2 Base image-text representation aligns with six visible fruit-freshness labels without fitting any model parameter or classifier on labelled target-dataset images.

## Frozen model and data

- Model: `google/siglip2-base-patch16-224`
- Pinned revision: `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`
- Checkpoint SHA-256: `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- Model parameters: 375,187,970 total; 92,884,224 image, 282,303,744 text, two similarity scalars; all frozen
- Frozen dataset: validation 270, test 2,698; split-report SHA-256 `d6a532e2360384a1e23967f9a6a1d26204f983e57ad67d7cc901291132539bea`
- Train split access: none

The native 224-pixel transform, 256,000-token Gemma tokenizer, and 768-dimensional projected image/text embeddings were used. Image and text vectors were L2-normalized. Similarity logits used the checkpoint's frozen scale 112.6689 and bias −16.7717. Nothing was trained or calibrated.

## Prompt protocol

The same P1–P4 registry was frozen before validation with SHA-256 `e902ba9a66fab36ff86c6440e34c241095d3f947e05c12d86a51d635ca2c67d7`. Selection used validation macro F1, accuracy, and prompt simplicity in that order.

| Family | Validation accuracy | Validation macro F1 | Validation weighted F1 |
| --- | ---: | ---: | ---: |
| P1 canonical | 98.89% | 98.76% | **98.89%** |
| P2 natural object | **98.89%** | **98.85%** | 98.89% |
| P3 condition description | 98.89% | 98.80% | 98.89% |
| P4 equal template ensemble | 98.52% | 98.34% | 98.52% |

P2 won the primary macro-F1 metric by 0.09 points over P1 and was frozen before test access. P4 did not improve validation performance.

## Single locked test run

Strict P1 and selected P2 were computed together from one image-encoding pass.

| Condition | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict P1 | **97.89%** | **97.69%** | **98.11%** | **97.86%** | **97.89%** |
| Validation-selected P2 | 96.96% | 96.78% | 97.39% | 96.99% | 96.96% |

| Class | Strict precision / recall / F1 | Selected precision / recall / F1 |
| --- | ---: | ---: |
| fresh apple | 93.38% / 100.00% / 96.58% | 89.37% / 100.00% / 94.38% |
| fresh banana | 98.70% / 99.74% / 99.22% | 98.70% / 99.74% / 99.22% |
| fresh orange | 95.77% / 99.23% / 97.47% | 95.07% / 99.48% / 97.23% |
| rotten apple | 100.00% / 94.84% / 97.35% | 100.00% / 91.01% / 95.30% |
| rotten banana | 99.81% / 99.06% / 99.43% | 99.81% / 99.06% / 99.43% |
| rotten orange | 98.47% / 95.78% / 97.11% | 97.70% / 95.04% / 96.35% |

The tiny validation advantage for P2 did not transfer to the held-out distribution: it added 25 errors and reduced test accuracy by 0.93 points relative to strict P1. The registered selection was retained and reported rather than revised after test access.

## Semantic errors and margins

Strict P1 made 57 errors: 54 condition errors (94.74%), 3 fruit-identity errors (5.26%), and no both errors. Selected P2 made 82 errors: 75 condition (91.46%), 7 identity (8.54%), and no both errors. The largest confusion was rotten apple predicted as fresh apple: 28 under P1 and 47 under P2.

Mean cosine margin was 0.04291 for correct strict predictions and 0.00909 for incorrect predictions. Under P2 it was 0.04085 versus 0.00968. These scores are not calibrated probabilities.

## Compute and limitations

- P1 text prototypes: 0.323 seconds; P2 text prototypes: 0.279 seconds
- Image encoding: 80.675 ms/image on the CPU-only reference host
- Similarity classification: approximately 0.00035 ms/image for P1 and 0.00028 for P2
- End-to-end model forward after preprocessing: 80.675 ms/image
- Observed peak process RSS: 2,622.43 MB
- Training time: not applicable; nothing was trained

This is one dataset, split, checkpoint, and prompt cohort. P2 is validation-selected zero-shot, not strict zero-shot. The result does not establish spoilage understanding, calibrated confidence, unseen pretraining concepts, cross-domain robustness, or food safety.
