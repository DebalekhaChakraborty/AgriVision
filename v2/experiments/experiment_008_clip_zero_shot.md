# Experiment 008 — Original CLIP ViT-B/16 Zero-Shot Classification

## Objective

Measure whether the original OpenAI CLIP image-text representation aligns with six visible fruit-freshness labels without fitting any model parameter or classifier on labelled target-dataset images.

## Frozen model and data

- Model: OpenCLIP `ViT-B-16`, pretrained identifier `openai`
- Pinned checkpoint: `timm/vit_base_patch16_clip_224.openai@977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d`
- Checkpoint SHA-256: `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`
- Model parameters: 149,620,737 total; 86,192,640 in the image encoder; all frozen
- Frozen dataset: validation 270, test 2,698; split-report SHA-256 `d6a532e2360384a1e23967f9a6a1d26204f983e57ad67d7cc901291132539bea`
- Train split access: none

The native CLIP evaluation transform and 512-dimensional projected image/text embeddings were used. Image and text vectors were L2-normalized. Similarity logits used the checkpoint's frozen scale of 100.0000 and zero bias. No classifier, prompt weight, temperature, or calibration parameter was trained.

## Prompt protocol

The complete P1–P4 registry was frozen before validation with SHA-256 `e902ba9a66fab36ff86c6440e34c241095d3f947e05c12d86a51d635ca2c67d7`. P1 is the strict canonical condition. Validation selection used macro F1, then accuracy, then the declared simplicity order P1–P4.

| Family | Validation accuracy | Validation macro F1 | Validation weighted F1 |
| --- | ---: | ---: | ---: |
| P1 canonical | 85.93% | 85.18% | 86.11% |
| P2 natural object | 87.41% | 86.72% | 87.62% |
| P3 condition description | **94.44%** | **93.77%** | **94.36%** |
| P4 equal template ensemble | 92.22% | 91.93% | 92.32% |

P3 was frozen as the validation-selected family before test access. P4 improved over P1 but did not beat P3.

## Single locked test run

Strict P1 and selected P3 were computed together from one image-encoding pass.

| Condition | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict P1 | 89.96% | 90.69% | 89.69% | 89.64% | 90.09% |
| Validation-selected P3 | **90.25%** | 89.83% | **91.09%** | **90.17%** | **90.23%** |

| Class | Strict precision / recall / F1 | Selected precision / recall / F1 |
| --- | ---: | ---: |
| fresh apple | 94.52% / 91.65% / 93.06% | 91.49% / 97.97% / 94.62% |
| fresh banana | 97.16% / 98.69% / 97.92% | 90.67% / 99.48% / 94.87% |
| fresh orange | 91.92% / 70.36% / 79.71% | 83.52% / 96.65% / 89.61% |
| rotten apple | 93.27% / 87.69% / 90.39% | 96.98% / 80.20% / 87.80% |
| rotten banana | 98.86% / 97.92% / 98.39% | 99.60% / 92.83% / 96.09% |
| rotten orange | 68.39% / 91.81% / 78.39% | 76.74% / 79.40% / 78.05% |

P3 improved the overall score only slightly and redistributed errors: fresh-orange recall rose sharply, while rotten-apple, rotten-banana, and rotten-orange recall fell.

## Semantic errors and margins

Strict P1 made 271 errors: 205 condition errors (75.65%), 62 fruit-identity errors (22.88%), and 4 both errors (1.48%). Selected P3 made 263 errors: 165 condition (62.74%), 91 identity (34.60%), and 7 both (2.66%). The largest strict confusion was `fresh_orange -> rotten_orange` (115); the largest selected confusion was `rotten_apple -> rotten_orange` (83).

Mean cosine top-1/top-2 margin was 0.01686 for correct strict predictions and 0.00643 for incorrect predictions. Under P3 it was 0.01642 versus 0.00811. The separation is descriptive only: cosine similarities and native logits are not calibrated probabilities.

## Compute and limitations

- P1 text prototypes: 0.206 seconds; P3 text prototypes: 0.182 seconds
- Image encoding: 79.268 ms/image on the CPU-only reference host
- Similarity classification: approximately 0.00022 ms/image
- End-to-end model forward after preprocessing: 79.268 ms/image
- Observed peak process RSS: 2,181.57 MB
- Training time: not applicable; nothing was trained

This is one dataset, split, checkpoint, and pre-registered prompt cohort. Validation-selected P3 is not pure strict zero-shot because validation labels chose it. Zero-shot means no target-dataset labels fitted model parameters; it does not mean CLIP lacked fruit or freshness concepts during pretraining. Visible-condition classification is not evidence of spoilage understanding or food safety.
