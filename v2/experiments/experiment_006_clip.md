# Experiment 006 — Original CLIP ViT-B/16 Linear Probe

## Objective

Measure the visible fruit-freshness information encoded by the frozen image tower from the original OpenAI CLIP ViT-B/16 generation.

## Registered Protocol

- OpenCLIP model/pretrained tag: `ViT-B-16` / `openai`
- Pinned mirror: `timm/vit_base_patch16_clip_224.openai@977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d`
- Deterministic original CLIP evaluation preprocessing; no augmentation
- One 512-dimensional L2-normalized image embedding per image
- Frozen encoder; `nn.Linear(512, 6)` probe only
- Adam, learning rate 0.001, batch size 64, 50 epochs, seed 42
- Frozen V1 split; validation-only checkpoint selection; one test extraction/evaluation
- No text input, prompt, or zero-shot classification

## Results

Exactly one registered run was executed. The probe completed all 50 epochs and validation selection retained epoch 50 (98.52% validation accuracy, 0.2717 validation loss). The selected checkpoint was then evaluated once on all 2,698 held-out images.

| Metric | Value |
| --- | ---: |
| Test loss | 0.3181 |
| Accuracy | 95.85% |
| Macro precision / recall / F1 | 95.67% / 96.42% / 95.90% |
| Weighted precision / recall / F1 | 96.14% / 95.85% / 95.84% |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| fresh apple | 86.98% | 99.75% | 92.92% | 395 |
| fresh banana | 98.96% | 99.74% | 99.35% | 381 |
| fresh orange | 94.32% | 98.45% | 96.34% | 388 |
| rotten apple | 99.43% | 87.52% | 93.10% | 601 |
| rotten banana | 99.81% | 99.25% | 99.53% | 530 |
| rotten orange | 94.50% | 93.80% | 94.15% | 403 |

The largest remaining errors were rotten apple predicted as fresh apple (59 images) and rotten orange predicted as fresh orange (23). The result exceeded every Phase 2 model in overall accuracy, but its rotten-apple recall remained below ResNet50's 91.18%.

## Reproducibility and compute

- Dataset: 1,539 train / 270 validation / 2,698 test; split-report SHA-256 `d6a532e2360384a1e23967f9a6a1d26204f983e57ad67d7cc901291132539bea`
- Vision encoder: 86,192,640 frozen parameters; 512-dimensional projected image embedding
- Probe: 3,078 trainable parameters; 2.15 seconds training time
- Combined CLIP checkpoint: 598,516,980 bytes; SHA-256 `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f`
- Test encoder forward: 79.00 ms/image; encoder + L2 normalization + probe: 79.02 ms/image
- Test-loop wall time: 234.70 seconds; observed peak process RSS: 2,457.46 MB
- Runtime: Python 3.11.2, PyTorch 2.13.0+cpu, OpenCLIP 3.3.0, CPU-only 20-vCPU host

`history.json`, `test_metrics.json`, `classification_report.json`, `confusion_matrix.png`, learning curves, and the complete environment/config/cache audit are in `v2/results/experiment_006/`. The ignored embedding caches and probe checkpoint are identified by SHA-256 in `metadata.json`.

No tokenizer, text prompt, or text embedding participated. The result is from one seed and one split and cannot by itself attribute the advantage to language supervision or establish zero-shot ability, robustness, or food safety.
