# Experiment 007 — SigLIP2 Base Linear Probe

## Objective

Measure the visible fruit-freshness information encoded by the frozen SigLIP2 Base Patch16/224 vision representation.

## Registered Protocol

- Exact checkpoint and revision: `google/siglip2-base-patch16-224@75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`
- Deterministic model-native evaluation preprocessing; no augmentation
- One 768-dimensional L2-normalized vision embedding per image
- Frozen vision encoder; `nn.Linear(768, 6)` probe only
- Adam, learning rate 0.001, batch size 64, 50 epochs, seed 42
- Frozen V1 split; validation-only checkpoint selection; one test extraction/evaluation
- No text input, prompt, or zero-shot classification

## Results

Exactly one registered run was executed. The probe completed all 50 epochs and validation selection retained epoch 50 (100.00% validation accuracy, 0.1412 validation loss). The selected checkpoint was then evaluated once on all 2,698 held-out images.

| Metric | Value |
| --- | ---: |
| Test loss | 0.1660 |
| Accuracy | 98.63% |
| Macro precision / recall / F1 | 98.51% / 98.90% / 98.69% |
| Weighted precision / recall / F1 | 98.67% / 98.63% / 98.62% |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| fresh apple | 96.34% | 100.00% | 98.14% | 395 |
| fresh banana | 99.74% | 99.74% | 99.74% | 381 |
| fresh orange | 99.23% | 99.74% | 99.49% | 388 |
| rotten apple | 100.00% | 94.84% | 97.35% | 601 |
| rotten banana | 99.62% | 99.81% | 99.72% | 530 |
| rotten orange | 96.15% | 99.26% | 97.68% | 403 |

SigLIP2 was the strongest Phase 3A system overall and on rotten apple. Its largest error groups were rotten apple predicted as fresh apple (15 images) and rotten apple predicted as rotten orange (15).

## Reproducibility and compute

- Dataset: 1,539 train / 270 validation / 2,698 test; split-report SHA-256 `d6a532e2360384a1e23967f9a6a1d26204f983e57ad67d7cc901291132539bea`
- Vision encoder: 92,884,224 frozen parameters; 768-dimensional pooled image embedding
- Probe: 4,614 trainable parameters; 4.20 seconds training time
- Combined SigLIP2 checkpoint: 1,500,800,904 bytes; SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- Test encoder forward: 82.32 ms/image; encoder + L2 normalization + probe: 82.34 ms/image
- Test-loop wall time: 238.15 seconds; observed peak process RSS: 1,506.88 MB
- Runtime: Python 3.11.2, PyTorch 2.13.0+cpu, Transformers 5.16.1, CPU-only 20-vCPU host

The combined checkpoint was loaded with its SigLIP-compatible top-level configuration; only the retained pretrained `vision_model` was frozen and executed for embeddings. No tokenizer, text encoder, prompt, or zero-shot path participated. Complete artifacts are in `v2/results/experiment_007/`; ignored caches and the probe checkpoint are content-hashed in `metadata.json`.

This is one seed, one split, and one fixed probe. The comparison cannot isolate the effect of image-text pretraining from architecture, data, scale, preprocessing, or other system differences, and visible appearance is not a food-safety measurement.
