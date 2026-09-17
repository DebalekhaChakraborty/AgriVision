# Experiment 014 — Multi-Domain Robustness Benchmark

## Research question

Are the model-generalization conclusions observed on the frozen Sultana benchmark stable across a second independently collected fruit-image domain?

## Locked three-domain protocol

- Domain S: frozen Kalluri source-test results, reused without inference.
- Domain A: frozen Sultana Experiment 013 results for 1,200 originals, reused without inference.
- Domain B: FruitVision Mendeley Data Version 2, DOI `10.17632/xkbjx8959c.2`; all 4,185 raw originals in the six overlapping apple/banana/orange × fresh/rotten classes.
- Excluded: all augmented, formalin-mixed, grape, and mango images.
- FruitVision manifest: `fruitvision_manifest.json`, SHA-256 `71651aee8816b01baaa1a752f1bcc6327ae26e98578ed95c2150276ad445cb3f`, frozen before inference.
- Target-domain use: evaluation only; no training, validation, calibration, prompt selection, prototype fitting, or adaptation.
- Systems: the exact nine frozen Experiment 013 systems, with all identity checks passed.
- FruitVision access: one inference pass per system, preserving each system's frozen preprocessing.
- Uncertainty: seed 42, 5,000 class-stratified bootstrap resamples.
- Ranking: worst external accuracy first, mean external accuracy second; frozen before FruitVision result inspection.
- Paired inference: exactly four pre-registered FruitVision comparisons.

## Integrity gates

Official metadata and archive packaging established raw-image provenance and the CC BY-NC-ND 4.0 dataset licence independently from the associated article licence. The 898,826,798-byte original archive matched SHA-256 `c72a8d8d9b1f44f8f4a7f31e1420da7f356de5eb4f18e698bbe692ca4d1a18fc` and passed CRC validation. All selected images decoded as RGB JPEG; class counts matched the publication.

The audit found 28 exact/pixel-equivalent duplicate groups, involving 0.669% of records under the pre-registered secondary-record calculation—below the 10% severe-duplication stop threshold. They were preserved as published records. Cross-domain screening found zero byte-identical and zero decoded-pixel-identical FruitVision matches against both source and Sultana. All 49 cross-domain pHash candidates (threshold ≤6) were visually reviewed and rejected as different photographs. No contamination exclusion was required.

## FruitVision results

| Condition | System | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| 014-A | Custom CNN | 38.95% | 34.79% | 35.25% |
| 014-B | ResNet50 | 76.20% | 73.85% | 74.19% |
| 014-C | EfficientNet-B0 | 70.30% | 68.53% | 68.77% |
| 014-D | MobileNetV3-Large | 70.37% | 67.32% | 67.46% |
| 014-E | DINOv2 + full-source linear probe | 66.38% | 61.27% | 61.75% |
| 014-F | CLIP + full-source linear probe | 55.24% | 46.07% | 44.81% |
| 014-G | SigLIP2 + full-source linear probe | 73.52% | 69.00% | 68.78% |
| 014-H | CLIP strict P1 zero-shot | 67.43% | 63.79% | 64.47% |
| 014-I | SigLIP2 strict P1 zero-shot | **86.31%** | **85.59%** | **85.95%** |

SigLIP2 strict zero-shot has 88.53% mean external accuracy and 86.31% worst external accuracy, leading the pre-registered robustness ranking. CLIP zero-shot beats its probe on both Sultana (+19.75 points) and FruitVision (+12.19); SigLIP2 zero-shot also beats its probe on both (+8.00 and +12.78).

## Result boundary

Experiment 014 provides evidence across two independently collected external fruit-image datasets, not proof of general real-world robustness. It evaluates visible fresh/rotten appearance only. It makes no claim about edibility, food safety, pathogens, toxins, internal spoilage, chemical contamination, preservatives, or formalin detection.
