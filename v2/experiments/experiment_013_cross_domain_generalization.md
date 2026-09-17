# Experiment 013 — Zero-Adaptation Cross-Domain Generalization

## Research question

How well do systems developed on the frozen source fruit-freshness dataset generalize to independently collected fruit images without target-domain adaptation?

## Frozen protocol

- Branch: `master` only.
- External release: Sultana, Jahan, and Uddin, Mendeley Data v1, DOI `10.17632/bdd69gyhv8.1`, CC BY 4.0.
- Evaluation population: all 1,200 original physical photographs in the six overlapping apple/banana/orange × fresh/rotten classes.
- Excluded: the entire augmented archive and all ten non-overlapping class folders.
- Manifest frozen before checkpoint access: SHA-256 `6fd074e514f3b79e6d4f1878516fdaa9a4ea2e283a2dee4453a397d9da5a7208`.
- External subsets: evaluation only; no training, validation, calibration, prompt selection, or adaptation.
- Systems: the selected checkpoints from Experiments 001–007 and strict P1 zero-shot conditions 008A/009A.
- Inference: one complete pass per system in fixed manifest order.
- Uncertainty: seed 42, 5,000 within-class bootstrap resamples preserving 200 observations per class.
- Paired inference: exactly the four comparisons pre-registered in `cross_domain_generalization.yaml`.

## Integrity gates

The image audit decoded all 1,200 files with no corruptions, unexpected extensions, zero-byte files, or class-count errors. It found seven exact duplicate pairs internal to the officially published FreshOrange originals; these were retained as distinct published records and are a limitation on sample independence. Cross-dataset screening found no byte-identical or decoded-pixel-identical images. All five pairs flagged at the pre-registered pHash distance threshold of six were visually reviewed and documented as different photographs, so no contamination blocker remained.

All nine system identity checks passed before inference. These cover checkpoint/probe SHA-256, encoder checkpoint SHA-256 and revision, preprocessing identity, canonical class order, and the strict prompt-registry SHA-256 where applicable.

## Conditions and external accuracy

| Condition | Frozen system | Accuracy | Macro F1 |
|---|---|---:|---:|
| 013-A | Custom CNN | 34.58% | 27.72% |
| 013-B | ResNet50 | 69.33% | 65.10% |
| 013-C | EfficientNet-B0 | 68.75% | 67.38% |
| 013-D | MobileNetV3-Large | 70.75% | 65.92% |
| 013-E | DINOv2 + full-source linear probe | 68.17% | 63.27% |
| 013-F | CLIP + full-source linear probe | 61.00% | 54.96% |
| 013-G | SigLIP2 + full-source linear probe | 82.75% | 81.24% |
| 013-H | CLIP strict P1 zero-shot | 80.75% | 80.03% |
| 013-I | SigLIP2 strict P1 zero-shot | **90.75%** | **90.70%** |

## Result boundary

These results establish external-domain generalization on one independently collected dataset, not general real-world performance. Labels refer to visible fresh/rotten appearance; they do not establish edibility, food safety, chemical or microbial spoilage, or shelf life. No action after observing results changed an image, label, prompt, preprocessing pipeline, threshold, or model.

