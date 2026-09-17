# V2 Phase 3B Zero-Shot Semantic Recognition Analysis

## Research boundary

Phase 3B asks whether language-aligned foundation representations can classify visible fruit freshness without fitting any parameter on labelled images from this target dataset. “Zero-shot” here means no target-dataset labels were used to train model weights, classifier heads, prompt vectors, temperatures, calibrators, centroids, or image prototypes. It does not mean the pretrained models never encountered fruit or freshness concepts during pretraining.

Experiment 008 uses the same original OpenAI CLIP ViT-B/16 checkpoint as Experiment 006. Experiment 009 uses the same pinned SigLIP2 Base Patch16/224 checkpoint as Experiment 007. DINOv2 is excluded because its Phase 3A image representation has no language-aligned text encoder for direct image-text classification.

The prompt registry was frozen before validation at SHA-256 `e902ba9a66fab36ff86c6440e34c241095d3f947e05c12d86a51d635ca2c67d7`. Strict condition A always uses canonical P1. Condition B is reported separately because validation labels selected among the four pre-registered prompt families. Each model's P1 and selected condition were evaluated together in one locked held-out test run.

## Phase 3A versus Phase 3B

| Exp | Model | Supervision | Trainable head | Test accuracy | Weighted F1 |
| --- | --- | --- | --- | ---: | ---: |
| 006 | CLIP ViT-B/16 | Full labelled train split | Linear probe | 95.85% | 95.84% |
| 007 | SigLIP2 Base | Full labelled train split | Linear probe | 98.63% | 98.62% |
| 008A | CLIP ViT-B/16 | Zero target labels; fixed P1 | None | 89.96% | 90.09% |
| 008B | CLIP ViT-B/16 | Validation-selected P3 prompts only | None | 90.25% | 90.23% |
| 009A | SigLIP2 Base | Zero target labels; fixed P1 | None | **97.89%** | **97.89%** |
| 009B | SigLIP2 Base | Validation-selected P2 prompts only | None | 96.96% | 96.96% |

CLIP strict zero-shot was 5.89 accuracy points below its supervised linear probe and retained 93.85% of the probe's accuracy on a relative basis. SigLIP2 strict zero-shot was only 0.74 points below its linear probe and retained 99.25% relatively. Validation selection reduced CLIP's gap slightly to 5.60 points, but widened SigLIP2's gap to 1.67 points because P2 did not transfer as well as P1.

## Validation prompt comparison

| Model | Metric | P1 | P2 | P3 | P4 ensemble | Selected |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| CLIP | Macro F1 | 85.18% | 86.72% | **93.77%** | 91.93% | P3 |
| CLIP | Accuracy | 85.93% | 87.41% | **94.44%** | 92.22% | P3 |
| SigLIP2 | Macro F1 | 98.76% | **98.85%** | 98.80% | 98.34% | P2 |
| SigLIP2 | Accuracy | **98.89%** | **98.89%** | **98.89%** | 98.52% | P2 by primary metric |

Equal-weight template ensembling was not consistently beneficial. It improved CLIP over canonical P1 but was weaker than P3; for SigLIP2 it was the weakest strategy. SigLIP2's P2 selection depended on a very small three-image validation difference in class balance reflected by a 0.09-point macro-F1 edge. On test, strict P1 was better by 0.93 accuracy points. The prompt was not changed retrospectively.

## Test class behavior

| Model/condition | Fresh apple recall | Fresh banana | Fresh orange | Rotten apple | Rotten banana | Rotten orange |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP strict P1 | 91.65% | 98.69% | 70.36% | 87.69% | 97.92% | 91.81% |
| CLIP selected P3 | 97.97% | 99.48% | 96.65% | 80.20% | 92.83% | 79.40% |
| SigLIP2 strict P1 | 100.00% | 99.74% | 99.23% | 94.84% | 99.06% | 95.78% |
| SigLIP2 selected P2 | 100.00% | 99.74% | 99.48% | 91.01% | 99.06% | 95.04% |

CLIP naturally represented banana classes strongly under P1, while orange freshness was less stable: 115 fresh oranges were classified as rotten orange. P3 recovered fresh-orange recall but shifted errors toward rotten fruit, especially rotten apple to rotten orange and rotten orange to fresh orange. SigLIP2 was strong across all classes without supervised adaptation. Its main strict weakness was rotten fruit predicted as the fresh counterpart, especially 28 rotten apples and 17 rotten oranges.

Relative to Phase 3A, CLIP's largest strict recall regression was fresh orange: 98.45% with the probe versus 70.36% zero-shot. This class therefore benefited most from supervised adaptation. SigLIP2's strict recalls remained close to its probe; its largest drop was rotten orange, from 99.26% to 95.78%.

## Semantic error analysis

| Model/condition | Total errors | Condition errors | Fruit-identity errors | Both errors |
| --- | ---: | ---: | ---: | ---: |
| CLIP strict P1 | 271 | 205 (75.65%) | 62 (22.88%) | 4 (1.48%) |
| CLIP selected P3 | 263 | 165 (62.74%) | 91 (34.60%) | 7 (2.66%) |
| SigLIP2 strict P1 | 57 | 54 (94.74%) | 3 (5.26%) | 0 |
| SigLIP2 selected P2 | 82 | 75 (91.46%) | 7 (8.54%) | 0 |

Percentages are shares of each condition's errors. Freshness-condition errors were more common than wrong-fruit errors for both models and both prompt conditions. SigLIP2 almost always retained fruit identity even when freshness was wrong. This indicates a remaining semantic boundary problem around visible condition, not evidence that either model reasons about physical spoilage.

## Similarity margins

| Model/condition | Correct mean cosine margin | Incorrect mean cosine margin | Correct median | Incorrect median |
| --- | ---: | ---: | ---: | ---: |
| CLIP strict P1 | 0.01686 | 0.00643 | 0.01549 | 0.00478 |
| CLIP selected P3 | 0.01642 | 0.00811 | 0.01556 | 0.00555 |
| SigLIP2 strict P1 | 0.04291 | 0.00909 | 0.04411 | 0.00729 |
| SigLIP2 selected P2 | 0.04085 | 0.00968 | 0.04074 | 0.00750 |

Correct predictions had larger top-1-minus-top-2 margins in every condition. SigLIP2's separation was particularly pronounced. These are descriptive cosine and native-logit margins, not calibrated uncertainty. Similarity values and softmax-derived scores are not guaranteed to be true predictive confidence, so no ECE, Brier, temperature scaling, or Platt scaling was fitted.

## Computational comparison

All figures were measured on the same 20-vCPU, CPU-only host. Forward timing begins after deterministic image preprocessing.

| Model | Total params | Image params | Checkpoint | P1 text generation | Image encoding | Similarity | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP ViT-B/16 | 149,620,737 | 86,192,640 | 598.52 MB | 0.206 s | 79.268 ms/image | 0.00022 ms/image | 2,181.57 MB |
| SigLIP2 Base | 375,187,970 | 92,884,224 | 1,500.80 MB | 0.323 s | 80.675 ms/image | 0.00035 ms/image | 2,622.43 MB |

There is no training time in Phase 3B. Text prototypes are a one-time cost, and image encoding dominates inference. The full checkpoint sizes include text towers and are not directly comparable with Phase 3A image-encoder-only parameter counts.

## Answers to the research questions

1. **How much performance remains without a linear probe?** CLIP lost 5.89 accuracy points; SigLIP2 lost 0.74 under strict P1.

2. **Can CLIP identify fruit identity and freshness semantically?** Yes to a substantial but imperfect degree. Strict accuracy was 89.96%, with strong banana recognition but a weak fresh/rotten orange boundary.

3. **Can SigLIP2?** Yes under this dataset and protocol. Strict accuracy was 97.89%, and only three of 57 errors preserved freshness while changing fruit identity.

4. **Is SigLIP2's Phase 3A advantage preserved?** Yes. Its strict zero-shot accuracy exceeded CLIP by 7.93 points, and its supervised-adaptation gap was much smaller.

5. **Does prompt ensembling improve validation?** Not reliably. P4 helped CLIP relative to P1 but did not win; it reduced SigLIP2 performance.

6. **Are freshness errors more common than identity errors?** Yes in all four reported conditions, ranging from 62.74% to 94.74% of errors.

7. **Which classes are naturally represented well?** Both models represented bananas well. SigLIP2 also represented all apple/orange classes strongly. CLIP's canonical orange classes were the weakest.

8. **Which classes depend most on supervised adaptation?** Fresh orange depended most strongly on CLIP's trained probe. SigLIP2 depended much less on adaptation, with its largest strict recall gap on rotten orange.

9. **How large is the supervised adaptation gap?** 5.89 accuracy points for CLIP and 0.74 for SigLIP2 under strict P1.

10. **What does this imply about alignment versus separability?** CLIP's image features are more task-separable with labelled supervision than they are directly aligned to the canonical class text, particularly for orange freshness. SigLIP2's canonical text alignment already captures nearly all of the separation exposed by its linear probe on this split.

## Limitations and claim boundary

These are single-checkpoint, single-split results with four hand-specified generic prompt families. Validation contains only 270 images, and SigLIP2 demonstrates that a small validation advantage can reverse on test. Model families differ in architecture, pretraining, tokenizer, parameter count, objective, and preprocessing, so no difference can be attributed solely to “newer” training or language supervision.

Zero-shot results do not establish spoilage understanding, causal reasoning, microbiological safety, calibration, cross-domain robustness, or performance on unseen fruits. The dataset records visible appearance labels only. No few-shot learning, image centroids, fine-tuning, prompt learning, generative VLM, or Phase 3C work was performed.
