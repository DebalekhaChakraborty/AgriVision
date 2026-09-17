# Experiment 015 — Representation Interpretability and Robustness Mechanisms

## Research question

What properties of frozen visual representations are associated with cross-domain fruit-freshness robustness?

## Frozen protocol

- Branch: `master`; Experiments 001–014, checkpoints, prompt registry, and dataset manifests remained frozen.
- Encoders: DINOv2 ViT-B/14, original CLIP ViT-B/16, and SigLIP2 Base.
- Decision comparison: frozen CLIP and SigLIP2 full-source linear probes versus their strict P1 zero-shot classifiers; frozen EfficientNet-B0 as the CNN attribution reference.
- Domains: source, Sultana, and FruitVision; no new data and no formalin samples.
- Balanced geometry pool: 3,600 samples, 200 per domain × class cell, deterministic SHA-256 ordering, seed 42.
- Analytical probes: five-fold stratified logistic regression with fold-local scaling and seed 42. They are exploratory measurements, not predictive candidates.
- Direction uncertainty: 2,000 class-stratified bootstrap resamples, seed 42.
- Retrieval: every external query against only the frozen source gallery, cosine similarity, top 1/5/10.
- Attribution: one common 7 × 7 patch-occlusion protocol on 109 preselected external images, with target margin and top-ranked-versus-random deletion.

## Principal results

| Finding | Result |
|---|---|
| Strongest freshness analytical probe | DINOv2, 99.47% accuracy |
| Least domain-accessible representation | DINOv2, 99.58% domain accuracy; all were near saturated |
| Highest cross-domain freshness-direction alignment | SigLIP2, mean 0.6686, range 0.5680–0.7697 |
| Strongest cross-fruit direction alignment | SigLIP2 in source (0.6423), Sultana (0.5223), and FruitVision (0.6197) |
| Smallest source-to-external class-centroid drift | CLIP, 0.0733 Sultana and 0.0791 FruitVision |
| Highest top-10 freshness retrieval | DINOv2, 0.7819 Sultana and 0.7654 FruitVision |
| Zero-shot-only versus probe-only recovery | Larger for CLIP and SigLIP2 in both external domains |
| Occlusion faithfulness | Top-ranked deletion beat random for 81.7–93.6% of images |

SigLIP2's strong freshness clustering and direction alignment are consistent with its previously observed external robustness, but the evidence is not a single-mechanism explanation. DINOv2 leads cross-domain retrieval, while CLIP has the smallest centroid drift. Decision geometry and the semantic text rule therefore both matter to the observed pattern.

## Local sensitivity boundary

Aggregate attribution is numeric and spatial. Edge-cell importance is not a segmentation-derived context measure. Source-trained probes were not consistently more edge-sensitive than zero-shot classifiers. No attention map is called an explanation; attention was omitted. Grad-CAM was also omitted because the shared perturbation analysis was sufficient and avoided a model-specific invasive extension.

## Failure analysis

CLIP zero-shot-only correctness was 28.17% on Sultana and 23.70% on FruitVision, versus probe-only correctness of 8.42% and 11.52%. SigLIP2 zero-shot-only correctness was 10.75% and 15.99%, versus probe-only correctness of 2.75% and 3.20%. Zero-shot-only cases had larger mean semantic margins than both-wrong cases in all four comparisons.

Probe error had only weak descriptive association with centroid drift (Spearman ρ 0.0838), direction alignment (−0.0925), and retrieval consistency (−0.1767). No causal claim follows.

## Artifact index

- Protocol: `v2/configs/representation_interpretability.yaml`
- Full analysis: `docs/REPRESENTATION_INTERPRETABILITY_ANALYSIS.md`
- Machine-readable results: `v2/results/experiment_015_interpretability/`
- Balanced manifest SHA-256: `e9998cd1ab4eadfc17edc5d59835cd35d4ff527171bb2d6c225b4f81147e8941`
- Attribution manifest SHA-256: `58b091dafac6b665547a1129384fccb8e5ed11dce5cb199cdc3f63c81202c790`

No raw photographs, nearest-neighbor contact sheets, image montages, or attribution overlays are committed. The task remains visible fresh/rotten appearance classification and makes no claim about food safety, edibility, or biological spoilage.
