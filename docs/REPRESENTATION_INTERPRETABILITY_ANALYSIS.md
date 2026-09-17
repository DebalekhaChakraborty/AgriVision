# Representation Interpretability and Robustness-Mechanism Analysis

## Scope and claim boundary

Experiment 015 asks which properties of frozen DINOv2 ViT-B/14, original CLIP ViT-B/16, and SigLIP2 Base representations are associated with cross-domain visible fruit-freshness robustness. It is a post-hoc descriptive study of the frozen source, Sultana, and FruitVision evidence from Experiments 001–014. It introduces no predictive model, changes no prompt or checkpoint, and performs no target-domain adaptation.

The terms *geometry*, *linear accessibility*, *alignment*, *retrieval consistency*, *attribution*, and *sensitivity* are used deliberately. These analyses do not reveal a model's reasoning and do not establish a causal mechanism. The labels concern visible fresh/rotten appearance only, not chemical or microbial spoilage, edibility, safety, or shelf life.

## Protocol

- Frozen embedding caches were verified for model/revision or checkpoint identity, preprocessing identity, sample and label ordering, dimensions, finite values, and L2 normalization. All nine encoder × domain caches passed.
- The balanced pool contains 3,600 records: 200 deterministic seed-42 samples from each of 18 domain × semantic-class cells. Its SHA-256 is `e9998cd1ab4eadfc17edc5d59835cd35d4ff527171bb2d6c225b4f81147e8941`.
- PCA uses that same pool. Cosine silhouette and Calinski–Harabasz scores are descriptive unsupervised summaries, not classification scores.
- Analytical probes use standardized multinomial/binary logistic regression, `C=1`, `lbfgs`, 2,000 maximum iterations, and shuffled five-fold stratified cross-validation with seed 42. Scaling is fitted inside each fold.
- An empirical fresh-to-rotten centroid direction is the normalized difference between rotten and fresh means for one encoder × domain × fruit cell. Cross-domain uncertainty uses 2,000 class-stratified bootstrap resamples with seed 42.
- External queries retrieve only from the frozen source gallery by cosine similarity. Top-1/5/10 fruit, freshness, and six-class consistency and hit rates are retained.
- The attribution manifest contains 109 external images, selected before attribution by deterministic hash ordering from domain × class × majority-correct/error strata. The same images are used for all five systems. Three error strata had fewer than five eligible images. Source images were omitted because no frozen per-image source prediction record exists and rescoring prior experiments was prohibited.
- Local sensitivity uses one fixed 7 × 7 input grid, a zero-valued normalized-input baseline, and the predicted-class score minus the strongest competing-class score. Deletion curves compare top-ranked regions with a seed-42 random ordering. Scores are margins, not probabilities.
- Raw attention and Grad-CAM were omitted. Raw attention was unnecessary and could not be treated as causal explanation; common occlusion was sufficient for the EfficientNet-B0 reference without introducing a model-specific method.
- No dataset photograph, contact sheet, montage, or attribution overlay is committed.

## A. Representation geometry

PCA plots colored separately by semantic class, fruit, freshness, and domain show the same broad pattern quantified below: fruit identity is the dominant coarse grouping, freshness contributes additional structure, and domain remains visible. The PCA projections are visual summaries only; apparent two-dimensional overlap must not be equated with separability in the full embedding space.

Cosine silhouette scores on the balanced pool were:

| Encoder | Six class | Fruit | Freshness | Domain |
|---|---:|---:|---:|---:|
| DINOv2 | 0.1881 | **0.4480** | 0.0730 | **0.0280** |
| CLIP | 0.1584 | 0.2983 | 0.0976 | 0.0950 |
| SigLIP2 | **0.2099** | 0.3489 | **0.1353** | 0.0815 |

DINOv2 has the clearest unsupervised fruit grouping and lowest domain silhouette. SigLIP2 has the clearest freshness and six-class grouping. These rankings do not exactly match the linear-probe rankings, illustrating that unsupervised compactness and linear accessibility measure different properties.

Six-class centroid matrices reinforce fruit dominance. Mean within-fruit fresh/rotten distance versus mean between-fruit distance was 0.2769 versus 0.8029 (source), 0.1849 versus 0.7623 (Sultana), and 0.1397 versus 0.7103 (FruitVision) for DINOv2; 0.0566 versus 0.1234, 0.0501 versus 0.1324, and 0.0430 versus 0.1315 for CLIP; and 0.0947 versus 0.1671, 0.0611 versus 0.1514, and 0.0399 versus 0.1341 for SigLIP2.

## B. Linear accessibility of semantic factors

Five-fold mean accuracy ± fold standard deviation (macro F1 is nearly identical and retained in the result JSON):

| Encoder | Six class | Fruit identity | Freshness | Freshness − fruit |
|---|---:|---:|---:|---:|
| DINOv2 | 99.56 ± 0.18% | **100.00 ± 0.00%** | **99.47 ± 0.21%** | −0.53 points |
| CLIP | 98.53 ± 0.41% | 99.69 ± 0.23% | 98.17 ± 0.35% | −1.53 points |
| SigLIP2 | 98.97 ± 0.36% | 99.94 ± 0.08% | 98.81 ± 0.40% | −1.14 points |

DINOv2 provides the strongest linear access to freshness in this balanced post-hoc probe. All encoders encode fruit slightly more accessibly than freshness, but the gap is small and performance is near saturation. SigLIP2 does not provide unusually strong linear freshness access by this metric; its distinctive evidence appears instead in direction consistency and its semantic decision layer.

Domain identity is also almost perfectly linearly accessible: DINOv2 99.58 ± 0.29%, CLIP 99.86 ± 0.14%, and SigLIP2 99.72 ± 0.26%. DINOv2 is the lowest, but the small difference should not obscure that all three retain strong dataset identity.

## C. Cross-domain alignment

Mean cross-domain cosine alignment of empirical fresh-to-rotten centroid directions, pooling three fruits and three domain pairs, was:

| Encoder | Mean | Range |
|---|---:|---:|
| DINOv2 | 0.6242 | 0.4722–0.7816 |
| CLIP | 0.5536 | 0.3717–0.7077 |
| SigLIP2 | **0.6686** | **0.5680–0.7697** |

SigLIP2 has both the highest mean and the highest worst-case alignment. Its per-fruit Source/Sultana, Source/FruitVision, and Sultana/FruitVision alignments are 0.5718/0.5935/0.7697 for apple, 0.6954/0.7004/0.7408 for banana, and 0.5680/0.6667/0.7107 for orange. The saved 2,000-resample intervals quantify sampling uncertainty for all 27 comparisons; they do not turn centroid directions into universal spoilage vectors.

Cross-fruit alignment within each domain is positive for every encoder, supporting a partially shared condition factor alongside fruit-specific structure. Mean apple/banana/orange pairwise alignment was 0.3062/0.2493/0.3094 across source/Sultana/FruitVision for DINOv2, 0.5359/0.3944/0.5917 for CLIP, and **0.6423/0.5223/0.6197** for SigLIP2. SigLIP2 is consistently strongest.

Class-centroid drift gives a different ranking. Mean source-to-Sultana/source-to-FruitVision cosine distance was 0.1825/0.1989 for DINOv2, **0.0733/0.0791 for CLIP**, and 0.0898/0.1079 for SigLIP2. CLIP has the smallest drift despite weaker external decisions than SigLIP2 strict zero-shot, so low centroid drift alone does not explain robustness.

## D. Retrieval behavior

Top-10 consistency for external queries against the frozen source gallery was:

| Encoder | Domain | Fruit | Freshness | Six class |
|---|---|---:|---:|---:|
| DINOv2 | Sultana | **0.9853** | **0.7819** | **0.7672** |
| DINOv2 | FruitVision | **0.9899** | **0.7654** | **0.7561** |
| CLIP | Sultana | 0.9384 | 0.6433 | 0.5898 |
| CLIP | FruitVision | 0.9679 | 0.5789 | 0.5516 |
| SigLIP2 | Sultana | 0.9698 | 0.7627 | 0.7327 |
| SigLIP2 | FruitVision | 0.9761 | 0.6650 | 0.6424 |

DINOv2 leads freshness retrieval in both external domains and is even stronger on fruit identity. This supports the view that its representation preserves useful neighbor structure while its frozen source-trained six-class head can still fail under domain shift. SigLIP2 does not lead this retrieval metric, so retrieval alone does not explain its strict zero-shot advantage.

## E. Local perturbation attribution

The deterministic subset has 55 Sultana and 54 FruitVision images (109 total): 60 majority-correct targets were available, plus 49 majority-error targets. Aggregate positive-importance mass is concentrated more toward interior regions than a uniform 7 × 7 map would be: edge cells form 24/49 = 49.0% of regions, while overall edge shares range from 31.0% to 39.0%. This is consistent with substantial sensitivity around centrally framed fruit, but the experiment has no segmentation masks; an edge cell is only a spatial proxy and cannot be called background.

| System | Median random − top AUC | Positive fraction | Edge importance share |
|---|---:|---:|---:|
| EfficientNet-B0 | **0.6315** | **93.58%** | 32.24% |
| CLIP probe | 0.3745 | 86.24% | 32.58% |
| CLIP strict zero-shot | 0.3346 | 82.57% | 39.02% |
| SigLIP2 probe | 0.3716 | 84.40% | 31.01% |
| SigLIP2 strict zero-shot | 0.2261 | 81.65% | 31.92% |

Top-ranked deletion lowers the target margin faster than random deletion for 81.7–93.6% of images, so the aggregate sensitivity ranking is generally faithful but not universal. Medians and positive fractions are emphasized because margin scales differ across systems and a few deletion curves produce large signed means.

Source-trained probes are not consistently more edge-sensitive than zero-shot decisions. On Sultana, zero-shot exceeds the probe edge share for both CLIP (48.52% versus 36.38%) and SigLIP2 (34.45% versus 29.66%). On FruitVision, CLIP is nearly equal (28.97% versus 28.78%) and SigLIP2 zero-shot is lower (29.64% versus 32.36%). Without object masks, these results cannot distinguish fruit surface, shadow, container, and background causally.

## F. Failure-mode association

Probe-versus-zero-shot outcomes expose where semantic text alignment recovers fitted-head errors:

| Family | Domain | Agreement | Probe only correct | Zero-shot only correct | Both wrong |
|---|---|---:|---:|---:|---:|
| CLIP | Sultana | 62.67% | 8.42% | **28.17%** | 10.83% |
| CLIP | FruitVision | 63.20% | 11.52% | **23.70%** | 21.05% |
| SigLIP2 | Sultana | 86.50% | 2.75% | **10.75%** | 6.50% |
| SigLIP2 | FruitVision | 80.62% | 3.20% | **15.99%** | 10.49% |

Zero-shot-only recoveries exceed probe-only recoveries for both families in both domains. Fresh banana is a prominent recovery class for both families; CLIP additionally recovers many fresh apples and rotten oranges. Mean zero-shot margins in zero-shot-only-correct cases exceed both-wrong margins in all four family/domain comparisons: 0.01243 versus 0.00990 and 0.00891 versus 0.00778 for CLIP, and 0.01828 versus 0.01412 and 0.01173 versus 0.01079 for SigLIP2. The difference is modest and descriptive.

Across 36 encoder × external-domain × class cells, Spearman association between probe error and centroid drift is only 0.0838; association with freshness-direction alignment is −0.0925; association with six-class retrieval consistency is −0.1767. These weak directions are broadly plausible but provide no strong single-variable explanation and no causal result.

## Explicit answers to the research questions

1. **Strongest linear freshness accessibility:** DINOv2, at 99.47% five-fold accuracy.
2. **Most consistent cross-domain direction:** SigLIP2, mean cosine 0.6686 and range 0.5680–0.7697.
3. **Shared across fruits:** Partially. All mean cross-fruit cosines are positive; SigLIP2 is strongest in all three domains, but variability rules out a universal direction claim.
4. **Least linearly accessible domain identity:** DINOv2 at 99.58%, although all encoders are near saturation.
5. **Smallest semantic-class centroid drift:** CLIP in both external domains.
6. **Highest cross-domain nearest-neighbor freshness consistency:** DINOv2 at top 10 in both Sultana and FruitVision.
7. **Is SigLIP2 geometry consistent with robustness?** Yes in freshness silhouette, cross-domain direction alignment, and cross-fruit alignment; no single geometry statistic explains the result because DINOv2 leads retrieval and CLIP has lower centroid drift.
8. **Does DINOv2 emphasize fruit more than freshness?** Yes descriptively: fruit silhouette 0.4480 versus freshness 0.0730, fruit probe 100.00% versus freshness 99.47%, and fruit retrieval exceeds freshness retrieval. Freshness remains highly linearly accessible.
9. **Why might the CLIP probe generalize worse than CLIP zero-shot?** The frozen source-fitted head can encode source-specific boundaries despite useful embeddings, whereas P1 text prototypes retain semantic decision anchors. This is an interpretation consistent with disagreement and retrieval evidence, not a causal demonstration.
10. **Why might SigLIP2 zero-shot generalize better than its probe?** SigLIP2 combines comparatively aligned freshness structure with a semantic text decision rule that recovers more probe errors than it introduces in both domains. The post-hoc study establishes association only.
11. **Are zero-shot-only-correct cases associated with stronger semantic margins?** Relative to both-wrong cases, yes in all four family/domain comparisons, though both-correct cases have still larger margins.
12. **Are probe failures associated with greater drift?** Only very weakly across the 36 cells (Spearman ρ = 0.0838); this is not persuasive evidence of a general relationship.
13. **Which regions are perturbationally important?** Aggregate 7 × 7 sensitivity is more concentrated in interior cells than uniform spatial mass, consistent with central fruit regions, with class/domain variation. Exact fruit-versus-background attribution is unavailable without segmentation masks.
14. **Are probes more context-sensitive than zero-shot classifiers?** Not consistently under the edge-cell proxy; the direction reverses by family/domain.
15. **Does faithfulness agree with the aggregate patterns?** Generally: top-ranked deletion beats random deletion on 81.7–93.6% of images. It is not universal, and qualitative image overlays were neither required nor committed.
16. **What replicates across both external domains?** Zero-shot-only recoveries exceed probe-only recoveries for both CLIP and SigLIP2; DINOv2 leads freshness/fruit/six-class retrieval; SigLIP2 has strong per-fruit source-to-external direction alignment; zero-shot-only cases have higher semantic margins than both-wrong cases; and top-ranked occlusion deletion is usually more disruptive than random deletion.

## Limitations

- PCA discards dimensions, and cluster statistics depend on representation geometry and sampling.
- Analytical probes measure linear accessibility on a balanced mixed-domain pool and are not deployable predictors.
- Centroid directions compress multimodal class distributions into a single vector and may be influenced by nuisance structure.
- Retrieval statistics depend on the source gallery composition and do not isolate a decision rule.
- Occlusion creates out-of-distribution inputs; zero baselines, grid boundaries, and margin scales affect results.
- Edge location is not a labeled background mask. No causal object/context separation is claimed.
- Attribution covers external images only because the frozen-evidence rule precluded regenerating source prediction records.
- Results span two external datasets, not unrestricted real-world conditions.

## Reproducibility and artifacts

The pre-registered protocol is in `v2/configs/representation_interpretability.yaml`. Computation is implemented in `v2/src/evaluation/interpretability_study.py` and `v2/src/evaluation/interpretability_attribution.py`. Machine-readable evidence, identifiers, manifest hashes, retrieval records, numeric occlusion maps, and aggregate figures are under `v2/results/experiment_015_interpretability/`. Embedding and local progress caches remain Git-ignored.
