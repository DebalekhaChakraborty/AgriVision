# FruitVision Domain-Shift Profile — Experiment 014

## Scope

This profile compares the frozen source test set, the Sultana originals from Experiment 013, and the FruitVision Version 2 originals used in Experiment 014. Statistics are descriptive: images were decoded as RGB and reduced to 64 × 64 only for aggregate brightness, contrast, and channel summaries. Evaluation retained each system's frozen model-native preprocessing. No profile statistic was used for model selection, calibration, or adaptation.

| Domain | Images | Formats | Median width × height | Median aspect ratio | Mean brightness | Mean contrast |
|---|---:|---|---:|---:|---:|---:|
| Source test | 2,698 | PNG | 404 × 366 | 1.119 | 0.650 | 0.290 |
| Sultana originals | 1,200 | JPEG | 2,720 × 2,434 | 1.072 | 0.640 | 0.219 |
| FruitVision originals | 4,185 | JPEG | 960 × 1,280 | 0.750 | 0.650 | 0.131 |

## Resolution and geometry

FruitVision spans 633–4,160 pixels in width and 721–4,472 in height, with 358 distinct width/height pairs. Its median portrait-like 0.75 aspect ratio differs from both source and Sultana medians above 1.0. Sultana is the highest-resolution domain on average (2,599 × 2,354); FruitVision averages 1,151 × 1,314, while the PNG source averages 417 × 353. These differences describe collection and packaging, not a causal explanation of accuracy.

## Photometric summaries

| Domain | Mean red | Mean green | Mean blue | Brightness 5th–95th percentile | Contrast 5th–95th percentile |
|---|---:|---:|---:|---:|---:|
| Source test | 0.737 | 0.636 | 0.532 | 0.442–0.865 | 0.136–0.403 |
| Sultana originals | 0.730 | 0.623 | 0.545 | 0.496–0.747 | 0.136–0.306 |
| FruitVision originals | 0.693 | 0.644 | 0.585 | 0.522–0.952 | 0.084–0.193 |

Mean brightness is similar across domains, but FruitVision has substantially lower mean contrast and a bright upper tail. Its mean blue channel is higher and mean red channel lower than the other domains. These aggregate summaries are compatible with the frequent centered-object/light-background composition observed during pHash review, but they cannot attribute model errors to any one visual property.

## Frozen-representation PCA

For a descriptive view, 1,200 deterministic records per domain were projected with two-component PCA from frozen DINOv2 and SigLIP2 embeddings. DINOv2 PC1/PC2 explain 24.08%/17.08% of variance; SigLIP2 explains 18.16%/15.08%. The plot is `v2/results/experiment_014_multi_domain/pca_three_domains.png`. It is exploratory only: no PCA coordinate, domain label, or separation pattern influenced the models or ranking.

## Interpretation boundary

FruitVision represents an additional independently collected image domain with different file format, geometry, resolution, contrast, backgrounds, and collection protocol. The profile supports the presence of dataset shift; it does not identify a causal mechanism or establish deployment-domain coverage.
