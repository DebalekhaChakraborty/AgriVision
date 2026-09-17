# Domain-Shift Profile — Experiment 013

This descriptive profile compares the frozen source test set (2,698 images) with all 1,200 external originals. Images were decoded to RGB and reduced to 64×64 with BOX resampling solely to calculate comparable brightness, contrast, and channel summaries. Evaluation inputs were not altered to make the domains resemble each other.

## Image geometry and encoding

| Statistic | Source test | External originals |
|---|---:|---:|
| Width, median (5th–95th percentile) | 404 px (214–646.3) | 2,720 px (1,037.95–4,160) |
| Height, median (5th–95th percentile) | 366 px (204–456) | 2,433.5 px (908.95–3,461.7) |
| Aspect ratio, median (5th–95th percentile) | 1.119 (0.866–1.684) | 1.072 (0.750–1.643) |
| Format | 2,698 PNG | 1,200 JPEG |

The most conspicuous measured shift is capture/encoding scale: the external photographs have much higher native resolution and use JPEG exclusively, while the source test data are lower-resolution PNG files.

## Appearance summaries

All values below are on a 0–1 scale.

| Statistic | Source mean ± SD | External mean ± SD |
|---|---:|---:|
| Brightness | 0.6501 ± 0.1314 | 0.6405 ± 0.0758 |
| Contrast | 0.2897 ± 0.0783 | 0.2194 ± 0.0542 |
| Red mean | 0.7369 ± 0.1347 | 0.7301 ± 0.0869 |
| Green mean | 0.6362 ± 0.1380 | 0.6235 ± 0.0816 |
| Blue mean | 0.5321 ± 0.1336 | 0.5450 ± 0.1117 |

Mean brightness and channels are similar in aggregate, but the external sample has lower mean contrast and substantially narrower brightness/channel distributions. These marginal summaries cannot capture background, framing, cultivar, defect texture, or other semantic/compositional differences.

## Representation-space description

The frozen DINOv2 and SigLIP2 representations were projected separately with two-component PCA using 1,200 class-balanced source-test embeddings and all 1,200 external embeddings. DINOv2 PC1/PC2 explain 23.36%/16.49%; SigLIP2 PC1/PC2 explain 18.01%/15.05%. The plot is [`pca_domain_shift.png`](../v2/results/experiment_013_cross_domain/pca_domain_shift.png).

PCA is exploratory evidence only. It was produced after the manifest and systems were frozen, was not used for model selection, and does not estimate a causal source of shift.

## Interpretation boundary

The domains may differ simultaneously in camera, lighting, background, fruit cultivar, ripeness distribution, annotation practice, geography, composition, resolution, and collection procedure. Experiment 013 therefore measures aggregate dataset shift. It does not isolate any one factor, and shared fresh/rotten vocabulary does not guarantee identical labelling standards.

