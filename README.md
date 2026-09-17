# AI-Based Detection of Rotten Fruits Using CNN

## Introduction

This project studies the visual classification of fresh and rotten fruit with a convolutional neural network (CNN). Version 1 supports photographs of apples, bananas, and oranges.

The complete pipeline covers provenance and integrity auditing, exploratory analysis, leakage-aware splitting, CNN training, evaluation, single-image prediction, and a local Flask demonstration. V1 has now been genuinely trained and evaluated while keeping the third-party dataset and large model binary outside Git.

## Project Evolution

### V1 — Classical Deep Learning (2018)

V1 is the completed historical study, **AI-Based Fruit Freshness Classification Using CNN (2018)**. It uses a custom CNN trained from scratch with TensorFlow 1.x, standalone Keras 2.2.4, and OpenCV. The `legacy` branch freezes its implementation, environment, genuine experiments, and documented limitations.

### V2 — Modern Vision AI Research (2026)

V2 is **AI-Based Fruit Freshness Intelligence Using Modern Vision AI**, a comparative research programme exploring:

- Modern from-scratch CNN baselines
- Transfer learning
- Self-supervised vision foundation models
- Vision-language representations
- Multimodal reasoning and explanation assessment
- Labelled-data efficiency and cross-domain generalization

**Research in progress.** V2 now includes the executed Phase 1 PyTorch baseline, the completed Phase 2 frozen-backbone transfer-learning benchmark, the completed Phase 3A frozen foundation-representation benchmark, the completed Phase 3B zero-shot semantic benchmark, the completed Phase 3C label-efficiency study, the completed Phase 3D zero-adaptation cross-domain benchmark, and the completed Phase 3E multi-domain robustness study.

The research framework begins with [V2_RESEARCH_BLUEPRINT.md](docs/handbook/V2_RESEARCH_BLUEPRINT.md). The complete protocol is in [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md), and planned model cohorts are defined without results in [MODEL_COMPARISON_MATRIX.md](docs/MODEL_COMPARISON_MATRIX.md).

## V2 Phase 1

### Modern CNN Baseline Reproduction

V1 is the historical TensorFlow 1.x study frozen on the `legacy` branch. V2 Phase 1 reproduces its fundamental unregularized CNN concept with Python 3.11 and PyTorch, using the same frozen leakage-safe train, validation, and test partitions. It keeps the 150 × 150 RGB input, 32/64/128 convolution stages, dense 128 classifier, Adam optimizer, batch size 32, learning rate 0.001, seed 42, and fixed 20 epochs. It adds no dropout, batch normalization, pretrained weights, residual blocks, or other accuracy-oriented changes.

Exactly one run was made. Validation-only checkpoint selection retained epoch 2 at 83.33% validation accuracy and 0.6173 loss. After selection, one held-out test pass measured 74.09% accuracy, 74.57% macro F1, and 74.03% weighted F1 across 2,698 images. These measurements establish a reproducible modern baseline; they are not evidence that modern tooling alone improves accuracy. The model reached 100% training accuracy while validation loss increased, showing substantial overfitting.

Run the registered pipeline from the repository root after creating the separate V2 environment and making the frozen V1 split available at the configured local path:

```bash
PYTHONPATH=. python -m v2.src.training.train --config v2/configs/baseline_cnn.yaml
PYTHONPATH=. python -m v2.src.evaluation.evaluate --config v2/configs/baseline_cnn.yaml
```

The evaluation command is intentionally guarded against a second test pass for the same experiment metadata. See [experiment_001_baseline_cnn.md](v2/experiments/experiment_001_baseline_cnn.md) for the complete run record and limitations. Generated metric JSON, learning curves, and the confusion matrix are in [v2/results/](v2/results/). The trained checkpoint and source images remain local and ignored by Git.

## V2 Phase 2 — Transfer Learning Benchmark

Phase 2 compares the frozen Experiment 001 control with three ImageNet-pretrained representations: ResNet50, EfficientNet-B0, and MobileNetV3-Large. Each backbone remains frozen and only a single six-class linear head is trained. All experiments use the same V1 leakage-safe split, seed 42, Adam optimizer, batch size 32, 20 fixed epochs, validation-based checkpoint selection, and one held-out test pass.

Phase 2 preprocessing uses 224 × 224 inputs, mild random horizontal flip, rotation, and color jitter for training, plus deterministic resize and center crop for validation/test. It applies ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. This differs materially from Experiment 001’s 150 × 150 scaling-only pipeline and is part of the registered transfer-learning protocol.

| Experiment | Model | Selected epoch | Test accuracy | Weighted F1 | Total / trainable parameters | CPU forward time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 001 | CNN from scratch | 2 | 74.09% | 74.03% | 4,829,126 / 4,829,126 | Not measured; test remained frozen |
| 002 | ResNet50 | 20 | 94.18% | 94.13% | 23,520,326 / 12,294 | 31.282 ms/image |
| 003 | EfficientNet-B0 | 8 | **94.55%** | **94.52%** | 4,015,234 / 7,686 | 12.555 ms/image |
| 004 | MobileNetV3-Large | 1 | 93.03% | 92.98% | 2,977,718 / 5,766 | **6.441 ms/image** |

All three transfer pipelines measured higher accuracy and weighted F1 than the control on this dataset. EfficientNet-B0 produced the strongest accuracy/compute balance in these runs, while MobileNetV3-Large was fastest and smallest. Rotten-apple recall increased from 51.41% in Experiment 001 to 91.18%, 89.52%, and 85.69%, respectively. These are dataset-specific observations, not proof that pretraining alone caused the difference: architecture, input resolution, augmentation, and normalization also changed.

Run a registered model with the shared pipeline:

```bash
PYTHONPATH=. python -m v2.src.training.train_transfer --config v2/configs/resnet50.yaml
PYTHONPATH=. python -m v2.src.evaluation.evaluate_transfer --config v2/configs/resnet50.yaml
```

Substitute `efficientnet.yaml` or `mobilenetv3.yaml` for the other registered experiments. Each evaluator refuses a second test pass once its metadata records completion. Full class-level analysis, timing definitions, and limitations are in [TRANSFER_LEARNING_ANALYSIS.md](docs/TRANSFER_LEARNING_ANALYSIS.md).

## V2 Phase 3A — Vision Foundation Representations

Phase 1 trained a custom CNN from scratch. Phase 2 trained identical linear heads on frozen ImageNet-pretrained CNN backbones. Phase 3A measures three frozen vision foundation representations by extracting one deterministic, model-native embedding per image, L2-normalizing it, and training the same single linear six-class probe.

All Phase 3A encoders remained frozen. Each used Adam, learning rate 0.001, batch size 64, 50 epochs, and seed 42 on the unchanged 1,539/270/2,698 train/validation/test split. Selection used validation accuracy with lower validation loss as the tie-break, followed by exactly one held-out test evaluation. No stochastic extraction augmentation, text prompt, zero-shot classification, nonlinear probe, or encoder fine-tuning was used.

| Exp | Model | Representation regime | Trainable params | Test accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: | ---: |
| 001 | Custom CNN | From scratch | 4,829,126 | 74.09% | 74.03% |
| 002 | ResNet50 | ImageNet transfer | 12,294 | 94.18% | 94.13% |
| 003 | EfficientNet-B0 | ImageNet transfer | 7,686 | 94.55% | 94.52% |
| 004 | MobileNetV3-Large | ImageNet transfer | 5,766 | 93.03% | 92.98% |
| 005 | DINOv2 ViT-B/14 | Self-supervised foundation | 4,614 | 89.07% | 89.01% |
| 006 | Original CLIP ViT-B/16 | Vision-language contrastive | 3,078 | 95.85% | 95.84% |
| 007 | SigLIP2 Base Patch16/224 | Modern vision-language foundation | 4,614 | **98.63%** | **98.62%** |

SigLIP2 was strongest overall and produced 94.84% rotten-apple recall, compared with 87.52% for CLIP and 66.06% for DINOv2. DINOv2 still exceeded the from-scratch CNN but trailed all Phase 2 transfer models. CLIP and SigLIP2 exceeded the strongest Phase 2 accuracy by 1.30 and 4.08 percentage points respectively. These observations compare complete pretrained systems; architecture, scale, pretraining data/objective, preprocessing, and representation dimension all differ, so the results do not isolate a causal benefit from self-supervision or language supervision.

Exact checkpoint revisions and hashes are in [FOUNDATION_MODEL_PROVENANCE.md](docs/FOUNDATION_MODEL_PROVENANCE.md). The full class-level, compute, and limitation analysis is in [FOUNDATION_REPRESENTATION_ANALYSIS.md](docs/FOUNDATION_REPRESENTATION_ANALYSIS.md), with individual run records under [v2/experiments/](v2/experiments/). Downloaded weights, embedding caches, source images, and probe checkpoints remain local and Git-ignored.

## V2 Phase 3B — Zero-Shot Semantic Recognition

Phase 3A trained a supervised linear classifier on frozen image embeddings. Phase 3B removes that classifier: frozen CLIP and SigLIP2 image embeddings are compared directly with frozen text prototypes, with no target-dataset parameter fitting, prompt learning, image prototypes, fine-tuning, or calibration.

The prompt registry was fixed before validation. Strict zero-shot uses the canonical P1 texts such as “a photo of a fresh apple.” A separate validation-selected condition compares four pre-registered prompt families and is not described as pure strict zero-shot. For each model, strict and selected conditions were computed together in one locked held-out test run.

| Exp | Model | Condition | Selected prompt | Test accuracy | Weighted F1 |
| --- | --- | --- | --- | ---: | ---: |
| 006 | Original CLIP ViT-B/16 | Supervised linear probe | Not applicable | 95.85% | 95.84% |
| 008A | Original CLIP ViT-B/16 | Strict zero-shot | P1 canonical | 89.96% | 90.09% |
| 008B | Original CLIP ViT-B/16 | Validation-selected zero-shot | P3 condition description | 90.25% | 90.23% |
| 007 | SigLIP2 Base | Supervised linear probe | Not applicable | 98.63% | 98.62% |
| 009A | SigLIP2 Base | Strict zero-shot | P1 canonical | **97.89%** | **97.89%** |
| 009B | SigLIP2 Base | Validation-selected zero-shot | P2 natural object | 96.96% | 96.96% |

CLIP strict zero-shot trailed its linear probe by 5.89 accuracy points. SigLIP2 strict zero-shot trailed its probe by only 0.74 points, showing substantially stronger direct alignment with these six class texts under this protocol. Validation selection helped CLIP slightly but hurt SigLIP2 on test; the registered P2 selection was retained rather than revised after seeing the result.

Most mistakes were freshness-condition errors rather than fruit-identity errors. This supports a narrow claim that the pretrained image-language spaces exhibit semantic alignment with the dataset labels; it does not show that the models understand spoilage or can assess food safety. Full prompt comparisons, class metrics, semantic error categories, similarity margins, compute measurements, and limitations are in [ZERO_SHOT_ANALYSIS.md](docs/ZERO_SHOT_ANALYSIS.md).

## V2 Phase 3C — Label-Efficiency Study

Phase 3A measured full-supervision linear probes on frozen foundation embeddings. Phase 3B measured zero-shot semantic recognition with frozen text prototypes. Phase 3C fills the controlled low-label region between them using nested, class-balanced subsets at 1, 5, 10, 25, 50, and 100 labelled images per class.

For DINOv2, original CLIP, and SigLIP2, five pre-registered subset draws (seeds 42–46) were evaluated at each budget while keeping optimization seed 42 and the Phase 3A `nn.Linear` protocol fixed. All 90 probes used cached L2-normalized embeddings from the unchanged frozen encoders, validation-only checkpoint selection, no augmentation, and one locked test evaluation after every checkpoint was frozen.

| Model | 1-shot mean accuracy | 5-shot | 25-shot | 100-shot | Frozen full-data |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 ViT-B/14 | 81.14% | 92.97% | 94.20% | 92.59% | 89.07% |
| Original CLIP ViT-B/16 | 74.28% | 89.85% | 94.37% | 95.30% | 95.85% |
| SigLIP2 Base | **87.52%** | **97.21%** | **98.12%** | **98.50%** | **98.63%** |

SigLIP2 led at every supervised budget and was within one percentage point of its full-data result at 10 labels/class. CLIP needed 100 labels/class to reach that threshold. DINOv2 was non-monotonic and exceeded its frozen full-data endpoint from 5-shot onward, a protocol-specific result that must not be interpreted as evidence that fewer labels are inherently better. Strict zero-shot markers for CLIP and SigLIP2 use a different text-prototype decision mechanism from the supervised probes.

These findings describe this frozen dataset, representation set, and linear-probe protocol; they do not establish universal food-freshness sample complexity. Aggregate intervals, per-class behavior, rotten-apple analysis, plots, and limitations are in [LABEL_EFFICIENCY_ANALYSIS.md](docs/LABEL_EFFICIENCY_ANALYSIS.md).

## V2 Phase 3D — Cross-Domain Generalization

Phase 3D challenges the nine frozen conditions from Experiments 001–009 on an independently collected dataset without changing their parameters, prompts, class mapping, or original model-specific preprocessing. The external benchmark uses all 1,200 original physical photographs—200 each for fresh/rotten apple, banana, and orange—from Sultana, Jahan, and Uddin's Mendeley Data release (DOI `10.17632/bdd69gyhv8.1`). The separately packaged augmentation archive and ten non-overlapping classes are excluded.

| Frozen system | Source accuracy | External accuracy | Retention |
|---|---:|---:|---:|
| Custom CNN | 74.09% | 34.58% | 46.68% |
| ResNet50 | 94.18% | 69.33% | 73.62% |
| EfficientNet-B0 | 94.55% | 68.75% | 72.71% |
| MobileNetV3-Large | 93.03% | 70.75% | 76.05% |
| DINOv2 + linear probe | 89.07% | 68.17% | 76.54% |
| CLIP + linear probe | 95.85% | 61.00% | 63.64% |
| SigLIP2 + linear probe | 98.63% | 82.75% | 83.90% |
| CLIP strict P1 zero-shot | 89.96% | 80.75% | 89.77% |
| SigLIP2 strict P1 zero-shot | 97.89% | **90.75%** | **92.71%** |

Strict zero-shot classification generalized better than the corresponding source-trained probe for both CLIP (+19.75 external accuracy points) and SigLIP2 (+8.00 points). Fresh-banana recall showed the largest average collapse across systems, while rotten orange had the lowest mean external recall. Condition errors dominated for all pretrained systems, indicating that fruit identity was generally more stable than the fresh/rotten boundary.

These results measure external-domain generalization on one independently collected dataset; they do not prove real-world generalization or food-safety capability. See [external provenance](docs/EXTERNAL_DATASET_PROVENANCE.md), the [domain profile](docs/DOMAIN_SHIFT_PROFILE.md), and the full [cross-domain analysis](docs/CROSS_DOMAIN_GENERALIZATION_ANALYSIS.md).

## V2 Phase 3E — Multi-Domain Robustness

Phase 3E reuses the frozen source and Sultana results, then evaluates the same nine systems once on 4,185 raw FruitVision originals from the six overlapping apple/banana/orange × fresh/rotten classes. FruitVision is evaluation-only: no target-domain training, validation, calibration, prompt selection, preprocessing change, or adaptation occurs. Augmented, formalin-mixed, grape, and mango images are excluded.

SigLIP2 strict P1 zero-shot remains strongest, with 88.53% mean external accuracy and 86.31% worst external accuracy across Sultana and FruitVision. Its worst external retention is 88.17%. Zero-shot beats the corresponding source-trained probe on both external datasets for CLIP (mean advantage 15.97 points) and SigLIP2 (10.39 points). External rankings are more concordant with each other (Spearman 0.833) than source rankings are with either external domain (0.567 and 0.617), showing that high source performance alone does not determine multi-domain robustness.

MobileNetV3-Large provides the strongest measured lightweight compromise at 7.26 CPU forward ms/image, 70.56% mean external accuracy, and 70.37% worst external accuracy. These results support robustness across two independently collected external fruit-image datasets, not general real-world robustness or food-safety capability. See the [FruitVision provenance](docs/FRUITVISION_DATASET_PROVENANCE.md), [domain-shift profile](docs/FRUITVISION_DOMAIN_SHIFT_PROFILE.md), and [multi-domain analysis](docs/MULTI_DOMAIN_ROBUSTNESS_ANALYSIS.md).

## Historical Runtime Contract

This `legacy` version has a strict technical cutoff of November 18, 2018. Its canonical runtime is CPython 3.6.x with TensorFlow 1.12.0 and standalone Keras 2.2.4. It uses generator-specific Keras methods, the classic `acc` and `val_acc` history keys, and HDF5 model storage.

The complete compatibility review is recorded in [TEMPORAL_AUDIT.md](TEMPORAL_AUDIT.md).

## Problem Statement

Fruit can show visible signs of spoilage such as discoloration, spotting, bruising, and surface degradation. Manual assessment can vary between observers. This project investigates whether a CNN can learn image patterns associated with the visible freshness condition of three common fruits.

The classifier evaluates visible appearance only. It cannot detect pathogens, toxins, internal spoilage, smell, or other microbiological hazards. It must not be used to decide whether food is safe for consumption.

## Objective

The objective is to train a six-class image classifier that returns the fruit type, visible condition, complete predicted class, and model confidence.

The supported directory labels are:

```text
fresh_apple     rotten_apple
fresh_banana    rotten_banana
fresh_orange    rotten_orange
```

## Dataset

The genuine experiment uses Sriram Reddy Kalluri's Kaggle dataset, [*Fruits fresh and rotten for classification*](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification), version 1 from August 24, 2018. Its six categories map directly to the V1 labels. Kaggle lists the license as unknown, so the archive and individual photographs are excluded from this repository.

Dataset images are not redistributed in this repository.

The download contained 10,901 source-train and 2,698 source-test PNGs. All 13,599 decoded successfully, all SHA-256 hashes were unique, and no exact cross-split duplicate was found. Filename analysis nevertheless revealed source-provided rotations, translations, flips, and noise variants derived from the same base photographs across the published train/test boundary.

The preparation therefore preserved the original test set, removed 9,092 source-train variants linked to test base photographs, grouped every remaining transformation family, and split those groups approximately 85/15 with seed 42. The frozen experiment counts are:

| Class | Train | Validation | Original test |
| --- | ---: | ---: | ---: |
| Fresh apple | 261 | 45 | 395 |
| Fresh banana | 270 | 45 | 381 |
| Fresh orange | 234 | 45 | 388 |
| Rotten apple | 207 | 36 | 601 |
| Rotten banana | 360 | 63 | 530 |
| Rotten orange | 207 | 36 | 403 |
| **Total** | **1,539** | **270** | **2,698** |

The full historical evidence, raw inventory, dimension ranges, leakage analysis, EDA observations, and redistribution boundary are in [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md). The generic splitter remains available for other legitimately sourced collections; see [dataset/README.md](dataset/README.md).

## Image Preprocessing

Images are loaded from class directories, resized to `150 × 150` pixels, and normalized by multiplying pixel values by `1/255`.

Experiment 1 and Experiment 2 use only normalization. Experiment 3 uses the canonical `ImageDataGenerator` rotation, width and height shifts, shear, zoom, and horizontal flip settings. Validation and test images are resized and normalized without online augmentation. The published source itself already contains offline transformations, which is disclosed separately from online augmentation.

## CNN Architecture

The classifier is a custom sequential CNN trained from scratch:

| Stage | Configuration |
| --- | --- |
| Input | 150 × 150 × 3 image |
| Convolution | 32 filters, 3 × 3 kernel, ReLU |
| Pooling | 2 × 2 max pooling |
| Convolution | 64 filters, 3 × 3 kernel, ReLU |
| Pooling | 2 × 2 max pooling |
| Convolution | 128 filters, 3 × 3 kernel, ReLU |
| Pooling | 2 × 2 max pooling |
| Classifier | Flatten, Dense 128 with ReLU, Dropout 0.5 |
| Output | Dense 6 with softmax |

Training uses Adam, categorical cross-entropy, and accuracy. The default batch size is 32. No pretrained model or transfer learning is used.

## V1 Installation

Create and activate a CPython 3.6 environment, then install the single canonical dependency manifest:

```bash
python3.6 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The exact environment is:

| Package | Version |
| --- | --- |
| Python | 3.6.x |
| TensorFlow | 1.12.0 |
| Keras | 2.2.4 |
| h5py | 2.8.0 |
| NumPy | 1.15.4 |
| Matplotlib | 3.0.2 |
| scikit-learn | 0.20.0 |
| opencv-python | 3.4.3.18 |
| Flask | 1.0.2 |
| Pillow | 5.3.0 |
| Jupyter | 1.0.0 |

## V2 Environment Boundary

V2 uses a separate Python 3.11+ research environment and never installs modern packages into the V1 environment:

```bash
python3.11 -m venv .venv-v2
source .venv-v2/bin/activate
python -m pip install -r requirements-v2.txt
```

`requirements-v2.txt` is the pinned V2 manifest for CPU PyTorch, torchvision, the analysis stack, Transformers, Hugging Face Hub, safetensors, OpenCLIP, timm, and ftfy. No V2 packages are required to inspect the historical V1 artifacts.

## Training

The three controlled variants are defined explicitly in [notebooks/03_cnn_experiments.ipynb](notebooks/03_cnn_experiments.ipynb). After preparing the documented local partition, the canonical augmentation-based training entry point remains:

```bash
python src/train.py --epochs 20 --batch-size 32
```

Training prints `model.summary()` and creates these local artifacts:

```text
model/fruit_freshness_cnn.h5
model/class_indices.json
outputs/training_accuracy.png
outputs/training_loss.png
```

The JSON file stores the exact class indices created by the training generator. Evaluation, command-line prediction, and the web page all read this mapping. For the genuine comparison, best-validation checkpoints were retained, and the validation-selected Experiment 2 checkpoint was copied to the canonical local model path. The model and mapping are ignored by Git.

## Evaluation

For a new, independently prepared run, evaluate once on its held-out test directory only after validation-based model selection:

```bash
python src/evaluate.py
```

The script prints test loss, test accuracy, and a classification report. It calculates the confusion matrix and draws it manually with Matplotlib, saving the result as `outputs/confusion_matrix.png`.

## Prediction

Classify one image from the repository root:

```bash
python src/predict.py path/to/image.jpg
```

The output contains the fruit, condition, predicted display class, and confidence. These values come from the saved HDF5 model and its class mapping.

## Web Application

Start the local demonstration after training:

```bash
python app.py
```

Open `http://127.0.0.1:5000/`, choose a JPG, JPEG, PNG, or BMP image of an apple, banana, or orange, and select **Predict**. The upload limit is 8 MB.

## Experimental Results

All variants used the same seed-42 partition, 150 × 150 inputs, Adam, categorical cross-entropy, batch size 32, and 20 fixed epochs. “Augmentation” here means additional online augmentation.

| Experiment | Dropout | Online augmentation | Selected epoch | Train acc @ selected epoch | Best validation acc | Validation loss |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 — baseline | 0 | No | 10 | 100.00% | 87.41% | 0.7773 |
| 2 — dropout | 0.5 | No | 8 | 92.85% | **90.00%** | **0.5937** |
| 3 — dropout + augmentation | 0.5 | Yes | 14 | 88.30% | 88.89% | 1.0751 |

Experiment 2 was selected strictly from validation behavior. The baseline showed the clearest overfitting, with perfect training accuracy and a 12.59-point checkpoint gap. Dropout improved validation accuracy and reduced that gap. Online augmentation constrained the fit and produced a small checkpoint gap, but it did not exceed dropout-only validation performance in this run.

The six genuine learning curves are in [results/](results/), and exact histories and observations are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

## Final Evaluation

After selection, the Experiment 2 epoch-8 checkpoint was evaluated once using a single inference pass across all 2,698 original held-out test images.

- Test loss: **0.6021**
- Test accuracy: **83.21%** (2,245 correct, 453 incorrect)
- Macro F1: **83.23%**
- Weighted F1: **82.93%**

Rotten banana had the strongest recall at 97.74%. Rotten apple and rotten orange were the weakest-recall classes at 69.55% and 69.48%, with confusion involving fresh fruit and other rotten-fruit classes. The genuine [confusion matrix](results/final_confusion_matrix.png), full six-class report, and a deterministic prediction table containing both correct and incorrect cases are in [EXPERIMENTS.md](EXPERIMENTS.md).

The real selected model also passed the unchanged CLI and Flask GET/model-backed POST checks with genuine held-out images. These metrics describe this specific historical dataset and split; they do not imply microbiological safety or broad real-world generalization.

## Observations

- Conservative transformation-family leakage removal reduced the usable development data substantially; honest independence was prioritized over a larger headline training count.
- The published images vary in crop, scale, lighting, and background, but many still resemble isolated or stock-style fruit photographs.
- Offline rotations sometimes introduce dark border artifacts that a model could learn.
- The validation peak occurred well before epoch 20 in every experiment, so preserving the best checkpoint mattered.
- Dropout-only won this comparison. The expected augmentation variant was not assumed to be superior.
- High-confidence mistakes occurred, so softmax confidence must not be interpreted as calibrated certainty.

## Limitations

- Predictions are limited to the six classes used during training.
- Performance depends on dataset size, quality, class balance, and labeling accuracy.
- Lighting, background, camera quality, viewpoint, occlusion, and surface damage can affect predictions.
- A single label cannot describe mixed conditions or several fruit items in one photograph.
- Visible appearance is not a microbiological food-safety measurement.

## Future Scope

V1 remains frozen. Future research is organized through the phased [V2 roadmap](docs/ROADMAP.md): research setup, modern baseline reproduction, transfer learning, foundation embeddings, vision-language reasoning, and paper preparation.

### V2 Phase 3F — Representation interpretability

Experiment 015 studies frozen DINOv2, CLIP, and SigLIP2 representation geometry across the source, Sultana, and FruitVision domains. It adds balanced PCA/separability analysis, exploratory factor and domain probes, empirical fresh-to-rotten centroid-direction alignment, centroid drift, source-gallery retrieval, probe-versus-zero-shot disagreement, and common 7 × 7 perturbation attribution with deletion-faithfulness checks. It does not change any predictive model or prior experiment.

The main result is deliberately plural rather than causal: SigLIP2 has the most consistent cross-domain and cross-fruit freshness directions, DINOv2 has the strongest linear freshness access and source-gallery retrieval, and CLIP has the smallest class-centroid drift. Strict zero-shot decisions recover more fitted-probe errors than they introduce for both CLIP and SigLIP2 in both external domains. See [the full representation analysis](docs/REPRESENTATION_INTERPRETABILITY_ANALYSIS.md) and [Experiment 015 record](v2/experiments/experiment_015_representation_interpretability.md).

## Repository Structure

```text
├── README.md
├── TEMPORAL_AUDIT.md
├── DATASET_PROVENANCE.md
├── EXPERIMENTS.md
├── requirements.txt
├── requirements-v2.txt
├── docs/
│   ├── handbook/
│   │   └── V2_RESEARCH_BLUEPRINT.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── MODEL_COMPARISON_MATRIX.md
│   ├── DATASET_STRATEGY.md
│   ├── REPRODUCIBILITY_GUIDE.md
│   └── ROADMAP.md
├── dataset/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_training.ipynb
│   └── 03_cnn_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset_split.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── model/
│   └── README.md
├── outputs/
│   └── README.md
├── results/
│   ├── experiment_01_accuracy.png
│   ├── experiment_01_loss.png
│   ├── experiment_02_accuracy.png
│   ├── experiment_02_loss.png
│   ├── experiment_03_accuracy.png
│   ├── experiment_03_loss.png
│   └── final_confusion_matrix.png
├── v2/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── splits/
│   ├── notebooks/
│   │   └── 01_baseline_analysis.ipynb
│   ├── src/
│   │   ├── datasets/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── visualization/
│   ├── experiments/
│   │   ├── experiment_001_baseline_cnn.md
│   │   └── experiment_001_metadata.json
│   ├── checkpoints/
│   │   └── class_indices.json
│   ├── configs/
│   │   ├── baseline_cnn.yaml
│   │   ├── transfer_learning.yaml
│   │   ├── foundation_embedding.yaml
│   │   └── vlm_evaluation.yaml
│   └── results/
│       ├── baseline_history.json
│       ├── baseline_test_metrics.json
│       ├── baseline_training_curve.png
│       ├── baseline_validation_curve.png
│       └── baseline_confusion_matrix.png
├── static/
│   ├── css/style.css
│   └── uploads/.gitkeep
├── templates/
│   ├── index.html
│   └── result.html
└── app.py
```
