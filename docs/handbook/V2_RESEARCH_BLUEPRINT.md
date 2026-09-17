# V2 Research Blueprint

## Research Motivation

Fruit-freshness classification offers a compact case study of how visual AI has changed across model generations. The completed V1 artifact asks a 2018 question: can a small convolutional neural network trained from scratch distinguish six visible fruit-condition classes? V2 asks a broader 2026 research question: how do learned representations, supervision regimes, and multimodal reasoning change data efficiency, generalization, and interpretability for the same underlying task?

The purpose is not to replace V1 or recast its historical result. V1 remains the fixed classical baseline. V2 will construct controlled comparisons between that baseline and later approaches while keeping dataset lineage, evaluation rules, and claim boundaries explicit.

This is a research programme, not a software-product roadmap. Its primary outputs are comparable evidence, documented limitations, and defensible conclusions about model generations.

## Research Questions

### RQ1 — Representation quality

How do modern visual representations compare with CNNs trained from scratch when evaluated on the same fruit-freshness task and held-out data?

### RQ2 — Labelled-data efficiency

How much labelled data is required by different AI generations to reach useful and stable classification performance?

### RQ3 — Generalization

Can foundation-model representations provide better generalization across acquisition conditions, backgrounds, fruit varieties, and dataset sources?

### RQ4 — Multimodal explanation

Can vision-language models provide explanations that are more meaningful to human readers while remaining faithful to visible evidence and explicit uncertainty?

## Research Evolution

### Stage 1 — Custom CNN

The frozen V1 model supplies the historical reference point: task-specific features learned from scratch, fixed six-class labels, and classical supervised evaluation. Its genuine results are evidence for V1 only and will not be silently recomputed under modern libraries.

### Stage 2 — Transfer learning

Modern convolutional backbones will be studied as pretrained feature extractors and, where justified, through controlled fine-tuning. This stage isolates the contribution of ImageNet-style pretraining from the larger architectural shift to foundation models.

### Stage 3 — Vision foundation models

Self-supervised and vision-language representations will be compared through frozen embeddings, lightweight probes, and data-efficiency curves. The study will distinguish representation quality from end-to-end training capacity.

### Stage 4 — Vision-language reasoning

Multimodal models will be evaluated for freshness descriptions, evidence localization in language, uncertainty communication, and failure awareness. Classification scores alone will not be treated as proof that an explanation is faithful.

## Comparative Study Design

Every stage will be evaluated under a registered protocol with immutable split manifests. Primary comparisons will use the same class ontology and test policy. Additional cross-domain datasets may measure robustness, but their results will be reported separately rather than pooled with the historical benchmark.

Model selection will use training and validation data only. The test set will remain closed until an experiment family and checkpoint are fixed. Data-efficiency experiments will use predetermined labelled-data fractions and repeated seeds where computationally feasible. Representation comparisons will report both predictive metrics and resource context so that accuracy is not detached from supervision and computation.

## Explanation Study Boundary

Explanation quality will be assessed as a separate research object. Candidate outputs may be judged for visible-evidence grounding, relevance, consistency, uncertainty, and human usefulness. Fluency will not be equated with correctness. No model output will be described as a microbiological food-safety assessment.

## Intended Contributions

V2 is designed to contribute:

1. A traceable comparison spanning a from-scratch CNN, transfer learning, frozen foundation embeddings, and multimodal reasoning.
2. Label-efficiency and cross-domain evidence rather than a single in-domain accuracy score.
3. A reproducibility record linking datasets, split manifests, configurations, checkpoints, and generated artifacts.
4. A claim framework that separates visible-condition classification from food-safety conclusions.

No V2 empirical result exists at framework-creation time. All comparative statements above are hypotheses or evaluation goals.
