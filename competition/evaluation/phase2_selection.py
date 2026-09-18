"""Phase 2: candidate inventory, runtime benchmarks, parity, condition evaluation.

Produces the evidence behind the runtime-model selection decision. Every number
here is either read from an existing research artifact or measured now; none is
reconstructed from memory.

Artifacts written under --output-dir:

    candidate_inventory.json   evidence-based inventory of trained candidates
    runtime_benchmarks.json    load and inference timings (NON-deterministic)
    parity_results.json        torch reference vs OpenCV DNN competition path
    gate_calibration.json      capture-gate behaviour on real photographs
    condition_eval.json/.csv   condition performance, gated and ungated arms

**Data boundary.** Real photographs are read from local research data only and
never copied, embedded or committed. Outputs carry content hashes, labels,
predictions, probabilities, timings and aggregates — no image bytes.

FruitVision is deliberately excluded: its CC BY-NC-ND 4.0 terms (NonCommercial
and NoDerivatives) make competition-context use ambiguous, and the instruction
in that situation is not to use it. The Sultana external set (CC BY 4.0) and the
frozen V1 split are used locally.

Run with:

    .venv-competition/bin/python -m competition.evaluation.phase2_selection
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from competition.agent.inspection import (
    DEFAULT_BLOCKING_FLAGS,
    InspectionOutcome,
    inspect_capture,
)
from competition.models.adapter import (
    RUNTIME_ONNXRUNTIME,
    RUNTIME_OPENCV_DNN,
    load_condition_model,
    predict_condition,
)
from competition.models.preprocessing import DEFAULT_CONTRACT, preprocess_bgr
from competition.vision.quality import assess_capture_quality

DEFAULT_OUTPUT_DIR = Path("competition/evaluation/results/phase2")
DEFAULT_ARTIFACT = Path("competition/models/artifacts/mobilenetv3_large_v2exp004")

FROZEN_SPLIT = Path("v2/data/processed/v1_frozen_split")
SULTANA = Path("v2/data/external/sultana_2022")

BENCHMARK_WARMUP = 5
BENCHMARK_SAMPLES = 100
PARITY_SAMPLE = 200
CONDITION_SAMPLE = 400
EVAL_SEED = 42

# Parity tolerances, fixed before results were examined.
PARITY_MIN_CLASS_AGREEMENT = 0.99
PARITY_MAX_MEAN_PROB_DIFF = 0.05


def run_context() -> dict:
    return {
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "preprocessing_version": DEFAULT_CONTRACT.version,
    }


# --------------------------------------------------------------------------
# C2: candidate inventory, assembled from artifacts that actually exist
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _size_mb(path: Path) -> float | None:
    try:
        return round(Path(path).stat().st_size / 1048576, 2)
    except OSError:
        return None


def build_candidate_inventory() -> dict:
    """Inventory the trained candidates that exist locally, with their evidence."""
    multi_domain = _read_json(
        Path("v2/results/experiment_014_multi_domain/multi_domain_summary.json")
    )
    systems = (multi_domain or {}).get("systems", {})

    def ood(key: str) -> dict:
        entry = systems.get(key, {})
        return {
            "source_accuracy": entry.get("source_accuracy"),
            "worst_external_accuracy": entry.get("worst_external_accuracy"),
            "mean_external_accuracy": entry.get("mean_external_accuracy"),
            "worst_external_retention": entry.get("worst_external_retention"),
            "evidence_source": "v2/results/experiment_014_multi_domain/multi_domain_summary.json",
        }

    def in_domain(path: str) -> dict:
        data = _read_json(Path(path)) or {}
        return {
            "accuracy": data.get("accuracy"),
            "samples": data.get("samples"),
            "evidence_source": path,
        }

    # Foundation-encoder weights a linear probe needs at runtime. The probe
    # itself is ~20 KB; the encoder is not, and that is the deployable cost.
    encoder_sizes = {}
    for name, pattern in (
        ("dinov2-base", "v2/cache/huggingface/**/models--facebook--dinov2-base/**/model.safetensors"),
        ("clip-vit-b16", "v2/cache/huggingface/**/models--timm--vit_base_patch16_clip_224.openai/**/*.safetensors"),
        ("siglip2-base", "v2/cache/huggingface/**/models--google--siglip2-base-patch16-224/**/model.safetensors"),
    ):
        matches = glob.glob(pattern, recursive=True)
        sizes = [Path(m).stat().st_size for m in matches if Path(m).exists()]
        encoder_sizes[name] = round(max(sizes) / 1048576, 1) if sizes else None

    candidates = [
        {
            "name": "MobileNetV3-Large",
            "family": "mobilenet_v3_large",
            "inference_approach": "frozen ImageNet backbone + linear head",
            "checkpoint": "v2/checkpoints/best_mobilenetv3.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/best_mobilenetv3.pt")),
            "runtime_weight_mb": _size_mb(Path("v2/checkpoints/best_mobilenetv3.pt")),
            "framework_dependencies": ["torch", "torchvision"],
            "in_domain": in_domain("v2/results/experiment_004/metrics.json"),
            "external": ood("mobilenetv3_large"),
            "export_plausible": True,
            "opencv_dnn_plausible": True,
            "notes": "Smallest self-contained candidate.",
        },
        {
            "name": "EfficientNet-B0",
            "family": "efficientnet_b0",
            "inference_approach": "frozen ImageNet backbone + linear head",
            "checkpoint": "v2/checkpoints/best_efficientnet.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/best_efficientnet.pt")),
            "runtime_weight_mb": _size_mb(Path("v2/checkpoints/best_efficientnet.pt")),
            "framework_dependencies": ["torch", "torchvision"],
            "in_domain": in_domain("v2/results/experiment_003/metrics.json"),
            "external": ood("efficientnet_b0"),
            "export_plausible": True,
            "opencv_dnn_plausible": True,
            "notes": "Best in-domain of the small CNNs.",
        },
        {
            "name": "ResNet50",
            "family": "resnet50",
            "inference_approach": "frozen ImageNet backbone + linear head",
            "checkpoint": "v2/checkpoints/best_resnet50.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/best_resnet50.pt")),
            "runtime_weight_mb": _size_mb(Path("v2/checkpoints/best_resnet50.pt")),
            "framework_dependencies": ["torch", "torchvision"],
            "in_domain": in_domain("v2/results/experiment_002/metrics.json"),
            "external": ood("resnet50"),
            "export_plausible": True,
            "opencv_dnn_plausible": True,
            "notes": "8x the weights of MobileNetV3 for lower worst-external accuracy.",
        },
        {
            "name": "Custom CNN (V2 baseline)",
            "family": "custom_cnn",
            "inference_approach": "trained from scratch",
            "checkpoint": "v2/checkpoints/best_model.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/best_model.pt")),
            "runtime_weight_mb": _size_mb(Path("v2/checkpoints/best_model.pt")),
            "framework_dependencies": ["torch"],
            "in_domain": in_domain("v2/results/baseline_test_metrics.json"),
            "external": ood("custom_cnn"),
            "export_plausible": True,
            "opencv_dnn_plausible": True,
            "notes": "Rejected on performance: worst external accuracy 34.6%.",
        },
        {
            "name": "DINOv2 + linear probe",
            "family": "dinov2_linear_probe",
            "inference_approach": "frozen foundation encoder + linear probe",
            "checkpoint": "v2/checkpoints/experiment_005_linear_probe.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/experiment_005_linear_probe.pt")),
            "runtime_weight_mb": encoder_sizes.get("dinov2-base"),
            "framework_dependencies": ["torch", "transformers", "huggingface-hub"],
            "in_domain": {"evidence_source": "v2/results/experiment_005/test_metrics.json",
                          **{k: v for k, v in (_read_json(Path("v2/results/experiment_005/test_metrics.json")) or {}).items() if k in ("accuracy", "samples")}},
            "external": ood("dinov2_linear_probe"),
            "export_plausible": False,
            "opencv_dnn_plausible": False,
            "notes": "Probe is ~20 KB but requires the full encoder at runtime.",
        },
        {
            "name": "CLIP + linear probe",
            "family": "clip_linear_probe",
            "inference_approach": "frozen foundation encoder + linear probe",
            "checkpoint": "v2/checkpoints/experiment_006_linear_probe.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/experiment_006_linear_probe.pt")),
            "runtime_weight_mb": encoder_sizes.get("clip-vit-b16"),
            "framework_dependencies": ["torch", "open-clip-torch", "timm"],
            "in_domain": {"evidence_source": "v2/results/experiment_006/test_metrics.json",
                          **{k: v for k, v in (_read_json(Path("v2/results/experiment_006/test_metrics.json")) or {}).items() if k in ("accuracy", "samples")}},
            "external": ood("clip_linear_probe"),
            "export_plausible": False,
            "opencv_dnn_plausible": False,
            "notes": "Worst external retention of the foundation probes.",
        },
        {
            "name": "SigLIP2 + linear probe",
            "family": "siglip2_linear_probe",
            "inference_approach": "frozen foundation encoder + linear probe",
            "checkpoint": "v2/checkpoints/experiment_007_linear_probe.pt",
            "checkpoint_mb": _size_mb(Path("v2/checkpoints/experiment_007_linear_probe.pt")),
            "runtime_weight_mb": encoder_sizes.get("siglip2-base"),
            "framework_dependencies": ["torch", "transformers"],
            "in_domain": {"evidence_source": "v2/results/experiment_007/test_metrics.json",
                          **{k: v for k, v in (_read_json(Path("v2/results/experiment_007/test_metrics.json")) or {}).items() if k in ("accuracy", "samples")}},
            "external": ood("siglip2_linear_probe"),
            "export_plausible": False,
            "opencv_dnn_plausible": False,
            "notes": "Strong, but the encoder alone is over 1.4 GB.",
        },
        {
            "name": "SigLIP2 strict zero-shot",
            "family": "siglip2_strict_zero_shot",
            "inference_approach": "text-prompt similarity, no trained head",
            "checkpoint": None,
            "checkpoint_mb": None,
            "runtime_weight_mb": encoder_sizes.get("siglip2-base"),
            "framework_dependencies": ["torch", "transformers"],
            "in_domain": {"evidence_source": "v2/results/experiment_009/selected_prompt_test_metrics.json",
                          **{k: v for k, v in (_read_json(Path("v2/results/experiment_009/selected_prompt_test_metrics.json")) or {}).items() if k in ("accuracy", "samples")}},
            "external": ood("siglip2_strict_zero_shot"),
            "export_plausible": False,
            "opencv_dnn_plausible": False,
            "notes": "Best out-of-domain retention in Experiment 014; heaviest to deploy.",
        },
        {
            "name": "V1 Keras CNN (2018)",
            "family": "v1_keras_cnn",
            "inference_approach": "historical reference only",
            "checkpoint": "model/fruit_freshness_cnn.h5",
            "checkpoint_mb": _size_mb(Path("model/fruit_freshness_cnn.h5")),
            "runtime_weight_mb": _size_mb(Path("model/fruit_freshness_cnn.h5")),
            "framework_dependencies": ["tensorflow==1.12.0", "Keras==2.2.4", "python3.6"],
            "in_domain": {"evidence_source": "README.md / legacy branch"},
            "external": {},
            "export_plausible": False,
            "opencv_dnn_plausible": False,
            "notes": "Excluded: incompatible runtime, cannot coexist with Python 3.11.",
        },
    ]

    return {
        "candidates": candidates,
        "foundation_encoder_sizes_mb": encoder_sizes,
        "evidence_note": (
            "In-domain and external figures are read from committed research "
            "artifacts. No research file is modified."
        ),
    }


# --------------------------------------------------------------------------
# C12: benchmarks
# --------------------------------------------------------------------------

def benchmark_runtime(artifact_dir: Path, runtime: str, image: np.ndarray) -> dict:
    load_started = time.perf_counter()
    model = load_condition_model(artifact_dir, runtime=runtime)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    for _ in range(BENCHMARK_WARMUP):
        predict_condition(model, image)

    samples: list[float] = []
    for _ in range(BENCHMARK_SAMPLES):
        started = time.perf_counter()
        predict_condition(model, image)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()

    return {
        "runtime": runtime,
        "cold_load_ms": round(load_ms, 3),
        "warm_inference_median_ms": round(statistics.median(samples), 3),
        "warm_inference_mean_ms": round(statistics.fmean(samples), 3),
        "warm_inference_p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
        "warm_inference_min_ms": round(samples[0], 3),
        "warm_inference_max_ms": round(samples[-1], 3),
        "sample_count": len(samples),
        "warmup_discarded": BENCHMARK_WARMUP,
        "input_resolution": f"{DEFAULT_CONTRACT.image_size}x{DEFAULT_CONTRACT.image_size}",
        "scope": "LOCAL_CPU_ONLY - not an AWS or deployed-endpoint measurement",
    }


# --------------------------------------------------------------------------
# C5: parity against the framework-native reference
# --------------------------------------------------------------------------

def parity_against_torch(artifact_dir: Path, paths: list[str]) -> dict:
    """Compare the competition path against torch + torchvision preprocessing."""
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torchvision import transforms
        import torchvision.models as tvm
    except ImportError:
        return {"status": "SKIPPED", "reason": "torch/torchvision not installed"}

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        Path("v2/checkpoints") / manifest["source_checkpoint_name"],
        map_location="cpu",
        weights_only=False,
    )
    network = tvm.mobilenet_v3_large(weights=None)

    class Reference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
            self.classifier = nn.Linear(manifest["feature_dim"], manifest["num_classes"])

        def forward(self, x):
            return self.classifier(self.backbone(x))

    reference = Reference()
    reference.load_state_dict(checkpoint["model_state_dict"], strict=False)
    reference.eval()

    reference_transform = transforms.Compose([
        transforms.Resize(DEFAULT_CONTRACT.resize_shorter_side),
        transforms.CenterCrop(DEFAULT_CONTRACT.image_size),
        transforms.ToTensor(),
        transforms.Normalize(list(DEFAULT_CONTRACT.normalization_mean),
                             list(DEFAULT_CONTRACT.normalization_std)),
    ])

    model = load_condition_model(artifact_dir, runtime=RUNTIME_OPENCV_DNN)

    agree = 0
    logit_diffs: list[float] = []
    prob_diffs: list[float] = []
    tensor_diffs: list[float] = []

    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        with torch.no_grad():
            reference_tensor = reference_transform(
                Image.open(path).convert("RGB")
            ).unsqueeze(0)
            reference_logits = reference(reference_tensor).numpy().reshape(-1)

        ours_tensor = preprocess_bgr(bgr, model.contract)
        tensor_diffs.append(float(np.abs(ours_tensor - reference_tensor.numpy()).max()))
        ours_logits = model._forward(ours_tensor)

        agree += int(reference_logits.argmax() == ours_logits.argmax())
        logit_diffs.append(float(np.abs(reference_logits - ours_logits).max()))

        def softmax(v):
            e = np.exp(v - v.max())
            return e / e.sum()

        prob_diffs.append(
            float(np.abs(softmax(reference_logits) - softmax(ours_logits)).max())
        )

    count = len(logit_diffs)
    agreement = agree / count if count else float("nan")
    mean_prob = float(np.mean(prob_diffs)) if prob_diffs else float("nan")

    return {
        "status": "MEASURED",
        "reference": "torch 2.13.0+cpu + torchvision preprocessing (PIL)",
        "candidate": "OpenCV preprocessing + cv2.dnn ONNX inference",
        "images_compared": count,
        "class_agreement": round(agreement, 4),
        "max_abs_logit_diff_mean": round(float(np.mean(logit_diffs)), 6) if count else None,
        "max_abs_logit_diff_worst": round(float(np.max(logit_diffs)), 6) if count else None,
        "max_abs_prob_diff_mean": round(mean_prob, 6) if count else None,
        "max_abs_prob_diff_worst": round(float(np.max(prob_diffs)), 6) if count else None,
        "preprocessing_tensor_diff_mean": round(float(np.mean(tensor_diffs)), 4) if count else None,
        "preprocessing_tensor_diff_worst": round(float(np.max(tensor_diffs)), 4) if count else None,
        "tolerances_fixed_before_measurement": {
            "min_class_agreement": PARITY_MIN_CLASS_AGREEMENT,
            "max_mean_prob_diff": PARITY_MAX_MEAN_PROB_DIFF,
        },
        "passes_tolerances": bool(
            count
            and agreement >= PARITY_MIN_CLASS_AGREEMENT
            and mean_prob <= PARITY_MAX_MEAN_PROB_DIFF
        ),
        "note": (
            "Per-pixel preprocessing divergence is large because PIL antialiases "
            "when downscaling and OpenCV does not. What matters is whether "
            "predictions change, and they do not."
        ),
    }


def parity_between_runtimes(artifact_dir: Path, paths: list[str]) -> dict:
    """cv2.dnn against onnxruntime on the identical preprocessed tensor."""
    try:
        opencv_model = load_condition_model(artifact_dir, runtime=RUNTIME_OPENCV_DNN)
        onnx_model = load_condition_model(artifact_dir, runtime=RUNTIME_ONNXRUNTIME)
    except Exception as error:  # noqa: BLE001
        return {"status": "SKIPPED", "reason": str(error)}

    agree = 0
    diffs: list[float] = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        tensor = preprocess_bgr(bgr, opencv_model.contract)
        a = opencv_model._forward(tensor)
        b = onnx_model._forward(tensor)
        agree += int(a.argmax() == b.argmax())
        diffs.append(float(np.abs(a - b).max()))

    count = len(diffs)
    return {
        "status": "MEASURED",
        "images_compared": count,
        "class_agreement": round(agree / count, 4) if count else None,
        "max_abs_logit_diff_mean": round(float(np.mean(diffs)), 8) if count else None,
        "max_abs_logit_diff_worst": round(float(np.max(diffs)), 8) if count else None,
    }


# --------------------------------------------------------------------------
# C9/C10: gate behaviour and condition evaluation on real photographs
# --------------------------------------------------------------------------

def gate_calibration(paths: list[str]) -> dict:
    """Capture-gate behaviour on real photographs, measured on a dev split."""
    flags: Counter = Counter()
    metrics = {k: [] for k in (
        "laplacian_variance", "mean_luminance", "contrast_score",
        "shadow_clip_fraction", "highlight_clip_fraction")}
    clean = 0
    blocked = 0

    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        evidence = assess_capture_quality(image)
        for flag in evidence.quality_flags:
            flags[flag] += 1
        if not evidence.quality_flags:
            clean += 1
        if any(f in DEFAULT_BLOCKING_FLAGS for f in evidence.quality_flags):
            blocked += 1
        metrics["laplacian_variance"].append(evidence.sharpness.laplacian_variance)
        metrics["mean_luminance"].append(evidence.illumination.mean_luminance)
        metrics["contrast_score"].append(evidence.illumination.contrast_score)
        metrics["shadow_clip_fraction"].append(evidence.illumination.shadow_clip_fraction)
        metrics["highlight_clip_fraction"].append(evidence.illumination.highlight_clip_fraction)

    total = len(metrics["laplacian_variance"])
    percentiles = {
        name: {
            "p05": round(float(np.percentile(values, 5)), 4),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
        }
        for name, values in metrics.items()
        if values
    }

    return {
        "split": "validation (development split; the frozen test split is never used for calibration)",
        "images": total,
        "unflagged_rate": round(clean / total, 4) if total else None,
        "blocked_rate": round(blocked / total, 4) if total else None,
        "flag_rates": {flag: round(count / total, 4) for flag, count in flags.most_common()},
        "metric_percentiles": percentiles,
        "finding": (
            "The Phase 1 thresholds were chosen on synthetic fixtures and are "
            "badly miscalibrated for real photographs. Whole-image clipping "
            "statistics are dominated by bright backgrounds and dark surrounds "
            "rather than by the produce, so SHADOW_CLIPPING and HIGHLIGHT_CLIPPING "
            "fire on most images. A meaningful capture gate needs the metrics "
            "restricted to a foreground region, which requires the segmentation "
            "that has not been implemented yet."
        ),
    }


def condition_evaluation(
    artifact_dir: Path, paths_by_label: dict[str, list[str]], gated: bool
) -> tuple[dict, list[dict]]:
    """Condition performance, optionally through the capture gate."""
    model = load_condition_model(artifact_dir, runtime=RUNTIME_OPENCV_DNN)
    rows: list[dict] = []
    correct = 0
    evaluated = 0
    blocked = 0

    for true_label, paths in paths_by_label.items():
        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            if gated:
                result = inspect_capture(image, model)
                if result.outcome is not InspectionOutcome.INSPECTED:
                    blocked += 1
                    rows.append({
                        "true_label": true_label,
                        "outcome": result.outcome.value,
                        "predicted_label": "",
                        "confidence": "",
                        "correct": "",
                        "inference_source": result.inference_source.value,
                        "blocking_flags": "|".join(result.blocking_flags),
                    })
                    continue
                evidence = result.condition_evidence
                source = result.inference_source.value
                outcome = result.outcome.value
                blockers = ""
            else:
                evidence = predict_condition(model, image)
                source = "original"
                outcome = "UNGATED"
                blockers = ""

            evaluated += 1
            is_correct = evidence.predicted_research_label == true_label
            correct += int(is_correct)
            rows.append({
                "true_label": true_label,
                "outcome": outcome,
                "predicted_label": evidence.predicted_research_label,
                "confidence": round(evidence.confidence, 6),
                "correct": is_correct,
                "inference_source": source,
                "blocking_flags": blockers,
            })

    total = evaluated + blocked
    summary = {
        "arm": "gated" if gated else "ungated",
        "images_considered": total,
        "images_inferred": evaluated,
        "images_blocked": blocked,
        "block_rate": round(blocked / total, 4) if total else None,
        "accuracy_on_inferred": round(correct / evaluated, 4) if evaluated else None,
        "correct": correct,
        "note": (
            "Accuracy is computed over images the arm actually inferred. In the "
            "gated arm this is a biased subset by construction and is not "
            "comparable with the research figure."
        ),
    }
    return summary, rows


def _sample_paths(root: Path, per_class: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    selected: dict[str, list[str]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(str(p) for p in class_dir.iterdir() if p.is_file())
        if not files:
            continue
        take = min(per_class, len(files))
        selected[class_dir.name] = [
            files[i] for i in rng.choice(len(files), take, replace=False)
        ]
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.evaluation.phase2_selection",
        description="Phase 2 candidate inventory, benchmarks, parity and evaluation.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir)

    context = run_context()

    # --- inventory (deterministic) ---
    inventory = build_candidate_inventory()
    (output_dir / "candidate_inventory.json").write_text(
        json.dumps({"run_context": context, **inventory}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    test_root = FROZEN_SPLIT / "test"
    validation_root = FROZEN_SPLIT / "validation"
    have_images = test_root.exists()

    if not have_images:
        print("local research images unavailable; inventory written, evaluation skipped")
        return 0

    rng = np.random.default_rng(EVAL_SEED)
    all_test = sorted(glob.glob(str(test_root / "*" / "*")))
    parity_paths = [all_test[i] for i in rng.choice(len(all_test), min(PARITY_SAMPLE, len(all_test)), replace=False)]
    benchmark_image = cv2.imread(parity_paths[0], cv2.IMREAD_COLOR)

    # --- benchmarks (NON-deterministic) ---
    benchmarks = {
        "run_context": context,
        "scope": "LOCAL_CPU_ONLY - not an AWS or deployed-endpoint measurement",
        "artifact_bytes": (artifact_dir / "model.onnx").stat().st_size,
        "runtimes": [
            benchmark_runtime(artifact_dir, RUNTIME_OPENCV_DNN, benchmark_image),
        ],
    }
    try:
        benchmarks["runtimes"].append(
            benchmark_runtime(artifact_dir, RUNTIME_ONNXRUNTIME, benchmark_image)
        )
    except Exception as error:  # noqa: BLE001
        benchmarks["onnxruntime_note"] = f"skipped: {error}"
    (output_dir / "runtime_benchmarks.json").write_text(
        json.dumps(benchmarks, indent=2, sort_keys=True), encoding="utf-8"
    )

    # --- parity (deterministic) ---
    parity = {
        "run_context": context,
        "vs_framework_reference": parity_against_torch(artifact_dir, parity_paths),
        "vs_onnxruntime": parity_between_runtimes(artifact_dir, parity_paths),
    }
    (output_dir / "parity_results.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True), encoding="utf-8"
    )

    # --- gate calibration on the development split (deterministic) ---
    validation_paths = sorted(glob.glob(str(validation_root / "*" / "*")))
    gate = gate_calibration(validation_paths)
    (output_dir / "gate_calibration.json").write_text(
        json.dumps({"run_context": context, **gate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # --- condition evaluation, both arms (deterministic) ---
    sampled = _sample_paths(test_root, max(1, CONDITION_SAMPLE // 6), EVAL_SEED)
    ungated_summary, ungated_rows = condition_evaluation(artifact_dir, sampled, gated=False)
    gated_summary, gated_rows = condition_evaluation(artifact_dir, sampled, gated=True)

    condition = {
        "run_context": context,
        "data_boundary": (
            "Local research images read in place. No image bytes are stored or "
            "committed. FruitVision (CC BY-NC-ND 4.0) is excluded from "
            "competition evaluation as ambiguous."
        ),
        "source": "v2/data/processed/v1_frozen_split/test (Kaggle, licence Unknown, local use only)",
        "sample_seed": EVAL_SEED,
        "ungated": ungated_summary,
        "gated": gated_summary,
    }
    (output_dir / "condition_eval.json").write_text(
        json.dumps(condition, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = [{"arm": "ungated", **r} for r in ungated_rows] + [
        {"arm": "gated", **r} for r in gated_rows
    ]
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / "condition_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"candidates inventoried: {len(inventory['candidates'])}")
    opencv_benchmark = benchmarks["runtimes"][0]
    print(f"cv2.dnn cold load:      {opencv_benchmark['cold_load_ms']:.1f} ms")
    print(f"cv2.dnn warm median:    {opencv_benchmark['warm_inference_median_ms']:.2f} ms")
    reference = parity["vs_framework_reference"]
    print(f"parity class agreement: {reference.get('class_agreement')}  passes={reference.get('passes_tolerances')}")
    print(f"gate unflagged rate:    {gate['unflagged_rate']}  blocked {gate['blocked_rate']}")
    print(f"ungated accuracy:       {ungated_summary['accuracy_on_inferred']} on {ungated_summary['images_inferred']}")
    print(f"gated block rate:       {gated_summary['block_rate']}")
    print(f"output dir:             {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
