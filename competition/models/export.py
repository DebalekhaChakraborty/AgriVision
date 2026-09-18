"""Export a research checkpoint to ONNX for the competition runtime.

**Build-time only.** This is the one place torch appears on the competition
side, and it never runs in the serving path. The whole point of the exercise is
that inference afterwards needs only OpenCV.

Boundaries this module respects:

* The research checkpoint is **read, never written**. No research file, record,
  manifest or weight is modified.
* The architecture is rebuilt from `torchvision` plus the `model_config` the
  checkpoint carries, rather than by importing `v2.src`. The competition tree
  does not import the research tree, and a test enforces that.
* Exported weights are a build product: gitignored, like every other weight in
  this repository. Only the manifest is committed.

Run with:

    .venv-competition/bin/python -m competition.models.export
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_CHECKPOINT = Path("v2/checkpoints/best_mobilenetv3.pt")
DEFAULT_ARTIFACT_DIR = Path("competition/models/artifacts/mobilenetv3_large_v2exp004")
OPSET_VERSION = 17

# Architectures this exporter knows how to rebuild without importing research
# code. Extend deliberately: a wrong rebuild silently produces a working model
# with the wrong weights loaded.
SUPPORTED_ARCHITECTURES = ("mobilenet_v3_large", "efficientnet_b0", "resnet50")


class ExportError(RuntimeError):
    """Raised when a checkpoint cannot be exported."""


def _build_module(architecture: str, feature_dim: int, num_classes: int):
    """Rebuild the frozen-backbone classifier used by the V2 transfer experiments."""
    import torch
    from torch import nn
    import torchvision.models as tvm

    if architecture == "mobilenet_v3_large":
        network = tvm.mobilenet_v3_large(weights=None)
        backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
    elif architecture == "efficientnet_b0":
        network = tvm.efficientnet_b0(weights=None)
        backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
    elif architecture == "resnet50":
        network = tvm.resnet50(weights=None)
        backbone = nn.Sequential(
            *list(network.children())[:-1], nn.Flatten(1)
        )
    else:
        raise ExportError(
            f"unsupported architecture {architecture!r}; "
            f"supported: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )

    class FrozenBackboneClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.backbone(x))

    return FrozenBackboneClassifier()


def export_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    opset: int = OPSET_VERSION,
) -> dict:
    """Export a checkpoint to ONNX and write the artifact manifest."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    artifact_dir = Path(artifact_dir)

    if not checkpoint_path.is_file():
        raise ExportError(f"checkpoint not found: {checkpoint_path.name}")

    # Read-only load. Nothing is written back to the research tree.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    class_to_idx = checkpoint["class_to_idx"]
    model_config = checkpoint.get("model_config", {})
    source_preprocessing = checkpoint.get("preprocessing", {})
    architecture = model_config.get("architecture")

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ExportError(
            f"checkpoint architecture {architecture!r} is not supported by this exporter"
        )

    class_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
    ]
    feature_dim = int(state_dict["classifier.weight"].shape[1])
    num_classes = int(state_dict["classifier.weight"].shape[0])
    if num_classes != len(class_names):
        raise ExportError("class count in the head does not match class_to_idx")

    module = _build_module(architecture, feature_dim, num_classes)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ExportError(
            f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected. "
            "The rebuilt architecture does not match the checkpoint."
        )
    module.eval()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifact_dir / "model.onnx"

    image_size = int(source_preprocessing.get("image_size", 224))
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        module,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
    )

    digest = hashlib.sha256()
    with onnx_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    manifest = {
        "model_id": f"{architecture}-v2exp{checkpoint.get('experiment_id', 'unknown')}",
        "model_family": architecture,
        "model_version": str(checkpoint.get("experiment_id", "unknown")),
        "source_experiment_id": checkpoint.get("experiment_id"),
        "source_checkpoint_name": checkpoint_path.name,
        "source_checkpoint_sha256_prefix": _file_prefix(checkpoint_path),
        "source_preprocessing": source_preprocessing,
        "class_names": class_names,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "input_shape": [1, 3, image_size, image_size],
        "opset_version": opset,
        "onnx_sha256_prefix": digest.hexdigest()[:32],
        "onnx_bytes": onnx_path.stat().st_size,
        "export_note": (
            "Build product derived from a gitignored research checkpoint. Not "
            "committed: the source dataset licence is recorded as Unknown with "
            "no redistribution grant, and this repository excludes trained "
            "weights from version control."
        ),
        "claim_boundary": (
            "Predicts visible surface condition only. Not a food-safety, "
            "edibility, contamination or internal-spoilage assessment."
        ),
    }

    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _file_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m competition.models.export",
        description="Export a V2 research checkpoint to ONNX for the competition runtime.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--opset", type=int, default=OPSET_VERSION)
    args = parser.parse_args(argv)

    manifest = export_checkpoint(
        Path(args.checkpoint), Path(args.artifact_dir), args.opset
    )
    print(f"model_id:    {manifest['model_id']}")
    print(f"classes:     {len(manifest['class_names'])}")
    print(f"onnx bytes:  {manifest['onnx_bytes']:,}")
    print(f"fingerprint: {manifest['onnx_sha256_prefix']}")
    print(f"artifact:    {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
