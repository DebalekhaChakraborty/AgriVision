"""Single registry for the three registered Phase 2 architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from v2.src.models.efficientnet_transfer import build_efficientnet_transfer
from v2.src.models.frozen_backbone import FrozenBackboneClassifier
from v2.src.models.mobilenetv3_transfer import build_mobilenetv3_transfer
from v2.src.models.resnet50_transfer import build_resnet50_transfer


MODEL_BUILDERS: dict[str, Callable[[int, str | None], FrozenBackboneClassifier]] = {
    "resnet50": build_resnet50_transfer,
    "efficientnet_b0": build_efficientnet_transfer,
    "mobilenet_v3_large": build_mobilenetv3_transfer,
}


def build_transfer_model(
    model_config: dict[str, Any],
    load_pretrained: bool,
) -> FrozenBackboneClassifier:
    architecture = str(model_config["architecture"])
    if architecture not in MODEL_BUILDERS:
        raise ValueError(f"Unregistered transfer architecture: {architecture}")
    weights_name = (
        str(model_config["pretrained_weights"]) if load_pretrained else None
    )
    return MODEL_BUILDERS[architecture](
        int(model_config["num_classes"]),
        weights_name,
    )
