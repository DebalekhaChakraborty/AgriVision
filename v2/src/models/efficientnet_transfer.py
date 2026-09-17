"""ImageNet-pretrained EfficientNet-B0 with a frozen six-class head."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from v2.src.models.frozen_backbone import FrozenBackboneClassifier


def build_efficientnet_transfer(
    num_classes: int,
    weights_name: str | None,
) -> FrozenBackboneClassifier:
    weights = EfficientNet_B0_Weights[weights_name] if weights_name else None
    network = efficientnet_b0(weights=weights)
    feature_dim = int(network.classifier[-1].in_features)
    backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
    return FrozenBackboneClassifier(backbone, feature_dim, num_classes)
