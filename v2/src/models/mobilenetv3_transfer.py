"""ImageNet-pretrained MobileNetV3-Large with a frozen six-class head."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from v2.src.models.frozen_backbone import FrozenBackboneClassifier


def build_mobilenetv3_transfer(
    num_classes: int,
    weights_name: str | None,
) -> FrozenBackboneClassifier:
    weights = MobileNet_V3_Large_Weights[weights_name] if weights_name else None
    network = mobilenet_v3_large(weights=weights)
    feature_dim = int(network.classifier[0].in_features)
    backbone = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
    return FrozenBackboneClassifier(backbone, feature_dim, num_classes)
