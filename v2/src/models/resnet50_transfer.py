"""ImageNet-pretrained ResNet50 with a frozen backbone and six-class head."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from v2.src.models.frozen_backbone import FrozenBackboneClassifier


def build_resnet50_transfer(
    num_classes: int,
    weights_name: str | None,
) -> FrozenBackboneClassifier:
    weights = ResNet50_Weights[weights_name] if weights_name else None
    network = resnet50(weights=weights)
    feature_dim = int(network.fc.in_features)
    network.fc = nn.Identity()
    return FrozenBackboneClassifier(network, feature_dim, num_classes)
