"""Common frozen-backbone classifier used by Phase 2 transfer models."""

from __future__ import annotations

import torch
from torch import nn


class FrozenBackboneClassifier(nn.Module):
    """Keep a pretrained feature extractor fixed and train one linear classifier."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenBackboneClassifier":
        super().train(mode)
        self.backbone.eval()
        self.classifier.train(mode)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(inputs)
        return self.classifier(features)

    @property
    def total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
