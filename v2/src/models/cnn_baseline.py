"""PyTorch reproduction of the conceptual V1 from-scratch CNN baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CNNBaseline(nn.Module):
    """Three convolution/pooling blocks followed by a 128-unit classifier."""

    def __init__(
        self,
        image_size: int,
        input_channels: int,
        conv_channels: Sequence[int],
        kernel_size: int,
        pool_size: int,
        dense_units: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError("The Phase 1 baseline requires exactly three conv blocks.")

        layers: list[nn.Module] = []
        channels_in = input_channels
        for channels_out in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=kernel_size,
                        padding=0,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size),
                ]
            )
            channels_in = channels_out
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            example = torch.zeros(1, input_channels, image_size, image_size)
            flattened_features = int(self.features(example).numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
