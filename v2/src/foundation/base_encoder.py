"""Common interface and integrity helpers for frozen foundation encoders."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch import nn


def checkpoint_file_record(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": digest.hexdigest(),
    }


class BaseFoundationEncoder(ABC):
    """Expose deterministic preprocessing and one frozen image embedding."""

    def __init__(self, encoder_config: dict[str, Any], device: torch.device) -> None:
        self.encoder_config = encoder_config
        self.device = device
        self.model: nn.Module
        self.preprocess: Callable[[Image.Image], torch.Tensor]
        self._checkpoint_record: dict[str, Any]

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized two-dimensional embedding tensor per image."""

    @abstractmethod
    def encoder_parameter_count(self) -> int:
        """Return parameters belonging to the image representation path only."""

    @abstractmethod
    def runtime_metadata(self) -> dict[str, Any]:
        """Return resolved representation and preprocessing metadata."""

    def freeze_and_validate(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("A foundation encoder parameter remains trainable.")
        if self.model.training:
            raise RuntimeError("The foundation encoder must remain in evaluation mode.")

    def validate_embeddings(self, embeddings: torch.Tensor) -> None:
        expected_dim = int(self.encoder_config["embedding_dim"])
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Expected embeddings shaped (batch, {expected_dim}), "
                f"received {tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("Encoder produced a non-finite embedding.")

    @property
    def checkpoint_record(self) -> dict[str, Any]:
        return dict(self._checkpoint_record)
