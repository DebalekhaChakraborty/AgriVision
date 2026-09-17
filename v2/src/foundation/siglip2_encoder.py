"""Pinned Transformers vision-only adapter for SigLIP2 Base Patch16/224."""

from __future__ import annotations

from typing import Any

import torch
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, SiglipModel

from v2.src.foundation.base_encoder import (
    BaseFoundationEncoder,
    checkpoint_file_record,
)


class SigLIP2Encoder(BaseFoundationEncoder):
    def __init__(self, encoder_config: dict[str, Any], device: torch.device) -> None:
        super().__init__(encoder_config, device)
        checkpoint_id = str(encoder_config["checkpoint_id"])
        revision = str(encoder_config["revision"])
        cache_dir = str(encoder_config["download_cache"])
        self.processor = AutoImageProcessor.from_pretrained(
            checkpoint_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        # This fixed-resolution SigLIP2 checkpoint intentionally retains the
        # SigLIP-compatible config/model type. Load the combined checkpoint so
        # Transformers applies the correct nested vision configuration, then
        # keep only the pretrained image tower for this image-only benchmark.
        combined_model = SiglipModel.from_pretrained(
            checkpoint_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
            use_safetensors=True,
        )
        self.model = combined_model.vision_model.to(device)
        del combined_model
        self.preprocess = self._preprocess
        checkpoint_path = hf_hub_download(
            repo_id=checkpoint_id,
            filename="model.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self._checkpoint_record = checkpoint_file_record(checkpoint_path)
        self.freeze_and_validate()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        embeddings = outputs.pooler_output
        self.validate_embeddings(embeddings)
        return embeddings

    def encoder_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "adapter": "SigLIP2Encoder",
            "checkpoint_id": self.encoder_config["checkpoint_id"],
            "revision": self.encoder_config["revision"],
            "library": "transformers",
            "library_version": transformers.__version__,
            "embedding_dim": int(self.encoder_config["embedding_dim"]),
            "encoder_parameters": self.encoder_parameter_count(),
            "all_parameters_frozen": True,
            "vision_encoder_only": True,
            "preprocessing": self.processor.to_dict(),
            "checkpoint_file": self.checkpoint_record,
        }
