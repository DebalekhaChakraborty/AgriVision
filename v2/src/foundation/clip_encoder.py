"""Pinned OpenCLIP adapter for the original OpenAI CLIP ViT-B/16."""

from __future__ import annotations

from typing import Any

import open_clip
import torch
from huggingface_hub import hf_hub_download

from v2.src.foundation.base_encoder import (
    BaseFoundationEncoder,
    checkpoint_file_record,
)


class CLIPEncoder(BaseFoundationEncoder):
    def __init__(self, encoder_config: dict[str, Any], device: torch.device) -> None:
        super().__init__(encoder_config, device)
        checkpoint_path = hf_hub_download(
            repo_id=str(encoder_config["checkpoint_id"]),
            filename=str(encoder_config["checkpoint_filename"]),
            revision=str(encoder_config["revision"]),
            cache_dir=str(encoder_config["download_cache"]),
            local_files_only=True,
        )
        mean = tuple(float(value) for value in encoder_config["image_mean"])
        std = tuple(float(value) for value in encoder_config["image_std"])
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            str(encoder_config["model_name"]),
            pretrained=checkpoint_path,
            force_quick_gelu=True,
            image_mean=mean,
            image_std=std,
            image_interpolation=str(encoder_config["interpolation"]),
            image_resize_mode=str(encoder_config["resize_mode"]),
            device=device,
        )
        self._checkpoint_record = checkpoint_file_record(checkpoint_path)
        self.freeze_and_validate()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode_image(images, normalize=False)
        self.validate_embeddings(embeddings)
        return embeddings

    def encoder_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.visual.parameters())

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "adapter": "CLIPEncoder",
            "model_name": self.encoder_config["model_name"],
            "pretrained_identifier": self.encoder_config["pretrained_identifier"],
            "checkpoint_id": self.encoder_config["checkpoint_id"],
            "revision": self.encoder_config["revision"],
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "embedding_dim": int(self.encoder_config["embedding_dim"]),
            "encoder_parameters": self.encoder_parameter_count(),
            "all_parameters_frozen": True,
            "preprocessing": {
                "input_resolution": int(self.encoder_config["input_resolution"]),
                "resize_mode": self.encoder_config["resize_mode"],
                "interpolation": self.encoder_config["interpolation"],
                "center_crop": True,
                "convert_rgb": True,
                "image_mean": list(self.encoder_config["image_mean"]),
                "image_std": list(self.encoder_config["image_std"]),
            },
            "checkpoint_file": self.checkpoint_record,
        }
