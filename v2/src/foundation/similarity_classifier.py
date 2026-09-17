"""Shared frozen image-text models and native similarity scoring for Phase 3B."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import open_clip
import torch
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel

from v2.src.foundation.base_encoder import checkpoint_file_record


class FrozenSimilarityModel(ABC):
    """One frozen image-text representation with model-native logit scaling."""

    def __init__(self, model_config: dict[str, Any], device: torch.device) -> None:
        self.model_config = model_config
        self.device = device
        self.model: nn.Module
        self.preprocess: Callable[[Image.Image], torch.Tensor]
        self.checkpoint_record: dict[str, Any]

    def freeze_and_validate(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("The zero-shot vision-language model is not frozen.")
        expected = str(self.model_config["checkpoint_sha256"])
        if self.checkpoint_record["sha256"] != expected:
            raise ValueError("Checkpoint SHA-256 differs from Phase 3A provenance.")
        if self.checkpoint_record["size_bytes"] != int(
            self.model_config["checkpoint_size_bytes"]
        ):
            raise ValueError("Checkpoint size differs from Phase 3A provenance.")

    @abstractmethod
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized projected image embedding per image."""

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return one unnormalized projected text embedding per text."""

    def scaled_logits(self, cosine_similarities: torch.Tensor) -> torch.Tensor:
        scale, bias = self.native_logit_parameters()
        return cosine_similarities * scale + bias

    @abstractmethod
    def native_logit_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frozen scalar native logit scale and bias."""

    @abstractmethod
    def runtime_metadata(self) -> dict[str, Any]:
        """Return exact model, parameter, processor, and checkpoint metadata."""

    def validate_embeddings(self, embeddings: torch.Tensor) -> None:
        expected_dim = int(self.model_config["embedding_dim"])
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Expected (*, {expected_dim}) embeddings, got {tuple(embeddings.shape)}."
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("Model produced non-finite embeddings.")


class CLIPZeroShotModel(FrozenSimilarityModel):
    """Original OpenAI CLIP ViT-B/16 through pinned OpenCLIP."""

    def __init__(self, model_config: dict[str, Any], device: torch.device) -> None:
        super().__init__(model_config, device)
        checkpoint_path = hf_hub_download(
            repo_id=str(model_config["checkpoint_id"]),
            filename=str(model_config["checkpoint_filename"]),
            revision=str(model_config["revision"]),
            cache_dir=str(model_config["download_cache"]),
            local_files_only=True,
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            str(model_config["model_name"]),
            pretrained=checkpoint_path,
            force_quick_gelu=True,
            image_mean=tuple(float(x) for x in model_config["image_mean"]),
            image_std=tuple(float(x) for x in model_config["image_std"]),
            image_interpolation=str(model_config["interpolation"]),
            image_resize_mode=str(model_config["resize_mode"]),
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(str(model_config["model_name"]))
        self.checkpoint_record = checkpoint_file_record(checkpoint_path)
        self.freeze_and_validate()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode_image(images, normalize=False)
        self.validate_embeddings(embeddings)
        return embeddings

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        embeddings = self.model.encode_text(tokens, normalize=False)
        self.validate_embeddings(embeddings)
        return embeddings

    def native_logit_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.model.logit_scale.exp()
        bias = (
            self.model.logit_bias
            if getattr(self.model, "logit_bias", None) is not None
            else torch.zeros((), device=self.device)
        )
        return scale, bias

    def runtime_metadata(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        visual = sum(parameter.numel() for parameter in self.model.visual.parameters())
        scale, bias = self.native_logit_parameters()
        return {
            "adapter": "CLIPZeroShotModel",
            "model_name": self.model_config["model_name"],
            "pretrained_identifier": self.model_config["pretrained_identifier"],
            "checkpoint_id": self.model_config["checkpoint_id"],
            "revision": self.model_config["revision"],
            "library": "open_clip_torch",
            "library_version": open_clip.__version__,
            "embedding_dim": int(self.model_config["embedding_dim"]),
            "total_model_parameters": total,
            "image_encoder_parameters": visual,
            "non_visual_and_shared_parameters": total - visual,
            "all_parameters_frozen": True,
            "native_logit_scale": float(scale.detach().cpu()),
            "native_logit_bias": float(bias.detach().cpu()),
            "preprocessing": {
                "input_resolution": int(self.model_config["input_resolution"]),
                "resize_mode": self.model_config["resize_mode"],
                "interpolation": self.model_config["interpolation"],
                "center_crop": True,
                "convert_rgb": True,
                "image_mean": list(self.model_config["image_mean"]),
                "image_std": list(self.model_config["image_std"]),
            },
            "checkpoint_file": self.checkpoint_record,
        }


class SigLIP2ZeroShotModel(FrozenSimilarityModel):
    """Pinned SigLIP2 Base fixed-resolution image-text checkpoint."""

    def __init__(self, model_config: dict[str, Any], device: torch.device) -> None:
        super().__init__(model_config, device)
        checkpoint_id = str(model_config["checkpoint_id"])
        revision = str(model_config["revision"])
        cache_dir = str(model_config["download_cache"])
        common = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": True,
        }
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_id, **common)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, **common)
        self.model = SiglipModel.from_pretrained(
            checkpoint_id,
            use_safetensors=True,
            **common,
        ).to(device)
        self.preprocess = self._preprocess
        checkpoint_path = hf_hub_download(
            repo_id=checkpoint_id,
            filename=str(model_config["checkpoint_filename"]),
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.checkpoint_record = checkpoint_file_record(checkpoint_path)
        self.freeze_and_validate()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    @staticmethod
    def _pooled(output: Any) -> torch.Tensor:
        return output if isinstance(output, torch.Tensor) else output.pooler_output

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self._pooled(self.model.get_image_features(pixel_values=images))
        self.validate_embeddings(embeddings)
        return embeddings

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=int(self.model_config["text_max_length"]),
            truncation=True,
            return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        embeddings = self._pooled(self.model.get_text_features(**tokens))
        self.validate_embeddings(embeddings)
        return embeddings

    def native_logit_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.logit_scale.exp(), self.model.logit_bias

    def runtime_metadata(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        visual = sum(
            parameter.numel() for parameter in self.model.vision_model.parameters()
        )
        text = sum(parameter.numel() for parameter in self.model.text_model.parameters())
        scale, bias = self.native_logit_parameters()
        return {
            "adapter": "SigLIP2ZeroShotModel",
            "checkpoint_id": self.model_config["checkpoint_id"],
            "revision": self.model_config["revision"],
            "library": "transformers",
            "library_version": transformers.__version__,
            "embedding_dim": int(self.model_config["embedding_dim"]),
            "total_model_parameters": total,
            "image_encoder_parameters": visual,
            "text_encoder_parameters": text,
            "shared_similarity_parameters": total - visual - text,
            "all_parameters_frozen": True,
            "native_logit_scale": float(scale.detach().cpu()),
            "native_logit_bias": float(bias.detach().cpu()),
            "text_max_length": int(self.model_config["text_max_length"]),
            "preprocessing": self.processor.to_dict(),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": int(self.tokenizer.vocab_size),
            "checkpoint_file": self.checkpoint_record,
        }


def build_zero_shot_model(
    model_config: dict[str, Any], device: torch.device
) -> FrozenSimilarityModel:
    adapters = {
        "clip_zero_shot": CLIPZeroShotModel,
        "siglip2_zero_shot": SigLIP2ZeroShotModel,
    }
    adapter = str(model_config["adapter"])
    if adapter not in adapters:
        raise ValueError(f"Unregistered zero-shot adapter: {adapter}")
    return adapters[adapter](model_config, device)
