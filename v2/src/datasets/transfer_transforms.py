"""Shared ImageNet preprocessing for all V2 Phase 2 experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torchvision import transforms


def build_transfer_transform(
    split: str,
    preprocessing: dict[str, Any],
) -> transforms.Compose:
    """Build the registered train or evaluation transform without model-specific drift."""
    image_size = int(preprocessing["image_size"])
    mean = [float(value) for value in preprocessing["normalization_mean"]]
    std = [float(value) for value in preprocessing["normalization_std"]]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("ImageNet normalization requires three mean and std values.")

    if split == "train":
        augmentation = preprocessing["train_augmentation"]
        jitter = augmentation["color_jitter"]
        operations: Sequence = (
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            transforms.RandomRotation(float(augmentation["rotation_degrees"])),
            transforms.ColorJitter(
                brightness=float(jitter["brightness"]),
                contrast=float(jitter["contrast"]),
                saturation=float(jitter["saturation"]),
                hue=float(jitter["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        )
    elif split in {"validation", "test"}:
        operations = (
            transforms.Resize(int(preprocessing["evaluation_resize_size"])),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        )
    else:
        raise ValueError(f"Unsupported transfer-learning split: {split}")
    return transforms.Compose(list(operations))
