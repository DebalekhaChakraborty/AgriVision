"""Deterministic directory loader for the frozen six-class fruit dataset."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = (
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def default_transform(image_size: int) -> Callable:
    """Match V1 baseline preprocessing: resize and scale pixels to [0, 1]."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class FruitFreshnessDataset(Dataset):
    """Read a frozen split and return an RGB tensor with its integer label."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_size: int,
        transform: Callable | None = None,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.split_root = self.dataset_root / split
        self.class_names = tuple(class_names)
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.transform = transform or default_transform(image_size)

        if not self.split_root.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {self.split_root}")

        actual_classes = {
            path.name for path in self.split_root.iterdir() if path.is_dir()
        }
        expected_classes = set(self.class_names)
        if actual_classes != expected_classes:
            missing = sorted(expected_classes - actual_classes)
            unexpected = sorted(actual_classes - expected_classes)
            raise ValueError(
                f"Invalid classes in {self.split_root}; "
                f"missing={missing}, unexpected={unexpected}"
            )

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_root = self.split_root / class_name
            class_files = sorted(
                path
                for path in class_root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not class_files:
                raise ValueError(f"No supported images found in {class_root}")
            class_index = self.class_to_idx[class_name]
            self.samples.extend((path, class_index) for path in class_files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = self.transform(rgb_image)
        return image_tensor, label

    def class_counts(self) -> dict[str, int]:
        counts = Counter(label for _, label in self.samples)
        return {
            class_name: counts[index]
            for class_name, index in self.class_to_idx.items()
        }

    def relative_paths(self) -> list[str]:
        return [str(path.relative_to(self.dataset_root)) for path, _ in self.samples]


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    dataset: FruitFreshnessDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
