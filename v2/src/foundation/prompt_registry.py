"""Load and validate the frozen Phase 3B semantic prompt registry."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as functional
import yaml


STRICT_P1_TEXTS = {
    "fresh_apple": "a photo of a fresh apple",
    "fresh_banana": "a photo of a fresh banana",
    "fresh_orange": "a photo of a fresh orange",
    "rotten_apple": "a photo of a rotten apple",
    "rotten_banana": "a photo of a rotten banana",
    "rotten_orange": "a photo of a rotten orange",
}


class TextEncoder(Protocol):
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return one unnormalized embedding per text."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenPromptRegistry:
    path: Path
    sha256: str
    class_order: tuple[str, ...]
    families: dict[str, dict[str, Any]]
    simplicity_order: tuple[str, ...]

    def prompts_for(self, family: str, class_name: str) -> list[str]:
        return list(self.families[family]["prompts"][class_name])


def load_prompt_registry(path: Path, expected_sha256: str) -> FrozenPromptRegistry:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Prompt registry differs from the pre-registered SHA-256: "
            f"{actual_sha256} != {expected_sha256}."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload.get("status") != "frozen_before_validation":
        raise ValueError("Prompt registry is not marked frozen before validation.")
    families = payload.get("families", {})
    if list(families) != ["P1", "P2", "P3", "P4"]:
        raise ValueError("Prompt registry must contain only ordered P1-P4 families.")
    class_order = tuple(payload.get("class_order", ()))
    if class_order != tuple(STRICT_P1_TEXTS):
        raise ValueError("Prompt class order differs from the registered mapping.")
    for family_name, family in families.items():
        prompts = family.get("prompts", {})
        if tuple(prompts) != class_order:
            raise ValueError(f"{family_name} prompt classes are incomplete or reordered.")
        expected_count = 4 if family_name == "P4" else 1
        if any(len(prompts[name]) != expected_count for name in class_order):
            raise ValueError(f"{family_name} contains an invalid prompt count.")
    p1 = {
        name: families["P1"]["prompts"][name][0] for name in class_order
    }
    if p1 != STRICT_P1_TEXTS or not families["P1"].get("strict_zero_shot"):
        raise ValueError("Strict P1 texts differ from the canonical protocol.")
    simplicity_order = tuple(payload["selection"]["simplicity_order"])
    if simplicity_order != ("P1", "P2", "P3", "P4"):
        raise ValueError("Prompt simplicity order differs from the protocol.")
    return FrozenPromptRegistry(
        path=path,
        sha256=actual_sha256,
        class_order=class_order,
        families=families,
        simplicity_order=simplicity_order,
    )


def encode_class_prototypes(
    model: TextEncoder,
    registry: FrozenPromptRegistry,
    family: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode, equally average, and renormalize one prototype per class."""

    if family not in registry.families:
        raise ValueError(f"Unregistered prompt family: {family}")
    class_prototypes: list[torch.Tensor] = []
    prompt_counts: dict[str, int] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for class_name in registry.class_order:
            prompts = registry.prompts_for(family, class_name)
            embeddings = model.encode_texts(prompts).float()
            embeddings = functional.normalize(embeddings, p=2, dim=1)
            prototype = embeddings.mean(dim=0, keepdim=True)
            prototype = functional.normalize(prototype, p=2, dim=1)
            class_prototypes.append(prototype)
            prompt_counts[class_name] = len(prompts)
    elapsed = time.perf_counter() - started
    prototypes = torch.cat(class_prototypes, dim=0)
    norms = torch.linalg.vector_norm(prototypes, ord=2, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6):
        raise ValueError("Text class prototypes are not L2-normalized.")
    return prototypes, {
        "family": family,
        "prompt_counts": prompt_counts,
        "total_prompts": sum(prompt_counts.values()),
        "equal_weighting": True,
        "normalize_each_prompt": True,
        "renormalize_class_prototype": True,
        "text_embedding_seconds": elapsed,
    }
