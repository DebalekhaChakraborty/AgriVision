"""One registry for the three Phase 3A frozen encoder adapters."""

from __future__ import annotations

from typing import Any

import torch

from v2.src.foundation.base_encoder import BaseFoundationEncoder
from v2.src.foundation.clip_encoder import CLIPEncoder
from v2.src.foundation.dinov2_encoder import DINOv2Encoder
from v2.src.foundation.siglip2_encoder import SigLIP2Encoder


ENCODERS: dict[str, type[BaseFoundationEncoder]] = {
    "dinov2": DINOv2Encoder,
    "clip": CLIPEncoder,
    "siglip2": SigLIP2Encoder,
}


def build_foundation_encoder(
    encoder_config: dict[str, Any],
    device: torch.device,
) -> BaseFoundationEncoder:
    adapter = str(encoder_config["adapter"])
    if adapter not in ENCODERS:
        raise ValueError(f"Unregistered foundation encoder adapter: {adapter}")
    return ENCODERS[adapter](encoder_config, device)
