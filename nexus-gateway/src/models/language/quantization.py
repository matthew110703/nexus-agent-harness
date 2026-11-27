"""Automatic 4-bit / 8-bit quantization configuration for large models on 16GB-24GB GPUs."""

from __future__ import annotations

import os
from typing import Any, Optional


def resolve_quantization_config(
    torch_mod: Any,
    *,
    model_size_bytes: int,
    total_vram_bytes: Optional[int] = None,
    force_4bit: bool = False,
) -> tuple[Optional[Any], Any]:
    """
    Determine if 4-bit (NF4 / bitsandbytes) quantization is needed to fit within VRAM.
    Returns: (quantization_config, torch_dtype).
    """
    from models.base import select_torch_dtype
    default_dtype = select_torch_dtype(torch_mod)

    # If force_4bit is requested or model weights > 20GB on a <= 24GB GPU, use bitsandbytes 4-bit NF4
    vram_gib = (total_vram_bytes / (1024**3)) if (total_vram_bytes and total_vram_bytes > 0) else 22.0
    model_gib = model_size_bytes / (1024**3)

    needs_4bit = force_4bit or (model_gib * 1.15 > vram_gib and model_gib > 10.0)
    if not needs_4bit:
        return None, default_dtype

    try:
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=default_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return q_config, default_dtype
    except Exception:
        # If bitsandbytes is not installed, fallback to native precision
        return None, default_dtype
