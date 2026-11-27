"""Automatic 4-bit / 8-bit quantization configuration for large models based on live GPU VRAM."""

from __future__ import annotations

from typing import Any, Optional


def _detect_live_vram_bytes(torch_mod: Any) -> Optional[int]:
    try:
        if torch_mod.cuda.is_available():
            props = torch_mod.cuda.get_device_properties(0)
            total = int(getattr(props, "total_memory", 0))
            if total > 0:
                return total
    except Exception:
        pass
    return None


def resolve_quantization_config(
    torch_mod: Any,
    *,
    model_size_bytes: int,
    total_vram_bytes: Optional[int] = None,
    force_4bit: bool = False,
) -> tuple[Optional[Any], Any]:
    """
    Determine if 4-bit (NF4 / bitsandbytes) quantization is needed to fit within live VRAM.
    Uses native bf16/fp16 when weights + KV headroom fit (e.g. A100 40/80 GiB), and automatically
    switches to 4-bit NF4 on smaller GPUs (e.g. L4 22 GiB / T4 15 GiB).
    Returns: (quantization_config, torch_dtype).
    """
    from models.base import select_torch_dtype

    default_dtype = select_torch_dtype(torch_mod)
    effective_vram = total_vram_bytes if (total_vram_bytes and total_vram_bytes > 0) else _detect_live_vram_bytes(torch_mod)
    vram_gib = (effective_vram / (1024 ** 3)) if (effective_vram and effective_vram > 0) else 22.0
    model_gib = model_size_bytes / (1024 ** 3)

    needs_4bit = force_4bit or (model_gib * 1.15 > vram_gib and model_gib > 8.0)
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
        return None, default_dtype
