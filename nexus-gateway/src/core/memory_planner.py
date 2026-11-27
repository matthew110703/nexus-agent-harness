"""Dynamic GPU VRAM + System RAM memory planner for maximum speed without CUDA OOM."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

GIB = 1024 ** 3
MIB = 1024 ** 2

CompatibilityTier = Literal["RECOMMENDED", "CPU_OFFLOAD_POSSIBLE", "INSUFFICIENT_MEMORY"]

# DeepSeek Harness system prompt + ~18 tool schemas consumes ~4.5k-6k tokens on Turn 1.
HARNESS_MIN_SAFE_CONTEXT = 12288
HARNESS_DEFAULT_CONTEXT = 16384
HARNESS_HIGH_VRAM_CONTEXT = 32768
CUDA_WORKSPACE_RESERVE_BYTES = int(2.5 * GIB)
SYSTEM_RAM_OS_RESERVE_BYTES = int(6.0 * GIB)
CORESIDENT_SAFETY_HEADROOM_BYTES = int(3.5 * GIB)


@dataclass(frozen=True)
class RuntimeMemoryPlan:
    tier: CompatibilityTier
    n_ctx: int
    n_gpu_layers: int
    batch_size: int
    ubatch_size: int
    threads: int
    estimated_vram_bytes: int
    estimated_ram_offload_bytes: int
    estimated_free_vram_after_bytes: int
    can_coreside_image: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _estimate_kv_bytes_per_token(weight_bytes: int, architecture: str) -> int:
    """
    Conservative KV-cache bytes/token estimate in Q8_0 (or F16 fallback) for Harness planning.
    GQA/MoE models (Qwen3.5-35B-A3B, Gemma4, Llama-4-Scout, Llama-3.1-70B) use ~96-256 KiB/token.
    """
    arch = (architecture or "").lower()
    if "70b" in arch or (weight_bytes >= 30 * GIB and "scout" not in arch and "moe" not in arch):
        return 262_144  # ~256 KiB/token (Llama-3.1-70B GQA 80 layers)
    if "scout" in arch or "llama4" in arch:
        return 196_608  # ~192 KiB/token
    if weight_bytes >= 15 * GIB:
        return 131_072  # ~128 KiB/token
    return 98_304  # ~96 KiB/token


def plan_gguf_memory(
    *,
    weight_bytes: int,
    mmproj_bytes: int = 0,
    architecture: str = "unknown",
    block_count: Optional[int] = None,
    max_model_ctx: int = 32768,
    total_vram_bytes: int = 0,
    free_vram_bytes: int = 0,
    available_ram_bytes: int = 0,
    cpu_count: int = 4,
    image_min_vram_gib: float = 0.0,
    n_ctx_override: int = 0,
    n_gpu_layers_override: int = -1,
    batch_size_override: int = 0,
    threads_override: int = 0,
    allow_cpu_offload: bool = True,
) -> RuntimeMemoryPlan:
    """
    Compute an optimal `llama-server` execution plan maximizing GPU VRAM and System RAM
    utilization while preventing CUDA OOM and Linux OOM-killer crashes.
    """
    effective_vram = free_vram_bytes if free_vram_bytes > 0 else total_vram_bytes
    usable_vram = max(0, effective_vram - CUDA_WORKSPACE_RESERVE_BYTES)
    usable_ram = max(0, available_ram_bytes - SYSTEM_RAM_OS_RESERVE_BYTES)

    total_weights = max(0, weight_bytes + mmproj_bytes)
    threads = threads_override if threads_override > 0 else max(2, min(16, cpu_count // 2 or 2))
    kv_per_tok = _estimate_kv_bytes_per_token(weight_bytes, architecture)

    if n_ctx_override > 0:
        target_ctx = min(n_ctx_override, max_model_ctx)
    elif usable_vram >= total_weights + HARNESS_HIGH_VRAM_CONTEXT * kv_per_tok:
        target_ctx = min(HARNESS_HIGH_VRAM_CONTEXT, max_model_ctx)
    elif usable_vram >= total_weights + HARNESS_DEFAULT_CONTEXT * kv_per_tok:
        target_ctx = min(HARNESS_DEFAULT_CONTEXT, max_model_ctx)
    else:
        target_ctx = min(HARNESS_MIN_SAFE_CONTEXT, max_model_ctx)
    target_ctx = max(4096, target_ctx)

    kv_bytes = target_ctx * kv_per_tok
    full_gpu_needed = total_weights + kv_bytes

    if effective_vram <= 0:
        if usable_ram >= full_gpu_needed:
            return RuntimeMemoryPlan(
                tier="CPU_OFFLOAD_POSSIBLE",
                n_ctx=target_ctx,
                n_gpu_layers=0,
                batch_size=batch_size_override or 1024,
                ubatch_size=256,
                threads=threads,
                estimated_vram_bytes=0,
                estimated_ram_offload_bytes=full_gpu_needed,
                estimated_free_vram_after_bytes=0,
                can_coreside_image=False,
                reason="No CUDA GPU detected; CPU RAM execution only",
            )
        return RuntimeMemoryPlan(
            tier="INSUFFICIENT_MEMORY",
            n_ctx=target_ctx,
            n_gpu_layers=0,
            batch_size=512,
            ubatch_size=256,
            threads=threads,
            estimated_vram_bytes=0,
            estimated_ram_offload_bytes=0,
            estimated_free_vram_after_bytes=0,
            can_coreside_image=False,
            reason="No CUDA GPU and insufficient System RAM",
        )

    # Case 1: Fits 100% in GPU VRAM with Harness-safe context + CUDA workspace reserve.
    if usable_vram >= full_gpu_needed:
        ngl = n_gpu_layers_override if n_gpu_layers_override >= 0 else 999
        est_vram = full_gpu_needed + CUDA_WORKSPACE_RESERVE_BYTES
        free_after = max(0, effective_vram - est_vram)
        can_coreside = image_min_vram_gib > 0 and free_after >= int(image_min_vram_gib * GIB) + CORESIDENT_SAFETY_HEADROOM_BYTES
        return RuntimeMemoryPlan(
            tier="RECOMMENDED",
            n_ctx=target_ctx,
            n_gpu_layers=ngl,
            batch_size=batch_size_override or 2048,
            ubatch_size=512,
            threads=threads,
            estimated_vram_bytes=est_vram,
            estimated_ram_offload_bytes=0,
            estimated_free_vram_after_bytes=free_after,
            can_coreside_image=can_coreside,
            reason=f"Full GPU placement ({est_vram / GIB:.1f} GiB / {effective_vram / GIB:.1f} GiB VRAM, n_ctx={target_ctx})",
        )

    # Case 2: Hybrid GPU VRAM + System RAM layer offload.
    total_layers = block_count or 48
    vram_for_weights = max(0, usable_vram - kv_bytes)
    if allow_cpu_offload and vram_for_weights > 0 and usable_ram >= (total_weights - vram_for_weights):
        ratio = min(0.95, vram_for_weights / max(1, total_weights))
        auto_ngl = max(1, min(total_layers, int(math.floor(total_layers * ratio))))
        ngl = n_gpu_layers_override if n_gpu_layers_override >= 0 else auto_ngl
        gpu_weights = int(total_weights * min(1.0, ngl / max(1, total_layers)))
        ram_offload = max(0, total_weights - gpu_weights)
        est_vram = gpu_weights + kv_bytes + CUDA_WORKSPACE_RESERVE_BYTES
        free_after = max(0, effective_vram - est_vram)
        return RuntimeMemoryPlan(
            tier="CPU_OFFLOAD_POSSIBLE",
            n_ctx=target_ctx,
            n_gpu_layers=ngl,
            batch_size=batch_size_override or 1024,
            ubatch_size=512,
            threads=threads,
            estimated_vram_bytes=est_vram,
            estimated_ram_offload_bytes=ram_offload,
            estimated_free_vram_after_bytes=free_after,
            can_coreside_image=False,
            reason=(
                f"Hybrid GPU+RAM ({ngl}/{total_layers} layers on GPU, "
                f"{ram_offload / GIB:.1f} GiB in System RAM, n_ctx={target_ctx})"
            ),
        )

    # Case 3: Insufficient combined VRAM + System RAM.
    needed_gib = (full_gpu_needed + CUDA_WORKSPACE_RESERVE_BYTES) / GIB
    have_gib = (effective_vram + usable_ram) / GIB
    return RuntimeMemoryPlan(
        tier="INSUFFICIENT_MEMORY",
        n_ctx=target_ctx,
        n_gpu_layers=0,
        batch_size=512,
        ubatch_size=256,
        threads=threads,
        estimated_vram_bytes=0,
        estimated_ram_offload_bytes=0,
        estimated_free_vram_after_bytes=0,
        can_coreside_image=False,
        reason=f"Insufficient memory: needs ~{needed_gib:.1f} GiB combined, only {have_gib:.1f} GiB usable VRAM+RAM",
    )


def can_coreside_image_model(
    *,
    free_vram_bytes: int,
    image_min_vram_gib: float,
    safety_headroom_bytes: int = CORESIDENT_SAFETY_HEADROOM_BYTES,
) -> bool:
    """Return True if live free VRAM can hold the image pipeline without unloading the LLM."""
    if free_vram_bytes <= 0 or image_min_vram_gib <= 0:
        return False
    required = int(image_min_vram_gib * GIB) + safety_headroom_bytes
    return free_vram_bytes >= required
