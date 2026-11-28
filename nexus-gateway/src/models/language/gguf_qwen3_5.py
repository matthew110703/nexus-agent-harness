"""Dedicated GGUF runtime for Qwen3.5 Defiant Fable / Qwen 3.5 GGUF models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.gguf_common import LlamaServerRuntimeBase


class GgufQwen35Model:
    """Dedicated GGUF runtime wrapping `llama-server` for Qwen 3.5 architecture GGUFs."""

    implementation = "gguf_qwen3_5"

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        architecture: str = "gguf",
        port: int = 0,
        n_ctx: int = 16384,
        n_gpu_layers: int = 999,
        batch_size: int = 2048,
        ubatch_size: int = 512,
        threads: int = 0,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.server = LlamaServerRuntimeBase(
            model_id=model_id,
            model_path=model_path,
            port=port,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            threads=threads,
            reasoning_format="deepseek",
            moe_cpu_offload=("a3b" in model_id.lower() or "moe" in architecture.lower()),
        )

    @property
    def last_usage(self) -> Optional[dict[str, int]]:
        return self.server.last_usage

    def is_ready(self) -> bool:
        return self.server.is_ready()

    def load(self, *, precision: Optional[str] = None) -> None:
        self.server.start()

    def unload(self) -> None:
        self.server.stop()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        yield from self.server.generate_chat_stream(req)
