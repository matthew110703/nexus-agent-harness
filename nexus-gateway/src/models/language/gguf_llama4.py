"""Dedicated GGUF runtime for Llama 4 Scout / Maverick MoE GGUF models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.gguf_common import LlamaServerRuntimeBase


class GgufLlama4Model:
    """Dedicated GGUF runtime wrapping `llama-server` for Llama 4 MoE architecture GGUFs."""

    implementation = "gguf_llama4"

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        architecture: str = "llama4",
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
            reasoning_format=None,
        )

    def is_ready(self) -> bool:
        return self.server.is_ready()

    def load(self, *, precision: Optional[str] = None) -> None:
        self.server.start()

    def unload(self) -> None:
        self.server.stop()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        yield from self.server.generate_chat_stream(req)
