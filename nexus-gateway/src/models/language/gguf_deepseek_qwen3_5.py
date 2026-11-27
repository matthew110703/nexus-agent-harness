"""Dedicated GGUF runtime for DeepSeek V4 Pro Qwen3.5 9B Uncensored GGUF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.gguf_common import LlamaServerRuntimeBase


class GgufDeepSeekQwen35Model:
    """Dedicated GGUF runtime wrapping `llama-server` for DeepSeek V4 Pro Qwen 3.5 architecture GGUFs."""

    implementation = "gguf_deepseek_qwen3_5"

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        architecture: str = "gguf",
        port: int = 0,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.server = LlamaServerRuntimeBase(
            model_id=model_id,
            model_path=model_path,
            port=port,
            n_ctx=32768,
            n_gpu_layers=999,
        )

    def is_ready(self) -> bool:
        return self.server.is_ready()

    def load(self, *, precision: Optional[str] = None) -> None:
        self.server.start()

    def unload(self) -> None:
        self.server.stop()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        yield from self.server.generate_chat_stream(req)
