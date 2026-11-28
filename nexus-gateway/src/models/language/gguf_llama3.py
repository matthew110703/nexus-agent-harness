"""Dedicated GGUF runtime for Llama 3 / 3.1 / 3.2 / 3.3 GGUF models (including 70B Q3_K_M)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from core.capabilities import resolve_curated_template_path
from models.gguf_metadata import inspect_gguf_directory
from models.language.gguf_common import LlamaServerRuntimeBase


def select_llama3_template_file(model_path: Path) -> Optional[Path]:
    """Select curated anti-compulsive `llama3_1_nexus.jinja` when GGUF lacks clean tool_calls or forces ipython."""
    layout = inspect_gguf_directory(model_path if model_path.is_dir() else model_path.parent)
    hdr = layout.header
    if hdr is None or (not hdr.template_has_tools) or hdr.template_has_ipython_compulsion or hdr.template_style != "llama3":
        return resolve_curated_template_path("llama3_1_nexus.jinja")
    return None


class GgufLlama3Model:
    """Dedicated GGUF runtime wrapping `llama-server` for Llama 3.x architecture GGUFs."""

    implementation = "gguf_llama3"

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        architecture: str = "llama",
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
            chat_template_file=select_llama3_template_file(model_path),
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
