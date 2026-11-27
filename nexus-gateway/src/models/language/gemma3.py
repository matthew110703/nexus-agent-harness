"""Dedicated Gemma 3 language and multimodal runtime (Gemma-3-12B-IT)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.prompt_tools import inject_tools_into_messages, parse_prompt_tool_calls
from models.language.quantization import resolve_quantization_config


class Gemma3LanguageModel:
    """Dedicated runtime for Gemma 3 (Gemma-3-12B-IT)."""

    implementation = "gemma3"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "Gemma3ForConditionalGeneration") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.processor: Any = None

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-not-found]

        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        # Compute size to decide 4-bit quantization on L4
        total_size = sum(p.stat().st_size for p in self.model_path.glob("*.safetensors"))
        q_cfg, dtype = resolve_quantization_config(torch, model_size_bytes=total_size)

        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if q_cfg is not None:
            load_kwargs["quantization_config"] = q_cfg

        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]
        import threading

        if not self.is_ready():
            raise RuntimeError(f"Gemma3LanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []

        # Gemma requires strict user/model alternation and prompt tool injection
        prepared_messages = inject_tools_into_messages(messages, tools, strict_alternation=True)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prompt = tokenizer.apply_chat_template(
            prepared_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": req.get("max_tokens") or 4096,
            "temperature": req.get("temperature", 0.7),
            "do_sample": True,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()
