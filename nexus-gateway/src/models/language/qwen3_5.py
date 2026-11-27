"""Dedicated Qwen3.5 / Qwen3.6 language model runtime (Qwen3.5-4B, Qwen3.6-35B-A3B)."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from models.base import select_torch_dtype
from models.language.common import deserialize_tool_call_arguments, extract_vlm_messages_and_images


class Qwen35LanguageModel:
    """
    Dedicated runtime for Qwen3.5 / Qwen3.6 checkpoints (`Qwen3_5ForConditionalGeneration`
    and `Qwen3_5MoeForConditionalGeneration`).
    - Deserializes assistant `tool_calls[].function.arguments` into dicts for `|items` in Jinja.
    - Passes binary `enable_thinking` directly into `apply_chat_template`.
    - Handles multimodal `data:image/*` blocks via `AutoProcessor`.
    """

    implementation = "qwen3_5"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "Qwen3_5ForConditionalGeneration") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None and self.tokenizer is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]

        dtype = select_torch_dtype(torch, precision)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def _format_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        enable_thinking: Optional[bool],
    ) -> str:
        prepared = deserialize_tool_call_arguments(messages)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True if enable_thinking is None else bool(enable_thinking),
        }
        if tools:
            kwargs["tools"] = tools
        return str(self.tokenizer.apply_chat_template(prepared, **kwargs))

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"Qwen35LanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        enable_thinking = req.get("enable_thinking")
        hf_messages, images = extract_vlm_messages_and_images(messages)
        prompt = self._format_prompt(hf_messages, tools, enable_thinking)

        proc_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        temperature = float(req.get("temperature", 0.7))
        cancel_event = threading.Event()
        from models.language.common import make_cancel_stopping_criteria

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "stopping_criteria": make_cancel_stopping_criteria(cancel_event),
            "max_new_tokens": int(req.get("max_tokens") or 4096),
            "do_sample": temperature > 0,
            "temperature": max(0.01, temperature),
            "top_p": float(req.get("top_p", 0.9)),
        }

        exc_holder: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)
            except BaseException as exc:
                exc_holder.append(exc)
                streamer.end()

        worker_thread = threading.Thread(target=_run, daemon=True)
        worker_thread.start()
        try:
            for piece in streamer:
                if piece:
                    yield str(piece)
        finally:
            cancel_event.set()
            worker_thread.join(timeout=5.0)
        if exc_holder:
            raise exc_holder[0]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
