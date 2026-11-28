"""Dedicated Qwen3-VL language model runtime (Qwen3-VL-4B / 8B / Heretic)."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from models.base import select_torch_dtype
from models.language.common import (
    deserialize_tool_call_arguments,
    enable_cuda_fast_math,
    extract_vlm_messages_and_images,
    make_cancel_stopping_criteria,
    normalize_tools_for_jinja,
    resolve_stop_token_ids,
)


class Qwen3VLLanguageModel:
    """
    Dedicated runtime for Qwen3-VL checkpoints (`Qwen3VLForConditionalGeneration`).
    Owns processor/model setup, multimodal image decoding, JSON `<tool_call>` template
    formatting, bf16 CUDA placement, streaming generation, and explicit unload.
    """

    implementation = "qwen3_vl"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "Qwen3VLForConditionalGeneration") -> None:
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

        enable_cuda_fast_math(torch)
        dtype = select_torch_dtype(torch, precision)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "attn_implementation": "sdpa",
        }
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(str(self.model_path), **load_kwargs)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForImageTextToText.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

    def _format_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        prepared = deserialize_tool_call_arguments(messages)
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = normalize_tools_for_jinja(tools)
        return str(self.tokenizer.apply_chat_template(prepared, **kwargs))

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"Qwen3VLLanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        hf_messages, images = extract_vlm_messages_and_images(messages)
        prompt = self._format_prompt(hf_messages, tools)

        proc_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        temperature = float(req.get("temperature", 0.7))
        cancel_event = threading.Event()
        stop_ids = resolve_stop_token_ids(self.tokenizer, ("<|im_end|>", "<|endoftext|>"))

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "stopping_criteria": make_cancel_stopping_criteria(cancel_event),
            "max_new_tokens": int(req.get("max_tokens") or 4096),
            "do_sample": temperature > 0,
            "temperature": max(0.01, temperature),
            "top_p": float(req.get("top_p", 0.9)),
        }
        if stop_ids is not None:
            gen_kwargs["eos_token_id"] = stop_ids

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
