"""Dedicated GPT-OSS Harmony language model runtime (GPT-OSS-20B-Abliterated)."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from models.base import select_torch_dtype
from models.language.common import (
    deserialize_tool_call_arguments,
    enable_cuda_fast_math,
    make_cancel_stopping_criteria,
    normalize_tools_for_jinja,
    resolve_stop_token_ids,
)


class GptOssLanguageModel:
    """
    Dedicated runtime for GPT-OSS checkpoints (`GptOssForCausalLM`, Harmony chat template).
    - Text-only causal LM (`AutoModelForCausalLM` + `AutoTokenizer`).
    - Deserializes `tool_calls[].function.arguments` into dicts before `|tojson`.
    - Passes `reasoning_effort` (`low` | `medium` | `high`) directly to `apply_chat_template`.
    """

    implementation = "gpt_oss"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "GptOssForCausalLM") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.tokenizer: Any = None

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        from models.language.quantization import resolve_quantization_config

        enable_cuda_fast_math(torch)
        total_size = sum(p.stat().st_size for p in self.model_path.glob("*.safetensors"))
        q_cfg, dtype = resolve_quantization_config(torch, model_size_bytes=total_size)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False)
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": False,
            "attn_implementation": "sdpa",
        }
        if q_cfg is not None:
            load_kwargs["quantization_config"] = q_cfg

        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

    def _format_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: Optional[str],
    ) -> str:
        prepared = deserialize_tool_call_arguments(messages)
        # Flatten list content blocks to text for text-only Harmony template.
        flat_messages: list[dict[str, Any]] = []
        for msg in prepared:
            copied = dict(msg)
            content = copied.get("content")
            if isinstance(content, list):
                copied["content"] = "\n".join(
                    str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            flat_messages.append(copied)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "reasoning_effort": reasoning_effort or "medium",
        }
        if tools:
            kwargs["tools"] = normalize_tools_for_jinja(tools)
        return str(self.tokenizer.apply_chat_template(flat_messages, **kwargs))

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"GptOssLanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        reasoning_effort = req.get("reasoning_effort")
        prompt = self._format_prompt(messages, tools, reasoning_effort)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        temperature = float(req.get("temperature", 0.7))
        cancel_event = threading.Event()
        stop_ids = resolve_stop_token_ids(self.tokenizer, ("<|return|>", "<|call|>"))

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
        self.tokenizer = None
        gc.collect()
