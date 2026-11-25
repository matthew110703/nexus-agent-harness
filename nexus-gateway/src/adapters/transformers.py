"""Transformers chat model adapter running inside the isolated GPU worker."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional


class TransformersChatAdapter:
    """Explicit-precision Hugging Face Transformers chat adapter."""

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "unknown") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        dtype = torch.bfloat16 if (precision == "bf16" or (precision is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported())) else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,
        )
        self.model.eval()

    def _apply_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], enable_thinking: Optional[bool]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return str(self.tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return str(self.tokenizer.apply_chat_template(messages, **kwargs))

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("TransformersChatAdapter is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        enable_thinking = req.get("enable_thinking")
        prompt = self._apply_template(messages, tools, enable_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(req.get("max_tokens") or 2048),
            "do_sample": float(req.get("temperature") or 0.7) > 0,
            "temperature": max(0.01, float(req.get("temperature") or 0.7)),
            "top_p": float(req.get("top_p") or 0.9),
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
        for piece in streamer:
            if piece:
                yield str(piece)
        worker_thread.join()
        if exc_holder:
            raise exc_holder[0]

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
