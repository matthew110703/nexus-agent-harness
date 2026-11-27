"""Dedicated DeepSeek-VL runtime (DeepSeek-VL-7B-Chat)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.prompt_tools import inject_tools_into_messages
from models.language.quantization import resolve_quantization_config


class DeepSeekVlLanguageModel:
    """Dedicated runtime for DeepSeek-VL (DeepSeek-VL-7B-Chat)."""

    implementation = "deepseek_vl"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "DeepseekVLForCausalLM") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.tokenizer: Any = None

    def is_ready(self) -> bool:
        return self.model is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        total_size = sum(p.stat().st_size for p in self.model_path.glob("*.safetensors"))
        q_cfg, dtype = resolve_quantization_config(torch, model_size_bytes=total_size)

        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        if q_cfg is not None:
            load_kwargs["quantization_config"] = q_cfg

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]
        import threading

        if not self.is_ready():
            raise RuntimeError(f"DeepSeekVlLanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        prepared_messages = inject_tools_into_messages(messages, tools)

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in prepared_messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel_event = threading.Event()
        from models.language.common import make_cancel_stopping_criteria

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "stopping_criteria": make_cancel_stopping_criteria(cancel_event),
            "max_new_tokens": req.get("max_tokens") or 4096,
            "temperature": req.get("temperature", 0.7),
            "do_sample": True,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            cancel_event.set()
            thread.join(timeout=5.0)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
