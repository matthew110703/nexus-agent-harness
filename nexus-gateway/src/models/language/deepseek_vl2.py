"""Dedicated DeepSeek-VL2 runtime (DeepSeek-VL2-Small)."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.common import enable_cuda_fast_math, make_cancel_stopping_criteria
from models.language.prompt_tools import inject_tools_into_messages
from models.language.quantization import resolve_quantization_config


class DeepSeekVl2LanguageModel:
    """Dedicated runtime for DeepSeek-VL2 (`DeepseekVLV2ForCausalLM`)."""

    implementation = "deepseek_vl2"

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "DeepseekVLV2ForCausalLM") -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.architecture = architecture
        self.model: Any = None
        self.tokenizer: Any = None
        self.vl_chat_processor: Any = None

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, *, precision: Optional[str] = None) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        enable_cuda_fast_math(torch)
        # Register DeepSeek-VL2 custom model classes BEFORE AutoModelForCausalLM.from_pretrained()
        vl_processor_cls: Any = None
        try:
            from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor  # type: ignore[import-not-found]  # noqa: F401

            vl_processor_cls = DeepseekVLV2Processor
        except ImportError:
            pass

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

        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

        if vl_processor_cls is not None:
            try:
                self.vl_chat_processor = vl_processor_cls.from_pretrained(str(self.model_path))
                self.tokenizer = self.vl_chat_processor.tokenizer
                return
            except Exception:
                pass
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"DeepSeekVl2LanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        prepared_messages = inject_tools_into_messages(messages, tools)

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in prepared_messages) + "\nassistant: "

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel_event = threading.Event()
        temperature = float(req.get("temperature", 0.7))
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

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield str(chunk)
        finally:
            cancel_event.set()
            thread.join(timeout=5.0)
        if exc_holder:
            raise exc_holder[0]

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.vl_chat_processor = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
