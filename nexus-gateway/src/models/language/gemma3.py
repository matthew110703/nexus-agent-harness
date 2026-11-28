"""Dedicated Gemma 3 language and multimodal runtime (Gemma-3-4B / 12B / 27B-IT)."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from models.language.common import (
    enable_cuda_fast_math,
    extract_vlm_messages_and_images,
    make_cancel_stopping_criteria,
    resolve_stop_token_ids,
)
from models.language.prompt_tools import inject_tools_into_messages, inject_tools_preserving_images
from models.language.quantization import resolve_quantization_config


class Gemma3LanguageModel:
    """Dedicated runtime for Gemma 3 (`Gemma3ForConditionalGeneration`) with true VLM image input."""

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
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]

        enable_cuda_fast_math(torch)
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        total_size = sum(p.stat().st_size for p in self.model_path.glob("*.safetensors"))
        q_cfg, dtype = resolve_quantization_config(torch, model_size_bytes=total_size)

        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        if q_cfg is not None:
            load_kwargs["quantization_config"] = q_cfg

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(str(self.model_path), **load_kwargs)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"Gemma3LanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        hf_messages, images = extract_vlm_messages_and_images(messages)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        device = next(self.model.parameters()).device

        if images and hasattr(self.processor, "apply_chat_template"):
            vlm_messages = inject_tools_preserving_images(hf_messages, tools, strict_alternation=True)
            prompt = self.processor.apply_chat_template(
                vlm_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        else:
            prepared_messages = inject_tools_into_messages(messages, tools, strict_alternation=True)
            prompt = tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel_event = threading.Event()
        temperature = float(req.get("temperature", 0.7))
        stop_ids = resolve_stop_token_ids(tokenizer, ("<end_of_turn>", "<eos>", "<|im_end|>"))
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
        self.processor = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
