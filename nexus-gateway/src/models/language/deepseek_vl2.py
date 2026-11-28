"""Dedicated DeepSeek-VL2 runtime (DeepSeek-VL2-Tiny / Small)."""

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


class DeepSeekVl2LanguageModel:
    """Dedicated runtime for DeepSeek-VL2 (`DeepseekVLV2ForCausalLM`) with multimodal PIL image support."""

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

    def _build_deepseek_vl_conversations(self, vlm_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        conv: list[dict[str, Any]] = []
        sys_prefix = ""
        for m in vlm_messages:
            role = m.get("role")
            blocks = m.get("content") if isinstance(m.get("content"), list) else [{"type": "text", "text": str(m.get("content") or "")}]
            text_segs: list[str] = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "image":
                    text_segs.append("<image>")
                elif isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                    text_segs.append(str(b["text"]))
            joined = "\n".join(text_segs).strip()
            if role == "system":
                sys_prefix = f"{sys_prefix}\n\n{joined}".strip()
                continue
            ds_role = "<|User|>" if role == "user" else "<|Assistant|>"
            if sys_prefix and ds_role == "<|User|>":
                joined = f"{sys_prefix}\n\n{joined}".strip()
                sys_prefix = ""
            conv.append({"role": ds_role, "content": joined})
        conv.append({"role": "<|Assistant|>", "content": ""})
        return conv

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if not self.is_ready():
            raise RuntimeError(f"DeepSeekVl2LanguageModel ({self.model_id}) is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        hf_messages, images = extract_vlm_messages_and_images(messages)
        device = next(self.model.parameters()).device

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel_event = threading.Event()
        temperature = float(req.get("temperature", 0.7))
        stop_ids = resolve_stop_token_ids(self.tokenizer, ("<｜end▁of▁sentence｜>", "<|EOT|>", "<|im_end|>"))

        gen_target = getattr(self.model, "language", None) if (images and self.vl_chat_processor is not None and hasattr(self.model, "prepare_inputs_embeds")) else None

        if images and self.vl_chat_processor is not None and gen_target is not None:
            vlm_messages = inject_tools_preserving_images(hf_messages, tools, strict_alternation=False)
            conversations = self._build_deepseek_vl_conversations(vlm_messages)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversations,
                images=images,
                force_batchify=True,
                system_prompt="",
            ).to(device)
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            gen_kwargs: dict[str, Any] = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": prepare_inputs.attention_mask,
                "pad_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "streamer": streamer,
                "stopping_criteria": make_cancel_stopping_criteria(cancel_event),
                "max_new_tokens": int(req.get("max_tokens") or 4096),
                "do_sample": temperature > 0,
                "temperature": max(0.01, temperature),
                "top_p": float(req.get("top_p", 0.9)),
                "use_cache": True,
            }
            runner_model = gen_target
        else:
            prepared_messages = inject_tools_into_messages(messages, tools)
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                prompt = self.tokenizer.apply_chat_template(
                    prepared_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in prepared_messages) + "\nassistant: "
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "stopping_criteria": make_cancel_stopping_criteria(cancel_event),
                "max_new_tokens": int(req.get("max_tokens") or 4096),
                "do_sample": temperature > 0,
                "temperature": max(0.01, temperature),
                "top_p": float(req.get("top_p", 0.9)),
            }
            runner_model = self.model

        if stop_ids is not None:
            gen_kwargs["eos_token_id"] = stop_ids

        exc_holder: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    runner_model.generate(**gen_kwargs)
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
