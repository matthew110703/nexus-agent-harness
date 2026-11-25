"""Qwen2.5-VL / Qwen3-VL multimodal chat adapter running inside the isolated GPU worker."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from adapters.transformers import TransformersChatAdapter, select_torch_dtype


def decode_vlm_image_url(url: str) -> Any:
    """
    Decode an authorized `data:` base64 image URL into a PIL RGB image.
    Explicitly rejects remote HTTP(S) URLs (no SSRF) and malformed base64/image bytes.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Empty image_url in multimodal message")
    if not url.startswith("data:image/"):
        raise ValueError(
            "Unsupported image_url scheme: only authorized data:image/* base64 attachments are accepted (remote URL fetching is disabled)"
        )
    if "," not in url:
        raise ValueError("Malformed data:image/* URL (missing comma separator)")
    header, b64 = url.split(",", 1)
    if ";base64" not in header.lower():
        raise ValueError("Unsupported data:image/* encoding: base64 is required")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc
    if not raw:
        raise ValueError("Decoded image payload is empty")
    from PIL import Image  # type: ignore[import-not-found]

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc


def extract_vlm_messages_and_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Convert OpenAI multimodal messages into HF chat messages + decoded PIL images."""
    images: list[Any] = []
    hf_messages: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": str(block.get("text") or "")})
                elif btype == "image_url":
                    img_obj = block.get("image_url")
                    url = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj or "")
                    images.append(decode_vlm_image_url(url))
                    parts.append({"type": "image"})
            hf_messages.append({"role": msg.get("role", "user"), "content": parts})
        else:
            hf_messages.append({"role": msg.get("role", "user"), "content": str(content or "")})
    return hf_messages, images


class QwenVLChatAdapter(TransformersChatAdapter):
    """Vision-language adapter for Qwen2.5-VL / Qwen3-VL / Qwen3.5+ checkpoints."""

    def __init__(self, *, model_id: str, model_path: Path, architecture: str = "unknown") -> None:
        super().__init__(model_id=model_id, model_path=model_path, architecture=architecture)
        self.processor: Any = None

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

    def generate_stream(self, req: dict[str, Any]) -> Iterator[str]:
        import torch  # type: ignore[import-not-found]
        from transformers import TextIteratorStreamer  # type: ignore[import-not-found]

        if self.model is None or self.processor is None:
            raise RuntimeError("QwenVLChatAdapter is not loaded")

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        enable_thinking = req.get("enable_thinking")
        reasoning_effort = req.get("reasoning_effort")

        hf_messages, images = extract_vlm_messages_and_images(messages)
        prompt = self._apply_template(hf_messages, tools, enable_thinking, reasoning_effort)
        proc_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

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
        self.processor = None
        super().unload()
        gc.collect()
