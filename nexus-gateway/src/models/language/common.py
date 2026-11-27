"""Shared VLM image decoding and multimodal message helpers for dedicated VLM runtimes."""

from __future__ import annotations

import base64
import binascii
import io
import json
import threading
from typing import Any


def make_cancel_stopping_criteria(cancel_event: threading.Event) -> Any:
    """Build a Transformers StoppingCriteriaList that halts generation as soon as cancel_event is set."""
    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore[import-not-found]

    class _CancelEventCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return cancel_event.is_set()

    return StoppingCriteriaList([_CancelEventCriteria()])


def _ensure_schema_descriptions(node: Any) -> Any:
    """Recursively ensure every JSON Schema object node has a `description` string for strict Jinja templates."""
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if "description" not in out or out["description"] is None:
        out["description"] = ""
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _ensure_schema_descriptions(v) for k, v in props.items()}
    if "items" in out:
        out["items"] = _ensure_schema_descriptions(out["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(key), list):
            out[key] = [_ensure_schema_descriptions(v) for v in out[key]]
    return out


def normalize_tools_for_jinja(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every OpenAI tool schema has `description`, `parameters.properties`, and `parameters.required`."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        t_copy = dict(tool)
        fn = t_copy.get("function")
        if isinstance(fn, dict):
            fn_copy = dict(fn)
            if "description" not in fn_copy or fn_copy["description"] is None:
                fn_copy["description"] = ""
            params = fn_copy.get("parameters")
            if isinstance(params, dict):
                p_copy = _ensure_schema_descriptions(params)
                p_copy.setdefault("type", "object")
                p_copy.setdefault("properties", {})
                p_copy.setdefault("required", [])
                fn_copy["parameters"] = p_copy
            else:
                fn_copy["parameters"] = {"type": "object", "properties": {}, "required": [], "description": ""}
            t_copy["function"] = fn_copy
        elif "description" not in t_copy or t_copy["description"] is None:
            t_copy["description"] = ""
        normalized.append(t_copy)
    return normalized


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


def deserialize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert JSON-string `tool_calls[].function.arguments` into dicts for chat templates
    whose Jinja iterates `tool_call.arguments|items` (Qwen3.5/3.6) or calls `|tojson` (GPT-OSS).
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        copied = dict(msg)
        raw_calls = copied.get("tool_calls")
        if isinstance(raw_calls, list):
            new_calls: list[dict[str, Any]] = []
            for tc in raw_calls:
                if not isinstance(tc, dict):
                    continue
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            if isinstance(parsed, dict):
                                fn_copy["arguments"] = parsed
                        except json.JSONDecodeError:
                            pass
                    tc_copy["function"] = fn_copy
                else:
                    args = tc_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            if isinstance(parsed, dict):
                                tc_copy["arguments"] = parsed
                        except json.JSONDecodeError:
                            pass
                new_calls.append(tc_copy)
            copied["tool_calls"] = new_calls
        out.append(copied)
    return out


def extract_vlm_messages_and_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Convert OpenAI multimodal messages into HF chat messages + decoded PIL images."""
    images: list[Any] = []
    hf_messages: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        entry = {k: v for k, v in msg.items() if k != "content"}
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
            entry["content"] = parts
        else:
            entry["content"] = str(content or "")
        hf_messages.append(entry)
    return hf_messages, images
