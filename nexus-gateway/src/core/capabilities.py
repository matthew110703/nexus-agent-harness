"""Dynamic Model Capability Engine for Nexus AI Lab (v2.3.0).

Inspects the locked language model + optional image pipeline + GGUF header metadata to
compute family-specific stop sequences, quantization-aware sampling bounds, tool-schema
compaction, and truthful runtime capability grounding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from models.base import SelectedImageSpec, SelectedLanguageSpec
from models.gguf_metadata import GgufHeaderInfo, inspect_gguf_directory

UNIVERSAL_TURN_STOPS: tuple[str, ...] = (
    "<|eot_id|>",
    "<|eom_id|>",
    "<|eot|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|start_header_id|>",
    "<|header_start|>",
)

FAMILY_EXTRA_STOPS: dict[str, tuple[str, ...]] = {
    "gguf_llama3": ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|start_header_id|>"),
    "gguf_llama4": ("<|eot|>", "<|end_of_text|>", "<|header_start|>", "<|python_end|>"),
    "gguf_gemma4": ("<end_of_turn>", "<eos>", "<start_of_turn>"),
    "gemma3": ("<end_of_turn>", "<eos>", "<start_of_turn>"),
    "qwen3_vl": ("<|im_end|>", "<|endoftext|>"),
    "qwen3_5": ("<|im_end|>", "<|endoftext|>"),
    "gguf_qwen3_5": ("<|im_end|>", "<|endoftext|>"),
    "gguf_deepseek_qwen3_5": ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"),
    "deepseek_vl2": ("<｜end▁of▁sentence｜>", "<|EOT|>", "<|im_end|>"),
    "deepseek_vl": ("<｜end▁of▁sentence｜>", "<|EOT|>", "<|im_end|>"),
    "gpt_oss": ("<|return|>", "<|end|>"),
}

AGGRESSIVE_QUANT_PREFIXES = ("Q2_", "Q3_", "IQ1_", "IQ2_", "IQ3_", "TQ1_", "TQ2_")


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """Resolved runtime capabilities, sampling bounds, stops, and template overrides."""

    model_id: str
    display_name: str
    implementation: str
    architecture: str
    quantization: str
    context_window: int
    vision_input_enabled: bool
    image_generation_enabled: bool
    image_model_id: Optional[str]
    is_aggressive_quant: bool
    compact_tool_schemas: bool
    max_tool_desc_chars: int
    chat_temperature: float
    tool_temperature: float
    top_p: float
    min_p: float
    repeat_penalty: float
    stop_sequences: list[str] = field(default_factory=list)
    template_override_name: Optional[str] = None
    template_has_tools: bool = True
    template_has_ipython_compulsion: bool = False
    template_style: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_aggressive_quantization(quant: str, model_id: str) -> bool:
    up_q = (quant or "").strip().upper()
    if any(up_q.startswith(p) for p in AGGRESSIVE_QUANT_PREFIXES):
        return True
    up_id = (model_id or "").upper()
    return any(tag in up_id for tag in ("-Q2", "_Q2", "-Q3", "_Q3", "IQ2", "IQ3"))


def resolve_curated_template_path(template_name: Optional[str]) -> Optional[Path]:
    """Return the absolute Path to a curated `.jinja` template inside `nexus-gateway/src/templates`."""
    if not template_name:
        return None
    candidate = Path(__file__).resolve().parents[1] / "templates" / template_name
    return candidate if candidate.is_file() else None


def build_model_capability_profile(
    lang_spec: SelectedLanguageSpec,
    img_spec: Optional[SelectedImageSpec] = None,
    gguf_header: Optional[GgufHeaderInfo] = None,
) -> ModelCapabilityProfile:
    """Compute the complete `ModelCapabilityProfile` for the active session models."""
    impl = lang_spec.implementation
    header = gguf_header
    mmproj_present = False
    if impl.startswith("gguf_"):
        model_dir = Path(lang_spec.model_path)
        if model_dir.exists():
            layout = inspect_gguf_directory(model_dir if model_dir.is_dir() else model_dir.parent)
            mmproj_present = layout.mmproj_file is not None
            if header is None:
                header = layout.header

    vision_enabled = "image" in (lang_spec.modalities or [])
    if impl.startswith("gguf_") and not mmproj_present and "image" not in (lang_spec.modalities or []):
        vision_enabled = False

    image_gen_enabled = img_spec is not None
    quant = lang_spec.quantization if lang_spec.quantization and lang_spec.quantization != "unknown" else (header.quantization if header else "unknown")
    aggressive_q = _is_aggressive_quantization(quant, lang_spec.id)

    ctx_win = int(lang_spec.n_ctx or lang_spec.context_window or 32768)
    compact_schemas = bool(ctx_win <= 32768 or impl.startswith("gguf_") or impl in ("gemma3", "deepseek_vl2", "deepseek_vl"))
    max_desc = 180 if ctx_win <= 16384 else 240

    if aggressive_q or impl in ("gguf_llama3", "gguf_llama4"):
        tool_temp = 0.2
        chat_temp = 0.5
        top_p = 0.85
        min_p = 0.05
        repeat_penalty = 1.05
    elif impl.startswith("gguf_"):
        tool_temp = 0.3
        chat_temp = 0.6
        top_p = 0.9
        min_p = 0.05
        repeat_penalty = 1.04
    else:
        tool_temp = 0.35
        chat_temp = 0.7
        top_p = 0.9
        min_p = 0.0
        repeat_penalty = 1.0

    stops: list[str] = []
    for tok in (*FAMILY_EXTRA_STOPS.get(impl, ()), *UNIVERSAL_TURN_STOPS):
        if tok == "<|python_end|>":
            # Llama 4 closes tool blocks with <|python_end|> before <|eot|>; let parser see <|python_end|>.
            continue
        if tok not in stops:
            stops.append(tok)

    tpl_has_tools = header.template_has_tools if header is not None else True
    tpl_ipython = header.template_has_ipython_compulsion if header is not None else False
    tpl_style = header.template_style if header is not None else "unknown"
    tpl_override: Optional[str] = None

    if impl == "gguf_llama3":
        if header is None or (not tpl_has_tools) or tpl_ipython or tpl_style != "llama3":
            tpl_override = "llama3_1_nexus.jinja"
    elif impl == "gguf_llama4":
        if header is None or (not tpl_has_tools) or tpl_ipython or tpl_style != "llama4":
            tpl_override = "llama4_scout_nexus.jinja"

    return ModelCapabilityProfile(
        model_id=lang_spec.id,
        display_name=lang_spec.display_name or lang_spec.id,
        implementation=impl,
        architecture=lang_spec.architecture,
        quantization=quant,
        context_window=ctx_win,
        vision_input_enabled=vision_enabled,
        image_generation_enabled=image_gen_enabled,
        image_model_id=img_spec.id if img_spec is not None else None,
        is_aggressive_quant=aggressive_q,
        compact_tool_schemas=compact_schemas,
        max_tool_desc_chars=max_desc,
        chat_temperature=chat_temp,
        tool_temperature=tool_temp,
        top_p=top_p,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        stop_sequences=stops,
        template_override_name=tpl_override,
        template_has_tools=tpl_has_tools,
        template_has_ipython_compulsion=tpl_ipython,
        template_style=tpl_style,
    )


def build_capability_grounding(profile: ModelCapabilityProfile) -> str:
    """Build a concise, factual self-awareness header prepended to the system prompt."""
    vision_line = (
        "ENABLED (multimodal vision active — you can directly see and analyze attached user images)."
        if profile.vision_input_enabled
        else "DISABLED (text-only checkpoint without a vision projector — you CANNOT see, inspect, OCR, or identify people/objects in uploaded images)."
    )
    if profile.image_generation_enabled and profile.image_model_id:
        img_gen_line = (
            f"ENABLED via `nexus_generate_image` ({profile.image_model_id}). "
            "IMPORTANT: `nexus_generate_image` is a blind image synthesizer — it CANNOT see, analyze, or identify images. "
            "ONLY call `nexus_generate_image` when the user explicitly asks to generate, draw, create, edit, or restyle an image."
        )
    else:
        img_gen_line = "DISABLED for this session."

    return (
        "[Nexus Runtime Capabilities]\n"
        f"- Active Language Model: {profile.display_name} ({profile.model_id}, context={profile.context_window} tokens)\n"
        f"- Vision Input (seeing uploaded images): {vision_line}\n"
        f"- Image Generation / Editing: {img_gen_line}\n"
        "- Response Discipline: Answer greetings, capability questions, and general questions directly in plain text without calling tools. "
        "Never output simulated `<|im_start|>` / `<|im_end|>` turns or raw `[func(...)]` text."
    )


def apply_capability_grounding(messages: list[dict[str, Any]], profile: ModelCapabilityProfile) -> list[dict[str, Any]]:
    """Prepend `build_capability_grounding(profile)` once to `messages[0]` (`role: system`)."""
    if not messages:
        return messages
    header = build_capability_grounding(profile)
    out = [dict(m) for m in messages]
    if out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
        existing = out[0]["content"]
        if "[Nexus Runtime Capabilities]" not in existing:
            out[0]["content"] = f"{header}\n\n{existing}".strip()
        return out
    return [{"role": "system", "content": header}, *out]


def _compact_description(desc: str, limit: int) -> str:
    cleaned = " ".join((desc or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    # Cut cleanly at the first sentence boundary if reasonable, otherwise hard-trim at word boundary.
    first_period = cleaned.find(". ")
    if 32 <= first_period + 1 <= limit:
        return cleaned[: first_period + 1]
    cut = cleaned[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{cut}..." if cut else cleaned[:limit]


def _compact_schema_node(node: Any, prop_limit: int) -> Any:
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if isinstance(out.get("description"), str):
        out["description"] = _compact_description(out["description"], prop_limit)
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: _compact_schema_node(v, prop_limit) for k, v in props.items()}
    if isinstance(out.get("items"), dict):
        out["items"] = _compact_schema_node(out["items"], prop_limit)
    return out


def compact_tools_for_context(tools: list[dict[str, Any]], *, max_desc_chars: int = 220) -> list[dict[str, Any]]:
    """Trim verbose multi-paragraph tool/property prose while keeping 100% of JSON Schema structure."""
    if not tools:
        return []
    prop_limit = max(90, max_desc_chars // 2)
    compacted: list[dict[str, Any]] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        t_copy = dict(entry)
        fn = t_copy.get("function")
        if isinstance(fn, dict):
            fn_copy = dict(fn)
            if isinstance(fn_copy.get("description"), str):
                fn_copy["description"] = _compact_description(fn_copy["description"], max_desc_chars)
            if isinstance(fn_copy.get("parameters"), dict):
                fn_copy["parameters"] = _compact_schema_node(fn_copy["parameters"], prop_limit)
            t_copy["function"] = fn_copy
        elif isinstance(t_copy.get("description"), str):
            t_copy["description"] = _compact_description(t_copy["description"], max_desc_chars)
        compacted.append(t_copy)
    return compacted


def resolve_sampling_params(
    profile: ModelCapabilityProfile,
    *,
    has_tools: bool,
    requested_temperature: Optional[float],
    requested_top_p: Optional[float],
) -> dict[str, Any]:
    """Compute safe sampling parameters (`temperature`, `top_p`, `min_p`, `repeat_penalty`, `stop`)."""
    target_temp = profile.tool_temperature if has_tools else profile.chat_temperature
    if requested_temperature is None:
        temp = target_temp
    elif profile.is_aggressive_quant or (has_tools and profile.implementation in ("gguf_llama3", "gguf_llama4")):
        # Clamp overly hot default Harness temperatures (0.7-1.0) on Q3/Llama tool turns while honoring explicit 0.0.
        temp = min(float(requested_temperature), target_temp)
    else:
        temp = float(requested_temperature)

    top_p = float(requested_top_p) if requested_top_p is not None else profile.top_p
    if profile.is_aggressive_quant:
        top_p = min(top_p, profile.top_p)

    return {
        "temperature": round(max(0.0, temp), 4),
        "top_p": round(max(0.05, min(1.0, top_p)), 4),
        "min_p": profile.min_p if profile.min_p > 0 else None,
        "repeat_penalty": profile.repeat_penalty if profile.repeat_penalty > 1.0 else None,
        "stop": list(profile.stop_sequences),
    }


def estimate_openai_usage(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    completion_text: str,
    reasoning_text: str = "",
    tool_calls: Optional[list[Any]] = None,
) -> dict[str, int]:
    """Deterministic token estimate (~3.6 chars/token) when worker runtime omits exact usage."""
    msg_chars = sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or "")) + 16 for m in messages)
    tool_chars = sum(len(str(t)) for t in (tools or []))
    prompt_tokens = max(1, (msg_chars + tool_chars) // 4)
    comp_chars = len(completion_text or "") + len(reasoning_text or "")
    if tool_calls:
        comp_chars += sum(len(str(tc)) for tc in tool_calls)
    completion_tokens = max(1, comp_chars // 4) if comp_chars > 0 else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
