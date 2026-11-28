"""OpenAI conversation normalization and model-aware reasoning-effort resolution (v2.3.0)."""

from __future__ import annotations

import re
from typing import Any, Optional

from models.base import SelectedLanguageSpec

_TOOL_DELTA_LINE_RE = re.compile(r"^\s*(?:Added|Removed|Available tools changed)\s*:\s*.*$", re.IGNORECASE | re.MULTILINE)

_TEXT_ONLY_USER_IMAGE_NOTE = (
    "[System Note: An image is attached to this message, but your active language checkpoint is text-only "
    "(no vision projector loaded), so you cannot see, read, OCR, or identify people/objects in the image. "
    "NEVER call `nexus_generate_image` or `read_image` to inspect, describe, or identify an image — "
    "`nexus_generate_image` is a blind image synthesizer that ONLY creates new artwork and cannot see anything. "
    "Only call `nexus_generate_image` (with `use_uploaded_image=true`) if the user explicitly asks to generate, "
    "draw, edit, or restyle an image; otherwise tell the user directly that this text-only model cannot view images "
    "and recommend selecting a Vision model (Qwen3-VL, Qwen3.5 HF, Gemma 3/4, or DeepSeek-VL2).]"
)

_HARNESS_RUNTIME_NOTICE_MARKERS: tuple[str, ...] = (
    "you are repeating the exact same tool call",
    "repeated tool call detected",
    "<system-reminder>",
    "[runtime notice]",
    "<context>",
    "<environment_details>",
    "current working directory:",
    "the working directory ",
)


def _clean_mid_system_notice(text: str) -> str:
    """Strip noisy tool-delta header lines (`Added: team_task_...`) from mid-turn system/developer notices."""
    stripped = _TOOL_DELTA_LINE_RE.sub("", text or "").strip()
    return stripped


def _is_harness_runtime_notice(msg: dict[str, Any]) -> bool:
    """True when a `user` message is an automated Harness runtime/guard reminder rather than a human turn."""
    if msg.get("_is_runtime_notice") is True:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        text = "\n".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    else:
        text = str(content or "")
    low = text.strip().lower()
    if not low:
        return True
    return any(marker in low for marker in _HARNESS_RUNTIME_NOTICE_MARKERS)


def _append_notice_to_content(existing: Any, notice_text: str) -> Any:
    if not notice_text.strip():
        return existing
    block_str = notice_text.strip()
    if not block_str.startswith("[Runtime Notice]"):
        block_str = f"[Runtime Notice]\n{block_str}"
    if isinstance(existing, list):
        return [*existing, {"type": "text", "text": f"\n\n{block_str}"}]
    base = str(existing or "").strip()
    return f"{base}\n\n{block_str}".strip() if base else block_str


def normalize_openai_messages(messages: list[dict[str, Any]], *, allow_images: bool = True) -> list[dict[str, Any]]:
    """
    Normalize OpenAI chat messages (including tool results, mid-conversation developer/system
    notices, and multimodal blocks) and repair genuinely cancelled tool turns without poisoning
    completed tool results.
    """
    normalized: list[dict[str, Any]] = []
    saw_initial_non_system = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        raw_role = str(msg.get("role") or "user")
        is_mid_system = False
        if raw_role in ("system", "developer"):
            if not saw_initial_non_system and not normalized:
                role = "system"
            elif not saw_initial_non_system and all(m.get("role") == "system" for m in normalized):
                role = "system"
            else:
                role = "user"
                is_mid_system = True
        else:
            role = raw_role
            saw_initial_non_system = True

        content = msg.get("content")
        entry: dict[str, Any] = {"role": role}
        if is_mid_system:
            entry["_is_runtime_notice"] = True

        if isinstance(content, list):
            if allow_images:
                entry["content"] = content
            else:
                text_parts: list[str] = []
                saw_image = False
                for part in content:
                    if isinstance(part, dict):
                        ptype = str(part.get("type") or "")
                        if ptype == "text" and part.get("text"):
                            text_parts.append(str(part["text"]))
                        elif ptype in {"image_url", "input_image", "image"}:
                            saw_image = True
                    elif isinstance(part, str) and part:
                        text_parts.append(part)
                if saw_image and role == "user" and not is_mid_system:
                    text_parts.append(_TEXT_ONLY_USER_IMAGE_NOTE)
                elif saw_image and role == "tool":
                    text_parts.append(
                        "[Tool returned an image attachment, which cannot be rendered because the active language model is text-only.]"
                    )
                entry["content"] = "\n".join(text_parts).strip()
        elif content is None:
            entry["content"] = ""
        else:
            entry["content"] = str(content)

        if is_mid_system:
            cleaned_notice = _clean_mid_system_notice(str(entry.get("content") or ""))
            if not cleaned_notice:
                continue
            entry["content"] = f"[Runtime Notice]\n{cleaned_notice}"

        # Merge consecutive initial system messages into messages[0]
        if role == "system" and normalized and normalized[-1].get("role") == "system":
            prev_sys = str(normalized[-1].get("content") or "").strip()
            cur_sys = str(entry.get("content") or "").strip()
            normalized[-1]["content"] = f"{prev_sys}\n\n{cur_sys}".strip()
            continue

        raw_tcalls = msg.get("tool_calls")
        if isinstance(raw_tcalls, list) and len(raw_tcalls) > 0:
            entry["tool_calls"] = raw_tcalls
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = str(msg["tool_call_id"])
        if msg.get("name"):
            entry["name"] = str(msg["name"])
        normalized.append(entry)

    # Repair dangling tool-call / cancelled-turn sequences while folding post-tool Harness runtime
    # notices (`repeat-tool-reminder`, `runtime-context`) onto the preceding `tool` message so
    # models never see a fake `"The previous tool call did not complete"` assistant message!
    repaired: list[dict[str, Any]] = []
    pending_calls: list[tuple[str, str]] = []  # (call_id, tool_name)

    for entry in normalized:
        is_notice = _is_harness_runtime_notice(entry)
        clean_entry = {k: v for k, v in entry.items() if k != "_is_runtime_notice"}
        role = clean_entry.get("role")

        if role == "tool":
            call_id = str(clean_entry.get("tool_call_id") or "")
            matched_name = ""
            remaining: list[tuple[str, str]] = []
            removed = False
            for pid, pname in pending_calls:
                if not removed and pid == call_id:
                    matched_name = pname
                    removed = True
                else:
                    remaining.append((pid, pname))
            pending_calls = remaining
            if not clean_entry.get("name") and matched_name:
                clean_entry["name"] = matched_name
            repaired.append(clean_entry)
            continue

        if pending_calls and role != "tool":
            for missing_id, missing_name in pending_calls:
                cancelled_tool: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": missing_id,
                    "content": "Error: tool execution was cancelled by user.",
                }
                if missing_name:
                    cancelled_tool["name"] = missing_name
                repaired.append(cancelled_tool)
            pending_calls.clear()

        if repaired and repaired[-1].get("role") == "tool" and role == "user":
            if is_notice:
                notice_txt = str(clean_entry.get("content") or "")
                repaired[-1] = {
                    **repaired[-1],
                    "content": _append_notice_to_content(repaired[-1].get("content"), notice_txt),
                }
                continue
            repaired.append({
                "role": "assistant",
                "content": "Understood. Proceeding with your new message.",
            })

        # Merge adjacent plain-text user runtime notices into preceding plain-text user message
        if (
            repaired
            and role == "user"
            and is_notice
            and repaired[-1].get("role") == "user"
            and isinstance(repaired[-1].get("content"), str)
            and isinstance(clean_entry.get("content"), str)
        ):
            repaired[-1] = {
                **repaired[-1],
                "content": _append_notice_to_content(repaired[-1]["content"], clean_entry["content"]),
            }
            continue

        if role == "assistant" and isinstance(clean_entry.get("tool_calls"), list):
            pending_calls = []
            for tc in clean_entry["tool_calls"]:
                if isinstance(tc, dict) and tc.get("id"):
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    pending_calls.append((str(tc["id"]), str(fn.get("name") or tc.get("name") or "")))
        repaired.append(clean_entry)

    return repaired


def resolve_reasoning_controls(
    spec: SelectedLanguageSpec,
    requested_effort: Optional[str],
    requested_enable_thinking: Optional[bool] = None,
) -> tuple[Optional[str], Optional[bool]]:
    """
    Resolve model-specific reasoning controls without inventing unsupported levels.
    Honors explicit `off` wire values so disabling thinking in Harness actually disables it.
    """
    mode = spec.reasoning_mode
    if mode == "none":
        return None, None

    efforts = spec.reasoning_efforts if isinstance(spec.reasoning_efforts, dict) else {}

    if mode == "binary_toggle":
        if requested_enable_thinking is not None:
            return ("high" if requested_enable_thinking else "off"), bool(requested_enable_thinking)
        if requested_effort in (None, "", "auto"):
            default_on = spec.default_reasoning_effort not in (None, "off")
            return (spec.default_reasoning_effort if default_on else "off"), default_on
        if requested_effort in ("off", "none"):
            return "off", False
        if requested_effort in efforts:
            wire = efforts[requested_effort]
            return (str(wire) if wire else "high"), True
        return None, None

    if mode == "effort_levels":
        if requested_effort in (None, "", "auto"):
            default_lvl = spec.default_reasoning_effort or "medium"
            return default_lvl, (default_lvl != "off")
        if requested_effort in ("off", "none"):
            return "low", False
        if requested_effort in efforts:
            wire = efforts[requested_effort]
            return (str(wire) if wire else spec.default_reasoning_effort or "medium"), True
        default_lvl = spec.default_reasoning_effort or "medium"
        return default_lvl, True

    return None, None
