"""OpenAI conversation normalization and model-aware reasoning-effort resolution."""

from __future__ import annotations

from typing import Any, Optional

from models.base import SelectedLanguageSpec


def normalize_openai_messages(messages: list[dict[str, Any]], *, allow_images: bool = True) -> list[dict[str, Any]]:
    """
    Normalize OpenAI chat messages (including tool results and multimodal blocks)
    and repair interrupted/cancelled tool turns for all language model templates.
    """
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        if role == "developer":
            role = "system"
        content = msg.get("content")
        entry: dict[str, Any] = {"role": role}
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
                if saw_image:
                    text_parts.append("[User attached a reference image; use nexus_generate_image with use_uploaded_image=true to edit or stylize it.]")
                entry["content"] = "\n".join(text_parts).strip()
        elif content is None:
            entry["content"] = ""
        else:
            entry["content"] = str(content)
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            entry["tool_call_id"] = str(msg["tool_call_id"])
        if msg.get("name"):
            entry["name"] = str(msg["name"])
        normalized.append(entry)

    # Repair dangling tool-call / cancelled-turn sequences so no model template stalls or errors:
    # 1. assistant(tool_calls) -> user (tool execution cancelled before result) => add cancelled tool results + assistant ack
    # 2. tool -> user (turn cancelled after tool result before assistant reply) => add assistant ack
    repaired: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []
    for entry in normalized:
        role = entry.get("role")
        if role == "tool":
            call_id = str(entry.get("tool_call_id") or "")
            if call_id in pending_call_ids:
                pending_call_ids.remove(call_id)
            repaired.append(entry)
            continue

        if pending_call_ids and role != "tool":
            for missing_id in pending_call_ids:
                repaired.append({
                    "role": "tool",
                    "tool_call_id": missing_id,
                    "content": "Error: tool execution was cancelled by user.",
                })
            pending_call_ids.clear()

        if repaired and repaired[-1].get("role") == "tool" and role == "user":
            repaired.append({
                "role": "assistant",
                "content": "The previous tool call did not complete. Continuing with your next message.",
            })

        if role == "assistant" and isinstance(entry.get("tool_calls"), list):
            pending_call_ids = [
                str(tc.get("id"))
                for tc in entry["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            ]
        repaired.append(entry)

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
