"""OpenAI conversation normalization and model-aware reasoning-effort resolution."""

from __future__ import annotations

from typing import Any, Optional

from models.base import SelectedLanguageSpec


def normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI chat messages (including tool results and multimodal blocks)."""
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
            entry["content"] = content
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
    return normalized


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
