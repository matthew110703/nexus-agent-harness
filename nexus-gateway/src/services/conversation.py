"""OpenAI conversation normalization and model-aware reasoning-effort resolution."""

from __future__ import annotations

from typing import Any, Optional

from core.model_registry import DiscoveredModel


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
    model: DiscoveredModel,
    requested_effort: Optional[str],
    requested_enable_thinking: Optional[bool] = None,
) -> tuple[Optional[str], Optional[bool]]:
    """
    Resolve model-specific reasoning controls without inventing unsupported levels.
    Keeps max output tokens completely separate from reasoning Effort.
    """
    cap = model.reasoning
    if cap.status != "supported":
        return None, None

    if cap.mode == "binary_toggle":
        if requested_enable_thinking is not None:
            return ("high" if requested_enable_thinking else None), bool(requested_enable_thinking)
        if requested_effort in (None, "", "auto"):
            default_on = cap.default not in (None, "off")
            return (cap.default if default_on else None), default_on
        if requested_effort == "off":
            return None, False
        if requested_effort in cap.levels:
            return cap.levels[requested_effort], True
        return None, None

    if cap.mode == "effort_levels":
        if requested_effort in (None, "", "auto"):
            return cap.default, (cap.default not in (None, "off"))
        if requested_effort == "off":
            return None, False
        if requested_effort in cap.levels:
            wire = cap.levels[requested_effort]
            return wire, (wire is not None)
        return cap.default, (cap.default not in (None, "off"))

    return None, None
