"""Tool-calling prompt injection and parser for models without native tool tokens in chat_template."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

TOOL_SYSTEM_PROMPT = """
# Available Tools
You have access to the following functions to help solve user requests:
{tools_json}

To call a function, respond with a JSON block in the following format:
```json
{{
  "name": "function_name",
  "arguments": {{
    "arg1": "value1"
  }}
}}
```
Only call tools when needed. When providing the final answer, do not use the tool block.
"""


def extract_text_content(content: Any) -> str:
    """Safely flatten OpenAI string or structured content-block lists into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "\n".join(p for p in parts if p)
    return str(content)


def build_tools_instruction(tools: list[dict[str, Any]]) -> str:
    """Generate system prompt instructions explaining tool calling."""
    if not tools:
        return ""
    tools_formatted = json.dumps(tools, indent=2)
    return TOOL_SYSTEM_PROMPT.format(tools_json=tools_formatted).strip()


def inject_tools_into_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    strict_alternation: bool = False,
) -> list[dict[str, Any]]:
    """
    Inject tool definitions into system prompt (or first user message if system not supported).
    Flattens structured content arrays to plain text and enforces strict alternation when requested.
    """
    normalized: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        copied = dict(m)
        copied["content"] = extract_text_content(copied.get("content"))
        normalized.append(copied)

    if not tools:
        return _format_roles(normalized, strict_alternation=strict_alternation)

    tool_instr = build_tools_instruction(tools)
    out: list[dict[str, Any]] = []
    has_system = any(m.get("role") == "system" for m in normalized)

    if has_system and not strict_alternation:
        injected = False
        for m in normalized:
            if m.get("role") == "system" and not injected:
                orig = m.get("content") or ""
                merged = f"{orig}\n\n{tool_instr}".strip()
                out.append({"role": "system", "content": merged})
                injected = True
            else:
                out.append(dict(m))
    elif strict_alternation:
        # Gemma models prefer system/tool instructions folded into the first user turn
        if normalized and normalized[0].get("role") == "user":
            first = dict(normalized[0])
            first["content"] = f"{tool_instr}\n\n{first.get('content') or ''}".strip()
            out.append(first)
            out.extend(dict(m) for m in normalized[1:])
        else:
            out.append({"role": "user", "content": tool_instr})
            out.extend(dict(m) for m in normalized)
    else:
        out.append({"role": "system", "content": tool_instr})
        out.extend(dict(m) for m in normalized)

    return _format_roles(out, strict_alternation=strict_alternation)


def _format_roles(messages: list[dict[str, Any]], *, strict_alternation: bool = False) -> list[dict[str, Any]]:
    if not strict_alternation:
        return messages

    # Gemma strict alternation: user, model, user, model...
    formatted: list[dict[str, Any]] = []
    current_role: Optional[str] = None
    accumulated_content: list[str] = []

    for m in messages:
        role = "model" if m.get("role") in ("assistant", "model") else "user"
        content = extract_text_content(m.get("content"))
        if m.get("role") == "tool" and content:
            content = f"[Tool Result]\n{content}"
        if m.get("tool_calls"):
            content = f"{content}\n```json\n{json.dumps(m['tool_calls'], indent=2)}\n```".strip()

        if role == current_role:
            accumulated_content.append(content)
        else:
            if current_role is not None:
                formatted.append({"role": current_role, "content": "\n\n".join(p for p in accumulated_content if p)})
            current_role = role
            accumulated_content = [content]

    if current_role is not None and accumulated_content:
        formatted.append({"role": current_role, "content": "\n\n".join(p for p in accumulated_content if p)})

    return formatted


def parse_prompt_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Detect fenced JSON tool call blocks from generated text."""
    tool_calls: list[dict[str, Any]] = []
    cleaned = text

    pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
    for m in pattern.finditer(text):
        raw_json = m.group(1)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                call_id = f"call_{len(tool_calls) + 1}"
                args = parsed["arguments"]
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args_str,
                    },
                })
        except Exception:
            pass

    return cleaned, tool_calls
