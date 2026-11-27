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
    Ensures messages adhere to model role constraints (e.g. strict user/model alternation for Gemma 3).
    """
    if not tools:
        return _format_roles(messages, strict_alternation=strict_alternation)

    tool_instr = build_tools_instruction(tools)
    out: list[dict[str, Any]] = []
    has_system = any(m.get("role") == "system" for m in messages)

    if has_system:
        for m in messages:
            if m.get("role") == "system":
                orig = m.get("content") or ""
                merged = f"{orig}\n\n{tool_instr}".strip()
                out.append({"role": "system", "content": merged})
            else:
                out.append(dict(m))
    else:
        # Prepend system or attach to first user message
        if strict_alternation:
            # Gemma models often dislike 'system' role
            if messages and messages[0].get("role") == "user":
                first = dict(messages[0])
                first_content = first.get("content") or ""
                first["content"] = f"{tool_instr}\n\n{first_content}".strip()
                out.append(first)
                out.extend(dict(m) for m in messages[1:])
            else:
                out.append({"role": "user", "content": tool_instr})
                out.extend(dict(m) for m in messages)
        else:
            out.append({"role": "system", "content": tool_instr})
            out.extend(dict(m) for m in messages)

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
        content = str(m.get("content") or "")
        # If tool_calls exist on assistant, stringify them
        if m.get("tool_calls"):
            content += f"\n```json\n{json.dumps(m['tool_calls'], indent=2)}\n```"

        if role == current_role:
            accumulated_content.append(content)
        else:
            if current_role is not None:
                formatted.append({"role": current_role, "content": "\n\n".join(accumulated_content)})
            current_role = role
            accumulated_content = [content]

    if current_role is not None and accumulated_content:
        formatted.append({"role": current_role, "content": "\n\n".join(accumulated_content)})

    return formatted


def parse_prompt_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Detect tool call JSON blocks from generated text:
    ```json
    { "name": ..., "arguments": ... }
    ```
    Returns: (cleaned_text, tool_calls_list).
    """
    tool_calls: list[dict[str, Any]] = []
    cleaned = text

    pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
    matches = list(pattern.finditer(text))

    for m in matches:
        raw_json = m.group(1)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                call_id = f"call_{len(tool_calls)+1}"
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
