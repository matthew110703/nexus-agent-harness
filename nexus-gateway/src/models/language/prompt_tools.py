"""Tool-calling prompt injection and parser for models without native tool tokens in chat_template (v2.3.0)."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

TOOL_SYSTEM_PROMPT = """# Available Tools
You can call these functions ONLY when needed to inspect files, run commands, search the web, or generate an explicitly requested image:
{tools_summary}

To call a function, emit:
<tool_call>{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}</tool_call>
For normal answers, greetings, or capability questions, reply directly in plain text without <tool_call>.
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


def _format_compact_tool(entry: dict[str, Any]) -> str:
    fn = entry.get("function") if entry.get("type") == "function" else entry
    if not isinstance(fn, dict):
        return ""
    name = str(fn.get("name") or "").strip()
    if not name:
        return ""
    desc = " ".join(str(fn.get("description") or "").split())[:140]
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    required = set(params.get("required") or [])
    arg_parts: list[str] = []
    for k, schema in props.items():
        ptype = schema.get("type", "any") if isinstance(schema, dict) else "any"
        if isinstance(schema, dict) and isinstance(schema.get("enum"), list):
            ptype = "|".join(repr(v) for v in schema["enum"][:5])
        req_mark = "*" if k in required else "?"
        arg_parts.append(f"{k}{req_mark}: {ptype}")
    sig = f"{name}({', '.join(arg_parts)})"
    return f"- {sig}: {desc}" if desc else f"- {sig}"


def build_tools_instruction(tools: list[dict[str, Any]]) -> str:
    """Generate compact system prompt instructions explaining tool calling (~75% fewer tokens)."""
    if not tools:
        return ""
    lines = [line for t in tools if isinstance(t, dict) and (line := _format_compact_tool(t))]
    summary = "\n".join(lines) if lines else json.dumps(tools)
    return TOOL_SYSTEM_PROMPT.format(tools_summary=summary).strip()


def _serialize_tool_calls_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    blocks: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(fn.get("name") or "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        blocks.append(f"<tool_call>{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}</tool_call>")
    return "\n".join(blocks)


def inject_tools_into_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    strict_alternation: bool = False,
) -> list[dict[str, Any]]:
    """
    Inject tool definitions into system prompt (or first user message if system not supported).
    Flattens structured content arrays to plain text while preserving historical `tool_calls` and `tool` results.
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


def _prepend_or_append_blocks(content: Any, text: str, *, prepend: bool = False) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        blocks = [dict(b) for b in content if isinstance(b, dict)]
    elif content:
        blocks = [{"type": "text", "text": str(content)}]
    if not text:
        return blocks or [{"type": "text", "text": ""}]
    txt_block = {"type": "text", "text": text}
    return [txt_block, *blocks] if prepend else [*blocks, txt_block]


def inject_tools_preserving_images(
    hf_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    strict_alternation: bool = False,
) -> list[dict[str, Any]]:
    """
    Inject prompt-based tool instructions into multimodal VLM messages while preserving
    `{"type": "image"}` blocks for `Gemma 3`, `DeepSeek-VL`, and `DeepSeek-VL2` processors!
    """
    tool_instr = build_tools_instruction(tools) if tools else ""
    staged: list[dict[str, Any]] = []

    for m in hf_messages:
        if not isinstance(m, dict):
            continue
        raw_role = str(m.get("role") or "user")
        content = m.get("content")
        blocks = _prepend_or_append_blocks(content, "")
        if raw_role == "tool":
            t_name = str(m.get("name") or m.get("tool_call_id") or "tool")
            blocks = _prepend_or_append_blocks(blocks, f"[Tool Result: {t_name}]", prepend=True)
            role = "user"
        elif raw_role in ("assistant", "model"):
            tc_str = _serialize_tool_calls_text(m.get("tool_calls"))
            if tc_str:
                blocks = _prepend_or_append_blocks(blocks, tc_str, prepend=False)
            role = "model" if strict_alternation else "assistant"
        elif raw_role == "system" and not strict_alternation:
            role = "system"
        else:
            role = "user"
        staged.append({"role": role, "content": blocks})

    if tool_instr:
        if strict_alternation:
            if staged and staged[0]["role"] == "user":
                staged[0]["content"] = _prepend_or_append_blocks(staged[0]["content"], tool_instr, prepend=True)
            else:
                staged.insert(0, {"role": "user", "content": [{"type": "text", "text": tool_instr}]})
        else:
            sys_idx = next((i for i, x in enumerate(staged) if x["role"] == "system"), -1)
            if sys_idx != -1:
                staged[sys_idx]["content"] = _prepend_or_append_blocks(staged[sys_idx]["content"], tool_instr, prepend=False)
            else:
                staged.insert(0, {"role": "system", "content": [{"type": "text", "text": tool_instr}]})

    if not strict_alternation:
        return staged

    merged: list[dict[str, Any]] = []
    for item in staged:
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1]["content"] = [*merged[-1]["content"], *item["content"]]
        else:
            merged.append(item)
    return merged


def _format_roles(messages: list[dict[str, Any]], *, strict_alternation: bool = False) -> list[dict[str, Any]]:
    if not strict_alternation:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = extract_text_content(m.get("content"))
            if role == "tool":
                t_name = str(m.get("name") or m.get("tool_call_id") or "tool")
                out.append({"role": "user", "content": f"[Tool Result: {t_name}]\n{content}".strip()})
            elif role in ("assistant", "model"):
                tc_str = _serialize_tool_calls_text(m.get("tool_calls"))
                combined = f"{content}\n{tc_str}".strip() if tc_str else content
                out.append({"role": "assistant", "content": combined})
            else:
                out.append({"role": role if role in ("system", "user") else "user", "content": content})
        return out

    formatted: list[dict[str, Any]] = []
    current_role: Optional[str] = None
    accumulated_content: list[str] = []

    for m in messages:
        role = "model" if m.get("role") in ("assistant", "model") else "user"
        content = extract_text_content(m.get("content"))
        if m.get("role") == "tool":
            t_name = str(m.get("name") or m.get("tool_call_id") or "tool")
            content = f"[Tool Result: {t_name}]\n{content}".strip()
        if m.get("tool_calls"):
            tc_str = _serialize_tool_calls_text(m["tool_calls"])
            content = f"{content}\n{tc_str}".strip()

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
    """Detect `<tool_call>` or fenced JSON tool call blocks from generated text."""
    tool_calls: list[dict[str, Any]] = []
    cleaned = text

    for pattern in (
        re.compile(r"<tool_call>([\s\S]*?)</tool_call>"),
        re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE),
    ):
        for m in pattern.finditer(text):
            raw_json = m.group(1).strip()
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    call_id = f"call_{len(tool_calls) + 1}"
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
