"""Model-specific native tool-call & reasoning stream parsers and recursive JSON Schema validator (v2.3.0)."""

from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

OPEN_TOOL_TAG = "<tool_call>"
CLOSE_TOOL_TAG = "</tool_call>"
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"

HARD_TURN_BOUNDARY_RE = re.compile(
    r"(?:<\|im_end\|>|<\|im_start\|>|<\|eot_id\|>|<\|eot\|>|<\|end_of_text\|>|<\|start_header_id\|>|<\|header_start\|>|<end_of_turn>|<start_of_turn>)"
)

CONTROL_TOKEN_RE = re.compile(
    r"<\|(?:im_end|im_start|endoftext|end_of_text|return|end|call|start|channel|message|constrain|python_tag|python_start|python_end|eot_id|eom_id|eot|tool_call|start_header_id|end_header_id|header_start|header_end)\|>"
)

IMPLEMENTED_PARSER_FAMILIES = frozenset({
    "qwen3_vl",
    "qwen3_5",
    "gpt_oss",
    "gemma3",
    "deepseek_vl2",
    "deepseek_vl",
    "gguf_qwen3_5",
    "gguf_gemma4",
    "gguf_deepseek_qwen3_5",
    "gguf_llama3",
    "gguf_llama4",
    "gguf_generic",
})

_XML_FUNC_RE = re.compile(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_HARMONY_TO_FUNC_RE = re.compile(r"to=functions\.([A-Za-z0-9_.-]+)")
_GEMMA4_CALL_RE = re.compile(
    r"(?:<\|tool_call\|>|<start_function_call>)\s*call:([A-Za-z0-9_.-]+)\s*(\{.*?\})\s*(?:<\|tool_call\|>|<end_function_call>)?",
    re.DOTALL,
)
_STANDALONE_BRACKET_RE = re.compile(r"\[\s*(?:[A-Za-z_][A-Za-z0-9_.-]*\s+)?[A-Za-z_][A-Za-z0-9_.-]*\s*\([^\[\]]*?\)\s*\]")


def _truncate_at_turn_boundary(text: str) -> str:
    """Hard-cut any simulated multi-turn leakage after `<|im_end|>`, `<|eot_id|>`, `<|eot|>`, etc."""
    m = HARD_TURN_BOUNDARY_RE.search(text)
    if m is not None:
        return text[: m.start()]
    return text


def has_native_tool_parser(implementation: str) -> bool:
    """True only when a dedicated native tool-call parser exists for `implementation`."""
    return implementation in IMPLEMENTED_PARSER_FAMILIES


@dataclass
class StreamDelta:
    content: str = ""
    reasoning: str = ""


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass
class ToolParseResult:
    text: str = ""
    trailing_text: str = ""
    reasoning: str = ""
    trailing_reasoning: str = ""
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ToolCallStreamParser(Protocol):
    def feed(self, chunk: str) -> StreamDelta: ...
    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult: ...


class QwenJsonToolStreamParser:
    """Incremental parser for Qwen3-VL JSON `<tool_call>{...}</tool_call>` with optional `<think>` support."""

    def __init__(self) -> None:
        self._delegate = Qwen35XmlToolStreamParser(started_in_think=False)

    def feed(self, chunk: str) -> StreamDelta:
        return self._delegate.feed(chunk)

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        return self._delegate.finish(advertised_tools, tool_choice=tool_choice)


class Qwen35XmlToolStreamParser:
    """
    Incremental parser for Qwen3.5 / Qwen3.6, Gemma 4, Llama 3.1/4, and GGUF models:
    - Streams `<think>...</think>` reasoning blocks into `StreamDelta.reasoning`.
    - Live-buffers `<tool_call>...</tool_call>`, `<|python_start|>...<|python_end|>`,
      `<|python_tag|>...`, `<|tool_call|>...<|tool_call|>`, and `<start_function_call>...<end_function_call>`
      so zero raw tool syntax leaks into visible chat text.
    - Halts visible output immediately at `<|im_end|>`, `<|im_start|>`, `<|eot_id|>`, `<|eot|>`, etc.
    """

    _OPEN_TAGS: tuple[str, ...] = (
        OPEN_TOOL_TAG,
        OPEN_THINK_TAG,
        "<|python_start|>",
        "<|python_tag|>",
        "<|tool_call|>",
        "<start_function_call>",
    )

    def __init__(self, *, started_in_think: bool = False, buffer_brackets: bool = False) -> None:
        self._buffer = ""
        self._state: str = "think" if started_in_think else "text"
        self._call_buf = ""
        self._bracket_buf = ""
        self._buffer_brackets = buffer_brackets
        self._halted = False
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._raw_calls: list[str] = []
        self._raw_chunks: list[str] = []
        self._errors: list[str] = []

    def feed(self, chunk: str) -> StreamDelta:
        if self._halted or not chunk:
            return StreamDelta()
        self._raw_chunks.append(chunk)
        self._buffer += chunk
        emitted_content: list[str] = []
        emitted_reasoning: list[str] = []

        while self._buffer and not self._halted:
            if self._state == "think":
                idx = self._buffer.find(CLOSE_THINK_TAG)
                if idx != -1:
                    if idx > 0:
                        r = CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._buffer[:idx]))
                        if r:
                            self._reasoning_parts.append(r)
                            emitted_reasoning.append(r)
                    self._buffer = self._buffer[idx + len(CLOSE_THINK_TAG) :]
                    self._state = "text"
                    continue
                keep = _partial_prefix_len(self._buffer, CLOSE_THINK_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    r = CONTROL_TOKEN_RE.sub("", self._buffer[:safe])
                    if r:
                        self._reasoning_parts.append(r)
                        emitted_reasoning.append(r)
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "call":
                idx = self._buffer.find(CLOSE_TOOL_TAG)
                if idx != -1:
                    self._call_buf += self._buffer[:idx]
                    self._raw_calls.append(self._call_buf.strip())
                    self._call_buf = ""
                    self._state = "text"
                    self._buffer = self._buffer[idx + len(CLOSE_TOOL_TAG) :]
                    continue
                keep = _partial_prefix_len(self._buffer, CLOSE_TOOL_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "python_start_call":
                stop_idx = -1
                stop_len = 0
                for end_tok in ("<|python_end|>", "<|eot|>", "<|eom_id|>", "<|eot_id|>"):
                    pos = self._buffer.find(end_tok)
                    if pos != -1 and (stop_idx == -1 or pos < stop_idx):
                        stop_idx = pos
                        stop_len = len(end_tok)
                if stop_idx != -1:
                    self._call_buf += self._buffer[:stop_idx]
                    body = self._call_buf.strip()
                    if body:
                        objs = _split_json_objects(body)
                        self._raw_calls.extend(objs if objs else [body])
                    self._call_buf = ""
                    self._state = "text"
                    self._buffer = self._buffer[stop_idx + stop_len :]
                    continue
                keep = max(
                    _partial_prefix_len(self._buffer, tok)
                    for tok in ("<|python_end|>", "<|eot|>", "<|eom_id|>", "<|eot_id|>")
                )
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "python_call":
                stop_idx = -1
                stop_len = 0
                for end_tok in ("<|eom_id|>", "<|eot_id|>", "<|eot|>", "<|im_end|>"):
                    pos = self._buffer.find(end_tok)
                    if pos != -1 and (stop_idx == -1 or pos < stop_idx):
                        stop_idx = pos
                        stop_len = len(end_tok)
                if stop_idx != -1:
                    self._call_buf += self._buffer[:stop_idx]
                    body = self._call_buf.strip()
                    if body:
                        objs = _split_json_objects(body)
                        self._raw_calls.extend(objs if objs else [body])
                    self._call_buf = ""
                    self._state = "text"
                    self._buffer = self._buffer[stop_idx + stop_len :]
                    continue
                keep = max(
                    _partial_prefix_len(self._buffer, tok)
                    for tok in ("<|eom_id|>", "<|eot_id|>", "<|eot|>", "<|im_end|>")
                )
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "gemma_call":
                stop_idx = -1
                stop_len = 0
                for end_tok in ("<|tool_call|>", "<end_function_call>", "<end_of_turn>"):
                    pos = self._buffer.find(end_tok)
                    if pos != -1 and (stop_idx == -1 or pos < stop_idx):
                        stop_idx = pos
                        stop_len = len(end_tok)
                if stop_idx != -1:
                    self._call_buf += self._buffer[:stop_idx]
                    gm = _GEMMA4_CALL_RE.search(f"<|tool_call|>{self._call_buf.strip()}<|tool_call|>")
                    if gm is not None:
                        self._raw_calls.append(
                            json.dumps({"name": gm.group(1), "arguments": _parse_gemma4_args(gm.group(2))})
                        )
                    elif self._call_buf.strip():
                        self._raw_calls.append(self._call_buf.strip())
                    self._call_buf = ""
                    self._state = "text"
                    self._buffer = self._buffer[stop_idx + stop_len :]
                    continue
                keep = max(
                    _partial_prefix_len(self._buffer, tok)
                    for tok in ("<|tool_call|>", "<end_function_call>", "<end_of_turn>")
                )
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "bracket_maybe":
                # Buffer `[...]` on Llama models so standalone `[func(...)]` never flashes in chat!
                nl_pos = self._buffer.find("\n")
                close_pos = self._buffer.find("]")
                if close_pos != -1 and (nl_pos == -1 or close_pos < nl_pos):
                    self._bracket_buf += self._buffer[: close_pos + 1]
                    self._buffer = self._buffer[close_pos + 1 :]
                    candidate = self._bracket_buf
                    self._bracket_buf = ""
                    self._state = "text"
                    if "(" in candidate and ")" in candidate:
                        self._raw_calls.append(candidate)
                    else:
                        self._text_parts.append(candidate)
                        emitted_content.append(candidate)
                    continue
                if nl_pos != -1 or len(self._bracket_buf) + len(self._buffer) > 512:
                    # Not a single-line tool bracket; flush cleanly as visible text.
                    flush_txt = CONTROL_TOKEN_RE.sub("", self._bracket_buf)
                    self._bracket_buf = ""
                    self._state = "text"
                    if flush_txt:
                        self._text_parts.append(flush_txt)
                        emitted_content.append(flush_txt)
                    continue
                self._bracket_buf += self._buffer
                self._buffer = ""
                break

            # state == "text"
            b_match = HARD_TURN_BOUNDARY_RE.search(self._buffer)
            candidates: list[tuple[int, str]] = []
            for tag in self._OPEN_TAGS:
                pos = self._buffer.find(tag)
                if pos != -1:
                    candidates.append((pos, tag))
            if self._buffer_brackets:
                b_pos = self._buffer.find("[")
                if b_pos != -1:
                    candidates.append((b_pos, "["))

            earliest_tag_pos = min((c[0] for c in candidates), default=-1)
            if b_match is not None and (earliest_tag_pos == -1 or b_match.start() < earliest_tag_pos):
                prefix = CONTROL_TOKEN_RE.sub("", self._buffer[: b_match.start()])
                if prefix:
                    self._text_parts.append(prefix)
                    emitted_content.append(prefix)
                self._buffer = ""
                self._halted = True
                break

            if candidates:
                idx, tag = min(candidates, key=lambda pair: pair[0])
                if idx > 0:
                    visible = CONTROL_TOKEN_RE.sub("", self._buffer[:idx])
                    if visible:
                        self._text_parts.append(visible)
                        emitted_content.append(visible)
                if tag == "[":
                    self._bracket_buf = "["
                    self._buffer = self._buffer[idx + 1 :]
                    self._state = "bracket_maybe"
                    continue
                self._buffer = self._buffer[idx + len(tag) :]
                if tag == OPEN_TOOL_TAG:
                    self._state = "call"
                    self._call_buf = ""
                elif tag == "<|python_start|>":
                    self._state = "python_start_call"
                    self._call_buf = ""
                elif tag == "<|python_tag|>":
                    self._state = "python_call"
                    self._call_buf = ""
                elif tag in ("<|tool_call|>", "<start_function_call>"):
                    self._state = "gemma_call"
                    self._call_buf = ""
                else:
                    self._state = "think"
                continue

            keep_candidates = [
                _partial_prefix_len(self._buffer, t)
                for t in (
                    *self._OPEN_TAGS,
                    "<|im_end|>",
                    "<|im_start|>",
                    "<|eot_id|>",
                    "<|eot|>",
                    "<|end_of_text|>",
                    "<|start_header_id|>",
                    "<|header_start|>",
                    "<end_of_turn>",
                )
            ]
            keep = max(keep_candidates) if keep_candidates else 0
            safe = len(self._buffer) - keep
            if safe > 0:
                visible = CONTROL_TOKEN_RE.sub("", self._buffer[:safe])
                if visible:
                    self._text_parts.append(visible)
                    emitted_content.append(visible)
                self._buffer = self._buffer[safe:]
            break

        return StreamDelta(content="".join(emitted_content), reasoning="".join(emitted_reasoning))

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        trailing_text = ""
        trailing_reasoning = ""
        if self._state == "call":
            tail = (self._call_buf + self._buffer).strip()
            if tail:
                # Attempt recovery if JSON/XML was complete before stream ended without closing tag
                recovered, _ = _validate_json_tool_calls([tail], advertised_tools, [])
                if recovered or _XML_FUNC_RE.search(tail):
                    self._raw_calls.append(tail)
                else:
                    self._errors.append("Unterminated <tool_call> block")
            else:
                self._errors.append("Unterminated <tool_call> block")
            self._state = "text"
            self._call_buf = ""
            self._buffer = ""
        elif self._state in ("python_call", "python_start_call"):
            tail = _truncate_at_turn_boundary(self._call_buf + self._buffer).strip()
            if tail:
                objs = _split_json_objects(tail)
                self._raw_calls.extend(objs if objs else [tail])
            self._state = "text"
            self._call_buf = ""
            self._buffer = ""
        elif self._state == "gemma_call":
            tail = _truncate_at_turn_boundary(self._call_buf + self._buffer).strip()
            if tail:
                gm = _GEMMA4_CALL_RE.search(f"<|tool_call|>{tail}<|tool_call|>")
                if gm is not None:
                    self._raw_calls.append(
                        json.dumps({"name": gm.group(1), "arguments": _parse_gemma4_args(gm.group(2))})
                    )
                else:
                    self._raw_calls.append(tail)
            self._state = "text"
            self._call_buf = ""
            self._buffer = ""
        elif self._state == "bracket_maybe":
            rem_b = CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._bracket_buf + self._buffer))
            self._bracket_buf = ""
            self._buffer = ""
            self._state = "text"
            if rem_b:
                self._text_parts.append(rem_b)
                trailing_text = rem_b
        elif self._state == "think" and self._buffer:
            r = CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._buffer))
            if r:
                self._reasoning_parts.append(r)
                trailing_reasoning = r
            self._buffer = ""
        elif self._buffer:
            visible = CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._buffer))
            if visible:
                self._text_parts.append(visible)
                trailing_text = visible
            self._buffer = ""

        full_text = _truncate_at_turn_boundary("".join(self._text_parts))

        if tool_choice == "none":
            return ToolParseResult(
                text=full_text.strip(),
                trailing_text=trailing_text,
                reasoning="".join(self._reasoning_parts).strip(),
                trailing_reasoning=trailing_reasoning,
            )

        schema_by_name = _index_tool_schemas(advertised_tools)

        # Fallback scan for any residual `[tool_name(...)]` in `full_text` if not buffered
        if advertised_tools and "[" in full_text and "]" in full_text:
            def _extract_inline_bracket(m: re.Match[str]) -> str:
                cand = m.group(0)
                py_calls = _parse_pythonic_calls(cand, set(schema_by_name.keys()))
                if py_calls:
                    self._raw_calls.append(cand)
                    return ""
                return cand

            full_text = _STANDALONE_BRACKET_RE.sub(_extract_inline_bracket, full_text)

        valid_calls: list[ParsedToolCall] = []
        errors = list(self._errors)
        non_tool_brackets: list[str] = []

        for raw in self._raw_calls:
            func_match = _XML_FUNC_RE.search(raw)
            if func_match is not None:
                name = func_match.group(1).strip()
                body = func_match.group(2)
                if name not in schema_by_name:
                    errors.append(f"Unknown or unadvertised tool: {name!r}")
                    continue
                schema = schema_by_name[name]
                props = schema.get("properties") if isinstance(schema, dict) else {}
                if not isinstance(props, dict):
                    props = {}
                arguments: dict[str, Any] = {}
                for p_match in _XML_PARAM_RE.finditer(body):
                    p_name = p_match.group(1).strip()
                    p_raw = p_match.group(2).strip()
                    arguments[p_name] = _coerce_xml_parameter(p_raw, props.get(p_name))
                arguments = _coerce_arguments_to_schema(arguments, schema, tool_name=name)
                err = validate_tool_arguments(arguments, schema)
                if err is not None:
                    errors.append(f"Invalid arguments for tool {name!r}: {err}")
                    continue
                valid_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))
                continue

            # If a buffered `[...]` was actually normal markdown/citation text (not a known tool), restore it!
            if raw.startswith("[") and raw.endswith("]"):
                py_try = _parse_pythonic_calls(raw, set(schema_by_name.keys()))
                if not py_try and _repair_and_parse_json_obj(raw) is None:
                    # Check if any advertised tool name appears before '('; if not, treat as prose, not error.
                    m_fn = re.search(r"([A-Za-z_][A-Za-z0-9_.-]*)\s*\(", raw)
                    if m_fn is None or (schema_by_name and m_fn.group(1) not in schema_by_name):
                        # Only drop silently if it looks like a hallucinated disabled tool call (`[list_agents()]`)
                        hallucinated_tool = m_fn is not None and (
                            m_fn.group(1) in {
                                "list_agents",
                                "wait_agent",
                                "subagent",
                                "subagent_fork",
                                "job_output",
                                "job_kill",
                                "job_list",
                                "ask_user_question",
                                "team_task_create",
                                "team_task_list",
                                "team_task_get",
                                "team_task_update",
                            }
                            or "()" in raw
                        )
                        if not hallucinated_tool:
                            non_tool_brackets.append(raw)
                        continue

            calls_from_json, json_errs = _validate_json_tool_calls([raw], advertised_tools, [])
            valid_calls.extend(calls_from_json)
            errors.extend(json_errs)

        if non_tool_brackets:
            restored = " ".join(non_tool_brackets)
            full_text = f"{full_text} {restored}".strip() if full_text else restored
            trailing_text = f"{trailing_text}{restored}"

        return ToolParseResult(
            text=full_text.strip(),
            trailing_text=trailing_text,
            reasoning="".join(self._reasoning_parts).strip(),
            trailing_reasoning=trailing_reasoning,
            tool_calls=valid_calls,
            errors=errors,
        )


class LlamaToolStreamParser(Qwen35XmlToolStreamParser):
    """Dedicated stream parser for Llama 3.1 / 3.3 and Llama 4 Scout GGUFs with live bracket buffering."""

    def __init__(self, *, started_in_think: bool = False) -> None:
        super().__init__(started_in_think=started_in_think, buffer_brackets=True)


class GptOssHarmonyStreamParser:
    """
    Live incremental parser for GPT-OSS Harmony channel format:
    - `<|channel|>analysis<|message|>...` -> streams live to `StreamDelta.reasoning`
    - `<|channel|>final<|message|>...` -> streams live to `StreamDelta.content`
    - `<|channel|>commentary to=functions.<name> [<|constrain|>json]<|message|>{...}<|call|>`
      (and `to=functions.<name>` in any attribute order) -> emits validated `tool_calls`.
    """

    _CHANNEL_TAG = "<|channel|>"
    _MESSAGE_TAG = "<|message|>"
    _STOP_TAGS = ("<|call|>", "<|return|>", "<|end|>", "<|start|>")

    def __init__(self) -> None:
        self._buffer = ""
        self._state = "scan"  # scan | header | analysis | final | call_body
        self._pre_channel_buf = ""
        self._header_buf = ""
        self._call_name = ""
        self._call_buf = ""
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._raw_calls: list[tuple[str, str]] = []

    def _find_earliest_stop(self, buf: str) -> tuple[int, str]:
        best_idx = -1
        best_tag = ""
        for tag in (*self._STOP_TAGS, self._CHANNEL_TAG):
            pos = buf.find(tag)
            if pos != -1 and (best_idx == -1 or pos < best_idx):
                best_idx = pos
                best_tag = tag
        return best_idx, best_tag

    def _max_partial_stop(self, buf: str) -> int:
        return max(
            _partial_prefix_len(buf, tag)
            for tag in (*self._STOP_TAGS, self._CHANNEL_TAG, self._MESSAGE_TAG)
        )

    def feed(self, chunk: str) -> StreamDelta:
        self._buffer += chunk
        emitted_content: list[str] = []
        emitted_reasoning: list[str] = []

        while self._buffer:
            if self._state == "scan":
                idx = self._buffer.find(self._CHANNEL_TAG)
                if idx != -1:
                    raw_prefix = self._buffer[:idx]
                    self._pre_channel_buf += raw_prefix
                    prefix = CONTROL_TOKEN_RE.sub("", raw_prefix).strip()
                    if prefix and "<|" not in raw_prefix and "to=functions." not in raw_prefix:
                        self._text_parts.append(prefix)
                        emitted_content.append(prefix)
                    self._buffer = self._buffer[idx + len(self._CHANNEL_TAG) :]
                    self._state = "header"
                    self._header_buf = ""
                    continue
                keep = max(
                    _partial_prefix_len(self._buffer, self._CHANNEL_TAG),
                    _partial_prefix_len(self._buffer, "<|start|>assistant"),
                )
                safe = len(self._buffer) - keep
                if safe > 0:
                    seg = self._buffer[:safe]
                    self._pre_channel_buf += seg
                    if "<|start|>" not in seg and "<|" not in seg and "to=functions." not in seg:
                        self._text_parts.append(seg)
                        emitted_content.append(seg)
                    self._buffer = self._buffer[safe:]
                break

            if self._state == "header":
                idx = self._buffer.find(self._MESSAGE_TAG)
                if idx != -1:
                    self._header_buf += self._buffer[:idx]
                    self._buffer = self._buffer[idx + len(self._MESSAGE_TAG) :]
                    hdr = self._header_buf
                    combined_hdr = self._pre_channel_buf + " " + hdr
                    self._header_buf = ""
                    self._pre_channel_buf = ""
                    fn_match = _HARMONY_TO_FUNC_RE.search(combined_hdr)
                    if fn_match is not None:
                        self._call_name = fn_match.group(1).strip()
                        self._call_buf = ""
                        self._state = "call_body"
                    elif "analysis" in hdr:
                        self._state = "analysis"
                    elif "final" in hdr:
                        self._state = "final"
                    else:
                        self._state = "analysis"
                    continue
                keep = _partial_prefix_len(self._buffer, self._MESSAGE_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._header_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            stop_idx, stop_tag = self._find_earliest_stop(self._buffer)
            if stop_idx != -1:
                body_seg = self._buffer[:stop_idx]
                if self._state == "analysis":
                    r = CONTROL_TOKEN_RE.sub("", body_seg)
                    if r:
                        self._reasoning_parts.append(r)
                        emitted_reasoning.append(r)
                elif self._state == "final":
                    t = CONTROL_TOKEN_RE.sub("", body_seg)
                    if t:
                        self._text_parts.append(t)
                        emitted_content.append(t)
                elif self._state == "call_body":
                    self._call_buf += body_seg
                    self._raw_calls.append((self._call_name, self._call_buf.strip()))
                    self._call_name = ""
                    self._call_buf = ""

                if stop_tag == self._CHANNEL_TAG:
                    self._buffer = self._buffer[stop_idx + len(stop_tag) :]
                    self._state = "header"
                    self._header_buf = ""
                else:
                    self._buffer = self._buffer[stop_idx + len(stop_tag) :]
                    self._state = "scan"
                continue

            keep = self._max_partial_stop(self._buffer)
            safe = len(self._buffer) - keep
            if safe > 0:
                seg = self._buffer[:safe]
                if self._state == "analysis":
                    r = CONTROL_TOKEN_RE.sub("", seg)
                    if r:
                        self._reasoning_parts.append(r)
                        emitted_reasoning.append(r)
                elif self._state == "final":
                    t = CONTROL_TOKEN_RE.sub("", seg)
                    if t:
                        self._text_parts.append(t)
                        emitted_content.append(t)
                elif self._state == "call_body":
                    self._call_buf += seg
                self._buffer = self._buffer[safe:]
            break

        return StreamDelta(content="".join(emitted_content), reasoning="".join(emitted_reasoning))

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        trailing_text = ""
        trailing_reasoning = ""
        if self._buffer:
            rem = CONTROL_TOKEN_RE.sub("", self._buffer)
            if self._state == "analysis" and rem:
                self._reasoning_parts.append(rem)
                trailing_reasoning = rem
            elif self._state == "final" and rem:
                self._text_parts.append(rem)
                trailing_text = rem
            elif self._state == "call_body":
                self._call_buf += self._buffer
                if self._call_name and self._call_buf.strip():
                    self._raw_calls.append((self._call_name, self._call_buf.strip()))
            elif self._state == "scan" and rem.strip() and not self._text_parts and not self._reasoning_parts and not self._raw_calls:
                self._text_parts.append(rem.strip())
                trailing_text = rem.strip()
            self._buffer = ""

        tool_calls: list[ParsedToolCall] = []
        errors: list[str] = []
        if tool_choice != "none" and self._raw_calls:
            schema_by_name = _index_tool_schemas(advertised_tools)
            for name, args_raw in self._raw_calls:
                if name not in schema_by_name:
                    errors.append(f"Unknown or unadvertised tool: {name!r}")
                    continue
                arguments = _repair_and_parse_json_obj(args_raw) if args_raw else {}
                if arguments is None:
                    errors.append(f"Malformed Harmony tool_call JSON for {name!r}")
                    continue
                if not isinstance(arguments, dict):
                    errors.append(f"Arguments for tool {name!r} must be an object")
                    continue
                arguments = _coerce_arguments_to_schema(arguments, schema_by_name[name], tool_name=name)
                err = validate_tool_arguments(arguments, schema_by_name[name])
                if err is not None:
                    errors.append(f"Invalid arguments for tool {name!r}: {err}")
                    continue
                tool_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))

        return ToolParseResult(
            text="".join(self._text_parts).strip(),
            trailing_text=trailing_text,
            reasoning="".join(self._reasoning_parts).strip(),
            trailing_reasoning=trailing_reasoning,
            tool_calls=tool_calls,
            errors=errors,
        )


_FENCED_JSON_TOOL_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


class FencedJsonToolStreamParser:
    """
    Incremental streaming parser for prompt-injected tool-calling models (Gemma 3, DeepSeek-VL, DeepSeek-VL2):
    Streams plain text live while holding back `<tool_call>...</tool_call>` and ```` ```json {...} ``` ```` fences.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "text"  # text | tag | fence
        self._hold = ""
        self._halted = False
        self._text_parts: list[str] = []
        self._raw_calls: list[str] = []

    def feed(self, chunk: str) -> StreamDelta:
        if self._halted or not chunk:
            return StreamDelta()
        self._buffer += chunk
        emitted: list[str] = []

        while self._buffer and not self._halted:
            if self._mode == "tag":
                idx = self._buffer.find(CLOSE_TOOL_TAG)
                if idx != -1:
                    self._hold += self._buffer[:idx]
                    self._raw_calls.append(self._hold.strip())
                    self._hold = ""
                    self._mode = "text"
                    self._buffer = self._buffer[idx + len(CLOSE_TOOL_TAG) :]
                    continue
                keep = _partial_prefix_len(self._buffer, CLOSE_TOOL_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._hold += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            if self._mode == "fence":
                idx = self._buffer.find("```")
                if idx != -1:
                    body = self._hold + self._buffer[:idx]
                    full_fence = f"```{body}```"
                    self._hold = ""
                    self._mode = "text"
                    self._buffer = self._buffer[idx + 3 :]
                    m = _FENCED_JSON_TOOL_RE.match(full_fence)
                    if m is not None:
                        cand = m.group(1).strip()
                        parsed = _repair_and_parse_json_obj(cand)
                        if isinstance(parsed, dict) and "name" in parsed and ("arguments" in parsed or "parameters" in parsed):
                            self._raw_calls.append(cand)
                            continue
                    self._text_parts.append(full_fence)
                    emitted.append(full_fence)
                    continue
                keep = _partial_prefix_len(self._buffer, "```")
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._hold += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break

            b_match = HARD_TURN_BOUNDARY_RE.search(self._buffer)
            idx_tag = self._buffer.find(OPEN_TOOL_TAG)
            idx_fence = self._buffer.find("```")
            candidates = [(p, t) for p, t in ((idx_tag, OPEN_TOOL_TAG), (idx_fence, "```")) if p != -1]
            earliest = min((c[0] for c in candidates), default=-1)

            if b_match is not None and (earliest == -1 or b_match.start() < earliest):
                prefix = CONTROL_TOKEN_RE.sub("", self._buffer[: b_match.start()])
                if prefix:
                    self._text_parts.append(prefix)
                    emitted.append(prefix)
                self._buffer = ""
                self._halted = True
                break

            if candidates:
                pos, tok = min(candidates, key=lambda x: x[0])
                if pos > 0:
                    vis = CONTROL_TOKEN_RE.sub("", self._buffer[:pos])
                    if vis:
                        self._text_parts.append(vis)
                        emitted.append(vis)
                self._buffer = self._buffer[pos + len(tok) :]
                self._hold = ""
                self._mode = "tag" if tok == OPEN_TOOL_TAG else "fence"
                continue

            keep = max(
                _partial_prefix_len(self._buffer, OPEN_TOOL_TAG),
                _partial_prefix_len(self._buffer, "```"),
                _partial_prefix_len(self._buffer, "<|im_end|>"),
                _partial_prefix_len(self._buffer, "<end_of_turn>"),
            )
            safe = len(self._buffer) - keep
            if safe > 0:
                vis = CONTROL_TOKEN_RE.sub("", self._buffer[:safe])
                if vis:
                    self._text_parts.append(vis)
                    emitted.append(vis)
                self._buffer = self._buffer[safe:]
            break

        return StreamDelta(content="".join(emitted))

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        trailing = ""
        if self._mode == "tag":
            tail = _truncate_at_turn_boundary(self._hold + self._buffer).strip()
            if tail:
                self._raw_calls.append(tail)
        elif self._mode == "fence":
            rem_fence = "```" + CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._hold + self._buffer))
            self._text_parts.append(rem_fence)
            trailing = rem_fence
        elif self._buffer:
            vis = CONTROL_TOKEN_RE.sub("", _truncate_at_turn_boundary(self._buffer))
            if vis:
                self._text_parts.append(vis)
                trailing = vis
        self._buffer = ""
        self._hold = ""
        self._mode = "text"

        full_text = _truncate_at_turn_boundary("".join(self._text_parts)).strip()
        if tool_choice == "none" or not advertised_tools:
            return ToolParseResult(text=full_text, trailing_text=trailing)

        valid_calls, errors = _validate_json_tool_calls(self._raw_calls, advertised_tools, [])
        return ToolParseResult(
            text=full_text,
            trailing_text=trailing,
            tool_calls=valid_calls,
            errors=errors,
        )


def create_tool_parser(implementation: str, *, started_in_think: bool = False) -> ToolCallStreamParser:
    """Create the dedicated stream parser for the selected language model implementation."""
    if implementation == "qwen3_vl":
        return QwenJsonToolStreamParser()
    if implementation in ("gguf_llama3", "gguf_llama4"):
        return LlamaToolStreamParser(started_in_think=started_in_think)
    if implementation in (
        "qwen3_5",
        "gguf_qwen3_5",
        "gguf_gemma4",
        "gguf_deepseek_qwen3_5",
        "gguf_generic",
    ):
        return Qwen35XmlToolStreamParser(started_in_think=started_in_think)
    if implementation == "gpt_oss":
        return GptOssHarmonyStreamParser()
    if implementation in ("gemma3", "deepseek_vl2", "deepseek_vl"):
        return FencedJsonToolStreamParser()
    raise ValueError(f"No dedicated tool/stream parser for implementation {implementation!r}")


def _split_json_objects(raw: str) -> list[str]:
    """Extract one or more JSON objects from Llama `<|python_tag|>` / `<|python_start|>` output."""
    decoder = json.JSONDecoder()
    idx = 0
    out: list[str] = []
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx] in " \t\r\n;":
            idx += 1
        if idx >= n:
            break
        if raw[idx] != "{":
            break
        try:
            obj, end_idx = decoder.raw_decode(raw, idx)
            if isinstance(obj, dict):
                out.append(json.dumps(obj))
            idx = end_idx
        except json.JSONDecodeError:
            return []
    return out


def _parse_gemma4_args(raw_obj: str) -> dict[str, Any]:
    """Parse Gemma 4 `call:name{key:<|"|>val<|"|>}` or JSON object arguments."""
    cleaned = raw_obj.replace('<|"|>', '"')
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    quoted_keys = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', cleaned)
    try:
        parsed = json.loads(quoted_keys)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _coerce_xml_parameter(raw_value: str, prop_schema: Any) -> Any:
    expected = prop_schema.get("type") if isinstance(prop_schema, dict) else None
    if expected == "string":
        return raw_value
    if expected == "boolean":
        low = raw_value.lower()
        if low == "true":
            return True
        if low == "false":
            return False
    if expected == "integer":
        try:
            return int(raw_value)
        except ValueError:
            return raw_value
    if expected == "number":
        try:
            return float(raw_value)
        except ValueError:
            return raw_value
    if expected in ("object", "array", "null") or expected is None:
        if raw_value.startswith(("{", "[")) or raw_value in ("true", "false", "null"):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                return raw_value
    return raw_value


def _repair_and_parse_json_obj(raw: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object with repair for fenced blocks, trailing commas, and Pythonic dicts."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    for candidate in (
        text,
        re.sub(r",\s*([}\]])", r"\1", text),
        re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', re.sub(r",\s*([}\]])", r"\1", text)),
    ):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        parsed_ast = ast.literal_eval(text)
        if isinstance(parsed_ast, dict):
            return parsed_ast
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        snippet = re.sub(r",\s*([}\]])", r"\1", text[start : end + 1])
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "file_path": ("path", "filepath", "filePath", "filename"),
    "path": ("file_path", "filepath", "filePath"),
    "command": ("cmd", "shell_command", "bash_command"),
    "queries": ("query", "search_query"),
    "query": ("queries",),
    "urls": ("url",),
    "url": ("urls",),
    "timeoutMs": ("timeout_ms", "timeout"),
    "timeout_ms": ("timeoutMs", "timeout"),
}


def _coerce_value_to_schema(value: Any, prop_schema: Any) -> Any:
    if not isinstance(prop_schema, dict):
        return value
    expected = prop_schema.get("type")
    if not isinstance(expected, str):
        return value
    if expected == "integer" and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                try:
                    fval = float(value.strip())
                    if fval.is_integer():
                        return int(fval)
                except ValueError:
                    pass
    elif expected == "number" and not isinstance(value, bool):
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
    elif expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes"}:
                return True
            if low in {"false", "0", "no"}:
                return False
    elif expected == "array":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [value] if value.strip() else []
    elif expected == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed_obj = _repair_and_parse_json_obj(value)
            if parsed_obj is not None:
                return parsed_obj
    elif expected == "string" and value is not None and not isinstance(value, str):
        return str(value)
    return value


def _coerce_arguments_to_schema(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    *,
    tool_name: str = "",
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return arguments
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return arguments
    required_keys = set(schema.get("required") or [])
    out = dict(arguments)

    # 1. Resolve common parameter aliases (`path` -> `file_path`, `timeout` -> `timeoutMs`, etc.)
    for target_key in props:
        if target_key not in out:
            for alias in _PARAM_ALIASES.get(target_key, ()):
                if alias in out:
                    out[target_key] = out.pop(alias)
                    break

    # 2. Strip unadvertised hallucinated keys (e.g. `run_in_background`, `sandbox_permissions`)
    for key in list(out.keys()):
        if key not in props:
            out.pop(key, None)

    # 3. Coerce types and sanitize timeout / numeric bounds
    for key, val in list(out.items()):
        prop_schema = props.get(key)
        coerced = _coerce_value_to_schema(val, prop_schema)

        if key in ("timeout_ms", "timeoutMs") or key.endswith(("_ms", "Ms")):
            if isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
                if coerced <= 0:
                    if key not in required_keys:
                        out.pop(key, None)
                        continue
                    coerced = 10000
                elif 1 <= coerced <= 300:
                    # Model passed seconds (e.g. `30`) to a millisecond field -> convert to ms!
                    coerced = int(coerced * 1000)
                if key == "timeout_ms":
                    coerced = max(10000, min(3600000, int(coerced)))

        if isinstance(prop_schema, dict) and isinstance(coerced, (int, float)) and not isinstance(coerced, bool):
            min_v = prop_schema.get("minimum")
            max_v = prop_schema.get("maximum")
            if isinstance(min_v, (int, float)) and coerced < min_v:
                coerced = int(min_v) if prop_schema.get("type") == "integer" else float(min_v)
            if isinstance(max_v, (int, float)) and coerced > max_v:
                coerced = int(max_v) if prop_schema.get("type") == "integer" else float(max_v)

        out[key] = coerced

    # 4. Auto-fill required `description` on `bash` / `pwsh` when omitted or blank by local models
    if tool_name in ("bash", "pwsh") and "description" in props:
        cur_desc = out.get("description")
        if not isinstance(cur_desc, str) or not cur_desc.strip():
            cmd_preview = str(out.get("command") or "shell command").strip().splitlines()[0][:64]
            out["description"] = f"Run: {cmd_preview}"

    return out


def _colon_kwargs_to_equals(expr: str) -> str:
    """Convert JSON-style `func(key: "val", n: 2)` outside quotes into valid Python `func(key="val", n=2)`."""
    out: list[str] = []
    in_quote: Optional[str] = None
    escaped = False
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and in_quote is not None:
            out.append(ch)
            escaped = True
            i += 1
            continue
        if ch in ('"', "'"):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
            out.append(ch)
            i += 1
            continue
        if in_quote is None and ch == ":":
            # Check if preceded by an identifier right after `(` or `,`
            prefix = "".join(out)
            if re.search(r"[(,]\s*[A-Za-z_][A-Za-z0-9_]*\s*$", prefix):
                out.append("=")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_python_literals(expr: str) -> str:
    """Convert unquoted JSON `true`/`false`/`null` keyword values into Python `True`/`False`/`None`."""
    out = re.sub(r"(=\s*)true\b", r"\1True", expr)
    out = re.sub(r"(=\s*)false\b", r"\1False", out)
    out = re.sub(r"(=\s*)null\b", r"\1None", out)
    return out


def _parse_pythonic_calls(
    raw: str,
    known_tools: Optional[set[str]] = None,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Parse `[func(k=v, ...)]`, `func(k: "v")`, or fused `[subagent_fork web_search(query: "...")]`
    into `(tool_name, kwargs)` pairs via AST.
    """
    text = (raw or "").strip()
    if not text:
        return []

    inner = text[1:-1].strip() if (text.startswith("[") and text.endswith("]")) else text
    # Strip fused hallucinated prefix tokens before an actual `tool_name(...)` call, e.g.:
    # `subagent_fork web_search(query: "gta 6")` -> `web_search(query: "gta 6")`
    m_fused = re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*\s+([A-Za-z_][A-Za-z0-9_.-]*\s*\([\s\S]*\))$", inner)
    if m_fused is not None:
        inner = m_fused.group(1).strip()

    repaired = _normalize_python_literals(_colon_kwargs_to_equals(inner))
    for candidate in (f"[{repaired}]", repaired):
        try:
            tree = ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        nodes: list[ast.expr] = list(tree.body.elts) if isinstance(tree.body, ast.List) else [tree.body]
        parsed: list[tuple[str, dict[str, Any]]] = []
        valid_tree = True
        for node in nodes:
            if not isinstance(node, ast.Call):
                valid_tree = False
                break
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            else:
                valid_tree = False
                break
            if known_tools is not None and known_tools and fn_name not in known_tools:
                valid_tree = False
                break
            kwargs: dict[str, Any] = {}
            for kw in node.keywords:
                if not kw.arg:
                    continue
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    valid_tree = False
                    break
            if not valid_tree:
                break
            parsed.append((fn_name, kwargs))
        if valid_tree and parsed:
            return parsed
    return []


def _validate_json_tool_calls(
    raw_calls: list[str],
    advertised_tools: list[dict[str, Any]],
    initial_errors: list[str],
) -> tuple[list[ParsedToolCall], list[str]]:
    schema_by_name = _index_tool_schemas(advertised_tools)
    valid_calls: list[ParsedToolCall] = []
    errors = list(initial_errors)
    for raw in raw_calls:
        payload = _repair_and_parse_json_obj(raw)
        if payload is None:
            py_calls = _parse_pythonic_calls(raw)
            if py_calls:
                for py_name, py_args in py_calls:
                    if py_name not in schema_by_name:
                        errors.append(f"Unknown or unadvertised tool: {py_name!r}")
                        continue
                    coerced = _coerce_arguments_to_schema(py_args, schema_by_name[py_name], tool_name=py_name)
                    err = validate_tool_arguments(coerced, schema_by_name[py_name])
                    if err is not None:
                        errors.append(f"Invalid arguments for tool {py_name!r}: {err}")
                        continue
                    valid_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=py_name, arguments=coerced))
                continue
            errors.append("Malformed tool_call JSON")
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = payload.get("parameters", {})
        if isinstance(arguments, str):
            repaired_args = _repair_and_parse_json_obj(arguments)
            if repaired_args is None:
                errors.append("Malformed tool_call arguments JSON string")
                continue
            arguments = repaired_args
        if not isinstance(name, str) or name not in schema_by_name:
            errors.append(f"Unknown or unadvertised tool: {name!r}")
            continue
        if not isinstance(arguments, dict):
            errors.append(f"Arguments for tool {name!r} must be an object")
            continue
        arguments = _coerce_arguments_to_schema(arguments, schema_by_name[name], tool_name=name)
        err = validate_tool_arguments(arguments, schema_by_name[name])
        if err is not None:
            errors.append(f"Invalid arguments for tool {name!r}: {err}")
            continue
        valid_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))
    return valid_calls, errors


def _partial_prefix_len(buf: str, tag: str) -> int:
    for length in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:length]):
            return length
    return 0


def _index_tool_schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entry in tools:
        fn = entry.get("function") if entry.get("type") == "function" else entry
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            indexed[fn["name"]] = fn.get("parameters") or {}
    return indexed


def _check_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


def _validate_node(value: Any, schema: dict[str, Any], path: str) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _check_json_type(value, expected_type):
        return f"field '{path}' expected type '{expected_type}'"
    if "enum" in schema and value not in schema["enum"]:
        return f"field '{path}' value {value!r} not in enum {schema['enum']!r}"

    if isinstance(value, dict):
        for req_key in schema.get("required") or []:
            if req_key not in value:
                sub = f"{path}.{req_key}" if path else str(req_key)
                return f"missing required field '{sub}'"
        properties = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for key, item_val in value.items():
            sub = f"{path}.{key}" if path else str(key)
            if key not in properties:
                if additional is False:
                    return f"unexpected additional property '{sub}'"
                continue
            err = _validate_node(item_val, properties[key], sub)
            if err is not None:
                return err

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for idx, elem in enumerate(value):
            err = _validate_node(elem, schema["items"], f"{path}[{idx}]")
            if err is not None:
                return err

    return None


def validate_tool_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> Optional[str]:
    """Recursively validate tool arguments against JSON Schema (nested objects, arrays, enums, required)."""
    return _validate_node(arguments, schema, "")
