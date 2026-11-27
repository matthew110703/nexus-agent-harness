"""Model-specific native tool-call & reasoning stream parsers and recursive JSON Schema validator."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

OPEN_TOOL_TAG = "<tool_call>"
CLOSE_TOOL_TAG = "</tool_call>"
OPEN_THINK_TAG = "<think>"
CLOSE_THINK_TAG = "</think>"
CONTROL_TOKEN_RE = re.compile(r"<\|(?:im_end|im_start|endoftext|return|end|call|start|channel|message)\|>")
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
})

_XML_FUNC_RE = re.compile(r"<function=([^\s>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_HARMONY_CALL_RE = re.compile(
    r"(?:<\|start\|>assistant\s+)?to=functions\.([A-Za-z0-9_.-]+)<\|channel\|>commentary(?:\s+json)?<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)
_HARMONY_ANALYSIS_RE = re.compile(
    r"(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)",
    re.DOTALL,
)
_HARMONY_FINAL_RE = re.compile(
    r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)


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
    """Incremental parser for Qwen3-VL JSON `<tool_call>{"name":...,"arguments":...}</tool_call>` blocks."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_call = False
        self._call_buf = ""
        self._text_parts: list[str] = []
        self._raw_calls: list[str] = []
        self._errors: list[str] = []

    def feed(self, chunk: str) -> StreamDelta:
        self._buffer += chunk
        emitted: list[str] = []
        while self._buffer:
            if not self._in_call:
                idx = self._buffer.find(OPEN_TOOL_TAG)
                if idx != -1:
                    if idx > 0:
                        visible = CONTROL_TOKEN_RE.sub("", self._buffer[:idx])
                        if visible:
                            self._text_parts.append(visible)
                            emitted.append(visible)
                    self._buffer = self._buffer[idx + len(OPEN_TOOL_TAG) :]
                    self._in_call = True
                    self._call_buf = ""
                    continue
                keep = _partial_prefix_len(self._buffer, OPEN_TOOL_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    visible = CONTROL_TOKEN_RE.sub("", self._buffer[:safe])
                    if visible:
                        self._text_parts.append(visible)
                        emitted.append(visible)
                    self._buffer = self._buffer[safe:]
                break
            else:
                idx = self._buffer.find(CLOSE_TOOL_TAG)
                if idx != -1:
                    self._call_buf += self._buffer[:idx]
                    self._raw_calls.append(self._call_buf.strip())
                    self._call_buf = ""
                    self._in_call = False
                    self._buffer = self._buffer[idx + len(CLOSE_TOOL_TAG) :]
                    continue
                keep = _partial_prefix_len(self._buffer, CLOSE_TOOL_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break
        return StreamDelta(content="".join(emitted))

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        trailing = ""
        if self._in_call:
            self._errors.append("Unterminated <tool_call> block")
            self._in_call = False
            self._call_buf = ""
            self._buffer = ""
        elif self._buffer:
            visible = CONTROL_TOKEN_RE.sub("", self._buffer)
            if visible:
                self._text_parts.append(visible)
                trailing = visible
            self._buffer = ""

        if tool_choice == "none":
            return ToolParseResult(text="".join(self._text_parts).strip(), trailing_text=trailing)

        valid_calls, errors = _validate_json_tool_calls(self._raw_calls, advertised_tools, self._errors)
        return ToolParseResult(
            text="".join(self._text_parts).strip(),
            trailing_text=trailing,
            tool_calls=valid_calls,
            errors=errors,
        )


class Qwen35XmlToolStreamParser:
    """
    Incremental parser for Qwen3.5 / Qwen3.6 models:
    - Streams `<think>...</think>` reasoning blocks (including when the chat template
      prefilled `<think>\n` before token 0) into `StreamDelta.reasoning`.
    - Parses `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
      (plus JSON fallback) with schema-aware value coercion and validation.
    """

    def __init__(self, *, started_in_think: bool = False) -> None:
        self._buffer = ""
        self._state: str = "think" if started_in_think else "text"
        self._call_buf = ""
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._raw_calls: list[str] = []
        self._errors: list[str] = []

    def feed(self, chunk: str) -> StreamDelta:
        self._buffer += chunk
        emitted_content: list[str] = []
        emitted_reasoning: list[str] = []

        while self._buffer:
            if self._state == "think":
                idx = self._buffer.find(CLOSE_THINK_TAG)
                if idx != -1:
                    if idx > 0:
                        r = CONTROL_TOKEN_RE.sub("", self._buffer[:idx])
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

            # state == "text"
            idx_tool = self._buffer.find(OPEN_TOOL_TAG)
            idx_think = self._buffer.find(OPEN_THINK_TAG)
            candidates = [(i, tag) for i, tag in ((idx_tool, OPEN_TOOL_TAG), (idx_think, OPEN_THINK_TAG)) if i != -1]
            if candidates:
                idx, tag = min(candidates, key=lambda pair: pair[0])
                if idx > 0:
                    visible = CONTROL_TOKEN_RE.sub("", self._buffer[:idx])
                    if visible:
                        self._text_parts.append(visible)
                        emitted_content.append(visible)
                self._buffer = self._buffer[idx + len(tag) :]
                if tag == OPEN_TOOL_TAG:
                    self._state = "call"
                    self._call_buf = ""
                else:
                    self._state = "think"
                continue

            keep = max(
                _partial_prefix_len(self._buffer, OPEN_TOOL_TAG),
                _partial_prefix_len(self._buffer, OPEN_THINK_TAG),
            )
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
            self._errors.append("Unterminated <tool_call> block")
            self._state = "text"
            self._call_buf = ""
            self._buffer = ""
        elif self._state == "think" and self._buffer:
            r = CONTROL_TOKEN_RE.sub("", self._buffer)
            if r:
                self._reasoning_parts.append(r)
                trailing_reasoning = r
            self._buffer = ""
        elif self._buffer:
            visible = CONTROL_TOKEN_RE.sub("", self._buffer)
            if visible:
                self._text_parts.append(visible)
                trailing_text = visible
            self._buffer = ""

        if tool_choice == "none":
            return ToolParseResult(
                text="".join(self._text_parts).strip(),
                trailing_text=trailing_text,
                reasoning="".join(self._reasoning_parts).strip(),
                trailing_reasoning=trailing_reasoning,
            )

        schema_by_name = _index_tool_schemas(advertised_tools)
        valid_calls: list[ParsedToolCall] = []
        errors = list(self._errors)
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
                err = validate_tool_arguments(arguments, schema)
                if err is not None:
                    errors.append(f"Invalid arguments for tool {name!r}: {err}")
                    continue
                valid_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))
                continue

            # Fallback to JSON payload inside <tool_call>...</tool_call>.
            calls_from_json, json_errs = _validate_json_tool_calls([raw], advertised_tools, [])
            valid_calls.extend(calls_from_json)
            errors.extend(json_errs)

        return ToolParseResult(
            text="".join(self._text_parts).strip(),
            trailing_text=trailing_text,
            reasoning="".join(self._reasoning_parts).strip(),
            trailing_reasoning=trailing_reasoning,
            tool_calls=valid_calls,
            errors=errors,
        )


class GptOssHarmonyStreamParser:
    """
    Parser for GPT-OSS Harmony channel format (`analysis`, `commentary json` `<|call|>`, `final`).
    Buffers channel tokens cleanly so raw Harmony control tokens never leak to the user.
    """

    def __init__(self) -> None:
        self._raw = ""

    def feed(self, chunk: str) -> StreamDelta:
        self._raw += chunk
        return StreamDelta()

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        raw = self._raw
        reasoning_blocks = [m.group(1).strip() for m in _HARMONY_ANALYSIS_RE.finditer(raw) if m.group(1).strip()]
        final_blocks = [m.group(1).strip() for m in _HARMONY_FINAL_RE.finditer(raw) if m.group(1).strip()]

        tool_calls: list[ParsedToolCall] = []
        errors: list[str] = []
        if tool_choice != "none":
            schema_by_name = _index_tool_schemas(advertised_tools)
            for m in _HARMONY_CALL_RE.finditer(raw):
                name = m.group(1).strip()
                args_raw = m.group(2).strip()
                if name not in schema_by_name:
                    errors.append(f"Unknown or unadvertised tool: {name!r}")
                    continue
                try:
                    arguments = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError as exc:
                    errors.append(f"Malformed Harmony tool_call JSON for {name!r}: {exc.msg}")
                    continue
                if not isinstance(arguments, dict):
                    errors.append(f"Arguments for tool {name!r} must be an object")
                    continue
                err = validate_tool_arguments(arguments, schema_by_name[name])
                if err is not None:
                    errors.append(f"Invalid arguments for tool {name!r}: {err}")
                    continue
                tool_calls.append(ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments))

        if final_blocks:
            text = "\n".join(final_blocks).strip()
        elif not tool_calls and not reasoning_blocks:
            text = CONTROL_TOKEN_RE.sub("", raw).strip()
        else:
            text = ""
        reasoning = "\n".join(reasoning_blocks).strip()
        return ToolParseResult(
            text=text,
            trailing_text=text,
            reasoning=reasoning,
            trailing_reasoning=reasoning,
            tool_calls=tool_calls,
            errors=errors,
        )


_FENCED_JSON_TOOL_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


class FencedJsonToolStreamParser:
    """
    Parser for prompt-injected tool-calling models (Gemma 3, DeepSeek-VL, DeepSeek-VL2):
    Supports both `<tool_call>{"name":...,"arguments":...}</tool_call>` and fenced
    ```` ```json {"name":...,"arguments":...} ``` ```` blocks.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def feed(self, chunk: str) -> StreamDelta:
        self._chunks.append(chunk)
        # Buffer until finish so fenced JSON tool calls are not leaked as visible assistant text.
        return StreamDelta()

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        raw = CONTROL_TOKEN_RE.sub("", "".join(self._chunks))
        if tool_choice == "none" or not advertised_tools:
            text = raw.strip()
            return ToolParseResult(text=text, trailing_text=text)

        raw_calls: list[str] = []
        cleaned = raw

        # 1. Extract <tool_call>...</tool_call> blocks if emitted
        def _collect_tag(match: re.Match[str]) -> str:
            raw_calls.append(match.group(1).strip())
            return ""

        cleaned = re.sub(r"<tool_call>([\s\S]*?)</tool_call>", _collect_tag, cleaned)

        # 2. Extract fenced JSON blocks that contain both "name" and "arguments"
        def _collect_fenced(match: re.Match[str]) -> str:
            candidate = match.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                    raw_calls.append(candidate)
                    return ""
            except json.JSONDecodeError:
                pass
            return match.group(0)

        cleaned = _FENCED_JSON_TOOL_RE.sub(_collect_fenced, cleaned)

        valid_calls, errors = _validate_json_tool_calls(raw_calls, advertised_tools, [])
        text = cleaned.strip()
        return ToolParseResult(
            text=text,
            trailing_text=text,
            tool_calls=valid_calls,
            errors=errors,
        )


def create_tool_parser(implementation: str, *, started_in_think: bool = False) -> ToolCallStreamParser:
    """Create the dedicated stream parser for the selected language model implementation."""
    if implementation == "qwen3_vl":
        return QwenJsonToolStreamParser()
    if implementation in ("qwen3_5", "gguf_qwen3_5", "gguf_gemma4", "gguf_deepseek_qwen3_5"):
        return Qwen35XmlToolStreamParser(started_in_think=started_in_think)
    if implementation == "gpt_oss":
        return GptOssHarmonyStreamParser()
    if implementation in ("gemma3", "deepseek_vl2", "deepseek_vl"):
        return FencedJsonToolStreamParser()
    raise ValueError(f"No dedicated tool/stream parser for implementation {implementation!r}")


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


def _validate_json_tool_calls(
    raw_calls: list[str],
    advertised_tools: list[dict[str, Any]],
    initial_errors: list[str],
) -> tuple[list[ParsedToolCall], list[str]]:
    schema_by_name = _index_tool_schemas(advertised_tools)
    valid_calls: list[ParsedToolCall] = []
    errors = list(initial_errors)
    for raw in raw_calls:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"Malformed tool_call JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append("Tool call payload must be a JSON object")
            continue
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                errors.append(f"Malformed tool_call arguments JSON string: {exc.msg}")
                continue
        if not isinstance(name, str) or name not in schema_by_name:
            errors.append(f"Unknown or unadvertised tool: {name!r}")
            continue
        if not isinstance(arguments, dict):
            errors.append(f"Arguments for tool {name!r} must be an object")
            continue
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
