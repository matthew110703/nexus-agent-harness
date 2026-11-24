"""Model-specific native tool-call streaming parsers and schema validators."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
CONTROL_TOKEN_RE = re.compile(r"<\|(?:im_end|im_start|endoftext)\|>")


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
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ToolCallStreamParser(Protocol):
    def feed(self, chunk: str) -> str: ...
    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult: ...


class QwenToolCallStreamParser:
    """Incremental parser for Qwen `<tool_call>...</tool_call>` blocks across arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_call = False
        self._call_buf = ""
        self._text_parts: list[str] = []
        self._raw_calls: list[str] = []
        self._errors: list[str] = []

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        emitted: list[str] = []
        while self._buffer:
            if not self._in_call:
                idx = self._buffer.find(OPEN_TAG)
                if idx != -1:
                    if idx > 0:
                        visible = CONTROL_TOKEN_RE.sub("", self._buffer[:idx])
                        if visible:
                            self._text_parts.append(visible)
                            emitted.append(visible)
                    self._buffer = self._buffer[idx + len(OPEN_TAG) :]
                    self._in_call = True
                    self._call_buf = ""
                    continue
                keep = _partial_prefix_len(self._buffer, OPEN_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    visible = CONTROL_TOKEN_RE.sub("", self._buffer[:safe])
                    if visible:
                        self._text_parts.append(visible)
                        emitted.append(visible)
                    self._buffer = self._buffer[safe:]
                break
            else:
                idx = self._buffer.find(CLOSE_TAG)
                if idx != -1:
                    self._call_buf += self._buffer[:idx]
                    self._raw_calls.append(self._call_buf.strip())
                    self._call_buf = ""
                    self._in_call = False
                    self._buffer = self._buffer[idx + len(CLOSE_TAG) :]
                    continue
                keep = _partial_prefix_len(self._buffer, CLOSE_TAG)
                safe = len(self._buffer) - keep
                if safe > 0:
                    self._call_buf += self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                break
        return "".join(emitted)

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        if self._in_call:
            self._errors.append("Unterminated <tool_call> block")
            self._in_call = False
            self._call_buf = ""
            self._buffer = ""
        elif self._buffer:
            visible = CONTROL_TOKEN_RE.sub("", self._buffer)
            if visible:
                self._text_parts.append(visible)
            self._buffer = ""

        if tool_choice == "none":
            return ToolParseResult(text="".join(self._text_parts).strip(), tool_calls=[], errors=[])

        schema_by_name = _index_tool_schemas(advertised_tools)
        valid_calls: list[ParsedToolCall] = []
        errors = list(self._errors)
        for raw in self._raw_calls:
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
        return ToolParseResult(text="".join(self._text_parts).strip(), tool_calls=valid_calls, errors=errors)


class PlainTextOnlyParser:
    """Fallback parser for model families without a registered native tool-call parser."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def feed(self, chunk: str) -> str:
        clean = CONTROL_TOKEN_RE.sub("", chunk)
        if clean:
            self._parts.append(clean)
        return clean

    def finish(self, advertised_tools: list[dict[str, Any]], tool_choice: Any = "auto") -> ToolParseResult:
        return ToolParseResult(text="".join(self._parts).strip(), tool_calls=[], errors=[])


def create_tool_parser(family: str) -> ToolCallStreamParser:
    """Select the model-specific native tool parser; never assumes every model is Qwen."""
    if family == "qwen":
        return QwenToolCallStreamParser()
    return PlainTextOnlyParser()


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


def validate_tool_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> Optional[str]:
    for key in schema.get("required") or []:
        if key not in arguments:
            return f"missing required field '{key}'"
    properties = schema.get("properties") or {}
    additional = schema.get("additionalProperties", True)
    for key, value in arguments.items():
        if key not in properties:
            if additional is False:
                return f"unexpected additional property '{key}'"
            continue
        prop = properties[key]
        if not isinstance(prop, dict):
            continue
        expected_type = prop.get("type")
        if isinstance(expected_type, str) and not _check_json_type(value, expected_type):
            return f"field '{key}' expected type '{expected_type}'"
        if "enum" in prop and value not in prop["enum"]:
            return f"field '{key}' value {value!r} not in enum {prop['enum']!r}"
    return None
