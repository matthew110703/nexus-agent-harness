"""Unit tests for Nexus AI Lab v2.2.0 gateway upgrades."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from adapters.tool_calls import QwenJsonToolStreamParser, create_tool_parser
from models.gguf_metadata import inspect_gguf_directory
from services.conversation import normalize_openai_messages
from services.image_generation import save_generated_image_artifact
from services.web_research import validate_public_http_url


def test_tool_call_repair_aliases_coercion_and_pythonic() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "nexus_generate_image",
                "parameters": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "steps": {"type": "integer"},
                        "use_uploaded_image": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        },
    ]
    parser = QwenJsonToolStreamParser()
    parser.feed(
        '<tool_call>{"name": "nexus_generate_image", "arguments": {"prompt": "neon city", "steps": "8", "use_uploaded_image": "true",}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
    )
    res = parser.finish(tools)
    assert res.errors == []
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].arguments == {"prompt": "neon city", "steps": 8, "use_uploaded_image": True}
    assert res.tool_calls[1].arguments == {"file_path": "README.md"}

    py_parser = create_tool_parser("gguf_llama4")
    py_parser.feed('<tool_call>[nexus_generate_image(prompt="mountain", steps="6")]</tool_call>')
    py_res = py_parser.finish(tools)
    assert py_res.errors == []
    assert len(py_res.tool_calls) == 1
    assert py_res.tool_calls[0].arguments == {"prompt": "mountain", "steps": 6}


def test_normalize_openai_messages_strips_images_for_text_models() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Make this cyberpunk"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    out = normalize_openai_messages(msgs, allow_images=False)
    assert isinstance(out[0]["content"], str)
    assert "Make this cyberpunk" in out[0]["content"]
    assert "nexus_generate_image" in out[0]["content"]


def test_save_generated_image_artifact_dual_paths(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    ws_dir = drive_root / "workspaces" / "my_project"
    ws_dir.mkdir(parents=True)
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
    saved = save_generated_image_artifact(
        output_dir=drive_root / "outputs" / "generated_images",
        b64_png=png_b64,
        model_id="FLUX.2-klein-4B",
        prompt="sunset",
        quality="detailed",
        width=1024,
        height=1024,
        steps=6,
        seed=42,
        operation_id="op_1",
        workspace=str(ws_dir),
        drive_root=drive_root,
        session_id="sess_abc",
        reference_image_used=True,
        strength=0.65,
    )
    assert Path(saved["drive_path"]).is_file()
    assert Path(saved["workspace_path"]).is_file()
    meta = json.loads(Path(saved["drive_path"] + ".json").read_text(encoding="utf-8"))
    assert meta["session_id"] == "sess_abc"
    assert meta["reference_image_used"] is True
    assert meta["strength"] == 0.65


def test_gguf_metadata_ignores_copying_temp_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "Llama-Test"
    model_dir.mkdir()
    (model_dir / "model.gguf.copying").write_bytes(b"partial")
    layout = inspect_gguf_directory(model_dir)
    assert not layout.valid


def test_validate_public_http_url_blocks_loopback() -> None:
    assert validate_public_http_url("https://example.com/docs") == "https://example.com/docs"
    with pytest.raises(ValueError):
        validate_public_http_url("http://127.0.0.1:7861/health")
    with pytest.raises(ValueError):
        validate_public_http_url("http://localhost:3080/")
    with pytest.raises(ValueError):
        validate_public_http_url("file:///etc/passwd")
