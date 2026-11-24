"""Unit and integration tests for the layered Nexus Gateway."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.tool_calls import create_tool_parser
from api.middleware.auth import assert_loopback_host, generate_bearer_token, verify_bearer_token
from core.model_registry import ModelRegistry, inspect_model_directory
from core.model_staging import stage_model_directory
from core.resource_manager import ResourceManager
from services.progress import ProgressService

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nexus_generate_image",
            "description": "Generate an image",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}, "steps": {"type": "integer"}},
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }
]


def test_loopback_and_auth() -> None:
    assert assert_loopback_host("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValueError):
        assert_loopback_host("0.0.0.0")
    tok = generate_bearer_token()
    assert verify_bearer_token(tok, f"Bearer {tok}") is True
    assert verify_bearer_token(tok, "Bearer bad") is False


def test_staging_progress_and_cache_hit(tmp_path: Path) -> None:
    drive_model = tmp_path / "drive" / "Qwen3-VL-4B-Heretic"
    drive_model.mkdir(parents=True)
    (drive_model / "weights.bin").write_bytes(b"0" * 1024)
    cache_root = tmp_path / "cache"

    progress = ProgressService()
    staged = stage_model_directory(
        drive_model, cache_root, model_id="Qwen3-VL-4B-Heretic", progress=progress, min_free_margin_bytes=0
    )
    assert (staged / "weights.bin").is_file()
    snap = progress.snapshot()
    assert snap["chat"]["bytes_copied"] == 1024
    assert snap["chat"]["total_bytes"] == 1024
    assert snap["chat"]["progress"] == 100.0


def test_resource_manager_single_active_model() -> None:
    mgr = ResourceManager()
    unloaded: list[str] = []
    with mgr.acquire("chat", "qwen3-vl", lambda: "m1", unload=lambda _: unloaded.append("qwen3-vl")):
        assert mgr.active_model_id == "qwen3-vl"
    with mgr.acquire("diffusion", "flux2", lambda: "m2", unload=lambda _: unloaded.append("flux2")):
        assert unloaded == ["qwen3-vl"]


def test_model_specific_tool_parsers_and_pseudo_marker_rejection() -> None:
    qwen = create_tool_parser("qwen")
    qwen.feed('<tool_call>{"name":"nexus_generate_image","arguments":{"prompt":"moon","steps":4}}</tool_call>')
    res = qwen.finish(SAMPLE_TOOLS)
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "nexus_generate_image"

    # Legacy pseudo-marker must never be dispatched
    qwen_legacy = create_tool_parser("qwen")
    qwen_legacy.feed('[[NEXUS_TOOL_CALL]]{"name":"nexus_generate_image","arguments":{"prompt":"moon"}}')
    assert qwen_legacy.finish(SAMPLE_TOOLS).tool_calls == []

    # Non-Qwen model family does not blindly parse Qwen tags
    other = create_tool_parser("generic_json")
    other.feed('<tool_call>{"name":"nexus_generate_image","arguments":{"prompt":"moon"}}</tool_call>')
    assert other.finish(SAMPLE_TOOLS).tool_calls == []


def test_registry_discovers_chat_and_flux(tmp_path: Path) -> None:
    vlm = tmp_path / "Qwen3-VL"
    vlm.mkdir()
    (vlm / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (vlm / "tokenizer_config.json").write_text(json.dumps({"chat_template": "tools tool_call"}))
    flux = tmp_path / "FLUX2"
    flux.mkdir()
    (flux / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinKVPipeline"}))

    reg = ModelRegistry(tmp_path)
    reg.scan()
    assert [m.id for m in reg.chat_models()] == ["Qwen3-VL"]
    assert [m.id for m in reg.image_models()] == ["FLUX2"]
    assert inspect_model_directory(vlm).tool_parser_family == "qwen"
