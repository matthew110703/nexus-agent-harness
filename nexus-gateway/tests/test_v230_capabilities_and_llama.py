"""Regression suite for Nexus AI Lab v2.3.0:
- Llama 3.1 / Llama 4 Scout bracket, fused prefix, colon-kwarg, and `<|python_start|>` tool parsing
- Hard turn-boundary truncation (`<|im_end|>`, `<|im_start|>`, `<|eot_id|>`, `<|eot|>`)
- Gemma 4 `<|tool_call|>call:...<|tool_call|>` live buffering (zero raw leakage)
- Schema-guided argument hygiene (`timeout_ms` clamping, seconds->ms, `bash.description` auto-fill, unadvertised key stripping)
- Multi-turn conversation normalization (post-tool runtime notice folding without fake cancellation, non-imperative text-only image note)
- Dynamic `ModelCapabilityProfile` & curated Jinja templates (`llama3_1_nexus.jinja`, `llama4_scout_nexus.jinja`)
- Clean `workspace/generated_images` without `.png.json` sidecar clutter
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from adapters.tool_calls import create_tool_parser
from core.capabilities import (
    apply_capability_grounding,
    build_model_capability_profile,
    compact_tools_for_context,
    estimate_openai_usage,
    resolve_curated_template_path,
    resolve_sampling_params,
)
from models.base import SelectedImageSpec, SelectedLanguageSpec
from models.gguf_metadata import GgufHeaderInfo, _classify_template
from models.language.gguf_common import LlamaServerRuntimeBase
from models.language.prompt_tools import inject_tools_into_messages, inject_tools_preserving_images
from services.conversation import normalize_openai_messages
from services.image_generation import save_generated_image_artifact


SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for public information. " * 10,
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page URL.",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command.",
            "parameters": {
                "type": "object",
                "required": ["description", "command"],
                "properties": {
                    "description": {"type": "string"},
                    "command": {"type": "string"},
                    "timeoutMs": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_agent",
            "description": "Wait for a subagent.",
            "parameters": {
                "type": "object",
                "required": ["ids"],
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "timeout_ms": {"type": "integer", "minimum": 10000, "maximum": 3600000},
                },
            },
        },
    },
]


def test_llama_exact_user_failure_fused_brackets_and_turn_leakage() -> None:
    """Replays the exact failing Llama output reported by the user and verifies clean extraction + truncation."""
    raw_output = (
        '[list_agents()]\n'
        '[subagent_fork web_search(query: "gta 6 XBox build leaked exe file PC port")]\n'
        'Please wait...<|im_end|><|im_end|> <|im_start|>assistant\n'
        '[subagent_fork web_search(query: "gta 6 XBox build leaked")]\n'
        'Would you like me to continue with the research?<|im_end|>'
    )
    parser = create_tool_parser("gguf_llama3")
    emitted: list[str] = []
    # Stream in small chunks to exercise live state-machine buffering
    for i in range(0, len(raw_output), 7):
        delta = parser.feed(raw_output[i : i + 7])
        if delta.content:
            emitted.append(delta.content)
    res = parser.finish(SAMPLE_TOOLS)
    visible = "".join(emitted) + res.trailing_text

    assert "<|im_end|>" not in visible
    assert "<|im_start|>" not in visible
    assert "Would you like me to continue" not in visible
    assert "[list_agents()]" not in visible
    assert "[subagent_fork" not in visible
    assert res.text == "Please wait..."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "web_search"
    assert res.tool_calls[0].arguments == {"query": "gta 6 XBox build leaked exe file PC port"}


def test_llama4_python_start_tag_live_buffering() -> None:
    parser = create_tool_parser("gguf_llama4")
    stream_text = 'Let me check that.<|python_start|>{"name": "web_search", "arguments": {"query": "llama 4 scout"}}<|python_end|><|eot|>extra garbage'
    streamed: list[str] = []
    for ch in stream_text:
        d = parser.feed(ch)
        if d.content:
            streamed.append(d.content)
    res = parser.finish(SAMPLE_TOOLS)
    assert "".join(streamed) + res.trailing_text == "Let me check that."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "web_search"
    assert res.tool_calls[0].arguments == {"query": "llama 4 scout"}


def test_gemma4_tool_call_tag_zero_raw_leakage() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "required": ["file_path"], "properties": {"file_path": {"type": "string"}}},
            },
        }
    ]
    parser = create_tool_parser("gguf_gemma4")
    raw = 'Reading file now.<|tool_call|>call:read{file_path:<|"|>README.md<|"|>}<|tool_call|><end_of_turn>Simulated turn'
    chunks_out = [parser.feed(raw[i : i + 5]).content for i in range(0, len(raw), 5)]
    res = parser.finish(tools)
    visible = "".join(chunks_out) + res.trailing_text
    assert "call:read" not in visible
    assert "Simulated turn" not in visible
    assert res.text == "Reading file now."
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "read"
    assert res.tool_calls[0].arguments == {"file_path": "README.md"}


def test_schema_hygiene_timeout_clamping_bash_description_and_strip_unadvertised() -> None:
    parser = create_tool_parser("gguf_llama3")
    parser.feed(
        '<tool_call>{"name": "bash", "arguments": {"command": "ls -la", "timeout": 15, "run_in_background": true, "sandbox_permissions": "danger-full-access"}}</tool_call>'
        '<tool_call>{"name": "wait_agent", "arguments": {"ids": ["a1"], "timeout_ms": 30}}</tool_call>'
    )
    res = parser.finish(SAMPLE_TOOLS)
    assert res.errors == []
    assert len(res.tool_calls) == 2
    bash_call = res.tool_calls[0]
    assert bash_call.arguments["command"] == "ls -la"
    assert bash_call.arguments["description"] == "Run: ls -la"
    assert bash_call.arguments["timeoutMs"] == 15000
    assert "run_in_background" not in bash_call.arguments
    assert "sandbox_permissions" not in bash_call.arguments

    wait_call = res.tool_calls[1]
    assert wait_call.arguments["timeout_ms"] == 30000


def test_post_tool_runtime_notice_folded_without_fake_incomplete_assistant_msg() -> None:
    """Verifies Root Cause #9 & Qwen `read_image` chain: post-tool runtime notices NEVER inject 'The previous tool call did not complete.'"""
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Who is in this photo?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_image", "arguments": '{"path":"a.png"}'}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Loaded a.png"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {
            "role": "user",
            "content": 'Current runtime context. This snapshot supersedes earlier runtime-context snapshots.\n\nCurrent working directory: "/workspace".',
        },
        {
            "role": "system",
            "content": "Added: team_task_create, team_task_list\nYou are repeating the exact same tool call with identical arguments.",
        },
    ]
    out = normalize_openai_messages(messages, allow_images=False)
    roles = [m["role"] for m in out]
    # System only at index 0; no fake assistant message after `tool`!
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = out[3]
    assert tool_msg["name"] == "read_image"
    assert "nexus_generate_image" not in tool_msg["content"]
    assert "did not complete" not in str(out)
    assert "Added: team_task_create" not in tool_msg["content"]
    assert "repeating the exact same tool call" in tool_msg["content"]


def test_text_only_user_image_note_forbids_nexus_generate_image() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Who is this person?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    out = normalize_openai_messages(msgs, allow_images=False)
    content = out[0]["content"]
    assert "text-only" in content
    assert "NEVER call `nexus_generate_image` or `read_image` to inspect" in content


def test_capability_profile_llama_templates_and_q3_sampling() -> None:
    _, has_compulsion, style = _classify_template(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nTools: {{ tools }}<|eot_id|>"
    )
    assert has_compulsion is True
    assert style == "llama3"

    hdr = GgufHeaderInfo(
        architecture="llama",
        quantization="Q3_K_M",
        context_length=16384,
        template_has_tools=True,
        template_has_ipython_compulsion=True,
        template_style="llama3",
    )
    lang = SelectedLanguageSpec(
        id="Llama-3.1-70B-Abliterated-Q3",
        display_name="Llama 3.1 70B Q3",
        model_path="/tmp/nonexistent-llama3",
        implementation="gguf_llama3",
        architecture="llama",
        modalities=["text"],
        context_window=16384,
        quantization="Q3_K_M",
    )
    img = SelectedImageSpec(
        id="LUSTIFY-v2.0",
        display_name="LUSTIFY v2.0",
        model_path="/tmp/nonexistent-img",
        implementation="sdxl",
        architecture="StableDiffusionXLPipeline",
    )
    prof = build_model_capability_profile(lang, img, gguf_header=hdr)
    assert prof.vision_input_enabled is False
    assert prof.image_generation_enabled is True
    assert prof.template_override_name == "llama3_1_nexus.jinja"
    tpl_path = resolve_curated_template_path(prof.template_override_name)
    assert tpl_path is not None and tpl_path.is_file()
    assert "Environment: ipython" not in tpl_path.read_text(encoding="utf-8")

    llama4_tpl = resolve_curated_template_path("llama4_scout_nexus.jinja")
    assert llama4_tpl is not None and llama4_tpl.is_file()
    assert "<|python_start|>" in llama4_tpl.read_text(encoding="utf-8")

    sampling = resolve_sampling_params(prof, has_tools=True, requested_temperature=0.7, requested_top_p=0.95)
    assert sampling["temperature"] == 0.2
    assert sampling["top_p"] == 0.85
    assert "<|im_end|>" in sampling["stop"]
    assert "<|eot_id|>" in sampling["stop"]

    grounded = apply_capability_grounding([{"role": "system", "content": "Base"}], prof)
    assert "DISABLED (text-only checkpoint" in grounded[0]["content"]
    assert "blind image synthesizer" in grounded[0]["content"]

    compacted = compact_tools_for_context(SAMPLE_TOOLS, max_desc_chars=120)
    assert len(compacted[0]["function"]["description"]) <= 125
    usage = estimate_openai_usage(grounded, compacted, "Hello world!")
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0


def test_gguf_fallback_and_prompt_tools_preserve_tool_history_and_images(tmp_path: Path) -> None:
    rt = LlamaServerRuntimeBase(model_id="test", model_path=tmp_path)
    history = [
        {"role": "user", "content": "Search for news"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"news"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "Top headline"},
    ]
    # Primary dict deserialization
    prep_dict = rt._prepare_messages(history, deserialize_args=True)
    assert prep_dict[1]["tool_calls"][0]["function"]["arguments"] == {"query": "news"}

    # Fallback prompt-injected tools preserves tool_calls and tool result
    prep_fb = rt._prepare_messages(history, inject_tools=SAMPLE_TOOLS[:1], deserialize_args=True)
    assert "<tool_call>" in prep_fb[2]["content"]
    assert "[Tool Result: web_search]" in prep_fb[3]["content"]

    # Prompt tools without strict_alternation preserves tool_calls too
    pt_msgs = inject_tools_into_messages(history, SAMPLE_TOOLS[:1], strict_alternation=False)
    assert any("<tool_call>" in m["content"] for m in pt_msgs if m["role"] == "assistant")
    assert any("[Tool Result: web_search]" in m["content"] for m in pt_msgs if m["role"] == "user")

    # VLM prompt tools preserves {"type": "image"} blocks
    vlm_in = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
    vlm_out = inject_tools_preserving_images(vlm_in, SAMPLE_TOOLS[:1], strict_alternation=True)
    assert any(b.get("type") == "image" for b in vlm_out[0]["content"])


def test_workspace_generated_images_excludes_json_sidecar(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    ws_dir = drive_root / "workspaces" / "projects" / "default-workspace"
    ws_dir.mkdir(parents=True)
    png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode("ascii")
    saved = save_generated_image_artifact(
        output_dir=drive_root / "outputs" / "generated_images",
        b64_png=png_b64,
        model_id="LUSTIFY-v2.0",
        prompt="studio portrait",
        quality="detailed",
        width=1024,
        height=1024,
        steps=30,
        seed=7,
        operation_id="op_ws",
        workspace=str(ws_dir),
        drive_root=drive_root,
    )
    assert Path(saved["drive_path"]).is_file()
    assert Path(saved["drive_path"] + ".json").is_file()
    assert Path(saved["workspace_path"]).is_file()
    assert not Path(saved["workspace_path"] + ".json").exists()
