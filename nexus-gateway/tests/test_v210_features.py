"""Tests for Nexus v2.1.0: GGUF split-shard inspection, MemoryPlanner, tool parsers, and co-resident worker."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GW_SRC = _REPO_ROOT / "nexus-gateway" / "src"
_COLAB_DIR = _REPO_ROOT / "colab"
for p in (str(_GW_SRC), str(_COLAB_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model_inventory  # noqa: E402
import staging  # noqa: E402
from adapters.tool_calls import (  # noqa: E402
    FencedJsonToolStreamParser,
    GptOssHarmonyStreamParser,
    Qwen35XmlToolStreamParser,
    create_tool_parser,
)
from api.app import GatewayConfig, create_app  # noqa: E402
from core.memory_planner import GIB, can_coreside_image_model, plan_gguf_memory  # noqa: E402
from models import gguf_metadata  # noqa: E402
from models.base import SelectedImageSpec, SelectedLanguageSpec, SessionRuntimeConfig  # noqa: E402
from models.language.prompt_tools import build_tools_instruction  # noqa: E402
from workers.model_worker import ModelWorkerState  # noqa: E402


def _make_min_gguf(path: Path, *, arch: str = "llama", ftype: int = 12, size: int = 1024) -> None:
    """Write a minimal valid GGUF v3 header padded to `size` bytes."""
    def _str_bytes(s: str) -> bytes:
        raw = s.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kv = bytearray()
    # 1. general.architecture (STRING = 8)
    kv.extend(_str_bytes("general.architecture"))
    kv.extend(struct.pack("<I", 8))
    kv.extend(_str_bytes(arch))
    # 2. general.file_type (UINT32 = 4)
    kv.extend(_str_bytes("general.file_type"))
    kv.extend(struct.pack("<II", 4, ftype))
    # 3. <arch>.block_count (UINT32 = 4)
    kv.extend(_str_bytes(f"{arch}.block_count"))
    kv.extend(struct.pack("<II", 4, 80))

    header = struct.pack("<4sIQQ", b"GGUF", 3, 0, 3) + bytes(kv)
    padded = header + b"\x00" * max(0, size - len(header))
    path.write_bytes(padded)


def test_gguf_metadata_and_split_parts_staging(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gguf_metadata, "MIN_VALID_GGUF_BYTES", 512)

    # 1. 0-byte GGUF (like interrupted Llama-3.1-70B download) is rejected cleanly.
    zero_dir = tmp_path / "models" / "Llama-3.1-70B-Abliterated-Q3"
    zero_dir.mkdir(parents=True)
    (zero_dir / "Llama-3.1-70B-Instruct-abliterated.Q3_K_M.gguf").write_bytes(b"")
    zero_cand = model_inventory.inspect_single_drive_dir(zero_dir, total_vram_bytes=80 * GIB)
    assert zero_cand.role == "excluded"
    assert "0 B" in zero_cand.reason

    # 2. Raw split `.gguf.part1of2` + `.part2of2` (Llama-4-Scout layout) + mmproj
    scout_dir = tmp_path / "models" / "Llama-4-Scout-Abliterated-Q3"
    scout_dir.mkdir(parents=True)
    full_buf = bytearray()
    tmp_gguf = tmp_path / "full.gguf"
    _make_min_gguf(tmp_gguf, arch="llama4", ftype=12, size=1024)
    full_buf.extend(tmp_gguf.read_bytes())
    (scout_dir / "Llama-4-Scout.Q3_K_M.gguf.part1of2").write_bytes(bytes(full_buf[:512]))
    (scout_dir / "Llama-4-Scout.Q3_K_M.gguf.part2of2").write_bytes(bytes(full_buf[512:]))
    (scout_dir / "mmproj-Llama-4-Scout-f16.gguf").write_bytes(b"GGUF" + b"\x00" * 128)

    cand = model_inventory.inspect_single_drive_dir(scout_dir, total_vram_bytes=80 * GIB)
    assert cand.role == "language"
    assert cand.implementation == "gguf_llama4"
    assert cand.quantization == "Q3_K_M"
    assert cand.shard_count == 2
    assert cand.modalities == ["text", "image"]

    # Stage & reassemble onto local SSD, then verify warm-cache reuse.
    stage_root = tmp_path / "staged"
    staged_dir = staging.stage_model_directory(
        scout_dir,
        stage_root,
        model_id="Llama-4-Scout-Abliterated-Q3",
        min_free_margin_bytes=0,
    )
    assembled = staged_dir / "Llama-4-Scout.Q3_K_M.gguf"
    assert assembled.is_file()
    assert assembled.read_bytes() == bytes(full_buf)
    assert (staged_dir / "mmproj-Llama-4-Scout-f16.gguf").is_file()


def test_memory_planner_tiers_and_coresidency() -> None:
    # A100 80GB + Qwen3.5-35B-A3B (19.7 GiB) + LUSTIFY-v2.0 (6.5 GiB) -> RECOMMENDED & co-resident!
    plan_a100 = plan_gguf_memory(
        weight_bytes=int(19.7 * GIB),
        architecture="qwen35moe",
        block_count=40,
        total_vram_bytes=80 * GIB,
        available_ram_bytes=120 * GIB,
        image_min_vram_gib=7.5,
    )
    assert plan_a100.tier == "RECOMMENDED"
    assert plan_a100.n_gpu_layers == 999
    assert plan_a100.n_ctx == 32768
    assert plan_a100.can_coreside_image is True

    # L4 24GB + Llama-3.1-70B (35 GiB) + 52 GiB RAM -> CPU_OFFLOAD_POSSIBLE
    plan_offload = plan_gguf_memory(
        weight_bytes=35 * GIB,
        architecture="llama",
        block_count=80,
        total_vram_bytes=24 * GIB,
        available_ram_bytes=52 * GIB,
    )
    assert plan_offload.tier == "CPU_OFFLOAD_POSSIBLE"
    assert 1 <= plan_offload.n_gpu_layers < 80
    assert plan_offload.can_coreside_image is False

    # Insufficient RAM + VRAM -> INSUFFICIENT_MEMORY
    plan_fail = plan_gguf_memory(
        weight_bytes=52 * GIB,
        architecture="llama4",
        block_count=48,
        total_vram_bytes=24 * GIB,
        available_ram_bytes=16 * GIB,
    )
    assert plan_fail.tier == "INSUFFICIENT_MEMORY"
    assert can_coreside_image_model(free_vram_bytes=40 * GIB, image_min_vram_gib=7.5) is True
    assert can_coreside_image_model(free_vram_bytes=8 * GIB, image_min_vram_gib=7.5) is False


def test_gpt_oss_harmony_live_streaming_and_tool_call() -> None:
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }]
    parser = GptOssHarmonyStreamParser()
    d1 = parser.feed("<|start|>assistant<|channel|>analysis<|message|>Let me check ")
    d2 = parser.feed("the directory.<|end|>")
    assert "Let me check " in d1.reasoning
    assert "the directory." in d2.reasoning

    d3 = parser.feed('<|start|>assistant<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{"command":"ls -la"}<|call|>')
    assert d3.content == ""
    res = parser.finish(tools)
    assert not res.errors
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "bash"
    assert res.tool_calls[0].arguments == {"command": "ls -la"}


def test_universal_gguf_and_compact_prompt_tool_parsers() -> None:
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }]
    # Llama 3.1 / 4 <|python_tag|> with "parameters"
    p_llama = create_tool_parser("gguf_llama4")
    p_llama.feed('<|python_tag|>{"name": "read_file", "parameters": {"path": "/tmp/a.txt"}}<|eom_id|>')
    res_llama = p_llama.finish(tools)
    assert not res_llama.errors
    assert len(res_llama.tool_calls) == 1
    assert res_llama.tool_calls[0].arguments == {"path": "/tmp/a.txt"}

    # Gemma 4 <|tool_call|>call:name{...}<|tool_call|>
    p_gemma = create_tool_parser("gguf_gemma4")
    p_gemma.feed('<|tool_call|>call:read_file{path:<|"|>/tmp/b.txt<|"|>}<|tool_call|>')
    res_gemma = p_gemma.finish(tools)
    assert not res_gemma.errors
    assert len(res_gemma.tool_calls) == 1
    assert res_gemma.tool_calls[0].arguments == {"path": "/tmp/b.txt"}

    # Compact prompt tool instruction is concise and parses <tool_call>
    instr = build_tools_instruction(tools)
    assert "read_file(path*: string)" in instr
    p_fenced = FencedJsonToolStreamParser()
    p_fenced.feed('<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/c.txt"}}</tool_call>')
    res_fenced = p_fenced.finish(tools)
    assert len(res_fenced.tool_calls) == 1


def test_coresident_worker_and_gateway_image_skip_swap(tmp_path: Path) -> None:
    class _FakeWorker:
        def __init__(self) -> None:
            self.active_lang: str | None = "Qwen3.5-35B-A3B-Uncensored-Q4"
            self.active_img: str | None = None
            self.unloaded_all = False

        def status(self):
            from workers.worker_protocol import WorkerStatus

            return WorkerStatus(
                pid=1234,
                alive=True,
                active_kind="language" if self.active_lang else ("image" if self.active_img else None),
                active_model_id=self.active_lang or self.active_img,
                active_language_id=self.active_lang,
                active_image_id=self.active_img,
                co_resident=bool(self.active_lang and self.active_img),
                device_total_vram_bytes=80 * GIB,
                device_free_vram_bytes=50 * GIB,
                device_used_vram_bytes=30 * GIB,
            )

        def load_model(self, req: Any) -> dict[str, Any]:
            if req.kind == "image":
                if not req.keep_other_slot:
                    self.active_lang = None
                self.active_img = req.model_id
            else:
                if not req.keep_other_slot:
                    self.active_img = None
                self.active_lang = req.model_id
            return {"status": "ready"}

        def generate_image(self, req: Any, on_progress=None) -> dict[str, Any]:
            import base64

            # 1x1 valid PNG
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
            return {
                "b64_png": base64.b64encode(png).decode("ascii"),
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "quality": req.quality,
            }

        def unload(self, *, kind: str = "all", terminate_on_failure: bool = True) -> dict[str, Any]:
            if kind in ("all", "language"):
                self.active_lang = None
            if kind in ("all", "image"):
                self.active_img = None
            if kind == "all":
                self.unloaded_all = True
            return {"status": "unloaded"}

    lang_dir = tmp_path / "lang"
    img_dir = tmp_path / "img"
    lang_dir.mkdir()
    img_dir.mkdir()
    cfg = GatewayConfig(
        api_key="test-key",
        session=SessionRuntimeConfig(
            language_model=SelectedLanguageSpec(
                id="Qwen3.5-35B-A3B-Uncensored-Q4",
                display_name="Qwen3.5-35B-A3B-Uncensored-Q4",
                model_path=str(lang_dir),
                implementation="gguf_qwen3_5",
                architecture="qwen35moe",
                can_coreside_image=True,
            ),
            image_model=SelectedImageSpec(
                id="LUSTIFY-v2.0",
                display_name="LUSTIFY-v2.0",
                model_path=str(img_dir),
                implementation="sdxl",
                architecture="StableDiffusionXLPipeline",
                min_vram_gib=7.5,
            ),
        ),
        output_image_dir=tmp_path / "out_images",
    )
    fake_worker = _FakeWorker()
    app = create_app(cfg, worker=fake_worker)  # type: ignore[arg-type]
    client = TestClient(app)
    res = client.post(
        "/v1/images/generations",
        json={"prompt": "sunset", "quality": "normal"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert res.status_code == 200
    assert fake_worker.unloaded_all is False
    assert fake_worker.active_lang == "Qwen3.5-35B-A3B-Uncensored-Q4"
    assert fake_worker.active_img == "LUSTIFY-v2.0"
