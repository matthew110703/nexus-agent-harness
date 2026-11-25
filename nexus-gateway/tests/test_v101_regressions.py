"""Stages 2, 3, and 4 regression tests for Nexus v1.0.1."""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from fastapi.testclient import TestClient
import pytest

from adapters.diffusers import (
    DiffusersImageAdapter,
    GenerationCancelledError,
    build_pipeline_call_kwargs,
    resolve_image_generation_params,
)
from adapters.qwen_vl import decode_vlm_image_url
from adapters.tool_calls import create_tool_parser, has_native_tool_parser, validate_tool_arguments
from adapters.transformers import select_torch_dtype
from api.app import GatewayConfig, create_app
from core.model_registry import ModelRegistry, inspect_model_directory
from core.model_staging import InsufficientDiskSpaceError, stage_model_directory
from services.progress import ProgressService
from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus


def _create_valid_hf_qwen(models_root: Path, name: str = "Qwen3-VL-8B-Instruct") -> Path:
    d = models_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}))
    (d / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% if tools %}<tool_call>{% endif %}{% if enable_thinking %}<think>{% endif %}"})
    )
    (d / "model.safetensors").write_bytes(b"SAFETENSORS_WEIGHTS")
    return d


def _create_valid_flux(models_root: Path, name: str = "FLUX.2-klein-9B") -> Path:
    d = models_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_index.json").write_text(
        json.dumps({
            "_class_name": "Flux2KleinPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "safety_checker": [None, None],  # Optional/null component must NOT reject pipeline!
        })
    )
    (d / "transformer").mkdir()
    (d / "transformer" / "config.json").write_text("{}")
    (d / "vae").mkdir()
    (d / "vae" / "config.json").write_text("{}")
    return d


def test_stage2_weight_and_diffusers_validation_and_adapter_dispatch(tmp_path: Path) -> None:
    # Config-only directory MUST be rejected.
    cfg_only = tmp_path / "ConfigOnly"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert inspect_model_directory(cfg_only).validation_status == "unsupported"

    # Arbitrary *.pt file MUST be rejected.
    pt_only = tmp_path / "ArbitraryPt"
    pt_only.mkdir()
    (pt_only / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (pt_only / "random.pt").write_bytes(b"not-hf")
    assert inspect_model_directory(pt_only).validation_status == "unsupported"

    # Missing indexed shard MUST be rejected.
    sharded = tmp_path / "ShardedMissing"
    sharded.mkdir()
    (sharded / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    )
    (sharded / "model-00001-of-00002.safetensors").write_bytes(b"ok")
    assert inspect_model_directory(sharded).validation_status == "unsupported"

    # Qwen3.6 (no "vl" in name) with qwen3_5_moe model_type -> dispatches to qwen_vl via metadata!
    qwen36 = tmp_path / "Qwen3.6-35B-A3B"
    qwen36.mkdir()
    (qwen36 / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    (qwen36 / "model.safetensors").write_bytes(b"ok")
    info36 = inspect_model_directory(qwen36)
    assert info36.validation_status == "compatible"
    assert info36.adapter_family == "qwen_vl"

    # Valid Diffusers pipeline with optional null safety_checker is accepted.
    flux = _create_valid_flux(tmp_path)
    flux_info = inspect_model_directory(flux)
    assert flux_info.structurally_valid is True
    assert flux_info.adapter_family == "diffusers"

    # Diffusers pipeline missing required `vae` component is rejected.
    bad_flux = tmp_path / "BrokenFlux"
    bad_flux.mkdir()
    (bad_flux / "model_index.json").write_text(json.dumps({"_class_name": "Flux2Pipeline", "vae": ["diffusers", "AutoencoderKL"]}))
    assert inspect_model_directory(bad_flux).validation_status == "unsupported"


def test_stage2_cpu_dtype_and_vlm_image_validation() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        float32 = "fp32"
        float16 = "fp16"
        bfloat16 = "bf16"
        cuda = FakeCuda()

    # CPU must default to float32, NEVER float16.
    assert select_torch_dtype(FakeTorch()) == "fp32"

    # Remote HTTP URL must be rejected (no SSRF, no silent drop).
    with pytest.raises(ValueError, match="Unsupported image_url scheme"):
        decode_vlm_image_url("https://example.com/evil.png")
    with pytest.raises(ValueError, match="Invalid base64"):
        decode_vlm_image_url("data:image/png;base64,!!!not-base64!!!")


def test_stage2_staging_large_file_same_size_same_mtime_invalidation_and_no_silent_fallback(tmp_path: Path) -> None:
    src = tmp_path / "drive" / "BigModel"
    src.mkdir(parents=True)
    big = src / "model.safetensors"
    big.write_bytes(b"A" * (64 * 1024))
    st_before = big.stat()

    cache = tmp_path / "cache"
    prog = ProgressService()
    staged = stage_model_directory(src, cache, model_id="BigModel", progress=prog, min_free_margin_bytes=0)
    assert (staged / "model.safetensors").read_bytes() == b"A" * (64 * 1024)

    # Modify content with EXACT SAME size and EXACT SAME mtime!
    big.write_bytes(b"B" * (64 * 1024))
    os.utime(big, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))
    stage_model_directory(src, cache, model_id="BigModel", progress=prog, min_free_margin_bytes=0)
    assert (staged / "model.safetensors").read_bytes() == b"B" * (64 * 1024)

    # Default allow_drive_fallback=False must raise InsufficientDiskSpaceError.
    with pytest.raises(InsufficientDiskSpaceError):
        stage_model_directory(src, tmp_path / "cache2", model_id="BigModel", progress=prog, min_free_margin_bytes=10 ** 18)


def test_stage3_nested_tool_validation_parser_alignment_and_error_surfacing() -> None:
    assert has_native_tool_parser("qwen") is True
    assert has_native_tool_parser("generic_json") is False

    nested_schema = {
        "type": "object",
        "properties": {
            "options": {
                "type": "object",
                "properties": {"count": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["count"],
                "additionalProperties": False,
            }
        },
        "required": ["options"],
    }
    assert validate_tool_arguments({"options": {"count": 2, "tags": ["a"]}}, nested_schema) is None
    assert validate_tool_arguments({"options": {"tags": ["a"]}}, nested_schema) is not None
    assert validate_tool_arguments({"options": {"count": 2, "tags": [123]}}, nested_schema) is not None
    assert validate_tool_arguments({"options": {"count": 2, "extra": True}}, nested_schema) is not None

    # Trailing buffered text flush without duplication.
    p = create_tool_parser("qwen")
    emitted = p.feed("hello <tool")  # partial prefix buffered
    assert emitted == "hello "
    res = p.finish([])
    assert res.trailing_text == "<tool"
    assert res.text == "hello <tool"


class CrashableFakeWorker:
    def __init__(self) -> None:
        self.alive = True
        self.active_kind: Optional[str] = None
        self.active_model_id: Optional[str] = None
        self.load_calls: list[tuple[str, str, str]] = []
        self.next_text = "ok"

    def crash(self) -> None:
        self.alive = False
        self.active_kind = None
        self.active_model_id = None

    def status(self) -> WorkerStatus:
        return WorkerStatus(pid=99, alive=self.alive, active_kind=self.active_kind, active_model_id=self.active_model_id)  # type: ignore[arg-type]

    def load_model(self, req: WorkerLoadRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        self.alive = True
        self.active_kind = req.kind
        self.active_model_id = req.model_id
        self.load_calls.append((req.kind, req.model_id, req.adapter_family))
        return {"status": "ready"}

    def generate(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> dict[str, Any]:
        return {"text": self.next_text}

    def generate_stream(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> Iterator[str]:
        yield self.next_text

    def generate_image(
        self,
        req: WorkerImageRequest,
        *,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        if on_progress is not None:
            on_progress({"phase": "denoising", "step": 1, "steps_total": 4, "progress": 25.0, "indeterminate": False})
            on_progress({"phase": "denoising", "step": 4, "steps_total": 4, "progress": 100.0, "indeterminate": False})
        return {
            "model": req.model_id,
            "quality": req.quality,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": None,
            "b64_png": base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode("ascii"),
        }

    def unload(self, *, terminate_on_failure: bool = True) -> dict[str, Any]:
        self.active_kind = None
        self.active_model_id = None
        return {"status": "unloaded"}

    def cancel_operation(self, operation_id: Optional[str] = None) -> dict[str, Any]:
        return {"status": "cancelled", "operations": [operation_id] if operation_id else []}


def test_stage2_and_stage4_worker_crash_reconciliation_and_explicit_image_model_and_cancel(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _create_valid_hf_qwen(models_root)
    _create_valid_flux(models_root)
    worker = CrashableFakeWorker()
    token = "tok"
    cfg = GatewayConfig(
        api_key=token,
        models_root=models_root,
        local_stage_root=tmp_path / "staged",
        output_image_dir=tmp_path / "out",
        min_stage_free_margin_bytes=0,
    )
    client = TestClient(create_app(cfg, worker=worker))  # type: ignore[arg-type]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Load & run chat -> crash worker -> next request MUST automatically reload!
    r1 = client.post("/v1/chat/completions", headers=headers, json={"model": "Qwen3-VL-8B-Instruct", "messages": [{"role": "user", "content": "hi"}]})
    assert r1.status_code == 200
    assert len(worker.load_calls) == 1
    assert worker.load_calls[0][2] == "qwen_vl"

    worker.crash()
    r2 = client.post("/v1/chat/completions", headers=headers, json={"model": "Qwen3-VL-8B-Instruct", "messages": [{"role": "user", "content": "hi again"}]})
    assert r2.status_code == 200
    assert len(worker.load_calls) == 2

    # 2. Malformed tool call MUST surface HTTP 422 instead of silent success.
    worker.next_text = "<tool_call>{bad json</tool_call>"
    bad_tc = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "Qwen3-VL-8B-Instruct",
            "messages": [{"role": "user", "content": "call"}],
            "tools": [{"type": "function", "function": {"name": "demo", "parameters": {"type": "object"}}}],
        },
    )
    assert bad_tc.status_code == 422

    # 3. Image generation MUST require explicit `model` and enforce `num_images == 1`.
    missing_model = client.post("/v1/images/generations", headers=headers, json={"prompt": "beach"})
    assert missing_model.status_code == 400
    multi_img = client.post("/v1/images/generations", headers=headers, json={"prompt": "beach", "model": "FLUX.2-klein-9B", "num_images": 2})
    assert multi_img.status_code == 400

    # 4. Real Diffusers step progress + cancellation inside callback_on_step_end.
    class FakePipe:
        def __call__(self, prompt: str, width: int, height: int, num_inference_steps: int, guidance_scale: float = 3.5, callback_on_step_end=None):
            assert guidance_scale == 3.5
            for s in range(num_inference_steps):
                if callback_on_step_end:
                    callback_on_step_end(self, s, 0, {})
            return type("Out", (), {"images": []})()

    adapter = DiffusersImageAdapter(model_id="FLUX.2-klein-9B", model_path=models_root / "FLUX.2-klein-9B", architecture="Flux2KleinPipeline")
    adapter.pipe = FakePipe()
    cancel_ev = threading.Event()
    seen_steps: list[int] = []

    def on_prog(ev: dict[str, Any]) -> None:
        seen_steps.append(ev["step"])
        if ev["step"] == 1:
            cancel_ev.set()

    with pytest.raises(GenerationCancelledError):
        adapter.generate({"prompt": "beach", "steps": 4}, on_progress=on_prog, cancel_event=cancel_ev)
    assert seen_steps == [1]
    assert getattr(adapter.pipe, "_interrupt", False) is True
