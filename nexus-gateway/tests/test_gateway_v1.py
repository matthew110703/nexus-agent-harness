"""Comprehensive v1.0.0 tests for Nexus Gateway API, worker lifecycle, Effort, staging, and Drive sync."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient
import pytest

from adapters.diffusers import resolve_image_generation_params
from api.app import GatewayConfig, create_app
from core.hardware_detector import classify_hardware_tier, detect_hardware
from core.model_registry import ModelRegistry
from core.model_staging import InsufficientDiskSpaceError, MANIFEST_FILENAME, stage_model_directory
from services.conversation import resolve_reasoning_controls
from services.progress import ProgressService
from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus

# Also import colab/sync_source.py for Drive layout & source-sync tests.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "colab") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "colab"))
import sync_source  # noqa: E402


class FakeWorkerManager:
    """In-process deterministic worker stub for API and streaming tests."""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []
        self.unloaded = 0
        self.terminated = 0
        self.active_kind: str | None = None
        self.active_model_id: str | None = None
        self.next_text = 'Hello <tool_call>{"name":"nexus_generate_image","arguments":{"prompt":"sunset"}}</tool_call>'
        self.last_gen_req: WorkerGenerateRequest | None = None

    def status(self) -> WorkerStatus:
        return WorkerStatus(
            pid=4242,
            alive=True,
            active_kind=self.active_kind,  # type: ignore[arg-type]
            active_model_id=self.active_model_id,
        )

    def load_model(self, req: WorkerLoadRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        self.loaded.append((req.kind, req.model_id))
        self.active_kind = req.kind
        self.active_model_id = req.model_id
        return {"status": "ready"}

    def generate(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> dict[str, Any]:
        self.last_gen_req = req
        return {"text": self.next_text, "model_id": req.model_id}

    def generate_stream(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> Iterator[str]:
        self.last_gen_req = req
        # Split across tag boundaries intentionally to exercise incremental parsing.
        for i in range(0, len(self.next_text), 7):
            yield self.next_text[i : i + 7]

    def generate_image(self, req: WorkerImageRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        png = base64.b64encode(b"\x89PNG\r\n\x1a\nFAKE").decode("ascii")
        return {
            "model": req.model_id,
            "quality": req.quality,
            "width": req.width or 1024,
            "height": req.height or 1024,
            "steps": req.steps or 6,
            "seed": req.seed,
            "b64_png": png,
        }

    def unload(self, *, terminate_on_failure: bool = True) -> dict[str, Any]:
        self.unloaded += 1
        self.active_kind = None
        self.active_model_id = None
        return {"status": "unloaded"}

    def terminate(self) -> None:
        self.terminated += 1
        self.active_kind = None
        self.active_model_id = None


def _build_fixture_env(tmp_path: Path) -> tuple[TestClient, FakeWorkerManager, str, Path]:
    models_root = tmp_path / "Nexus" / "models"
    qwen = models_root / "Qwen3-VL-8B-Instruct"
    qwen.mkdir(parents=True)
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "max_position_embeddings": 65536}))
    (qwen / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% if tools %}<tool_call>{% endif %}{% if enable_thinking %}<think>{% endif %}"})
    )

    flux = models_root / "FLUX.2-klein-9B"
    flux.mkdir(parents=True)
    (flux / "model_index.json").write_text(json.dumps({"_class_name": "Flux2KleinPipeline"}))

    sidecar = tmp_path / "Nexus" / "colab-src" / "config" / "model-capabilities.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps({
            "models": {
                "Qwen3-VL-8B-Instruct": {
                    "tool_behavior_verified": True,
                    "reasoning": {
                        "status": "supported",
                        "mode": "binary_toggle",
                        "levels": {"off": None, "high": "high"},
                        "default": "high",
                        "source": "verified_sidecar",
                    },
                }
            }
        })
    )

    token = "test-secret-token"
    worker = FakeWorkerManager()
    out_dir = tmp_path / "Nexus" / "outputs" / "generated_images"
    cfg = GatewayConfig(
        api_key=token,
        models_root=models_root,
        capabilities_sidecar=sidecar,
        local_stage_root=tmp_path / "staged",
        output_image_dir=out_dir,
        plugin_index_path=_REPO_ROOT / "colab" / "plugin-index.json",
        min_stage_free_margin_bytes=0,
    )
    app = create_app(cfg, worker=worker)  # type: ignore[arg-type]
    return TestClient(app), worker, token, out_dir


def test_gateway_auth_models_reasoning_and_plugin_index(tmp_path: Path) -> None:
    client, _worker, token, _out = _build_fixture_env(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401

    headers = {"Authorization": f"Bearer {token}"}
    res = client.get("/v1/models", headers=headers)
    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "Qwen3-VL-8B-Instruct"
    assert data[0]["reasoning_efforts"] == {"off": None, "high": "high"}
    assert data[0]["reasoning"]["confidence"] == "verified"

    img_res = client.get("/v1/image-models", headers=headers)
    assert img_res.status_code == 200
    assert img_res.json()["quality_profiles"] == ["normal", "detailed", "maximum"]
    assert img_res.json()["data"][0]["id"] == "FLUX.2-klein-9B"

    idx = client.get("/v1/plugin-index.json")
    assert idx.status_code == 200
    assert idx.json()["format"] == 1
    assert len(idx.json()["plugins"]) >= 1


def test_chat_completions_streaming_and_non_streaming_tool_calls(tmp_path: Path) -> None:
    client, worker, token, _out = _build_fixture_env(tmp_path)
    headers = {"Authorization": f"Bearer {token}"}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "nexus_generate_image",
                "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
            },
        }
    ]

    # Non-streaming with binary thinking off
    resp = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "Qwen3-VL-8B-Instruct",
            "messages": [{"role": "user", "content": "draw sunset"}],
            "tools": tools,
            "reasoning_effort": "off",
            "stream": False,
        },
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "nexus_generate_image"
    assert worker.last_gen_req is not None
    assert worker.last_gen_req.enable_thinking is False

    # Streaming SSE across arbitrary chunk splits
    stream_resp = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "Qwen3-VL-8B-Instruct",
            "messages": [{"role": "user", "content": "draw sunset"}],
            "tools": tools,
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    body = stream_resp.text
    assert '"nexus_generate_image"' in body
    assert "data: [DONE]" in body


def test_image_generation_quality_profiles_and_drive_artifact_metadata(tmp_path: Path) -> None:
    client, _worker, token, out_dir = _build_fixture_env(tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    bad = client.post(
        "/v1/images/generations",
        headers=headers,
        json={"prompt": "beach", "quality": "ultra-fake"},
    )
    assert bad.status_code == 400

    ok = client.post(
        "/v1/images/generations",
        headers=headers,
        json={"prompt": "beach at twilight", "model": "FLUX.2-klein-9B", "quality": "maximum"},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["quality"] == "maximum"
    assert payload["steps"] == 8  # Distilled FLUX.2-klein maximum profile
    assert Path(payload["master_path"]).is_file()
    meta = json.loads(Path(payload["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["prompt"] == "beach at twilight"
    assert meta["quality"] == "maximum"
    assert out_dir in Path(payload["master_path"]).parents

    prog = client.get(f"/v1/progress/{payload['operation_id']}", headers=headers)
    assert prog.status_code == 200
    assert prog.json()["status"] == "completed"


def test_hardware_tiers_and_image_param_validation() -> None:
    assert classify_hardware_tier(False, 0) == "cpu_only"
    assert classify_hardware_tier(True, 24 * 1024 ** 3) == "l4_baseline"
    assert classify_hardware_tier(True, 80 * 1024 ** 3) == "high_vram_datacenter"
    snap = detect_hardware()
    assert snap.cpu_count >= 1

    with pytest.raises(ValueError):
        resolve_image_generation_params(model_id="FLUX.2-klein-9B", architecture="Flux2KleinPipeline", num_images=9)


def test_staging_manifest_integrity_and_invalidation(tmp_path: Path) -> None:
    src = tmp_path / "drive" / "ModelA"
    src.mkdir(parents=True)
    (src / "config.json").write_text('{"v": 1}')
    cache = tmp_path / "cache"
    prog = ProgressService()

    staged = stage_model_directory(src, cache, model_id="ModelA", progress=prog, min_free_margin_bytes=0)
    assert (staged / MANIFEST_FILENAME).is_file()

    # Corrupt staged file -> must re-stage rather than blindly returning corrupted cache
    (staged / "config.json").write_text('{"v": 999}')
    stage_model_directory(src, cache, model_id="ModelA", progress=prog, min_free_margin_bytes=0)
    assert (staged / "config.json").read_text() == '{"v": 1}'

    # Disk space rejection when fallback is disabled
    with pytest.raises(InsufficientDiskSpaceError):
        stage_model_directory(
            src,
            tmp_path / "cache2",
            model_id="ModelA",
            progress=prog,
            min_free_margin_bytes=10 ** 18,
            allow_drive_fallback=False,
        )


def test_drive_layout_and_idempotent_source_sync_with_conflict_and_rollback(tmp_path: Path) -> None:
    drive_root = tmp_path / "MyDrive" / "Nexus"
    # Existing user assets must never be touched.
    existing_model = drive_root / "models" / "Qwen3-VL-8B-Instruct"
    existing_model.mkdir(parents=True)
    (existing_model / "weights.safetensors").write_bytes(b"abc")

    layout = sync_source.init_drive_layout(drive_root)
    assert "colab-src/nexus-gateway" in layout["created"]
    assert (existing_model / "weights.safetensors").read_bytes() == b"abc"

    runtime_root = tmp_path / "runtime"
    (runtime_root / "nexus-gateway" / "src").mkdir(parents=True)
    (runtime_root / "nexus-gateway" / "src" / "hello.py").write_text("X = 1\n")

    drive_src = drive_root / "colab-src"
    manifest = drive_src / "sync-manifest.json"
    sync_source.seed_drive_baseline(
        runtime_root,
        drive_src,
        manifest,
        git_commit="abc123",
        rel_pairs=[("nexus-gateway/src/hello.py", "nexus-gateway/src/hello.py")],
    )

    # Idempotent no-op when unchanged
    noop = sync_source.apply_sync(drive_src, runtime_root, manifest, git_commit="abc123")
    assert noop.applied == []
    assert noop.restart_targets == []

    # Edit in Drive -> detected & applied with gateway_api restart target
    (drive_src / "nexus-gateway" / "src" / "hello.py").write_text("X = 2\n")
    applied = sync_source.apply_sync(drive_src, runtime_root, manifest, git_commit="abc123")
    assert applied.applied == ["nexus-gateway/src/hello.py"]
    assert "gateway_api" in applied.restart_targets
    assert (runtime_root / "nexus-gateway" / "src" / "hello.py").read_text() == "X = 2\n"

    # Syntax error in Drive -> rejected before modifying runtime
    (drive_src / "nexus-gateway" / "src" / "hello.py").write_text("def broken(:\n")
    bad = sync_source.apply_sync(drive_src, runtime_root, manifest, git_commit="abc123")
    assert bad.errors
    assert (runtime_root / "nexus-gateway" / "src" / "hello.py").read_text() == "X = 2\n"

    # Conflict detection: both Drive and runtime modified since last sync
    (drive_src / "nexus-gateway" / "src" / "hello.py").write_text("X = 3\n")
    (runtime_root / "nexus-gateway" / "src" / "hello.py").write_text("X = 99\n")
    conflict = sync_source.apply_sync(drive_src, runtime_root, manifest, git_commit="def456")
    assert conflict.conflicts == ["nexus-gateway/src/hello.py"]
    assert conflict.errors
    assert (runtime_root / "nexus-gateway" / "src" / "hello.py").read_text() == "X = 99\n"
