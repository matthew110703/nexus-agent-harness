"""Regression tests for Nexus v1.0.1 runtime load, deduplication, progress, and crash reconciliation."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi.testclient import TestClient

GATEWAY_SRC = Path(__file__).resolve().parents[1] / "src"
if str(GATEWAY_SRC) not in sys.path:
    sys.path.insert(0, str(GATEWAY_SRC))

from api.app import GatewayConfig, create_app
from core.worker_manager import WorkerError
from workers.worker_protocol import WorkerLoadRequest, WorkerStatus


class FakeAsyncWorker:
    """Thread-safe fake worker with controllable load delay, failure, and crash simulation."""

    def __init__(self, *, load_delay_s: float = 0.05) -> None:
        self.load_delay_s = load_delay_s
        self.load_calls: list[WorkerLoadRequest] = []
        self.unload_calls: int = 0
        self.alive: bool = False
        self.active_kind: Optional[str] = None
        self.active_model_id: Optional[str] = None
        self.fail_next_load: Optional[str] = None
        self._lock = threading.Lock()
        self.load_started = threading.Event()
        self.allow_load_finish = threading.Event()
        self.allow_load_finish.set()

    def status(self) -> WorkerStatus:
        with self._lock:
            return WorkerStatus(
                pid=4242 if self.alive else 0,
                alive=self.alive,
                active_kind=self.active_kind,  # type: ignore[arg-type]
                active_model_id=self.active_model_id,
                cuda_allocated_bytes=4 * 1024 * 1024 * 1024 if self.active_model_id else 0,
                cuda_reserved_bytes=5 * 1024 * 1024 * 1024 if self.active_model_id else 0,
            )

    def load_model(self, req: WorkerLoadRequest) -> dict[str, Any]:
        self.load_calls.append(req)
        self.load_started.set()
        self.allow_load_finish.wait(timeout=5.0)
        if self.load_delay_s > 0:
            time.sleep(self.load_delay_s)
        if self.fail_next_load:
            msg = self.fail_next_load
            self.fail_next_load = None
            raise WorkerError(msg, status_code=503)
        with self._lock:
            self.alive = True
            self.active_kind = req.kind
            self.active_model_id = req.model_id
        return {"status": "loaded", "model_id": req.model_id}

    def unload(self, *, terminate_on_failure: bool = True) -> dict[str, Any]:
        self.unload_calls += 1
        with self._lock:
            self.active_kind = None
            self.active_model_id = None
        return {"status": "unloaded"}

    def terminate(self) -> None:
        with self._lock:
            self.alive = False
            self.active_kind = None
            self.active_model_id = None

    def generate(self, req: Any) -> dict[str, Any]:
        return {"text": f"reply from {req.model_id}"}

    def simulate_crash(self) -> None:
        with self._lock:
            self.alive = False
            self.active_kind = None
            self.active_model_id = None


def _make_chat_model(models_root: Path, name: str, *, include_weights: bool = True) -> None:
    d = models_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "vision_config": {"hidden_size": 1280},
            }
        ),
        encoding="utf-8",
    )
    (d / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% if enable_thinking %}think{% endif %}"}),
        encoding="utf-8",
    )
    if include_weights:
        (d / "model.safetensors").write_bytes(b"fake-weights-1234")


def _wait_terminal(client: TestClient, headers: dict[str, str], op_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        res = client.get(f"/v1/progress/{op_id}", headers=headers)
        assert res.status_code == 200
        last = res.json()
        if last["status"] in ("completed", "failed", "cancelled"):
            return last
        time.sleep(0.02)
    raise AssertionError(f"Operation {op_id} did not reach terminal state: {last}")


def test_runtime_load_validation_async_dedup_switch_crash_and_chat(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    _make_chat_model(models_root, "Qwen-3-VL-8B-Instruct-heretic")
    _make_chat_model(models_root, "Qwen3-VL-4b-Heretic")
    _make_chat_model(models_root, "Broken-No-Weights", include_weights=False)

    worker = FakeAsyncWorker(load_delay_s=0.02)
    token = "test-secret-token"
    app = create_app(
        GatewayConfig(
            api_key=token,
            models_root=models_root,
            local_stage_root=tmp_path / "stage",
            output_image_dir=tmp_path / "out",
        ),
        worker=worker,  # type: ignore[arg-type]
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Missing model -> 400 before touching worker.
    r_missing = client.post("/v1/runtime/load", json={}, headers=headers)
    assert r_missing.status_code == 400
    assert "model is required" in r_missing.json()["detail"]
    assert len(worker.load_calls) == 0

    # 2. Unknown model -> 404 before touching worker.
    r_unknown = client.post("/v1/runtime/load", json={"model": "nonexistent"}, headers=headers)
    assert r_unknown.status_code == 404
    assert len(worker.load_calls) == 0

    # 3. Incomplete weights -> 422 before touching worker.
    r_broken = client.post("/v1/runtime/load", json={"model": "Broken-No-Weights"}, headers=headers)
    assert r_broken.status_code == 422
    assert len(worker.load_calls) == 0

    # 4. Async load + duplicate click deduplication.
    worker.allow_load_finish.clear()
    r1 = client.post(
        "/v1/runtime/load",
        json={"model": "Qwen-3-VL-8B-Instruct-heretic", "kind": "chat", "async": True},
        headers=headers,
    )
    assert r1.status_code == 202
    op_id = r1.json()["operation_id"]
    assert isinstance(op_id, str) and op_id.startswith("op_")

    assert worker.load_started.wait(timeout=3.0)
    # Second click while in-flight must reuse exact same operation_id and NOT spawn a duplicate load.
    r2 = client.post(
        "/v1/runtime/load",
        json={"model": "Qwen-3-VL-8B-Instruct-heretic", "kind": "chat", "async": True},
        headers=headers,
    )
    assert r2.status_code == 202
    assert r2.json()["operation_id"] == op_id

    # Progress while loading weights is indeterminate (not fabricated percentage).
    in_flight = client.get(f"/v1/progress/{op_id}", headers=headers).json()
    assert in_flight["status"] == "running"
    assert in_flight["indeterminate"] is True

    worker.allow_load_finish.set()
    terminal = _wait_terminal(client, headers, op_id)
    assert terminal["status"] == "completed"
    assert terminal["phase"] == "ready"
    assert len(worker.load_calls) == 1
    assert worker.load_calls[0].adapter_family == "qwen_vl"

    # Runtime confirms worker alive & active model matches.
    rt = client.get("/v1/runtime", headers=headers).json()
    assert rt["worker"]["alive"] is True
    assert rt["worker"]["active_model_id"] == "Qwen-3-VL-8B-Instruct-heretic"
    assert rt["active_model"] == "Qwen-3-VL-8B-Instruct-heretic"

    # 5. Ordinary chat works with NO image model selected.
    chat_res = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen-3-VL-8B-Instruct-heretic",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=headers,
    )
    assert chat_res.status_code == 200
    assert "reply from Qwen-3-VL-8B-Instruct-heretic" in chat_res.json()["choices"][0]["message"]["content"]

    # 6. Switching to another model serializes unload + load.
    r_switch = client.post(
        "/v1/runtime/load",
        json={"model": "Qwen3-VL-4b-Heretic", "kind": "chat", "async": True},
        headers=headers,
    )
    assert r_switch.status_code == 202
    switch_op = r_switch.json()["operation_id"]
    switch_terminal = _wait_terminal(client, headers, switch_op)
    assert switch_terminal["status"] == "completed"
    assert worker.unload_calls >= 1
    assert client.get("/v1/runtime", headers=headers).json()["worker"]["active_model_id"] == "Qwen3-VL-4b-Heretic"

    # 7. Worker crash invalidates stale ResourceManager Ready state.
    worker.simulate_crash()
    rt_after_crash = client.get("/v1/runtime", headers=headers).json()
    assert rt_after_crash["worker"]["alive"] is False
    assert rt_after_crash["active_model"] is None

    # 8. Failed async load records structured terminal error.
    worker.fail_next_load = "GPU memory is insufficient (CUDA OOM)"
    r_fail = client.post(
        "/v1/runtime/load",
        json={"model": "Qwen-3-VL-8B-Instruct-heretic", "kind": "chat", "async": True},
        headers=headers,
    )
    assert r_fail.status_code == 202
    fail_terminal = _wait_terminal(client, headers, r_fail.json()["operation_id"])
    assert fail_terminal["status"] == "failed"
    assert "CUDA OOM" in (fail_terminal.get("error") or "")
