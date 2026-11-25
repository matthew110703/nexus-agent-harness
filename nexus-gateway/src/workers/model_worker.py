"""Isolated GPU model worker subprocess owning CUDA context, weights, inference, and cancellation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adapters.diffusers import DiffusersImageAdapter, GenerationCancelledError  # noqa: E402
from adapters.qwen_vl import QwenVLChatAdapter  # noqa: E402
from adapters.transformers import TransformersChatAdapter  # noqa: E402
from api.middleware.auth import assert_loopback_host, make_auth_dependency  # noqa: E402


class ModelWorkerState:
    """Owns the single active CUDA model or diffusion pipeline inside this worker process."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_kind: Optional[str] = None
        self.active_model_id: Optional[str] = None
        self.adapter: Any = None
        self.last_error: Optional[str] = None
        self._cancel_lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}

    def register_cancel_event(self, operation_id: str) -> threading.Event:
        ev = threading.Event()
        with self._cancel_lock:
            self._cancel_events[operation_id] = ev
        return ev

    def clear_cancel_event(self, operation_id: str) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(operation_id, None)

    def cancel_operation(self, operation_id: Optional[str] = None) -> list[str]:
        cancelled: list[str] = []
        with self._cancel_lock:
            targets = [operation_id] if operation_id else list(self._cancel_events.keys())
            for op_id in targets:
                ev = self._cancel_events.get(op_id)
                if ev is not None:
                    ev.set()
                    cancelled.append(op_id)
        return cancelled

    def _cuda_stats(self) -> tuple[int, int]:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
        except Exception:
            pass
        return 0, 0

    def status(self) -> dict[str, Any]:
        alloc, reserved = self._cuda_stats()
        return {
            "pid": os.getpid(),
            "alive": True,
            "active_kind": self.active_kind,
            "active_model_id": self.active_model_id,
            "cuda_allocated_bytes": alloc,
            "cuda_reserved_bytes": reserved,
            "last_error": self.last_error,
        }

    def unload(self) -> None:
        with self.lock:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        adapter = self.adapter
        self.adapter = None
        self.active_kind = None
        self.active_model_id = None
        if adapter is not None:
            try:
                adapter.unload()
            except Exception:
                pass
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def load(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload.get("kind")
        model_id = str(payload.get("model_id") or "")
        model_path = Path(str(payload.get("model_path") or ""))
        architecture = str(payload.get("architecture") or "")
        adapter_family = str(payload.get("adapter_family") or "")
        if kind not in ("chat", "diffusion") or not model_id or not model_path.exists():
            raise HTTPException(status_code=400, detail="Valid kind, model_id, and existing model_path are required")

        with self.lock:
            if self.active_kind == kind and self.active_model_id == model_id and self.adapter is not None:
                return {"status": "ready", "reused": True, **self.status()}
            self._unload_unlocked()
            try:
                if adapter_family == "diffusers":
                    adapter: Any = DiffusersImageAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                elif adapter_family == "qwen_vl":
                    adapter = QwenVLChatAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                elif adapter_family == "transformers_causal":
                    adapter = TransformersChatAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported adapter_family {adapter_family!r} for model {model_id}",
                    )
                adapter.load(precision=payload.get("precision"))
                self.adapter = adapter
                self.active_kind = kind
                self.active_model_id = model_id
                self.last_error = None
                return {"status": "ready", "reused": False, **self.status()}
            except HTTPException:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._unload_unlocked()
                raise HTTPException(status_code=500, detail=self.last_error) from exc


def create_worker_app(secret_token: str) -> FastAPI:
    app = FastAPI(title="Nexus Isolated Model Worker", version="1.0.1")
    state = ModelWorkerState()
    require_auth = make_auth_dependency(secret_token)

    @app.get("/worker/health")
    def worker_health(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return state.status()

    @app.post("/worker/load")
    def worker_load(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        return state.load(body)

    @app.post("/worker/unload")
    def worker_unload(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        state.unload()
        return {"status": "unloaded", **state.status()}

    @app.post("/worker/cancel")
    def worker_cancel(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        op_id = (body or {}).get("operation_id")
        cancelled = state.cancel_operation(str(op_id) if op_id else None)
        return {"status": "cancelled", "operations": cancelled}

    @app.post("/worker/generate")
    def worker_generate(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        if state.active_kind != "chat" or state.adapter is None:
            raise HTTPException(status_code=409, detail="No chat model loaded in worker")
        stream = bool(body.get("stream", False))
        if not stream:
            with state.lock:
                try:
                    text = "".join(state.adapter.generate_stream(body))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"text": text, "model_id": state.active_model_id}

        def event_stream():
            state.lock.acquire()
            try:
                for chunk in state.adapter.generate_stream(body):
                    if chunk:
                        yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"
            finally:
                state.lock.release()

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post("/worker/generate-image")
    def worker_generate_image(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        if state.active_kind != "diffusion" or state.adapter is None:
            raise HTTPException(status_code=409, detail="No diffusion pipeline loaded in worker")
        op_id = str(body.get("operation_id") or "img_op")
        cancel_event = state.register_cancel_event(op_id)
        ev_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()

        def _on_progress(event: dict[str, Any]) -> None:
            ev_queue.put({"type": "progress", "operation_id": op_id, **event})

        def _run_generation() -> None:
            with state.lock:
                try:
                    result = state.adapter.generate(body, on_progress=_on_progress, cancel_event=cancel_event)
                    ev_queue.put({"type": "result", "operation_id": op_id, "data": result})
                except GenerationCancelledError as exc:
                    ev_queue.put({"type": "cancelled", "operation_id": op_id, "message": str(exc)})
                except Exception as exc:
                    ev_queue.put({"type": "error", "operation_id": op_id, "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    state.clear_cancel_event(op_id)
                    ev_queue.put(None)

        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()

        def ndjson_stream():
            try:
                while True:
                    item = ev_queue.get()
                    if item is None:
                        break
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:
                cancel_event.set()
                thread.join(timeout=5.0)

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    @app.post("/worker/shutdown")
    def worker_shutdown(_auth: None = Depends(require_auth)) -> dict[str, str]:
        state.cancel_operation(None)
        state.unload()
        threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        return {"status": "shutting_down"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus isolated GPU model worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    host = assert_loopback_host(args.host)
    token = os.environ.get("NEXUS_WORKER_SECRET", "")
    if not token:
        raise RuntimeError("NEXUS_WORKER_SECRET environment variable is required")
    app = create_worker_app(token)
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
