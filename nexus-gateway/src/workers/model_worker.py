"""Isolated GPU model worker subprocess owning CUDA context, dual-slot co-resident runtimes, and cancellation."""

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

from api.middleware.auth import assert_loopback_host, make_auth_dependency  # noqa: E402
from models.base import (  # noqa: E402
    GenerationCancelledError,
    load_dedicated_image_runtime,
    load_dedicated_language_runtime,
)


class ModelWorkerState:
    """Owns dual CUDA model slots (`language_runtime` + `image_runtime`) inside this isolated process."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.language_runtime: Any = None
        self.active_language_id: Optional[str] = None
        self.image_runtime: Any = None
        self.active_image_id: Optional[str] = None
        self.last_error: Optional[str] = None
        self._cancel_lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}

    @property
    def active_kind(self) -> Optional[str]:
        if self.language_runtime is not None:
            return "language"
        if self.image_runtime is not None:
            return "image"
        return None

    @property
    def active_model_id(self) -> Optional[str]:
        if self.active_language_id is not None:
            return self.active_language_id
        return self.active_image_id

    @property
    def runtime(self) -> Any:
        return self.language_runtime or self.image_runtime

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

    def _cuda_stats(self) -> tuple[int, int, int, int, int]:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                alloc = int(torch.cuda.memory_allocated())
                reserved = int(torch.cuda.memory_reserved())
                try:
                    free_b, total_b = torch.cuda.mem_get_info(0)
                    used_b = max(0, int(total_b) - int(free_b))
                    return alloc, reserved, int(total_b), int(free_b), used_b
                except Exception:
                    return alloc, reserved, 0, 0, alloc
        except Exception:
            pass
        return 0, 0, 0, 0, 0

    def status(self) -> dict[str, Any]:
        alloc, reserved, dev_total, dev_free, dev_used = self._cuda_stats()
        return {
            "pid": os.getpid(),
            "alive": True,
            "active_kind": self.active_kind,
            "active_model_id": self.active_model_id,
            "active_language_id": self.active_language_id,
            "active_image_id": self.active_image_id,
            "co_resident": self.language_runtime is not None and self.image_runtime is not None,
            "cuda_allocated_bytes": alloc,
            "cuda_reserved_bytes": reserved,
            "device_total_vram_bytes": dev_total,
            "device_free_vram_bytes": dev_free,
            "device_used_vram_bytes": dev_used,
            "last_error": self.last_error,
        }

    def unload(self, kind: str = "all") -> None:
        with self.lock:
            self._unload_unlocked(kind)

    def _unload_unlocked(self, kind: str = "all") -> None:
        targets: list[Any] = []
        if kind in ("all", "language") and self.language_runtime is not None:
            targets.append(self.language_runtime)
            self.language_runtime = None
            self.active_language_id = None
        if kind in ("all", "image") and self.image_runtime is not None:
            targets.append(self.image_runtime)
            self.image_runtime = None
            self.active_image_id = None
        for rt in targets:
            try:
                rt.unload()
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
        implementation = str(payload.get("implementation") or "")
        architecture = str(payload.get("architecture") or "unknown")
        keep_other_slot = bool(payload.get("keep_other_slot", False))
        runtime_options = payload.get("runtime_options") if isinstance(payload.get("runtime_options"), dict) else None
        if kind not in ("language", "image") or not model_id or not implementation or not model_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Valid kind ('language'|'image'), model_id, implementation, and existing staged model_path are required",
            )

        with self.lock:
            if kind == "language" and self.active_language_id == model_id and self.language_runtime is not None:
                if not keep_other_slot:
                    self._unload_unlocked("image")
                return {"status": "ready", "reused": True, **self.status()}
            if kind == "image" and self.active_image_id == model_id and self.image_runtime is not None:
                if not keep_other_slot:
                    self._unload_unlocked("language")
                return {"status": "ready", "reused": True, **self.status()}

            if keep_other_slot:
                self._unload_unlocked(kind)
            else:
                self._unload_unlocked("all")

            try:
                if kind == "language":
                    runtime: Any = load_dedicated_language_runtime(
                        implementation=implementation,
                        model_id=model_id,
                        model_path=model_path,
                        architecture=architecture,
                        runtime_options=runtime_options,
                    )
                else:
                    runtime = load_dedicated_image_runtime(
                        implementation=implementation,
                        model_id=model_id,
                        model_path=model_path,
                        architecture=architecture,
                    )
                runtime.load(precision=payload.get("precision"))
                if not runtime.is_ready():
                    raise RuntimeError(f"Dedicated runtime for {model_id} did not report ready after load")
                if kind == "language":
                    self.language_runtime = runtime
                    self.active_language_id = model_id
                else:
                    self.image_runtime = runtime
                    self.active_image_id = model_id
                self.last_error = None
                return {"status": "ready", "reused": False, **self.status()}
            except HTTPException:
                raise
            except ValueError as exc:
                self.last_error = str(exc)
                self._unload_unlocked(kind)
                raise HTTPException(status_code=400, detail=self.last_error) from exc
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._unload_unlocked(kind)
                raise HTTPException(status_code=500, detail=self.last_error) from exc


def create_worker_app(secret_token: str) -> FastAPI:
    app = FastAPI(title="Nexus Isolated Model Worker", version="2.1.0")
    state = ModelWorkerState()
    require_auth = make_auth_dependency(secret_token)

    @app.get("/worker/health")
    def worker_health(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return state.status()

    @app.post("/worker/load")
    def worker_load(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        return state.load(body)

    @app.post("/worker/unload")
    def worker_unload(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        kind = str((body or {}).get("kind") or "all")
        state.unload(kind)
        return {"status": "unloaded", **state.status()}

    @app.post("/worker/cancel")
    def worker_cancel(body: Optional[dict[str, Any]] = None, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        op_id = (body or {}).get("operation_id")
        cancelled = state.cancel_operation(str(op_id) if op_id else None)
        return {"status": "cancelled", "operations": cancelled}

    @app.post("/worker/generate")
    def worker_generate(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        if state.language_runtime is None:
            raise HTTPException(status_code=409, detail="No language model loaded in worker")
        stream = bool(body.get("stream", False))
        if not stream:
            with state.lock:
                try:
                    text = "".join(state.language_runtime.generate_stream(body))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                usage = getattr(state.language_runtime, "last_usage", None)
            resp_payload: dict[str, Any] = {"text": text, "model_id": state.active_language_id}
            if isinstance(usage, dict):
                resp_payload["usage"] = usage
            return resp_payload

        def event_stream():
            state.lock.acquire()
            try:
                for chunk in state.language_runtime.generate_stream(body):
                    if chunk:
                        yield json.dumps({"chunk": chunk}, ensure_ascii=False) + "\n"
                usage = getattr(state.language_runtime, "last_usage", None)
                if isinstance(usage, dict):
                    yield json.dumps({"usage": usage}, ensure_ascii=False) + "\n"
            except ValueError as exc:
                yield json.dumps({"error": str(exc), "status_code": 400}, ensure_ascii=False) + "\n"
            except Exception as exc:
                yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n"
            finally:
                state.lock.release()

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @app.post("/worker/generate-image")
    def worker_generate_image(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        if state.image_runtime is None:
            raise HTTPException(status_code=409, detail="No image pipeline loaded in worker")
        op_id = str(body.get("operation_id") or "img_op")
        cancel_event = state.register_cancel_event(op_id)
        ev_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()

        def _on_progress(event: dict[str, Any]) -> None:
            ev_queue.put({"type": "progress", "operation_id": op_id, **event})

        def _run_generation() -> None:
            with state.lock:
                try:
                    result = state.image_runtime.generate(body, on_progress=_on_progress, cancel_event=cancel_event)
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
        state.unload("all")
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
