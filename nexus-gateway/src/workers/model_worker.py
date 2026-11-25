"""Isolated GPU model worker subprocess owning CUDA context, weights, and inference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

# Ensure nexus-gateway/src is importable when launched as a subprocess script.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adapters.diffusers import DiffusersImageAdapter  # noqa: E402
from adapters.qwen_vl import QwenVLChatAdapter  # noqa: E402
from adapters.transformers import TransformersChatAdapter  # noqa: E402
from api.middleware.auth import assert_loopback_host, make_auth_dependency  # noqa: E402
from services.progress import ProgressService  # noqa: E402


class ModelWorkerState:
    """Owns the single active CUDA model or diffusion pipeline inside this worker process."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_kind: Optional[str] = None
        self.active_model_id: Optional[str] = None
        self.adapter: Any = None
        self.last_error: Optional[str] = None
        self.progress = ProgressService(max_records=32)

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
        if kind not in ("chat", "diffusion") or not model_id or not model_path.exists():
            raise HTTPException(status_code=400, detail="Valid kind, model_id, and existing model_path are required")

        with self.lock:
            if self.active_kind == kind and self.active_model_id == model_id and self.adapter is not None:
                return {"status": "ready", "reused": True, **self.status()}
            self._unload_unlocked()
            try:
                if kind == "diffusion":
                    adapter = DiffusersImageAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                elif "vl" in architecture.lower() or "vl" in model_id.lower():
                    adapter = QwenVLChatAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                else:
                    adapter = TransformersChatAdapter(model_id=model_id, model_path=model_path, architecture=architecture)
                adapter.load(precision=payload.get("precision"))
                self.adapter = adapter
                self.active_kind = kind
                self.active_model_id = model_id
                self.last_error = None
                return {"status": "ready", "reused": False, **self.status()}
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._unload_unlocked()
                raise HTTPException(status_code=500, detail=self.last_error) from exc


def create_worker_app(secret_token: str) -> FastAPI:
    app = FastAPI(title="Nexus Isolated Model Worker", version="1.0.0")
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

    @app.post("/worker/generate")
    def worker_generate(body: dict[str, Any], _auth: None = Depends(require_auth)) -> Any:
        if state.active_kind != "chat" or state.adapter is None:
            raise HTTPException(status_code=409, detail="No chat model loaded in worker")
        stream = bool(body.get("stream", False))
        if not stream:
            with state.lock:
                text = "".join(state.adapter.generate_stream(body))
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
    def worker_generate_image(body: dict[str, Any], _auth: None = Depends(require_auth)) -> dict[str, Any]:
        if state.active_kind != "diffusion" or state.adapter is None:
            raise HTTPException(status_code=409, detail="No diffusion pipeline loaded in worker")
        with state.lock:
            return state.adapter.generate(body)

    @app.post("/worker/shutdown")
    def worker_shutdown(_auth: None = Depends(require_auth)) -> dict[str, str]:
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
