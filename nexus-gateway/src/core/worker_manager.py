"""Lifecycle manager for the isolated GPU model worker subprocess."""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus

# Allocated (not reserved!) CUDA VRAM ceiling after unload before triggering worker recycle.
MAX_LEAKED_ALLOCATED_VRAM_BYTES = 512 * 1024 * 1024  # 512 MiB


def _pick_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class WorkerError(RuntimeError):
    """Structured error raised when worker lifecycle or RPC fails."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class WorkerManager:
    """
    Manages a dedicated isolated subprocess for GPU model execution.
    Redirects worker stdout/stderr to a managed log file (never undrained pipes),
    streams NDJSON image progress, supports real cancellation, and verifies actual
    allocated VRAM release on unload.
    """

    def __init__(
        self,
        *,
        worker_script: Optional[Path] = None,
        python_executable: Optional[str] = None,
        startup_timeout_s: float = 30.0,
        log_path: Optional[Path] = None,
    ) -> None:
        self._worker_script = worker_script or (Path(__file__).resolve().parents[1] / "workers" / "model_worker.py")
        self._python = python_executable or sys.executable
        self._startup_timeout_s = startup_timeout_s
        self._log_path = log_path or Path(tempfile.gettempdir()) / "nexus-model-worker.log"
        self._log_fh: Any = None
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._port: Optional[int] = None
        self._secret: str = ""
        self._active_kind: Optional[str] = None
        self._active_model_id: Optional[str] = None
        self._last_error: Optional[str] = None
        self.last_stream_usage: Optional[dict[str, int]] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _redact(self, text: str) -> str:
        out = text
        if self._secret:
            out = out.replace(self._secret, "[REDACTED_WORKER_SECRET]")
        api_key = os.environ.get("NEXUS_API_KEY", "")
        if api_key:
            out = out.replace(api_key, "[REDACTED_NEXUS_API_KEY]")
        return out

    def _read_log_tail(self, max_chars: int = 800) -> str:
        try:
            if self._log_path.is_file():
                raw = self._log_path.read_text(encoding="utf-8", errors="replace")
                return self._redact(raw[-max_chars:])
        except OSError:
            pass
        return ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._secret}",
            "Content-Type": "application/json",
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        if self._port is None:
            raise WorkerError("Model worker is not running", status_code=503)
        url = f"http://127.0.0.1:{self._port}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            raise WorkerError(f"Worker error ({exc.code}): {self._redact(str(detail))}", status_code=exc.code) from exc
        except Exception as exc:
            raise WorkerError(f"Worker RPC failed: {type(exc).__name__}: {self._redact(str(exc))}", status_code=502) from exc

    def ensure_started(self) -> WorkerStatus:
        with self._lock:
            if self.is_running:
                try:
                    info = self._request_json("GET", "/worker/health", timeout=5.0)
                    return WorkerStatus(**info)
                except WorkerError:
                    self._terminate_unlocked()

            port = _pick_free_loopback_port()
            secret = secrets.token_urlsafe(32)
            env = {
                **os.environ,
                "NEXUS_WORKER_SECRET": secret,
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            }
            cmd = [self._python, str(self._worker_script), "--host", "127.0.0.1", "--port", str(port)]
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self._log_path.open("ab")
            popen_kwargs: dict[str, Any] = {
                "env": env,
                "stdout": self._log_fh,
                "stderr": subprocess.STDOUT,
            }
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **popen_kwargs)
            self._proc = proc
            self._port = port
            self._secret = secret
            self._active_kind = None
            self._active_model_id = None

            deadline = time.monotonic() + self._startup_timeout_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = self._read_log_tail()
                    self._last_error = f"Worker exited early (code {proc.returncode}): {tail}"
                    self._terminate_unlocked()
                    raise WorkerError(self._last_error, status_code=503)
                try:
                    info = self._request_json("GET", "/worker/health", timeout=2.0)
                    self._last_error = None
                    return WorkerStatus(**info)
                except WorkerError:
                    time.sleep(0.15)

            self._terminate_unlocked()
            self._last_error = f"Model worker did not become healthy within {self._startup_timeout_s}s"
            raise WorkerError(self._last_error, status_code=504)

    def status(self) -> WorkerStatus:
        with self._lock:
            if not self.is_running:
                return WorkerStatus(
                    pid=0,
                    alive=False,
                    active_kind=None,
                    active_model_id=None,
                    last_error=self._last_error,
                )
            try:
                info = self._request_json("GET", "/worker/health", timeout=3.0)
                return WorkerStatus(**info)
            except WorkerError as exc:
                self._last_error = str(exc)
                return WorkerStatus(
                    pid=self._proc.pid if self._proc else 0,
                    alive=False,
                    active_kind=None,
                    active_model_id=None,
                    last_error=self._last_error,
                )

    def load_model(self, req: WorkerLoadRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        self.ensure_started()
        res = self._request_json("POST", "/worker/load", req.to_dict(), timeout=timeout)
        self._active_kind = req.kind
        self._active_model_id = req.model_id
        return res

    def generate(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> dict[str, Any]:
        self.ensure_started()
        return self._request_json("POST", "/worker/generate", req.to_dict(), timeout=timeout)

    def generate_stream(self, req: WorkerGenerateRequest, *, timeout: float = 600.0) -> Iterator[str]:
        self.ensure_started()
        if self._port is None:
            raise WorkerError("Model worker is not running", status_code=503)
        self.last_stream_usage = None
        payload = {**req.to_dict(), "stream": True}
        url = f"http://127.0.0.1:{self._port}/worker/generate"
        http_req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    parsed_line = json.loads(line)
                    if "error" in parsed_line and parsed_line["error"]:
                        err_msg = self._redact(str(parsed_line["error"]))
                        st_code = int(parsed_line.get("status_code") or (400 if "maximum context length" in err_msg.lower() or "exceed_context_size" in err_msg.lower() else 502))
                        raise WorkerError(f"Worker stream error: {err_msg}", status_code=st_code)
                    usage_obj = parsed_line.get("usage")
                    if isinstance(usage_obj, dict):
                        self.last_stream_usage = {
                            "prompt_tokens": int(usage_obj.get("prompt_tokens") or 0),
                            "completion_tokens": int(usage_obj.get("completion_tokens") or 0),
                            "total_tokens": int(usage_obj.get("total_tokens") or 0),
                        }
                    chunk = parsed_line.get("chunk")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
        except WorkerError:
            raise
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise WorkerError(f"Worker stream error ({exc.code}): {self._redact(body)}", status_code=exc.code) from exc
        except Exception as exc:
            raise WorkerError(f"Worker stream failed: {type(exc).__name__}: {self._redact(str(exc))}", status_code=502) from exc

    def cancel_operation(self, operation_id: Optional[str] = None) -> dict[str, Any]:
        if not self.is_running:
            return {"status": "not_running", "operations": []}
        try:
            return self._request_json("POST", "/worker/cancel", {"operation_id": operation_id}, timeout=5.0)
        except WorkerError:
            return {"status": "unreachable", "operations": []}

    def generate_image(
        self,
        req: WorkerImageRequest,
        *,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        """Consume NDJSON image-generation stream, relaying real progress and handling disconnects."""
        self.ensure_started()
        if self._port is None:
            raise WorkerError("Model worker is not running", status_code=503)
        url = f"http://127.0.0.1:{self._port}/worker/generate-image"
        http_req = urllib.request.Request(
            url,
            data=json.dumps(req.to_dict()).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        result_data: Optional[dict[str, Any]] = None
        try:
            with urllib.request.urlopen(http_req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    ev_type = event.get("type")
                    if ev_type == "progress" and on_progress is not None:
                        on_progress(event)
                    elif ev_type == "result":
                        result_data = event.get("data")
                    elif ev_type == "cancelled":
                        raise WorkerError(str(event.get("message") or "Image generation cancelled"), status_code=499)
                    elif ev_type == "error":
                        raise WorkerError(str(event.get("message") or "Worker image generation error"), status_code=500)
        except WorkerError:
            raise
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise WorkerError(f"Worker image error ({exc.code}): {self._redact(body)}", status_code=exc.code) from exc
        except Exception as exc:
            self.cancel_operation(req.operation_id or None)
            raise WorkerError(f"Worker image stream disconnected: {type(exc).__name__}: {self._redact(str(exc))}", status_code=502) from exc

        if not isinstance(result_data, dict):
            raise WorkerError("Worker image stream ended without returning a result", status_code=502)
        return result_data

    def unload(self, *, kind: str = "all", terminate_on_failure: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self.is_running:
                self._active_kind = None
                self._active_model_id = None
                return {"status": "not_running"}
            try:
                res = self._request_json("POST", "/worker/unload", {"kind": kind}, timeout=30.0)
                if kind == "all":
                    self._active_kind = None
                    self._active_model_id = None
                    # Check actual allocated VRAM (not reserved pool) + active_model_id.
                    allocated = int(res.get("cuda_allocated_bytes") or 0)
                    still_loaded = res.get("active_model_id") is not None
                    if still_loaded or allocated > MAX_LEAKED_ALLOCATED_VRAM_BYTES:
                        if terminate_on_failure:
                            self._terminate_unlocked()
                            return {"status": "terminated_for_vram_release", "leaked_allocated_bytes": allocated}
                return res
            except WorkerError:
                if terminate_on_failure:
                    self._terminate_unlocked()
                    return {"status": "terminated_for_vram_release"}
                raise

    def restart(self) -> WorkerStatus:
        """Restart the isolated worker subprocess while preserving staged model caches."""
        self.terminate()
        return self.ensure_started()

    def terminate(self) -> None:
        with self._lock:
            self._terminate_unlocked()

    def _terminate_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        self._port = None
        self._active_kind = None
        self._active_model_id = None
        if proc is not None and proc.poll() is None:
            pgid: Optional[int] = None
            if os.name != "nt":
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = None
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait(timeout=3.0)
                except Exception:
                    pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None
