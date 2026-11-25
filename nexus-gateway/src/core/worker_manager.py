"""Lifecycle manager for the isolated GPU model worker subprocess."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

from workers.worker_protocol import WorkerGenerateRequest, WorkerImageRequest, WorkerLoadRequest, WorkerStatus


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
    Supports on-demand spawn, health checks, graceful unload, forced process
    termination when VRAM is not reclaimed, and preserves local staged caches.
    """

    def __init__(
        self,
        *,
        worker_script: Optional[Path] = None,
        python_executable: Optional[str] = None,
        startup_timeout_s: float = 30.0,
    ) -> None:
        self._worker_script = worker_script or (Path(__file__).resolve().parents[1] / "workers" / "model_worker.py")
        self._python = python_executable or sys.executable
        self._startup_timeout_s = startup_timeout_s
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._port: Optional[int] = None
        self._secret: str = ""
        self._active_kind: Optional[str] = None
        self._active_model_id: Optional[str] = None
        self._last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

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
            raise WorkerError(f"Worker error ({exc.code}): {detail}", status_code=exc.code) from exc
        except Exception as exc:
            raise WorkerError(f"Worker RPC failed: {type(exc).__name__}: {exc}", status_code=502) from exc

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
            env = {**os.environ, "NEXUS_WORKER_SECRET": secret}
            cmd = [self._python, str(self._worker_script), "--host", "127.0.0.1", "--port", str(port)]
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._proc = proc
            self._port = port
            self._secret = secret
            self._active_kind = None
            self._active_model_id = None

            deadline = time.monotonic() + self._startup_timeout_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    stderr_tail = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="replace")[-500:]
                    self._last_error = f"Worker exited early (code {proc.returncode}): {stderr_tail}"
                    self._proc = None
                    self._port = None
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
                    chunk = json.loads(line).get("chunk")
                    if isinstance(chunk, str) and chunk:
                        yield chunk
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise WorkerError(f"Worker stream error ({exc.code}): {body}", status_code=exc.code) from exc
        except Exception as exc:
            raise WorkerError(f"Worker stream failed: {type(exc).__name__}: {exc}", status_code=502) from exc

    def generate_image(self, req: WorkerImageRequest, *, timeout: float = 900.0) -> dict[str, Any]:
        self.ensure_started()
        return self._request_json("POST", "/worker/generate-image", req.to_dict(), timeout=timeout)

    def unload(self, *, terminate_on_failure: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self.is_running:
                self._active_kind = None
                self._active_model_id = None
                return {"status": "not_running"}
            try:
                res = self._request_json("POST", "/worker/unload", {}, timeout=30.0)
                self._active_kind = None
                self._active_model_id = None
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
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=3.0)
                except Exception:
                    pass
