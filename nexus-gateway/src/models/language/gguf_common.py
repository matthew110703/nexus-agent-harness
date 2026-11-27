"""Base process controller and HTTP client for local `llama-server` GGUF instances."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from models.base import GenerationCancelledError


def find_llama_server_binary() -> str:
    """Locate `llama-server` binary on PATH or common installation locations."""
    for cand in (
        shutil.which("llama-server"),
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/content/llama-b*-bin-ubuntu-cuda-12.8-x64/build/bin/llama-server",
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(cand)
    # Check glob under /content
    content_matches = sorted(Path("/content").glob("llama-*-bin-*/build/bin/llama-server"))
    if content_matches and os.access(content_matches[-1], os.X_OK):
        return str(content_matches[-1])
    raise RuntimeError(
        "llama-server binary not found. Run Colab runtime setup to download llama.cpp CUDA binaries."
    )


def find_primary_gguf_file(model_path: Path) -> Path:
    """Find the single or primary shard `.gguf` file in a directory."""
    candidates = sorted(model_path.glob("*.gguf"))
    if not candidates:
        raise ValueError(f"No .gguf files found in model directory {model_path}")
    # Prioritize non-mmproj GGUFs
    non_mmproj = [p for p in candidates if "mmproj" not in p.name.lower()]
    return non_mmproj[0] if non_mmproj else candidates[0]


def find_mmproj_file(model_path: Path) -> Optional[Path]:
    """Find an optional multimodal projector mmproj-*.gguf in the directory."""
    candidates = sorted(model_path.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None


class LlamaServerRuntimeBase:
    """Manages an isolated `llama-server` child process with an OpenAI-compatible HTTP interface."""

    def __init__(
        self,
        *,
        model_id: str,
        model_path: Path,
        port: int = 8081,
        n_ctx: int = 32768,
        n_gpu_layers: int = 999,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen[Any]] = None
        self.base_url = f"http://127.0.0.1:{self.port}"

    def is_ready(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/health")
                return res.status_code == 200
        except Exception:
            return False

    def start(self) -> None:
        if self.is_ready():
            return
        binary = find_llama_server_binary()
        gguf_path = find_primary_gguf_file(self.model_path)
        mmproj_path = find_mmproj_file(self.model_path)

        cmd = [
            binary,
            "-m", str(gguf_path),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--jinja",
        ]
        if mmproj_path is not None:
            cmd.extend(["--mmproj", str(mmproj_path)])
        cmd.extend(self.extra_args)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server health
        start_t = time.monotonic()
        while time.monotonic() - start_t < 90:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited prematurely with code {self.process.returncode}")
            try:
                with httpx.Client(timeout=1.0) as client:
                    res = client.get(f"{self.base_url}/health")
                    if res.status_code == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)

        self.stop()
        raise TimeoutError("llama-server failed to become healthy within 90s")

    def stop(self) -> None:
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except Exception:
                    pass
            self.process = None

    def generate_chat_stream(self, req: dict[str, Any]) -> Iterator[str]:
        if not self.is_ready():
            raise RuntimeError(f"llama-server ({self.model_id}) is not running or not healthy")

        messages = req.get("messages") or []
        tools = req.get("tools")
        temperature = req.get("temperature", 0.7)
        max_tokens = req.get("max_tokens") or 4096

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}/v1/chat/completions"
        try:
            with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"llama-server returned {response.status_code}: {err_text}")

                    for line in response.iter_lines():
                        if not line or not line.strip():
                            continue
                        line_str = line.strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as exc:
            raise RuntimeError(f"llama-server streaming failed: {exc}") from exc
