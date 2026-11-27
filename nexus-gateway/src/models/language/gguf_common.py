"""Base process controller and HTTP client for local `llama-server` GGUF instances."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from models.base import GenerationCancelledError


def _pick_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_llama_server_binary() -> str:
    """Locate `llama-server` binary on PATH or common installation locations."""
    for cand in (
        shutil.which("llama-server"),
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(Path(cand).resolve())
    for search_root in (Path("/content/llama-cpp-cuda"), Path("/content")):
        if not search_root.is_dir():
            continue
        matches = sorted(search_root.rglob("llama-server"))
        for match in reversed(matches):
            if match.is_file() and os.access(match, os.X_OK):
                return str(match.resolve())
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
        port: int = 0,
        n_ctx: int = 16384,
        n_gpu_layers: int = 999,
        extra_args: Optional[list[str]] = None,
        merge_consecutive_roles: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = extra_args or []
        self.merge_consecutive_roles = merge_consecutive_roles
        self.has_mmproj = find_mmproj_file(model_path) is not None
        self.process: Optional[subprocess.Popen[Any]] = None
        self._active_port = port or 0
        self._log_path: Optional[Path] = None
        self.base_url = f"http://127.0.0.1:{self._active_port}" if self._active_port else ""

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
        self.stop()
        binary = find_llama_server_binary()
        gguf_path = find_primary_gguf_file(self.model_path)
        mmproj_path = find_mmproj_file(self.model_path)

        self._active_port = self.port if self.port > 0 else _pick_free_loopback_port()
        self.base_url = f"http://127.0.0.1:{self._active_port}"

        cmd = [
            binary,
            "-m", str(gguf_path),
            "-c", str(self.n_ctx),
            "-ngl", str(self.n_gpu_layers),
            "-fa", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--port", str(self._active_port),
            "--host", "127.0.0.1",
            "--jinja",
        ]
        if mmproj_path is not None:
            cmd.extend(["--mmproj", str(mmproj_path)])
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        bin_dir = Path(binary).resolve().parent
        lib_dirs = [str(bin_dir)]
        install_root = Path("/content/llama-cpp-cuda")
        if install_root.is_dir():
            for so_file in install_root.rglob("*.so*"):
                if so_file.is_file() and str(so_file.parent) not in lib_dirs:
                    lib_dirs.append(str(so_file.parent))
        cur_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, *([cur_ld] if cur_ld else [])])

        log_path = Path("/content/nexus-llama-server.log") if Path("/content").is_dir() else Path("nexus-llama-server.log")
        self._log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Wait for server health
        start_t = time.monotonic()
        while time.monotonic() - start_t < 180:
            if self.process.poll() is not None:
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                except Exception:
                    pass
                raise RuntimeError(f"llama-server exited prematurely with code {self.process.returncode}:\n{tail}")
            try:
                with httpx.Client(timeout=1.0) as client:
                    res = client.get(f"{self.base_url}/health")
                    if res.status_code == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)

        self.stop()
        raise TimeoutError("llama-server failed to become healthy within 180s")

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
        log_fh = getattr(self, "_log_file", None)
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass
            self._log_file = None

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        inject_tools: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Normalize OpenAI messages for llama-server Jinja templates."""
        prepared: list[dict[str, Any]] = []
        for raw in messages:
            if not isinstance(raw, dict):
                continue
            msg = dict(raw)
            content = msg.get("content")
            if isinstance(content, list):
                has_image = self.has_mmproj and any(
                    isinstance(b, dict) and b.get("type") == "image_url" for b in content
                )
                if not has_image:
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and isinstance(block.get("text"), str):
                                parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    msg["content"] = "\n".join(parts)
            elif content is None:
                msg["content"] = ""
            prepared.append(msg)

        if inject_tools:
            tools_json = json.dumps(inject_tools, indent=2)
            tool_block = (
                "# Available Tools\n"
                "You may call a tool by emitting:\n"
                "<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>\n\n"
                f"Tools:\n{tools_json}"
            )
            if prepared and prepared[0].get("role") == "system" and isinstance(prepared[0].get("content"), str):
                prepared[0] = {**prepared[0], "content": f"{prepared[0]['content']}\n\n{tool_block}".strip()}
            else:
                prepared.insert(0, {"role": "system", "content": tool_block})

        if not self.merge_consecutive_roles:
            return prepared

        merged: list[dict[str, Any]] = []
        for msg in prepared:
            role = msg.get("role")
            if merged and merged[-1].get("role") == "tool" and role == "user":
                # A cancelled turn after a tool call leaves `tool -> user` with no assistant reply;
                # insert an assistant separator so Gemma's strict Jinja alternation stays valid.
                merged.append({"role": "assistant", "content": "(Previous tool turn was cancelled.)"})
            if (
                merged
                and role in ("system", "user")
                and merged[-1].get("role") == role
                and isinstance(merged[-1].get("content"), str)
                and isinstance(msg.get("content"), str)
                and not merged[-1].get("tool_calls")
                and not msg.get("tool_calls")
            ):
                merged[-1] = {
                    **merged[-1],
                    "content": f"{merged[-1]['content']}\n\n{msg['content']}".strip(),
                }
            else:
                merged.append(msg)
        return merged

    def _stream_once(self, payload: dict[str, Any]) -> Iterator[str]:
        url = f"{self.base_url}/v1/chat/completions"
        in_think = False
        pending_tool_calls: dict[int, dict[str, str]] = {}
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_text = response.read().decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        f"llama-server returned {response.status_code}: {err_text}",
                        request=response.request,
                        response=response,
                    )

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
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}

                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                if not in_think:
                                    yield "<think>"
                                    in_think = True
                                yield reasoning

                            content = delta.get("content")
                            if content:
                                if in_think:
                                    yield "</think>"
                                    in_think = False
                                yield content

                            raw_tool_calls = delta.get("tool_calls")
                            if isinstance(raw_tool_calls, list):
                                for tc in raw_tool_calls:
                                    if not isinstance(tc, dict):
                                        continue
                                    idx = int(tc.get("index", 0))
                                    entry = pending_tool_calls.setdefault(idx, {"name": "", "arguments": ""})
                                    fn = tc.get("function") or {}
                                    if isinstance(fn.get("name"), str) and fn["name"]:
                                        entry["name"] += fn["name"]
                                    if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                                        entry["arguments"] += fn["arguments"]
                        except json.JSONDecodeError:
                            pass

        if in_think:
            yield "</think>"
            in_think = False

        for idx in sorted(pending_tool_calls):
            entry = pending_tool_calls[idx]
            if not entry["name"]:
                continue
            args_raw = entry["arguments"].strip() or "{}"
            try:
                args_obj = json.loads(args_raw)
            except json.JSONDecodeError:
                args_obj = args_raw
            yield f"<tool_call>{json.dumps({'name': entry['name'], 'arguments': args_obj}, ensure_ascii=False)}</tool_call>"

    def generate_chat_stream(self, req: dict[str, Any]) -> Iterator[str]:
        if not self.is_ready():
            rc = self.process.poll() if self.process is not None else "no-process"
            tail = ""
            if self._log_path is not None and self._log_path.is_file():
                try:
                    tail = self._log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                except Exception:
                    pass
            raise RuntimeError(
                f"llama-server ({self.model_id}) is not running or not healthy (rc={rc}, port={self._active_port}):\n{tail}"
            )

        messages = req.get("messages") or []
        tools = req.get("tools") or []
        temperature = req.get("temperature", 0.7)
        max_tokens = req.get("max_tokens") or 4096

        payload: dict[str, Any] = {
            "messages": self._prepare_messages(messages),
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        try:
            yield from self._stream_once(payload)
        except httpx.HTTPStatusError as exc:
            if tools and exc.response.status_code in (400, 500):
                # Fallback when a GGUF's embedded Jinja template rejects OpenAI `tools`
                fallback_payload: dict[str, Any] = {
                    "messages": self._prepare_messages(messages, inject_tools=tools),
                    "stream": True,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                try:
                    yield from self._stream_once(fallback_payload)
                    return
                except Exception as fb_exc:
                    raise RuntimeError(f"llama-server fallback streaming failed: {fb_exc}") from fb_exc
            raise RuntimeError(f"llama-server streaming failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"llama-server streaming failed: {exc}") from exc
