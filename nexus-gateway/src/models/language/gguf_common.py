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
from models.gguf_metadata import inspect_gguf_directory
from models.language.common import deserialize_tool_call_arguments, normalize_tools_for_jinja
from models.language.prompt_tools import build_tools_instruction


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
    """Find the primary `.gguf` file (or first shard) in a directory, excluding auxiliary files."""
    layout = inspect_gguf_directory(model_path)
    if layout.valid and layout.source_files and layout.layout in ("single", "gguf_split"):
        return layout.source_files[0]
    candidates = sorted(model_path.glob("*.gguf"))
    if not candidates:
        raise ValueError(f"No .gguf files found in model directory {model_path} ({layout.reason})")
    non_aux = [
        p
        for p in candidates
        if not any(tag in p.name.lower() for tag in ("mmproj", "mtp-", "imatrix", "draft"))
    ]
    pool = non_aux or candidates
    return max(pool, key=lambda p: p.stat().st_size)


def find_mmproj_file(model_path: Path) -> Optional[Path]:
    """Find an optional multimodal projector mmproj-*.gguf in the directory."""
    candidates = sorted(p for p in model_path.glob("*mmproj*.gguf") if p.stat().st_size > 0)
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
        batch_size: int = 2048,
        ubatch_size: int = 512,
        threads: int = 0,
        cache_reuse: int = 256,
        reasoning_format: Optional[str] = "deepseek",
        extra_args: Optional[list[str]] = None,
        merge_consecutive_roles: bool = False,
        allow_quantized_kv: bool = True,
        moe_cpu_offload: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.port = port
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.cache_reuse = cache_reuse
        self.reasoning_format = reasoning_format
        self.extra_args = extra_args or []
        self.merge_consecutive_roles = merge_consecutive_roles
        self.allow_quantized_kv = allow_quantized_kv
        self.moe_cpu_offload = moe_cpu_offload
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

    def _launch_with_flags(
        self,
        *,
        binary: str,
        gguf_path: Path,
        mmproj_path: Optional[Path],
        n_ctx: int,
        n_gpu_layers: int,
        flag_set: list[str],
        env: dict[str, str],
        log_path: Path,
    ) -> None:
        self.stop()
        self._active_port = self.port if self.port > 0 else _pick_free_loopback_port()
        self.base_url = f"http://127.0.0.1:{self._active_port}"

        cmd = [
            binary,
            "-m", str(gguf_path),
            "-c", str(n_ctx),
            "-ngl", str(n_gpu_layers),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            *flag_set,
            "--port", str(self._active_port),
            "--host", "127.0.0.1",
            "--jinja",
        ]
        if self.threads > 0:
            cmd.extend(["-t", str(self.threads)])
        if self.moe_cpu_offload and n_gpu_layers < 999:
            cmd.extend(["-ot", ".ffn_.*_exps.=CPU"])
        if mmproj_path is not None:
            cmd.extend(["--mmproj", str(mmproj_path)])
        cmd.extend(self.extra_args)

        self._log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

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

    def start(self) -> None:
        if self.is_ready():
            return
        binary = find_llama_server_binary()
        gguf_path = find_primary_gguf_file(self.model_path)
        mmproj_path = find_mmproj_file(self.model_path)

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

        perf_flags: list[str] = []
        if self.cache_reuse > 0:
            perf_flags.extend(["--cache-reuse", str(self.cache_reuse)])
        if self.reasoning_format:
            perf_flags.extend(["--reasoning-format", self.reasoning_format])

        # Try full perf + Flash Attention + Q8_0 KV cache first; step down gracefully for hybrid
        # architectures (e.g. Qwen 3.5 Gated DeltaNet) or tight VRAM (smaller n_ctx / partial -ngl).
        ctx_primary = max(4096, self.n_ctx)
        ctx_stepdown = max(8192, ctx_primary // 2) if ctx_primary > 8192 else ctx_primary
        ngl_primary = self.n_gpu_layers
        ngl_stepdown = int(ngl_primary * 0.8) if 0 < ngl_primary < 999 else ngl_primary

        attempts: list[tuple[int, int, list[str]]] = []
        if self.allow_quantized_kv:
            attempts.append((ctx_primary, ngl_primary, ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", *perf_flags]))
        attempts.extend([
            (ctx_primary, ngl_primary, ["-fa", "on", *perf_flags]),
            (ctx_primary, ngl_primary, ["-fa", "on"]),
            (ctx_stepdown, ngl_stepdown, ["-fa", "on"]),
            (ctx_stepdown, ngl_stepdown, []),
        ])
        last_exc: Optional[Exception] = None
        for ctx_val, ngl_val, flags in attempts:
            try:
                self._launch_with_flags(
                    binary=binary,
                    gguf_path=gguf_path,
                    mmproj_path=mmproj_path,
                    n_ctx=ctx_val,
                    n_gpu_layers=ngl_val,
                    flag_set=flags,
                    env=env,
                    log_path=log_path,
                )
                return
            except RuntimeError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc

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
        deserialize_args: bool = False,
    ) -> list[dict[str, Any]]:
        """Normalize OpenAI messages for llama-server Jinja templates."""
        source_messages = deserialize_tool_call_arguments(messages) if deserialize_args else messages
        prepared: list[dict[str, Any]] = []
        for raw in source_messages:
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
            tool_block = build_tools_instruction(inject_tools)
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
        raw_tools = req.get("tools") or []
        tools = normalize_tools_for_jinja(raw_tools) if raw_tools else []
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
            if exc.response.status_code in (400, 500):
                # Retry once with deserialized tool_call arguments for templates using `|items`.
                if tools:
                    try:
                        dict_args_payload: dict[str, Any] = {
                            "messages": self._prepare_messages(messages, deserialize_args=True),
                            "stream": True,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "tools": tools,
                        }
                        yield from self._stream_once(dict_args_payload)
                        return
                    except httpx.HTTPStatusError:
                        pass
                    # Final fallback when a GGUF's embedded Jinja template rejects OpenAI `tools`
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
