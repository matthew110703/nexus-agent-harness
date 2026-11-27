"""Dynamic installer for model-specific dependencies (llama-server, bitsandbytes, deepseek-vl2)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_llama_server() -> None:
    """Download and extract pre-built official CUDA llama-server binary if missing."""
    if shutil.which("llama-server") or list(Path("/content").glob("llama-*-bin-*/build/bin/llama-server")):
        return

    print("[DEPS] llama-server not found. Downloading llama.cpp CUDA 12.8 binaries...")
    url = "https://github.com/ggml-org/llama.cpp/releases/download/b4816/llama-b4816-bin-ubuntu-cuda-12.8-x64.tar.gz"
    tar_path = Path("/content/llama-cuda.tar.gz")
    dest_dir = Path("/content/llama-b4816-bin-ubuntu-cuda-12.8-x64")

    subprocess.run(["curl", "-L", "-o", str(tar_path), url], check=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(dest_dir)], check=True)

    binary = dest_dir / "build" / "bin" / "llama-server"
    if binary.is_file():
        os.chmod(binary, 0o755)
        # Symlink to /usr/local/bin for convenience
        try:
            target_symlink = Path("/usr/local/bin/llama-server")
            if not target_symlink.exists():
                target_symlink.symlink_to(binary)
        except Exception:
            pass
        print(f"[DEPS] Successfully installed llama-server: {binary}")
    else:
        print("[DEPS] Warning: llama-server binary extraction did not find build/bin/llama-server")


def ensure_bitsandbytes() -> None:
    """Ensure bitsandbytes is installed for 4-bit NF4 quantized inference."""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("[DEPS] Installing bitsandbytes for 4-bit GPU inference...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0"], check=True)


def ensure_deepseek_vl2() -> None:
    """Ensure deepseek-vl2 package is installed if running DeepSeek-VL2 models."""
    try:
        import deepseek_vl2  # noqa: F401
    except ImportError:
        print("[DEPS] Installing deepseek-vl2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/deepseek-ai/DeepSeek-VL2.git"], check=False)


def prepare_dependencies_for_model(implementation: str) -> None:
    """Check and fetch required runtime packages for the chosen implementation."""
    if implementation.startswith("gguf_"):
        ensure_llama_server()
    elif implementation in ("gemma3", "deepseek_vl2", "gpt_oss"):
        ensure_bitsandbytes()
        if implementation == "deepseek_vl2":
            ensure_deepseek_vl2()
