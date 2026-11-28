"""Dynamic installer for model-specific dependencies (llama-server, bitsandbytes, deepseek-vl2)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

FALLBACK_LLAMA_TAG = "b12808"
LLAMA_INSTALL_DIR = Path("/content/llama-cpp-cuda")


def _find_installed_llama_server() -> Optional[Path]:
    for cand in (
        shutil.which("llama-server"),
        "/usr/local/bin/llama-server",
    ):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return Path(cand).resolve()
    if LLAMA_INSTALL_DIR.is_dir():
        for p in sorted(LLAMA_INSTALL_DIR.rglob("llama-server")):
            if p.is_file() and os.access(p, os.X_OK):
                return p.resolve()
    return None


def _resolve_latest_llama_cuda_urls() -> tuple[str, str, Optional[str]]:
    """Resolve the newest official Ubuntu CUDA 12.8 llama.cpp release tarball URLs."""
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10",
            headers={"User-Agent": "nexus-colab-setup/1.2.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            assets = rel.get("assets") or []
            bin_asset = next(
                (
                    a
                    for a in assets
                    if a.get("name", "").startswith("llama-")
                    and a.get("name", "").endswith("-bin-ubuntu-cuda-12.8-x64.tar.gz")
                ),
                None,
            )
            if bin_asset:
                tag = str(rel.get("tag_name") or FALLBACK_LLAMA_TAG)
                cudart_asset = next(
                    (
                        a
                        for a in assets
                        if a.get("name", "").startswith("cudart-llama-")
                        and a.get("name", "").endswith("-bin-ubuntu-cuda-12.8-x64.tar.gz")
                    ),
                    None,
                )
                return (
                    tag,
                    str(bin_asset["browser_download_url"]),
                    str(cudart_asset["browser_download_url"]) if cudart_asset else None,
                )
    except Exception as exc:
        print(f"[DEPS] GitHub API lookup fallback ({exc}); using pinned {FALLBACK_LLAMA_TAG}.")

    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{FALLBACK_LLAMA_TAG}"
    return (
        FALLBACK_LLAMA_TAG,
        f"{base}/llama-{FALLBACK_LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        f"{base}/cudart-llama-{FALLBACK_LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    )


def _verify_llama_cuda_support(binary: Path) -> None:
    """Verify CUDA GPU backend is active in `llama-server` when a CUDA GPU is present."""
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return
    except Exception:
        return
    env = os.environ.copy()
    lib_dirs = sorted({str(p.parent) for p in LLAMA_INSTALL_DIR.rglob("*.so*") if p.is_file()})
    if lib_dirs:
        cur_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, *([cur_ld] if cur_ld else [])])
    res = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, env=env, check=False)
    output = (res.stdout or "") + "\n" + (res.stderr or "")
    if "CUDA" not in output.upper():
        raise RuntimeError(
            f"llama-server ({binary}) did not report a CUDA device (--list-devices output:\n{output.strip()}).\n"
            "Refusing silent CPU-only fallback for large GGUF inference."
        )


MIN_LLAMA_SERVER_BUILD = 5100


def _check_llama_server_min_build(binary: Path) -> bool:
    """Return True if `llama-server --version` build is >= MIN_LLAMA_SERVER_BUILD (required for Llama 4)."""
    import re

    env = os.environ.copy()
    lib_dirs = sorted({str(p.parent) for p in LLAMA_INSTALL_DIR.rglob("*.so*") if p.is_file()}) if LLAMA_INSTALL_DIR.is_dir() else []
    if lib_dirs:
        cur_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, *([cur_ld] if cur_ld else [])])
    try:
        res = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env, timeout=10, check=False)
        output = (res.stdout or "") + "\n" + (res.stderr or "")
        m = re.search(r"\bversion:\s*(\d{4,6})\b", output, re.IGNORECASE) or re.search(r"\bb(\d{4,6})\b", output)
        if m is not None:
            return int(m.group(1)) >= MIN_LLAMA_SERVER_BUILD
    except Exception:
        return False
    return True


def ensure_llama_server() -> None:
    """Download and extract pre-built official CUDA llama-server binary if missing or outdated (< b5100)."""
    existing = _find_installed_llama_server()
    if existing is not None and _check_llama_server_min_build(existing):
        print(f"[DEPS] llama-server already available: {existing}")
        _verify_llama_cuda_support(existing)
        return
    if existing is not None:
        print(f"[DEPS] Existing llama-server ({existing}) is older than b{MIN_LLAMA_SERVER_BUILD}; upgrading for Llama 4 support...")

    tag, bin_url, cudart_url = _resolve_latest_llama_cuda_urls()
    print(f"[DEPS] llama-server not found. Downloading llama.cpp {tag} CUDA 12.8 binaries...")

    tar_path = Path("/content/llama-cuda.tar.gz")
    cudart_tar_path = Path("/content/llama-cudart.tar.gz")
    if LLAMA_INSTALL_DIR.exists():
        shutil.rmtree(LLAMA_INSTALL_DIR)
    LLAMA_INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(["curl", "-fSL", "--retry", "3", "-o", str(tar_path), bin_url], check=True)
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(LLAMA_INSTALL_DIR)], check=True)
    tar_path.unlink(missing_ok=True)

    if cudart_url:
        try:
            subprocess.run(["curl", "-fSL", "--retry", "3", "-o", str(cudart_tar_path), cudart_url], check=True)
            subprocess.run(["tar", "-xzf", str(cudart_tar_path), "-C", str(LLAMA_INSTALL_DIR)], check=True)
            cudart_tar_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[DEPS] Note: optional cudart tarball skipped ({exc}).")

    binary = next((p for p in sorted(LLAMA_INSTALL_DIR.rglob("llama-server")) if p.is_file()), None)
    if binary is None:
        raise RuntimeError(f"llama-server binary not found after extracting {bin_url} into {LLAMA_INSTALL_DIR}")

    os.chmod(binary, 0o755)
    target_symlink = Path("/usr/local/bin/llama-server")
    try:
        if target_symlink.exists() or target_symlink.is_symlink():
            target_symlink.unlink()
        target_symlink.symlink_to(binary)
    except Exception:
        pass

    # Export shared library directories so llama-server finds libllama.so / libggml-cuda.so / libcudart.so
    lib_dirs = sorted({str(p.parent) for p in LLAMA_INSTALL_DIR.rglob("*.so*") if p.is_file()})
    if lib_dirs:
        cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, *([cur_ld] if cur_ld else [])])

    print(f"[DEPS] Successfully installed llama-server ({tag}): {binary}")
    _verify_llama_cuda_support(binary)


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


def ensure_deepseek_vl() -> None:
    """Ensure deepseek-vl package is installed if running DeepSeek-VL-7B-Chat."""
    try:
        import deepseek_vl  # noqa: F401
    except ImportError:
        print("[DEPS] Installing deepseek-vl...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/deepseek-ai/DeepSeek-VL.git"], check=False)


def prepare_dependencies_for_model(implementation: str) -> None:
    """Check and fetch required runtime packages for the chosen implementation."""
    if implementation.startswith("gguf_"):
        ensure_llama_server()
    elif implementation in ("gemma3", "deepseek_vl2", "deepseek_vl", "gpt_oss"):
        ensure_bitsandbytes()
        if implementation == "deepseek_vl2":
            ensure_deepseek_vl2()
        elif implementation == "deepseek_vl":
            ensure_deepseek_vl()
