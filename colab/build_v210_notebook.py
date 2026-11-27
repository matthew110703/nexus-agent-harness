"""Reproducible generator for Nexus_Model_Lab_03_Nexus_Harness_Fork_Integration_v2.1.0.ipynb."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def _md(text: str) -> dict[str, Any]:
    lines = text.strip("\n").splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def _code(text: str) -> dict[str, Any]:
    body = text.strip("\n")
    ast.parse(body)
    lines = body.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_v210_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        _md(
            """
# Nexus AI Lab — Final Chat Model Expansion, Max VRAM+RAM & Co-Resident Harness (v2.1.0)

Production **v2.1.0** Colab-first runtime connecting **Google Drive Inventory + GGUF Header Inspector → VRAM+RAM MemoryPlanner → Fast SSD Staging & Split-GGUF Reassembly → Loopback Nexus Gateway (`127.0.0.1:7861`) → Dual-Slot Co-Resident GPU Worker → Locked DeepSeek Harness (`127.0.0.1:3080`)**.

### Highlights of v2.1.0
- **New & Future GGUF Models Supported Natively**:
  - `Qwen3.5-35B-A3B-Uncensored-Q4` (19.7 GB — L4 24GB & A100)
  - `Llama-3.1-70B-Abliterated-Q3` (~35 GB — A100 80GB or hybrid GPU+RAM offload)
  - `Llama-4-Scout-Abliterated-Q3` (~52 GB — automatic `.gguf.part1ofN` SSD stream-reassembly, A100 80GB or High-RAM offload)
  - Plus **any future GGUF** dropped into `MyDrive/Nexus/models/` via pure-Python GGUF binary header inspection (`gguf_generic`), with automatic `0 B` / incomplete download rejection.
- **Max VRAM + System RAM Performance (Zero OOM)**:
  - `MemoryPlanner` classifies every model (`RECOMMENDED` / `CPU_OFFLOAD_POSSIBLE` / `INSUFFICIENT_MEMORY`), maximizes GPU layer placement (`-ngl`), enables `llama-server --cache-reuse 256 -b 2048 -ub 512`, `sdpa` + TF32 on PyTorch, and expandable CUDA segments.
- **Co-Resident Language + Image GPU Worker**:
  - When live free VRAM fits your selected image model (`+ 3.5 GiB` headroom), both models stay resident in GPU VRAM simultaneously (instant ~4–10s image generation, zero LLM unload/reload), falling back automatically to swap-and-restore on smaller GPUs or OOM.
- **Bulletproof Tool Calling & Live GPT-OSS Streaming**:
  - Fixed GPT-OSS Harmony channel regex (`<|channel|>commentary to=functions.<name> <|constrain|>json`), stop tokens (`<|call|>`, `<|return|>`), and live `analysis`/`final` streaming; added universal GGUF tool parsing (`<tool_call>`, Llama `<|python_tag|>`, Gemma 4 `<|tool_call|>`); compacted prompt-tool schemas (~75% fewer tokens).
"""
        ),
        _md("## 1. Configuration and paths"),
        _code(
            """
# @title
import os
import secrets
from pathlib import Path

NEXUS_DRIVE_ROOT = Path("/content/drive/MyDrive/Nexus")
NEXUS_HARNESS_REPO_URL = "https://github.com/matthew110703/nexus-agent-harness.git"
NEXUS_HARNESS_GIT_REF = ""  # Empty -> latest origin/main

NEXUS_HARNESS_DIR = Path("/content/nexus-agent-harness")
NEXUS_STAGE_DIR = Path("/content/nexus-staged-models")
NEXUS_SESSION_MANIFEST = NEXUS_STAGE_DIR / "session-runtime.json"
NEXUS_WORKSPACE_ROOT = NEXUS_DRIVE_ROOT / "workspaces" / "projects"
NEXUS_SHARED_WORKSPACE_ROOT = NEXUS_DRIVE_ROOT / "workspaces" / "shared"
NEXUS_OUTPUTS_ROOT = NEXUS_DRIVE_ROOT / "outputs"
NEXUS_IMAGE_OUTPUT_DIR = NEXUS_OUTPUTS_ROOT / "generated_images"
NEXUS_HARNESS_HOME = Path("/content/nexus-harness-home")
NEXUS_HARNESS_HOME_DRIVE_BACKUP = NEXUS_DRIVE_ROOT / "harness-home"
NEXUS_HF_CACHE_DIR = Path("/content/nexus-hf-cache")
NEXUS_COLAB_SRC_DIR = NEXUS_DRIVE_ROOT / "colab-src"
NEXUS_SYNC_MANIFEST = NEXUS_COLAB_SRC_DIR / "sync-manifest.json"
NEXUS_RUNTIME_MANIFEST_DIR = NEXUS_DRIVE_ROOT / "runtime-manifests"
NEXUS_BUILD_CACHE_DIR = NEXUS_DRIVE_ROOT / "build-cache"

NEXUS_GATEWAY_HOST = "127.0.0.1"
NEXUS_GATEWAY_PORT = 7861
NEXUS_HARNESS_PORT = 3080
NEXUS_GATEWAY_ROOT_URL = f"http://{NEXUS_GATEWAY_HOST}:{NEXUS_GATEWAY_PORT}"
NEXUS_GATEWAY_URL = f"{NEXUS_GATEWAY_ROOT_URL}/v1"

NEXUS_API_KEY = os.environ.get("NEXUS_API_KEY") or secrets.token_urlsafe(32)
os.environ["NEXUS_API_KEY"] = NEXUS_API_KEY
os.environ["NEXUS_DRIVE_ROOT"] = str(NEXUS_DRIVE_ROOT)
os.environ["NEXUS_STAGE_DIR"] = str(NEXUS_STAGE_DIR)
os.environ["NEXUS_SESSION_MANIFEST"] = str(NEXUS_SESSION_MANIFEST)
os.environ["NEXUS_IMAGE_OUTPUT_DIR"] = str(NEXUS_IMAGE_OUTPUT_DIR)
os.environ["NEXUS_WORKSPACE_ROOT"] = str(NEXUS_WORKSPACE_ROOT)
os.environ["NEXUS_SHARED_WORKSPACE_ROOT"] = str(NEXUS_SHARED_WORKSPACE_ROOT)
os.environ["NEXUS_OUTPUTS_ROOT"] = str(NEXUS_OUTPUTS_ROOT)
os.environ["NEXUS_GATEWAY_HOST"] = NEXUS_GATEWAY_HOST
os.environ["NEXUS_GATEWAY_PORT"] = str(NEXUS_GATEWAY_PORT)
os.environ["NEXUS_GATEWAY_ROOT_URL"] = NEXUS_GATEWAY_ROOT_URL
os.environ["NEXUS_GATEWAY_URL"] = NEXUS_GATEWAY_URL
os.environ["NEXUS_HARNESS_PORT"] = str(NEXUS_HARNESS_PORT)
os.environ["NEXUS_PLUGIN_INDEX_URL"] = f"{NEXUS_GATEWAY_ROOT_URL}/v1/plugin-index.json"
os.environ["HF_HOME"] = str(NEXUS_HF_CACHE_DIR)
os.environ["DSH_HOME"] = str(NEXUS_HARNESS_HOME)
os.environ["ELECTRON_SKIP_BINARY_DOWNLOAD"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

NEXUS_MANAGED_PIDS = {}
print("[CONFIG] Nexus v2.1.0 paths configured.")
"""
        ),
        _md("## 2. Mount Drive, Checkout Harness, Detect GPU+RAM & List Available Models"),
        _code(
            """
# @title
import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
assert NEXUS_DRIVE_ROOT.is_dir(), f"Expected Drive directory not found: {NEXUS_DRIVE_ROOT}"

for rel in (
    "colab-src/nexus-gateway",
    "colab-src/supervisor",
    "colab-src/scripts",
    "colab-src/config",
    "workspaces/projects/default-workspace",
    "workspaces/shared",
    "harness-home/snapshots",
    "runtime-manifests/source",
    "runtime-manifests/staging",
    "runtime-manifests/runtime",
    "build-cache",
):
    (NEXUS_DRIVE_ROOT / rel).mkdir(parents=True, exist_ok=True)
for local_dir in (NEXUS_IMAGE_OUTPUT_DIR, NEXUS_STAGE_DIR, NEXUS_HARNESS_HOME, NEXUS_HF_CACHE_DIR):
    local_dir.mkdir(parents=True, exist_ok=True)

if not NEXUS_HARNESS_DIR.exists():
    subprocess.run(
        ["git", "clone", "--branch", "main", "--single-branch", NEXUS_HARNESS_REPO_URL, str(NEXUS_HARNESS_DIR)],
        check=True,
    )
else:
    dirty = subprocess.check_output(["git", "-C", str(NEXUS_HARNESS_DIR), "status", "--porcelain"], text=True).strip()
    if dirty:
        raise RuntimeError("Refusing to overwrite dirty runtime checkout:\\n" + dirty)
subprocess.run(["git", "-C", str(NEXUS_HARNESS_DIR), "fetch", "origin", "main"], check=True)
target_ref = NEXUS_HARNESS_GIT_REF.strip() if NEXUS_HARNESS_GIT_REF.strip() else "origin/main"
resolved_sha = subprocess.check_output(["git", "-C", str(NEXUS_HARNESS_DIR), "rev-parse", target_ref], text=True).strip()
subprocess.run(["git", "-C", str(NEXUS_HARNESS_DIR), "checkout", "--detach", resolved_sha], check=True)
NEXUS_ACTIVE_COMMIT = resolved_sha
print(f"[GIT] Checked out {target_ref} -> {NEXUS_ACTIVE_COMMIT}")

for p in (NEXUS_HARNESS_DIR / "colab", NEXUS_HARNESS_DIR / "nexus-gateway" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import sync_source
sync_source = importlib.reload(sync_source)
sync_source.seed_drive_baseline(
    NEXUS_HARNESS_DIR,
    NEXUS_COLAB_SRC_DIR,
    NEXUS_SYNC_MANIFEST,
    git_commit=NEXUS_ACTIVE_COMMIT,
    rel_pairs=[
        ("supervisor/supervisor.mjs", "colab/supervisor.mjs"),
        ("scripts/sync_source.py", "colab/sync_source.py"),
        ("scripts/harness_home_snapshot.py", "colab/harness_home_snapshot.py"),
        ("scripts/model_inventory.py", "colab/model_inventory.py"),
        ("scripts/runtime_deps.py", "colab/runtime_deps.py"),
        ("scripts/drive_fast_transfer.py", "colab/drive_fast_transfer.py"),
        ("scripts/build_cache.py", "colab/build_cache.py"),
        ("scripts/staging.py", "colab/staging.py"),
        ("config/colab.overlay.yml", "apps/cli/config/examples/colab.overlay.yml"),
        ("config/plugin-index.json", "colab/plugin-index.json"),
    ],
)
sync_report = sync_source.apply_sync(
    NEXUS_COLAB_SRC_DIR,
    NEXUS_HARNESS_DIR,
    NEXUS_SYNC_MANIFEST,
    git_commit=NEXUS_ACTIVE_COMMIT,
    dry_run=False,
)
if sync_report.errors:
    raise RuntimeError("Source sync halted:\\n" + "\\n".join(sync_report.errors))

from core.hardware_detector import detect_hardware
import model_inventory
model_inventory = importlib.reload(model_inventory)

NEXUS_HARDWARE = detect_hardware(Path("/content"))
gpu_label = (
    f"{NEXUS_HARDWARE.gpus[0].name} ({NEXUS_HARDWARE.total_vram_bytes / (1024**3):.1f} GiB VRAM, "
    f"{NEXUS_HARDWARE.system_ram_available_bytes / (1024**3):.1f} GiB free RAM, tier={NEXUS_HARDWARE.tier})"
    if NEXUS_HARDWARE.gpus
    else "CPU only (no CUDA GPU detected)"
)
NEXUS_INVENTORY = model_inventory.inspect_drive_models(
    NEXUS_DRIVE_ROOT / "models",
    total_vram_bytes=NEXUS_HARDWARE.total_vram_bytes,
    available_ram_bytes=NEXUS_HARDWARE.system_ram_available_bytes,
    cpu_count=NEXUS_HARDWARE.cpu_count,
)
print(model_inventory.format_inventory_report(NEXUS_INVENTORY, gpu_summary=gpu_label))
print(f"\\n[DISK] Local /content free: {shutil.disk_usage('/content').free / (1024**3):.1f} GiB")
"""
        ),
        _md(
            """
## 3. Select Your Single Language Model, Image Model & Tuning Overrides

Copy **one** language model name and **one** image model name (or leave `SELECTED_IMAGE_MODEL = ""` to disable image generation) from the inventory output above. Leave numeric overrides at `0` / `-1` to let `MemoryPlanner` pick optimal values for your GPU + System RAM.
"""
        ),
        _code(
            """
# @title
import importlib
import shutil

import runtime_deps
runtime_deps = importlib.reload(runtime_deps)

SELECTED_LANGUAGE_MODEL = "Qwen3.5-35B-A3B-Uncensored-Q4"  # @param {type:"string"}
SELECTED_IMAGE_MODEL = "LUSTIFY-v2.0"  # @param {type:"string"}

# Advanced MemoryPlanner overrides (0 / -1 = auto-detect optimal for current GPU+RAM)
CONTEXT_LENGTH_OVERRIDE = 0  # @param {type:"integer"}
GPU_LAYERS_OVERRIDE = -1  # @param {type:"integer"}
BATCH_SIZE_OVERRIDE = 0  # @param {type:"integer"}
ALLOW_CPU_OFFLOAD = True  # @param {type:"boolean"}
ENABLE_IMAGE_CORESIDENCY = True  # @param {type:"boolean"}

NEXUS_SELECTED_LANGUAGE, NEXUS_SELECTED_IMAGE = model_inventory.validate_selected_pair(
    NEXUS_DRIVE_ROOT / "models",
    SELECTED_LANGUAGE_MODEL,
    SELECTED_IMAGE_MODEL,
    total_vram_bytes=NEXUS_HARDWARE.total_vram_bytes,
    available_ram_bytes=NEXUS_HARDWARE.system_ram_available_bytes,
    cpu_count=NEXUS_HARDWARE.cpu_count,
    free_disk_bytes=shutil.disk_usage("/content").free,
    enforce_gpu_fit=NEXUS_HARDWARE.cuda_available,
    n_ctx_override=CONTEXT_LENGTH_OVERRIDE,
    n_gpu_layers_override=GPU_LAYERS_OVERRIDE,
    batch_size_override=BATCH_SIZE_OVERRIDE,
    allow_cpu_offload=ALLOW_CPU_OFFLOAD,
    enable_image_coresidency=ENABLE_IMAGE_CORESIDENCY,
)
print(
    f"[SELECT] Language model locked: {NEXUS_SELECTED_LANGUAGE.id} "
    f"(impl={NEXUS_SELECTED_LANGUAGE.implementation}, quant={NEXUS_SELECTED_LANGUAGE.quantization}, "
    f"{NEXUS_SELECTED_LANGUAGE.size_bytes / (1024**3):.2f} GiB, tier={NEXUS_SELECTED_LANGUAGE.compatibility_tier})"
)
if NEXUS_SELECTED_LANGUAGE.n_ctx is not None:
    print(
        f"[PLAN]   n_ctx={NEXUS_SELECTED_LANGUAGE.n_ctx}, -ngl={NEXUS_SELECTED_LANGUAGE.n_gpu_layers}, "
        f"batch={NEXUS_SELECTED_LANGUAGE.batch_size}, threads={NEXUS_SELECTED_LANGUAGE.threads}, "
        f"co-resident-eligible={NEXUS_SELECTED_LANGUAGE.can_coreside_image}"
    )
if NEXUS_SELECTED_IMAGE is not None:
    print(
        f"[SELECT] Image model locked:    {NEXUS_SELECTED_IMAGE.id} "
        f"(impl={NEXUS_SELECTED_IMAGE.implementation}, {NEXUS_SELECTED_IMAGE.size_bytes / (1024**3):.2f} GiB)"
    )
else:
    print("[SELECT] Image model: disabled for this session.")

runtime_deps.prepare_dependencies_for_model(NEXUS_SELECTED_LANGUAGE.implementation)
"""
        ),
        _md("## 4. Fast Web-Only Harness Build (Skip Desktop + Drive Build Cache)"),
        _code(
            """
# @title
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
assert torch.cuda.is_available(), "CUDA GPU runtime is required for Nexus AI Lab v2.1.0."
print(f"[DEPS] Torch {torch.__version__} | CUDA={torch.version.cuda}")

USE_DRIVE_BUILD_CACHE = True  # @param {type:"boolean"}

LIGHTWEIGHT_PY = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "accelerate": "accelerate",
    "PIL": "pillow",
    "xxhash": "xxhash",
}
missing_pkgs = []
for mod, pip_name in LIGHTWEIGHT_PY.items():
    try:
        __import__(mod)
    except ImportError:
        missing_pkgs.append(pip_name)
if missing_pkgs:
    subprocess.run(["pip", "install", "-q", *missing_pkgs], check=True)
    print("[DEPS] Installed:", missing_pkgs)
else:
    print("[DEPS] Python ML/Gateway dependencies verified.")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retry(url: str, dest: Path, *, retries: int = 4, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nexus-colab-setup/2.1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            os.replace(tmp, dest)
            return dest
        except Exception as exc:
            last_exc = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {dest.name}: {last_exc}") from last_exc


NODE_VERSION = "24.22.3"
NODE_ARCHIVE = f"node-v{NODE_VERSION}-linux-x64.tar.xz"
NODE_ROOT = Path("/content/nexus-node")
NODE_BIN = NODE_ROOT / f"node-v{NODE_VERSION}-linux-x64" / "bin"
if not (NODE_BIN / "node").is_file():
    shasums = _download_with_retry(f"https://nodejs.org/dist/v{NODE_VERSION}/SHASUMS256.txt", NODE_ROOT / "SHASUMS256.txt")
    expected_sha = next((ln.split()[0] for ln in shasums.read_text("utf-8").splitlines() if len(ln.split()) == 2 and ln.split()[1] == NODE_ARCHIVE), None)
    assert expected_sha, f"SHA-256 missing for {NODE_ARCHIVE}"
    archive = _download_with_retry(f"https://nodejs.org/dist/v{NODE_VERSION}/{NODE_ARCHIVE}", NODE_ROOT / NODE_ARCHIVE)
    assert _sha256_file(archive) == expected_sha, "Node SHA-256 mismatch"
    subprocess.run(["tar", "-xJf", str(archive), "-C", str(NODE_ROOT)], check=True)
    archive.unlink(missing_ok=True)
os.environ["PATH"] = f"{NODE_BIN}:{os.environ['PATH']}"
os.environ["ELECTRON_SKIP_BINARY_DOWNLOAD"] = "1"

root_pkg = json.loads((NEXUS_HARNESS_DIR / "package.json").read_text("utf-8"))
pkg_manager = str(root_pkg.get("packageManager", "pnpm@11.28.5"))
pinned_pnpm = pkg_manager.split("@", 1)[1] if pkg_manager.startswith("pnpm@") else "11.28.5"
cur_pnpm = subprocess.check_output(["pnpm", "--version"], text=True).strip() if shutil.which("pnpm") else ""
if cur_pnpm != pinned_pnpm:
    subprocess.run(["npm", "install", "-g", f"pnpm@{pinned_pnpm}"], check=True)

cf_path = Path("/usr/local/bin/cloudflared")
if shutil.which("cloudflared") is None or not cf_path.is_file() or cf_path.stat().st_size == 0:
    meta_path = Path("/content/cloudflared-release.json")
    _download_with_retry("https://api.github.com/repos/cloudflare/cloudflared/releases/latest", meta_path, timeout=30)
    rel_meta = json.loads(meta_path.read_text("utf-8"))
    asset = next((a for a in rel_meta.get("assets", []) if a.get("name") == "cloudflared-linux-amd64"), None)
    assert asset is not None, "cloudflared-linux-amd64 asset missing"
    digest_field = str(asset.get("digest") or "")
    expected_cf = digest_field.split(":", 1)[1].strip().lower() if digest_field.startswith("sha256:") else None
    if not expected_cf:
        m = re.search(r"cloudflared-linux-amd64[^\\n]*?([0-9a-fA-F]{64})", str(rel_meta.get("body") or ""))
        expected_cf = m.group(1).lower() if m else None
    assert expected_cf, "Missing official SHA-256 for cloudflared-linux-amd64"
    _download_with_retry(asset["browser_download_url"], cf_path, timeout=60)
    if _sha256_file(cf_path).lower() != expected_cf:
        cf_path.unlink(missing_ok=True)
        raise RuntimeError("cloudflared SHA-256 mismatch")
    cf_path.chmod(0o755)
    meta_path.unlink(missing_ok=True)

import build_cache
build_cache = importlib.reload(build_cache)

BUILD_STAMP = Path("/content/nexus-build-fingerprint.txt")
fp = sync_source.compute_build_fingerprint(NEXUS_HARNESS_DIR, NEXUS_COLAB_SRC_DIR, git_commit=NEXUS_ACTIVE_COMMIT)
if not (NEXUS_HARNESS_DIR / "node_modules").is_dir():
    subprocess.run(["pnpm", "install", "--frozen-lockfile"], cwd=NEXUS_HARNESS_DIR, check=True)

if (BUILD_STAMP.read_text("utf-8").strip() if BUILD_STAMP.is_file() else "") != fp:
    restored = False
    if USE_DRIVE_BUILD_CACHE:
        restored = build_cache.restore_build_cache(
            NEXUS_HARNESS_DIR,
            NEXUS_BUILD_CACHE_DIR,
            fp,
            on_status=print,
        )
    if not restored:
        print("[BUILD] Running fast web-only build (--scope web-artifacts --no-check, skipping Desktop/Electron)...")
        subprocess.run(
            ["pnpm", "exec", "tsx", "scripts/build.ts", "--scope", "web-artifacts", "--no-check"],
            cwd=NEXUS_HARNESS_DIR,
            check=True,
        )
        if USE_DRIVE_BUILD_CACHE:
            build_cache.save_build_cache(
                NEXUS_HARNESS_DIR,
                NEXUS_BUILD_CACHE_DIR,
                fp,
                keep_latest=2,
                on_status=print,
            )
    BUILD_STAMP.write_text(fp + "\\n", encoding="utf-8")
print(f"[BUILD] Harness Web artifacts ready ({fp[:12]}).")
"""
        ),
        _md("## 5. Fast SSD Staging, Start Gateway & Warm-Load Language (+ Co-Resident Image)"),
        _code(
            """
# @title
import importlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import drive_fast_transfer
drive_fast_transfer = importlib.reload(drive_fast_transfer)
import staging
staging = importlib.reload(staging)

STAGING_TRANSPORT = "auto"  # @param ["auto", "drive_api", "fuse"]
STAGING_VERIFY_MODE = "fast"  # @param ["fast", "local_full", "strict_source"]
STAGE_IMAGE_IN_BACKGROUND = True  # @param {type:"boolean"}

img_lock_path = (NEXUS_STAGE_DIR / f".{NEXUS_SELECTED_IMAGE.id}.staging.lock") if NEXUS_SELECTED_IMAGE is not None else None
if img_lock_path is not None:
    if STAGE_IMAGE_IN_BACKGROUND:
        img_lock_path.parent.mkdir(parents=True, exist_ok=True)
        img_lock_path.write_text("staging\\n", encoding="utf-8")
    else:
        img_lock_path.unlink(missing_ok=True)

NEXUS_SESSION_CONFIG = staging.stage_selected_models(
    local_stage_root=NEXUS_STAGE_DIR,
    language_candidate=NEXUS_SELECTED_LANGUAGE,
    image_candidate=NEXUS_SELECTED_IMAGE,
    manifest_path=NEXUS_SESSION_MANIFEST,
    drive_manifest_copy=NEXUS_RUNTIME_MANIFEST_DIR / "runtime" / "session-runtime.json",
    verify_mode=STAGING_VERIFY_MODE,
    transport=STAGING_TRANSPORT,
    defer_image_staging=STAGE_IMAGE_IN_BACKGROUND,
    on_status=print,
)

prev_gw = NEXUS_MANAGED_PIDS.pop("gateway", None)
if prev_gw is not None:
    try:
        req_un = urllib.request.Request(
            f"{NEXUS_GATEWAY_ROOT_URL}/v1/runtime/unload",
            data=b'{"force_terminate": true}',
            headers={"Authorization": f"Bearer {NEXUS_API_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req_un, timeout=10)
    except Exception:
        pass
    try:
        os.kill(prev_gw, signal.SIGTERM)
        time.sleep(1.0)
    except ProcessLookupError:
        pass

gw_log_path = Path("/content/nexus-gateway.log")
gw_log = open(gw_log_path, "w", encoding="utf-8")
gw_proc = subprocess.Popen(
    [
        "python3",
        str(NEXUS_HARNESS_DIR / "nexus-gateway" / "src" / "main.py"),
        "--host",
        NEXUS_GATEWAY_HOST,
        "--port",
        str(NEXUS_GATEWAY_PORT),
    ],
    stdout=gw_log,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
)
NEXUS_MANAGED_PIDS["gateway"] = gw_proc.pid

deadline = time.time() + 30
while time.time() < deadline:
    if gw_proc.poll() is not None:
        print(gw_log_path.read_text("utf-8", errors="replace")[-3000:])
        raise RuntimeError("Nexus Gateway exited early. Inspect /content/nexus-gateway.log")
    try:
        with urllib.request.urlopen(f"{NEXUS_GATEWAY_ROOT_URL}/health", timeout=2) as resp:
            if resp.status == 200:
                break
    except Exception:
        time.sleep(0.4)
else:
    gw_proc.terminate()
    NEXUS_MANAGED_PIDS.pop("gateway", None)
    raise RuntimeError("Nexus Gateway did not respond on /health within 30s.")

print(f"[BOOTSTRAP] Loading {NEXUS_SELECTED_LANGUAGE.id} into isolated GPU worker...")
try:
    req = urllib.request.Request(
        f"{NEXUS_GATEWAY_ROOT_URL}/v1/runtime/bootstrap",
        data=json.dumps({"model": NEXUS_SELECTED_LANGUAGE.id}).encode("utf-8"),
        headers={"Authorization": f"Bearer {NEXUS_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        boot = json.loads(resp.read().decode("utf-8"))
    rt_req = urllib.request.Request(
        f"{NEXUS_GATEWAY_ROOT_URL}/v1/runtime",
        headers={"Authorization": f"Bearer {NEXUS_API_KEY}"},
    )
    with urllib.request.urlopen(rt_req, timeout=15) as resp:
        rt = json.loads(resp.read().decode("utf-8"))
    assert rt.get("ready") is True and rt.get("worker", {}).get("active_model_id") == NEXUS_SELECTED_LANGUAGE.id
    used_gb = (rt["worker"].get("device_used_vram_bytes") or rt["worker"].get("cuda_allocated_bytes") or 0) / (1024 ** 3)
    print(f"[READY] Language model '{NEXUS_SELECTED_LANGUAGE.id}' is live in VRAM ({used_gb:.2f} GiB used).")
except Exception as exc:
    detail = ""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    if detail:
        print(f"[BOOTSTRAP ERROR BODY] {detail}")
    for lp in (gw_log_path, Path("/content/nexus-llama-server.log")):
        if lp.is_file():
            tail = lp.read_text("utf-8", errors="replace")[-2000:].strip()
            if tail:
                print(f"--- tail {lp.name} ---\\n{tail}")
    gw_proc.terminate()
    NEXUS_MANAGED_PIDS.pop("gateway", None)
    if img_lock_path is not None:
        img_lock_path.unlink(missing_ok=True)
    raise RuntimeError(f"Initial language model load failed; gateway terminated cleanly: {exc} {detail}") from exc


def _try_preload_coresident_image() -> None:
    if not ENABLE_IMAGE_CORESIDENCY or NEXUS_SELECTED_IMAGE is None:
        return
    try:
        pre_req = urllib.request.Request(
            f"{NEXUS_GATEWAY_ROOT_URL}/v1/runtime/preload-image",
            data=b"{}",
            headers={"Authorization": f"Bearer {NEXUS_API_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(pre_req, timeout=300) as resp:
            pre_res = json.loads(resp.read().decode("utf-8"))
        if pre_res.get("co_resident"):
            print(f"[CO-RESIDENT] '{NEXUS_SELECTED_IMAGE.id}' pre-loaded alongside '{NEXUS_SELECTED_LANGUAGE.id}' in VRAM!")
        else:
            print(f"[CO-RESIDENT] Skipped preload ({pre_res.get('reason')}); will swap automatically on image generation.")
    except Exception as pre_exc:
        print(f"[CO-RESIDENT] Preload note: {pre_exc}")


if STAGE_IMAGE_IN_BACKGROUND and NEXUS_SELECTED_IMAGE is not None and img_lock_path is not None:
    def _bg_stage_image() -> None:
        try:
            staging.stage_model_directory(
                Path(NEXUS_SELECTED_IMAGE.source_path),
                NEXUS_STAGE_DIR,
                model_id=NEXUS_SELECTED_IMAGE.id,
                verify_mode=STAGING_VERIFY_MODE,
                transport=STAGING_TRANSPORT,
                on_status=lambda msg: print(f"[BG-IMAGE] {msg}"),
            )
            img_lock_path.unlink(missing_ok=True)
            _try_preload_coresident_image()
        except Exception as bg_exc:
            print(f"[BG-IMAGE] Staging failed for {NEXUS_SELECTED_IMAGE.id}: {bg_exc}")
        finally:
            img_lock_path.unlink(missing_ok=True)

    bg_thread = threading.Thread(target=_bg_stage_image, daemon=True)
    bg_thread.start()
    print(f"[STAGE] Background staging started for image model '{NEXUS_SELECTED_IMAGE.id}'. You can run Section 6 now!")
elif NEXUS_SELECTED_IMAGE is not None:
    _try_preload_coresident_image()
"""
        ),
        _md("## 6. Restore Session Snapshot & Start Locked Single-Model Harness + Quick Tunnel"),
        _code(
            """
# @title
import importlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import harness_home_snapshot
harness_home_snapshot = importlib.reload(harness_home_snapshot)

restore_info = harness_home_snapshot.restore_latest_snapshot(
    NEXUS_HARNESS_HOME_DRIVE_BACKUP,
    NEXUS_HARNESS_HOME,
)
print("[SNAPSHOT] Restore before launch:", restore_info)

prev_sup = NEXUS_MANAGED_PIDS.pop("supervisor", None)
if prev_sup is not None:
    try:
        os.killpg(os.getpgid(prev_sup), signal.SIGTERM)
        time.sleep(1.5)
    except Exception:
        pass

sup_log_path = Path("/content/nexus-supervisor.log")
sup_log = open(sup_log_path, "w", encoding="utf-8")
sup_proc = subprocess.Popen(
    ["node", str(NEXUS_HARNESS_DIR / "colab" / "supervisor.mjs")],
    cwd=str(NEXUS_HARNESS_DIR),
    stdout=sup_log,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
    start_new_session=True,
)
NEXUS_MANAGED_PIDS["supervisor"] = sup_proc.pid

launch_url = None
deadline = time.time() + 120
offset = 0
while time.time() < deadline:
    if sup_proc.poll() is not None:
        print(sup_log_path.read_text("utf-8", errors="replace")[-3000:])
        raise RuntimeError(f"Supervisor exited with code {sup_proc.returncode}")
    text = sup_log_path.read_text("utf-8", errors="replace")
    if len(text) > offset:
        print(text[offset:], end="", flush=True)
        offset = len(text)
        m = re.search(r"Nexus Harness Ready:\\s*(https://[^\\s]+)", text)
        if m:
            launch_url = m.group(1)
            break
    time.sleep(0.5)

if not launch_url:
    raise RuntimeError("Timed out waiting for Harness Ready URL. Check /content/nexus-supervisor.log")
NEXUS_LAUNCH_URL = launch_url
print("\\n[READY] Open your private Nexus Harness URL:", NEXUS_LAUNCH_URL)
"""
        ),
        _md("## 7. Live v2.1.0 Acceptance Suite (Lock, Chat, Tool Call, Co-Resident / Swap Image)"),
        _code(
            """
# @title
import json
import urllib.error
import urllib.request

headers = {"Authorization": f"Bearer {NEXUS_API_KEY}", "Content-Type": "application/json"}
results = []


def _record(step: str, status: str, detail: str = "") -> None:
    safe = detail.replace(NEXUS_API_KEY, "<REDACTED>")
    results.append((step, status, safe))
    print(f"[{status}] {step}: {safe}")


def _get(path: str, auth: bool = True):
    req = urllib.request.Request(f"{NEXUS_GATEWAY_ROOT_URL}{path}", headers=headers if auth else {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict, timeout: int = 300):
    req = urllib.request.Request(
        f"{NEXUS_GATEWAY_ROOT_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


st, h = _get("/health", auth=False)
_record("01_health_ready", "PASS" if st == 200 and h.get("ready") is True else "FAIL", f"v={h.get('version')}, ready={h.get('ready')}")

st, rt = _get("/v1/runtime")
w = rt.get("worker") or {}
_record("02_runtime_ready", "PASS" if st == 200 and rt.get("ready") and w.get("active_model_id") == NEXUS_SELECTED_LANGUAGE.id else "FAIL", f"active={w.get('active_model_id')}")

st, models_payload = _get("/v1/models")
data = models_payload.get("data", [])
_record("03_single_model_catalog", "PASS" if len(data) == 1 and data[0]["id"] == NEXUS_SELECTED_LANGUAGE.id else "FAIL", f"count={len(data)}")

try:
    _post("/v1/runtime/bootstrap", {"model": "__unauthorized_second_model__"})
    _record("04_second_model_rejected", "FAIL", "Second model was accepted")
except urllib.error.HTTPError as err:
    _record("04_second_model_rejected", "PASS" if err.code == 409 else "FAIL", f"HTTP {err.code}")

st, chat_resp = _post(
    "/v1/chat/completions",
    {"model": NEXUS_SELECTED_LANGUAGE.id, "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 32},
    timeout=300,
)
_record("05_live_chat_completion", "PASS" if st == 200 and chat_resp.get("choices") else "FAIL", f"model={NEXUS_SELECTED_LANGUAGE.id}")

test_tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
st, tool_resp = _post(
    "/v1/chat/completions",
    {
        "model": NEXUS_SELECTED_LANGUAGE.id,
        "messages": [{"role": "user", "content": "Call get_weather for Tokyo."}],
        "tools": test_tools,
        "max_tokens": 256,
    },
    timeout=300,
)
msg_out = (tool_resp.get("choices") or [{}])[0].get("message") or {}
tcalls = msg_out.get("tool_calls") or []
_record("06_live_tool_call_roundtrip", "PASS" if st == 200 and len(tcalls) >= 1 else "FAIL", f"tool_calls={len(tcalls)}")

if NEXUS_SELECTED_IMAGE is not None:
    st, img_resp = _post(
        "/v1/images/generations",
        {"prompt": "minimal geometric prism on dark background", "quality": "normal", "num_images": 1},
        timeout=600,
    )
    ok_img = st == 200 and bool(img_resp.get("filename")) and bool(img_resp.get("b64_png"))
    _record("07_image_generation", "PASS" if ok_img else "FAIL", f"file={img_resp.get('filename')}")

    st, rt_after = _get("/v1/runtime")
    w_after = rt_after.get("worker") or {}
    _record(
        "08_llm_ready_after_image",
        "PASS" if rt_after.get("ready") and w_after.get("active_language_id") == NEXUS_SELECTED_LANGUAGE.id else "FAIL",
        f"co_resident={w_after.get('co_resident')}, active={w_after.get('active_language_id')}",
    )
else:
    _record("07_image_generation", "SKIP", "Image model disabled")
    _record("08_llm_ready_after_image", "SKIP", "Image model disabled")

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{NEXUS_HARNESS_PORT}/", timeout=10) as resp:
        _record("09_harness_responding", "PASS" if resp.status < 500 else "FAIL", f"HTTP {resp.status}")
except urllib.error.HTTPError as err:
    _record("09_harness_responding", "PASS" if err.code in (200, 302, 401, 403) else "FAIL", f"HTTP {err.code}")

print(
    f"\\n[SUMMARY] total={len(results)} pass={sum(1 for r in results if r[1]=='PASS')} "
    f"skip={sum(1 for r in results if r[1]=='SKIP')} fail={sum(1 for r in results if r[1]=='FAIL')}"
)
"""
        ),
        _md("## 8. Graceful Shutdown & Versioned Secret-Excluding Snapshot Backup"),
        _code(
            """
# @title
import os
import signal
import time
import urllib.request

try:
    req = urllib.request.Request(
        f"{NEXUS_GATEWAY_ROOT_URL}/v1/runtime/unload",
        data=b'{"force_terminate": true}',
        headers={"Authorization": f"Bearer {NEXUS_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15)
    print("[STOP] Isolated GPU worker terminated cleanly.")
except Exception as exc:
    print("[STOP] Worker unload note:", exc)

for label in ("supervisor", "gateway"):
    pid = NEXUS_MANAGED_PIDS.pop(label, None)
    if pid is None:
        continue
    try:
        if label == "supervisor":
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        print(f"[STOP] Stopped {label} (PID {pid}).")
    except ProcessLookupError:
        pass

time.sleep(1.5)
if "harness_home_snapshot" in globals():
    snap = harness_home_snapshot.create_snapshot(NEXUS_HARNESS_HOME, NEXUS_HARNESS_HOME_DRIVE_BACKUP)
    print("[SNAPSHOT] Verified Drive snapshot saved:", snap)
"""
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "L4"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def write_v210_notebook(output_path: Path) -> Path:
    nb = build_v210_notebook()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "Nexus_Model_Lab_03_Nexus_Harness_Fork_Integration_v2.1.0.ipynb"
    write_v210_notebook(out)
    print(f"Wrote {out}")
