"""Reproducible generator for Nexus_Model_Lab_03_Nexus_Harness_Fork_Integration_v1.2.0.ipynb."""

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


def build_v120_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        _md(
            """
# Nexus AI Lab — Single-Model Colab Runtime & Dedicated Harness (v1.2.0)

Colab-first, single-active-model runtime connecting **Google Drive Inventory → Pre-Run Model Selection → Local SSD Staging → Loopback Nexus Gateway (`127.0.0.1:7861`) → Isolated GPU Worker (Dedicated Lazy-Imported Model Modules) → Locked Single-Model Nexus Harness (`127.0.0.1:3080`)**.

- **What's New in v1.2.0**:
  - **GGUF Models (`llama-server`)**: `Qwen3.5-9B-Defiant-Fable-Uncensored`, `Gemma4-26B-A4B-QAT-Uncensored-MTP`, `DeepSeek-V4-Pro-Qwen3.5-9B-Uncensored`.
  - **Safetensors Language / VLM Models**: `Qwen3.5-4B`, `Gemma-3-12B-IT` (auto 4-bit NF4 on L4), `DeepSeek-VL2-Small` (auto 4-bit NF4 on L4), `DeepSeek-VL-7B-Chat`, `GPT-OSS-20B-Abliterated` (auto 4-bit NF4 on L4).
  - **SDXL FP16 & Single-File Fix**: Resolves `shootstuff/LUSTIFY-v2.0` (`variant="fp16"` + monolithic `.safetensors` fallback).
  - **Workspace-Scoped Image Persistence**: Generated images save into `<active_workspace>/generated_images/` on Drive, and Harness snapshots preserve `storages/` across Colab resets.
- **Two-cell selection flow**: Run **Section 2** to mount Drive, inspect your GPU, and list all runnable language & image models in `MyDrive/Nexus/models/`, then paste **one** language model and **one** image model into **Section 3**.
- **No mid-session model switching**: Harness is locked to the single selected language model and single selected image model. To change models, update **Section 3** and re-run from Section 3 (or restart the runtime).
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

NEXUS_MANAGED_PIDS = {}
print("[CONFIG] Nexus v1.2.0 paths configured.")
"""
        ),
        _md("## 2. Mount Drive, Checkout Harness, Detect GPU & List Available Models"),
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
):
    (NEXUS_DRIVE_ROOT / rel).mkdir(parents=True, exist_ok=True)
for local_dir in (NEXUS_IMAGE_OUTPUT_DIR, NEXUS_STAGE_DIR, NEXUS_HARNESS_HOME, NEXUS_HF_CACHE_DIR):
    local_dir.mkdir(parents=True, exist_ok=True)

# Clone or fast-forward Harness checkout (refuse dirty tree).
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

# Sync Drive colab-src/ before running inventory.
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
    f"{NEXUS_HARDWARE.gpus[0].name} ({NEXUS_HARDWARE.total_vram_bytes / (1024**3):.1f} GiB VRAM, tier={NEXUS_HARDWARE.tier})"
    if NEXUS_HARDWARE.gpus
    else "CPU only (no CUDA GPU detected)"
)
NEXUS_INVENTORY = model_inventory.inspect_drive_models(
    NEXUS_DRIVE_ROOT / "models",
    total_vram_bytes=NEXUS_HARDWARE.total_vram_bytes,
)
print(model_inventory.format_inventory_report(NEXUS_INVENTORY, gpu_summary=gpu_label))
print(f"\\n[DISK] Local /content free: {shutil.disk_usage('/content').free / (1024**3):.1f} GiB")
"""
        ),
        _md(
            """
## 3. Select Your Single Language Model & Single Image Model

Copy **one** language model name and **one** image model name (or leave `SELECTED_IMAGE_MODEL = ""` to disable image generation) from the inventory output above, then run this cell to validate VRAM and disk fit and install model-specific dependencies before staging.
"""
        ),
        _code(
            """
# @title
import importlib
import shutil

import runtime_deps
runtime_deps = importlib.reload(runtime_deps)

SELECTED_LANGUAGE_MODEL = "Qwen-3-VL-8B-Instruct-heretic"  # @param {type:"string"}
SELECTED_IMAGE_MODEL = "FLUX.2-klein-4B"  # @param {type:"string"}

NEXUS_SELECTED_LANGUAGE, NEXUS_SELECTED_IMAGE = model_inventory.validate_selected_pair(
    NEXUS_DRIVE_ROOT / "models",
    SELECTED_LANGUAGE_MODEL,
    SELECTED_IMAGE_MODEL,
    total_vram_bytes=NEXUS_HARDWARE.total_vram_bytes,
    free_disk_bytes=shutil.disk_usage("/content").free,
    enforce_gpu_fit=NEXUS_HARDWARE.cuda_available,
)
print(
    f"[SELECT] Language model locked: {NEXUS_SELECTED_LANGUAGE.id} "
    f"(impl={NEXUS_SELECTED_LANGUAGE.implementation}, {NEXUS_SELECTED_LANGUAGE.size_bytes / (1024**3):.2f} GiB)"
)
if NEXUS_SELECTED_IMAGE is not None:
    print(
        f"[SELECT] Image model locked:    {NEXUS_SELECTED_IMAGE.id} "
        f"(impl={NEXUS_SELECTED_IMAGE.implementation}, {NEXUS_SELECTED_IMAGE.size_bytes / (1024**3):.2f} GiB)"
    )
else:
    print("[SELECT] Image model: disabled for this session.")

# Prepare model-specific runtime dependencies (e.g. llama-server for GGUFs, bitsandbytes for 4-bit)
runtime_deps.prepare_dependencies_for_model(NEXUS_SELECTED_LANGUAGE.implementation)
"""
        ),
        _md("## 4. Validate Dependencies, Pinned Node/pnpm, cloudflared & Fingerprint-Cached Harness Build"),
        _code(
            """
# @title
import hashlib
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
assert torch.cuda.is_available(), "CUDA GPU runtime is required for Nexus AI Lab v1.2.0."
print(f"[DEPS] Torch {torch.__version__} | CUDA={torch.version.cuda}")

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
            req = urllib.request.Request(url, headers={"User-Agent": "nexus-colab-setup/1.2.0"})
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

BUILD_STAMP = Path("/content/nexus-build-fingerprint.txt")
fp = sync_source.compute_build_fingerprint(NEXUS_HARNESS_DIR, NEXUS_COLAB_SRC_DIR, git_commit=NEXUS_ACTIVE_COMMIT)
if (BUILD_STAMP.read_text("utf-8").strip() if BUILD_STAMP.is_file() else "") != fp or not (NEXUS_HARNESS_DIR / "node_modules").is_dir():
    subprocess.run(["pnpm", "install", "--frozen-lockfile"], cwd=NEXUS_HARNESS_DIR, check=True)
    subprocess.run(["pnpm", "exec", "tsx", "scripts/build.ts", "--scope", "artifacts", "--serial"], cwd=NEXUS_HARNESS_DIR, check=True)
    BUILD_STAMP.write_text(fp + "\\n", encoding="utf-8")
print(f"[BUILD] Harness artifacts ready ({fp[:12]}).")
"""
        ),
        _md("## 5. Fast SSD Staging, Start Gateway & Warm-Load Selected Language Model"),
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
import urllib.request

import drive_fast_transfer
drive_fast_transfer = importlib.reload(drive_fast_transfer)
import staging
staging = importlib.reload(staging)

STAGING_TRANSPORT = "auto"  # @param ["auto", "drive_api", "fuse"]
STAGING_VERIFY_MODE = "fast"  # @param ["fast", "local_full", "strict_source"]
STAGE_IMAGE_IN_BACKGROUND = True  # @param {type:"boolean"}

# 1. Stage language model (and image model upfront unless background staging is enabled).
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

# 2. Restart Gateway bound strictly to loopback 127.0.0.1:7861.
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

gw_log = open("/content/nexus-gateway.log", "w", encoding="utf-8")
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

# 3. Warm-load the single selected language model in the isolated GPU worker BEFORE Harness starts.
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
    alloc_gb = rt["worker"].get("cuda_allocated_bytes", 0) / (1024 ** 3)
    print(f"[READY] Language model '{NEXUS_SELECTED_LANGUAGE.id}' is live in VRAM ({alloc_gb:.2f} GiB allocated).")
except Exception as exc:
    # Failed load MUST NOT leave a false-ready gateway running.
    gw_proc.terminate()
    NEXUS_MANAGED_PIDS.pop("gateway", None)
    if img_lock_path is not None:
        img_lock_path.unlink(missing_ok=True)
    raise RuntimeError(f"Initial language model load failed; gateway terminated cleanly: {exc}") from exc

# 4. Stage image model in background while you start chatting immediately!
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
        except Exception as bg_exc:
            print(f"[BG-IMAGE] Staging failed for {NEXUS_SELECTED_IMAGE.id}: {bg_exc}")
        finally:
            img_lock_path.unlink(missing_ok=True)

    bg_thread = threading.Thread(target=_bg_stage_image, daemon=True)
    bg_thread.start()
    print(f"[STAGE] Background staging started for image model '{NEXUS_SELECTED_IMAGE.id}'. You can run Section 6 now!")
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
        _md("## 7. Live v1.2.0 Acceptance Suite (Single-Model Lock, Chat, Tool Loop & GPU Swap)"),
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


# 1. Live readiness on /health and /v1/runtime
st, h = _get("/health", auth=False)
_record("01_health_ready", "PASS" if st == 200 and h.get("ready") is True else "FAIL", f"ready={h.get('ready')}, state={h.get('state')}")

st, rt = _get("/v1/runtime")
w = rt.get("worker") or {}
_record("02_runtime_ready", "PASS" if st == 200 and rt.get("ready") and w.get("active_model_id") == NEXUS_SELECTED_LANGUAGE.id else "FAIL", f"active={w.get('active_model_id')}")

# 2. Single-model lock: /v1/models returns strictly 1 model; loading another model returns HTTP 409
st, models_payload = _get("/v1/models")
data = models_payload.get("data", [])
_record("03_single_model_catalog", "PASS" if len(data) == 1 and data[0]["id"] == NEXUS_SELECTED_LANGUAGE.id else "FAIL", f"count={len(data)}")

try:
    _post("/v1/runtime/bootstrap", {"model": "__unauthorized_second_model__"})
    _record("04_second_model_rejected", "FAIL", "Second model was accepted")
except urllib.error.HTTPError as err:
    _record("04_second_model_rejected", "PASS" if err.code == 409 else "FAIL", f"HTTP {err.code}")

# 3. Live chat completion against locked language model
st, chat_resp = _post(
    "/v1/chat/completions",
    {"model": NEXUS_SELECTED_LANGUAGE.id, "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 32},
    timeout=300,
)
_record("05_live_chat_completion", "PASS" if st == 200 and chat_resp.get("choices") else "FAIL", f"model={NEXUS_SELECTED_LANGUAGE.id}")

# 4. Live image generation swap-and-restore (if image model selected)
if NEXUS_SELECTED_IMAGE is not None:
    st, img_resp = _post(
        "/v1/images/generations",
        {"prompt": "minimal geometric prism on dark background", "quality": "normal", "num_images": 1},
        timeout=600,
    )
    ok_img = st == 200 and bool(img_resp.get("filename")) and bool(img_resp.get("b64_png"))
    _record("06_image_swap_generation", "PASS" if ok_img else "FAIL", f"file={img_resp.get('filename')}")

    st, rt_after = _get("/v1/runtime")
    w_after = rt_after.get("worker") or {}
    _record(
        "07_llm_auto_restored_after_image",
        "PASS" if rt_after.get("ready") and w_after.get("active_model_id") == NEXUS_SELECTED_LANGUAGE.id else "FAIL",
        f"ready={rt_after.get('ready')}, active={w_after.get('active_model_id')}",
    )
else:
    _record("06_image_swap_generation", "SKIP", "Image model disabled")
    _record("07_llm_auto_restored_after_image", "SKIP", "Image model disabled")

# 5. Harness HTTP check
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{NEXUS_HARNESS_PORT}/", timeout=10) as resp:
        _record("08_harness_responding", "PASS" if resp.status < 500 else "FAIL", f"HTTP {resp.status}")
except urllib.error.HTTPError as err:
    _record("08_harness_responding", "PASS" if err.code in (200, 302, 401, 403) else "FAIL", f"HTTP {err.code}")

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


def write_v120_notebook(output_path: Path) -> Path:
    nb = build_v120_notebook()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "Nexus_Model_Lab_03_Nexus_Harness_Fork_Integration_v1.2.0.ipynb"
    write_v120_notebook(out)
    print(f"Wrote {out}")
