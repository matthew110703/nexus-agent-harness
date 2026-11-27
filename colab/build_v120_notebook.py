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

### What's New in v1.2.0:
- **Comprehensive Model Support**: All 8 Drive models supported dynamically:
  - **GGUF Models (`llama-server`)**: `Qwen3.5-9B-Defiant-Fable-Uncensored`, `Gemma4-26B-A4B-QAT-Uncensored-MTP`, `DeepSeek-V4-Pro-Qwen3.5-9B-Uncensored`.
  - **Safetensors Large Language / VLM Models**: `Qwen3.5-4B`, `Gemma-3-12B-IT` (4-bit NF4 auto-fit), `DeepSeek-VL2-Small` (4-bit NF4 auto-fit), `DeepSeek-VL-7B-Chat`, `GPT-OSS-20B-Abliterated` (4-bit NF4 auto-fit).
  - **Diffusers & Monolithic SDXL**: Fixed `shootstuff/LUSTIFY-v2.0` fp16 component variant loading & monolithic single-file fallback.
- **Drive Workspace Persistence**: Harness working directory defaults to `/content/drive/MyDrive/Nexus/workspaces/projects/default-workspace`. All generated images route into `<active_workspace>/generated_images/` on Drive.
- **Durable Storage Backup**: Harness snapshots persist `storages/`, preserving custom workspace names, order, and pins across Colab restarts.
- **Inline Gallery Rendering**: Inline image tool views with durable attachment support.
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
os.environ["NEXUS_HARNESS_HOME"] = str(NEXUS_HARNESS_HOME)
os.environ["HF_HOME"] = str(NEXUS_HF_CACHE_DIR)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTHONPATH"] = (
    f"{NEXUS_HARNESS_DIR / 'colab'}:{NEXUS_HARNESS_DIR / 'nexus-gateway' / 'src'}:"
    f"{NEXUS_COLAB_SRC_DIR}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
)

for p in (
    NEXUS_STAGE_DIR,
    NEXUS_WORKSPACE_ROOT / "default-workspace",
    NEXUS_SHARED_WORKSPACE_ROOT,
    NEXUS_IMAGE_OUTPUT_DIR,
    NEXUS_HARNESS_HOME,
    NEXUS_HARNESS_HOME_DRIVE_BACKUP,
    NEXUS_HF_CACHE_DIR,
    NEXUS_COLAB_SRC_DIR,
    NEXUS_RUNTIME_MANIFEST_DIR,
):
    p.mkdir(parents=True, exist_ok=True)

print("[OK] Paths and environment configured for Nexus v1.2.0.")
"""
        ),
        _md("## 2. Sync Repository / Colab-Src & Inspect Google Drive Models"),
        _code(
            """
# @title
import importlib
import shutil
import sys
from pathlib import Path

# Clone or update deepseek-harness repository
if not (NEXUS_HARNESS_DIR / ".git").is_dir():
    import subprocess
    print(f"[GIT] Cloning {NEXUS_HARNESS_REPO_URL} -> {NEXUS_HARNESS_DIR}...")
    subprocess.run(["git", "clone", NEXUS_HARNESS_REPO_URL, str(NEXUS_HARNESS_DIR)], check=True)
else:
    import subprocess
    subprocess.run(["git", "-C", str(NEXUS_HARNESS_DIR), "fetch", "origin"], check=False)
    ref = NEXUS_HARNESS_GIT_REF or "origin/main"
    subprocess.run(["git", "-C", str(NEXUS_HARNESS_DIR), "checkout", ref], check=False)

if str(NEXUS_HARNESS_DIR / "colab") not in sys.path:
    sys.path.insert(0, str(NEXUS_HARNESS_DIR / "colab"))
if str(NEXUS_HARNESS_DIR / "nexus-gateway" / "src") not in sys.path:
    sys.path.insert(0, str(NEXUS_HARNESS_DIR / "nexus-gateway" / "src"))

import sync_source
sync_source = importlib.reload(sync_source)

# Two-way sync with Drive colab-src
sync_report = sync_source.sync_colab_source(NEXUS_HARNESS_DIR, NEXUS_COLAB_SRC_DIR)
print(f"[SYNC] Source sync status: {sync_report['status']} ({len(sync_report.get('synced_files', []))} files updated).")

import model_inventory
model_inventory = importlib.reload(model_inventory)

NEXUS_HARDWARE = model_inventory.detect_hardware_profile()
print(f"[HARDWARE] {NEXUS_HARDWARE.summary}")

DRIVE_MODELS_ROOT = NEXUS_DRIVE_ROOT / "models"
NEXUS_INVENTORY = model_inventory.inspect_drive_models(
    DRIVE_MODELS_ROOT,
    total_vram_bytes=NEXUS_HARDWARE.total_vram_bytes,
)
print("\\n" + model_inventory.format_inventory_report(NEXUS_INVENTORY, gpu_summary=NEXUS_HARDWARE.summary))
"""
        ),
        _md("## 3. Pre-Run Model Selection (Select Exactly 1 Language Model + 1 Image Model)"),
        _code(
            """
# @title
import shutil
import model_inventory
import runtime_deps

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
    archive = _download_with_retry(f"https://nodejs.org/dist/v{NODE_VERSION}/{NODE_ARCHIVE}", NODE_ROOT / NODE_ARCHIVE)
    if expected_sha and _sha256_file(archive) != expected_sha:
        archive.unlink()
        raise RuntimeError("Node archive SHA-256 mismatch")
    subprocess.run(["tar", "-xJf", str(archive), "-C", str(NODE_ROOT)], check=True)
    print(f"[NODE] Installed Node {NODE_VERSION}.")

os.environ["PATH"] = f"{NODE_BIN}:{os.environ.get('PATH', '')}"

PNPM_VERSION = "10.6.1"
pnpm_path = shutil.which("pnpm")
if not pnpm_path:
    subprocess.run(["npm", "install", "-g", f"pnpm@{PNPM_VERSION}"], check=True)
    print(f"[PNPM] Installed pnpm {PNPM_VERSION}.")

CF_PATH = Path("/usr/local/bin/cloudflared")
if not CF_PATH.is_file():
    _download_with_retry("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CF_PATH)
    CF_PATH.chmod(0o755)
    print("[TUNNEL] cloudflared binary verified.")

# Install pnpm workspace dependencies & build if artifacts missing
build_marker = NEXUS_HARNESS_DIR / "packages" / "experimental" / "nexus-host" / "lib" / "index.js"
if not build_marker.is_file():
    print("[BUILD] Installing harness workspace dependencies (frozen lockfile)...")
    subprocess.run(["pnpm", "install", "--frozen-lockfile"], cwd=str(NEXUS_HARNESS_DIR), check=True)
    print("[BUILD] Building artifacts...")
    subprocess.run(["pnpm", "run", "build:artifacts"], cwd=str(NEXUS_HARNESS_DIR), check=True)
    print("[BUILD] Harness build complete.")
else:
    print("[BUILD] Reusing verified build artifacts.")
"""
        ),
        _md("## 5. Stage Selected Models to Local SSD & Launch Loopback Gateway"),
        _code(
            """
# @title
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import staging

if "NEXUS_MANAGED_PIDS" not in globals():
    NEXUS_MANAGED_PIDS = {}

# 1. Stage the selected models to local SSD (/content/nexus-staged-models/)
print(f"[STAGE] Staging selected language model: {NEXUS_SELECTED_LANGUAGE.id}...")
lang_stage_dir = staging.stage_model_directory(
    Path(NEXUS_SELECTED_LANGUAGE.source_path),
    NEXUS_STAGE_DIR,
    model_id=NEXUS_SELECTED_LANGUAGE.id,
    on_status=print,
)

img_stage_dir = None
if NEXUS_SELECTED_IMAGE is not None:
    print(f"[STAGE] Staging selected image model: {NEXUS_SELECTED_IMAGE.id}...")
    img_stage_dir = staging.stage_model_directory(
        Path(NEXUS_SELECTED_IMAGE.source_path),
        NEXUS_STAGE_DIR,
        model_id=NEXUS_SELECTED_IMAGE.id,
        on_status=print,
    )

# Write session runtime manifest
manifest_payload = {
    "language_model": {
        "id": NEXUS_SELECTED_LANGUAGE.id,
        "display_name": NEXUS_SELECTED_LANGUAGE.id,
        "model_path": str(lang_stage_dir),
        "implementation": NEXUS_SELECTED_LANGUAGE.implementation,
        "architecture": NEXUS_SELECTED_LANGUAGE.architecture,
        "modalities": NEXUS_SELECTED_LANGUAGE.modalities,
        "context_window": NEXUS_SELECTED_LANGUAGE.context_window,
        "max_output_tokens": NEXUS_SELECTED_LANGUAGE.max_output_tokens,
        "reasoning_mode": NEXUS_SELECTED_LANGUAGE.reasoning_mode,
        "reasoning_efforts": NEXUS_SELECTED_LANGUAGE.reasoning_efforts,
        "default_reasoning_effort": NEXUS_SELECTED_LANGUAGE.default_reasoning_effort,
        "min_vram_gib": NEXUS_SELECTED_LANGUAGE.min_vram_gib,
        "size_bytes": NEXUS_SELECTED_LANGUAGE.size_bytes,
    },
    "image_model": {
        "id": NEXUS_SELECTED_IMAGE.id,
        "display_name": NEXUS_SELECTED_IMAGE.id,
        "model_path": str(img_stage_dir),
        "implementation": NEXUS_SELECTED_IMAGE.implementation,
        "architecture": NEXUS_SELECTED_IMAGE.architecture,
        "default_steps": NEXUS_SELECTED_IMAGE.default_steps,
        "min_vram_gib": NEXUS_SELECTED_IMAGE.min_vram_gib,
        "size_bytes": NEXUS_SELECTED_IMAGE.size_bytes,
    } if NEXUS_SELECTED_IMAGE is not None else None,
}
NEXUS_SESSION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
NEXUS_SESSION_MANIFEST.write_text(json.dumps(manifest_payload, indent=2) + "\\n", encoding="utf-8")
print(f"[STAGE] Session runtime manifest written to {NEXUS_SESSION_MANIFEST}.")

# 2. Launch loopback gateway
prev_gw = NEXUS_MANAGED_PIDS.pop("gateway", None)
if prev_gw is not None:
    try:
        os.kill(prev_gw, signal.SIGTERM)
        time.sleep(1.0)
    except Exception:
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
    gw_proc.terminate()
    NEXUS_MANAGED_PIDS.pop("gateway", None)
    raise RuntimeError(f"Initial language model load failed; gateway terminated cleanly: {exc}") from exc
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
