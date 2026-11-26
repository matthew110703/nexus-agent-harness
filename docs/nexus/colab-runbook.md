# Nexus AI Lab Colab Runbook (v1.1.0)

Run Nexus AI Lab on Google Colab (`L4 24 GB`, `A100 40 GB`, or `A100 80 GB`) using the single-active-model workflow.

## Workflow Summary

1. **Section 1 (Paths & Env):** Generates per-session `NEXUS_API_KEY` (server-side only) and sets local SSD paths (`DSH_HOME`, `HF_HOME`, `/content/nexus-staged-models`).
2. **Section 2 (Mount Drive, Checkout & Inventory):** Mounts `/content/drive`, syncs `colab-src/`, inspects your live GPU VRAM, and prints tables of runnable language and image models in `MyDrive/Nexus/models/` with GPU fit badges.
3. **Section 3 (Pre-Run Selection):** Paste **one** `SELECTED_LANGUAGE_MODEL` name and **one** `SELECTED_IMAGE_MODEL` name (or `""` to disable image generation). Validates checkpoint completeness, GPU VRAM fit, and local SSD staging space.
4. **Section 4 (Dependencies & Build):** Verifies PyTorch/CUDA, pinned Node `24.22.3`, pinned `pnpm`, SHA-256-verified `cloudflared`, and fingerprint-cached `build:artifacts`.
5. **Section 5 (Stage & Warm Load):** Stages both selected models from Drive to `/content/nexus-staged-models/` (SHA-256 verified, skipping `.cache/`), writes `session-runtime.json`, starts Gateway (`127.0.0.1:7861`), warm-loads `SELECTED_LANGUAGE_MODEL`, and verifies live GPU readiness.
6. **Section 6 (Start Locked Harness):** Restores the latest verified non-secret Drive snapshot and runs `node colab/supervisor.mjs`, which verifies Gateway live readiness and launches `dsh web` locked to the single selected model.
7. **Section 7 (Live Acceptance Suite):** Verifies live readiness, single-model lock (HTTP 409 on second model), live chat completion, and image swap + automatic LLM restore.
8. **Section 8 (Shutdown & Snapshot):** Terminates the isolated GPU worker, supervisor, and Gateway, then saves a verified non-secret snapshot to `MyDrive/Nexus/harness-home/snapshots/`.
