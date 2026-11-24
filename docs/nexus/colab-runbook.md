# Nexus Harness Colab Runbook

Run Nexus Harness on Google Colab with loopback-only model serving (`127.0.0.1:7861`), authenticated Cloudflare quick tunnel (`127.0.0.1:3080`), and constrained workspace browsing.

## Cell 1: Mount Drive & prepare directories

```python
from google.colab import drive
import os, secrets

drive.mount('/content/drive')
os.makedirs('/content/nexus-harness-workspace', exist_ok=True)
os.makedirs('/content/nexus_staged_models', exist_ok=True)
os.makedirs('/content/drive/MyDrive/Nexus/outputs/generated_images', exist_ok=True)

# Generate per-session bearer token (never printed)
os.environ['NEXUS_API_KEY'] = secrets.token_urlsafe(32)
os.environ['NEXUS_WORKSPACE_ROOT'] = '/content/nexus-harness-workspace'
os.environ['NEXUS_GATEWAY_URL'] = 'http://127.0.0.1:7861/v1'
print('Environment initialized.')
```

## Cell 2: Install Node >=24.22.3, pnpm, cloudflared, and build Nexus Harness

```bash
%%bash
set -euo pipefail
git clone https://github.com/matthew110703/nexus-agent-harness.git /content/nexus-harness || (cd /content/nexus-harness && git pull)
cd /content/nexus-harness
corepack enable
pnpm install
pnpm run build:artifacts
pip install -q -e ./nexus-gateway
```

## Cell 3: Start Nexus Gateway (loopback-only `127.0.0.1:7861`)

Run the FastAPI gateway bound strictly to `127.0.0.1:7861` using `os.environ['NEXUS_API_KEY']`. Do **not** expose port `7861` via Cloudflare.

## Cell 4: Start Supervised Nexus Harness + Cloudflare Quick Tunnel

```bash
%%bash
cd /content/nexus-harness
node colab/supervisor.mjs
```

The supervisor will:
1. Verify `http://127.0.0.1:7861/health` and `NEXUS_API_KEY` (never printing the token).
2. Launch `cloudflared tunnel --url http://127.0.0.1:3080` and parse the exact `trycloudflare.com` hostname.
3. Launch `dsh web --patch apps/cli/config/examples/colab.overlay.yml --public-url https://<host> --trusted-host <host> --no-open`.
4. Print the one-time authenticated Harness launch URL once.

## Rollback & Upstream Rebase

- Core patches are isolated to two small, opt-in flags: `trustForwardedProto` in `packages/host/authentication-token` and `allowedRoots` in `packages/host/directory-picker-browse`.
- All Nexus-specific code lives in additive directories (`nexus-gateway/`, `packages/experimental/nexus-host/`, `colab/`, `apps/cli/config/examples/colab.overlay.yml`, `docs/nexus/`).
- To revert to stock upstream behavior, omit `--patch apps/cli/config/examples/colab.overlay.yml`.
