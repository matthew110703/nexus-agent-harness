# Nexus Gateway (v1.1.0 — Single-Model Runtime)

Loopback-only (`127.0.0.1:7861`) OpenAI-compatible inference and Diffusers image-generation gateway for **Nexus AI Lab** on Google Colab.

Model discovery, selection, and local SSD staging happen strictly during initial Colab bootstrap (`colab/model_inventory.py` + `colab/staging.py`), which writes a locked `session-runtime.json` manifest consumed by the Gateway.

## Architecture

```text
nexus-gateway/src/
├── main.py                       # Loopback-only uvicorn entrypoint reading session-runtime.json
├── api/
│   ├── app.py                    # FastAPI app: single-model lock, bootstrap, swap-and-restore image gen
│   └── middleware/auth.py        # Constant-time Bearer auth + loopback enforcement
├── core/
│   ├── hardware_detector.py      # Dynamic GPU/VRAM/CUDA/RAM/disk detection (L4 / A100-40GB / A100-80GB)
│   ├── resource_manager.py       # Exclusive single-GPU Lock serializing inference & image swaps
│   └── worker_manager.py         # Isolated subprocess lifecycle, live status, forced VRAM release
├── workers/
│   ├── model_worker.py           # Isolated GPU worker lazy-importing only the selected model module
│   └── worker_protocol.py        # Typed worker RPC schemas
├── models/
│   ├── base.py                   # SessionRuntimeConfig, minimal runtime protocols, lazy importlib loader
│   ├── language/
│   │   ├── common.py             # Authorized data:image/* decoder & tool-call argument dict normalization
│   │   ├── qwen3_vl.py           # Dedicated Qwen3-VL runtime (4B/8B/Heretic)
│   │   ├── qwen3_5.py            # Dedicated Qwen3.5 / Qwen3.6 VLM + Thinking runtime
│   │   └── gpt_oss.py            # Dedicated GPT-OSS Harmony runtime
│   └── image/
│       ├── flux2_klein.py        # Dedicated FLUX.2-klein (4B/9B) Diffusers runtime
│       └── sdxl.py               # Dedicated SDXL Diffusers runtime
├── adapters/
│   └── tool_calls.py             # Qwen JSON, Qwen3.5 XML+Think, and GPT-OSS Harmony stream parsers
└── services/
    ├── progress.py               # Bounded operation progress registry (including GPU swap phases)
    ├── conversation.py           # Message normalization & model-aware reasoning control resolution
    └── image_generation.py       # Atomic PNG + JSON metadata sidecar persistence to Google Drive
```

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | None | Live worker readiness (`ready`, `state`), locked model IDs, uptime |
| `GET /v1/models` | Bearer | Single-element list containing the locked Colab language model |
| `GET /v1/runtime` | Bearer | Live hardware, worker status, locked language/image specs, VRAM |
| `POST /v1/runtime/bootstrap` | Bearer | Colab-only warm load of the locked language model (HTTP 409 on any other model) |
| `POST /v1/runtime/unload` | Bearer | Graceful unload / `force_terminate` worker cleanup for Colab shutdown |
| `POST /v1/runtime/cancel` | Bearer | Cancel an in-flight image generation operation |
| `POST /v1/chat/completions` | Bearer | Streaming SSE & non-streaming chat with native tool calls & reasoning (HTTP 503 if not ready) |
| `POST /v1/images/generations` | Bearer | Atomic GPU swap (Unload LLM → Load pre-staged Image → Denoise & Save → Reload LLM) |
| `GET /v1/progress` | Bearer | Compact progress snapshot for Harness read-only UI |
| `GET /v1/progress/{operation_id}` | Bearer | Individual operation progress |
| `GET /v1/plugin-index.json` | Loopback | Curated format-1 Nexus plugin index |

## Running Tests

```sh
pytest nexus-gateway/tests -v
```
