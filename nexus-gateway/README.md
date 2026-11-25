# Nexus Gateway (v1.0.0)

Loopback-only (`127.0.0.1:7861`) OpenAI-compatible inference, Diffusers image-generation, capability-discovery, and isolated GPU worker gateway for **Nexus Harness**.

## Architecture

```text
nexus-gateway/src/
├── main.py                      # Loopback-only uvicorn entrypoint
├── api/
│   ├── app.py                   # FastAPI app & all /v1/* + /health routes
│   └── middleware/auth.py       # Constant-time Bearer auth + loopback enforcement
├── core/
│   ├── hardware_detector.py     # Dynamic GPU/VRAM/CUDA/RAM/disk detection (L4 baseline -> A100/H100)
│   ├── model_registry.py        # Drive models/adapters/upscalers + adaptive Effort & sidecar overrides
│   ├── model_staging.py         # Manifest-backed 4 MiB chunk staging with SHA-256 integrity & resume
│   ├── resource_manager.py      # Non-thread-owned Lock serializing single active GPU workload
│   └── worker_manager.py        # Subprocess lifecycle, health, graceful unload, forced VRAM release
├── workers/
│   ├── model_worker.py          # Isolated GPU worker subprocess owning CUDA context & weights
│   └── worker_protocol.py       # Typed worker RPC schemas
├── adapters/
│   ├── transformers.py          # Transformers chat adapter (explicit precision, enable_thinking)
│   ├── qwen_vl.py               # Qwen2.5-VL / Qwen3-VL multimodal adapter
│   ├── tool_calls.py            # Incremental native <tool_call> streaming parser + JSON schema validator
│   └── diffusers.py             # Independent Diffusers/FLUX.2 adapter with explicit quality profiles
└── services/
    ├── progress.py              # Bounded operation progress registry
    ├── conversation.py          # Message normalization & model-aware reasoning control resolution
    └── image_generation.py      # Atomic PNG + JSON metadata sidecar persistence to Google Drive
```

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | None | Gateway uptime, worker status, discovered model counts |
| `GET /v1/models` | Bearer | Chat/VLM models with `modalities`, `context_window`, `reasoning`, `reasoning_efforts` |
| `GET /v1/image-models` | Bearer | Independent Diffusers/FLUX.2 image models and quality profiles (`normal`, `detailed`, `maximum`) |
| `POST /v1/chat/completions` | Bearer | OpenAI-compatible streaming SSE & non-streaming chat with native tool calls & adaptive Effort |
| `POST /v1/images/generations` | Bearer | Independent image generation with explicit quality profiles & Drive PNG + JSON persistence |
| `GET /v1/runtime` | Bearer | Live hardware snapshot, worker status, active workload, adapters, upscalers, diagnostics |
| `POST /v1/runtime/load` | Bearer | Explicit model staging + worker load |
| `POST /v1/runtime/unload` | Bearer | Graceful unload (with optional `force_terminate: true` to release CUDA context) |
| `GET /v1/progress` | Bearer | Compact progress snapshot for Harness UI |
| `GET /v1/progress/{operation_id}` | Bearer | Individual operation progress |
| `POST /v1/tools/parse` | Bearer | Structured native tool-call parse & schema validation |
| `GET /v1/plugin-index.json` | Loopback | Curated format-1 Nexus plugin index for Harness **Plugins → Discover** |

## Running Tests

```sh
pytest nexus-gateway/tests -v
```
