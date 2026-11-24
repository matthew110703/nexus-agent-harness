# Nexus Gateway

Loopback-only (`127.0.0.1:7861`) OpenAI-compatible inference, staging, and FLUX.2 Diffusers gateway for Nexus Harness.

## Layout

- `src/api/` — FastAPI app, bearer-token middleware, and route modules (`models`, `chat`, `runtime`, `tools`, `images`).
- `src/core/` — Dynamic Drive model registry, single-active-model resource manager, Drive-to-local staging with byte-level progress, model loader, and inference runner.
- `src/adapters/` — Model-specific adapters (`qwen_vl.py`, extensible `tool_calls.py` parser registry, `diffusers.py` FLUX.2 adapters).
- `src/services/` — Operation-tracked progress service, conversation normalizer, and image-generation service.

## Running Tests

```sh
pytest nexus-gateway/tests -v
```
