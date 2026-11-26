# Nexus Harness Architecture (v1.1.0)

Nexus Harness v1.1.0 runs a Colab-first, single-active-model runtime over DeepSeek Harness without modifying the agent loop or duplicating session persistence.

## Topology and trust boundaries

```mermaid
flowchart LR
  Browser["User Browser"] -->|"HTTPS (Cloudflare Quick Tunnel)"| CF["cloudflared (Colab VM)"]
  CF -->|"HTTP 127.0.0.1:3080 (X-Forwarded-Proto: https)"| Host["Locked Nexus Harness Host"]
  subgraph ColabVM["Colab VM (Loopback Only)"]
    Colab["Colab Bootstrap (Inventory -> Select -> Stage -> Warm Load)"] --> SSD["Local SSD (/content/nexus-staged-models)"]
    Colab --> GW["Nexus Gateway (127.0.0.1:7861)"]
    Host -->|"Bearer Token -> /v1/chat/completions & /v1/images/generations"| GW
    Host -->|"Bearer Token -> /v1/runtime & /v1/progress"| GW
    GW --> Lock["Exclusive Single-GPU Lock"]
    Lock --> Worker["Isolated GPU Worker (Lazy-Imported Dedicated Model)"]
    Worker --> SSD
    GW --> Drive["Google Drive Outputs (/content/drive/MyDrive/Nexus/outputs/)"]
  end
  Host -->|"Same-origin read-only /api/nexus.runtime*"| Browser
```

## Ownership boundaries

| Surface | Owner | Invariant |
|---|---|---|
| Model inventory, selection, SSD staging, warm load | Colab Bootstrap (`colab/model_inventory.py`, `colab/staging.py`, v1.1.0 notebook) | Stages 1 language model and 1 optional image model upfront; fails loud before Harness starts if load fails. |
| Single-model inference & GPU swap | Nexus Gateway (`127.0.0.1:7861`) + Isolated GPU Worker | Rejects unselected models (HTTP 400/409) and pre-readiness chat (HTTP 503); image generation atomically swaps LLM → Image → LLM inside one exclusive lock. |
| Native tool execution & session log | Harness Agent Loop (`@deepseek-ai/dsh-session`) | Gateway parses & validates model-specific tool calls (Qwen3-VL JSON, Qwen3.5/3.6 XML, GPT-OSS Harmony); Harness executes tools and appends `role: "tool"`. |
| Runtime UI | `nexus-runtime-ui` + `nexus-host` | Strictly read-only status badge, VRAM display, and live swap/denoising progress; zero model switching or loading controls in Harness. |
