# Nexus Harness Source Audit

This audit records the exact checked-out baseline of the Nexus Harness fork and maps every subsystem touched by the Colab-first Nexus integration. All findings come directly from the checked-out source tree and type declarations, not README assumptions.

## 1. Checked-out baseline

| Property | Value |
|---|---|
| Repository | `https://github.com/matthew110703/nexus-agent-harness.git` |
| Commit SHA | `388ed04b7b2447565468096edb1de813f67ece8a` |
| Upstream baseline | `4b399c39a6` (`@deepseek-ai/dsh-root@0.2.3-alpha.2`) |
| Node engine range | `^24.22.3 \|\| >=26.13.1` (`package.json`) |
| Package manager | `pnpm@11.28.5` (`pnpm-lock.yaml`) |
| Note vs. `0.2.2-rc.5` | The checked-out tree is newer than `0.2.2-rc.5`. All Nexus plugins and patches target the verified `0.2.3-alpha.2` APIs below. |

## 2. Subsystem component map

### 2.1 CLI entrypoint and option parsing order

- **Files**: `apps/cli/src/bin.ts`, `apps/cli/src/args.ts`, `apps/cli/src/profile-boot.ts`.
- **Behavior**: `parseDshArgs` parses launcher flags first (`--profile`, `--from-default-profile`, repeatable `--patch`, `--dump-config*`). The first unrecognized token starts `args`, which are forwarded verbatim to `ctx.cmdlineArgs` for injected app plugins (`web-startup`, etc.).
- **Failure mode**: Placing `--patch` after `web` app flags (e.g. `dsh web --public-url ... --patch ...`) passes `--patch` into app args instead of the launcher. Correct order: `dsh web --patch <overlay.yml> --public-url <url> --trusted-host <host> --no-open`.

### 2.2 Web profile composition and Cordis plugin injection

- **Files**: `packages/util/official-bundles/src/bundles.ts`, `packages/bundle/base/cordis.patch.yml`, `packages/bundle/web-app/package.json`, `packages/bundle/web-app/cordis.patch.yml`, `packages/bundle/web-app/src/startup.ts`.
- **Composition order**: (1) `@deepseek-ai/dsh-base`, (2) `@deepseek-ai/dsh-web-app` (plus presets), (3) profile `cordis.patch.yml`, (4) `$DSH_HOME/cordis.patch.yml`, (5) `--patch` overlays in argv order.
- **Startup injection**: `web-startup` parses `--host`, `--port`, `--public-url`, `--trusted-host`, `--no-open`, `--tls-cert`, `--tls-key` and provides `ctx.webStartup`. Dependent rows (`webserver`, `web-runtime`, `connection`) read `ctx.webStartup.*` via `!!js`.

### 2.3 WebServer, Host/Origin validation, and launch-token authentication

- **Files**:
  - `packages/host/webserver/src/index.ts` (`WebServer`, rejects wildcard bind addresses like `0.0.0.0`).
  - `packages/client/connection/src/api-request-trust.ts` (`isTrustedApiRequest`), `packages/client/connection/src/rpc-host.ts` (`HostConnectionService`).
  - `packages/host/authentication-token/src/index.ts`, `packages/host/authentication-token/src/browser-auth.ts` (`BrowserAuth`).
  - `packages/bundle/web-app/src/public-url.ts`, `packages/bundle/web-app/src/index.ts` (`WebApp`).
- **Exact trust checks**:
  1. `Host` header must match loopback, `webServer.host`, or an exact entry in `connection.trustedHosts` (`--trusted-host`).
  2. `sec-fetch-site` must not be `'cross-site'`.
  3. If `Origin` is present, `originUrl.host` must equal `hostUrl.host` exactly.
- **Concrete failure modes**:
  - `--public-url` is advertisement-only and grants zero trust. Omitting `--trusted-host <tunnel-host>` causes HTTP `403` on `/api/directoryPicker/list` and `/api/llm/listProviders`.
  - `BrowserAuth` ignores `X-Forwarded-Proto`, so behind a TLS-terminating loopback proxy (`cloudflared` -> `http://127.0.0.1:3080`) `dsh-auth-*` cookies omit the `Secure` attribute unless an explicit, opt-in trusted-proxy `X-Forwarded-Proto` check is added.

### 2.4 Workspace picker and root confinement

- **Files**:
  - `packages/host/directory-picker/src/index.ts` (seam & error codes).
  - `packages/host/directory-picker-browse/src/index.ts` (`BrowseDirectoryPicker`).
  - `packages/host/directory-picker-auto/src/resolve.ts` (`resolveDirectoryPickerBackend`).
  - `packages/api/workspace-controller/src/directory-picker.ts` (`DirectoryPickerController`).
  - `packages/workspace/workspace/src/index.ts` (`WorkspaceRegistry`).
- **Concrete failure modes**:
  - `BrowseDirectoryPicker.Config` only has `{ maxEntries: number }` and defaults omitted `path` to `os.homedir()`, exposing the entire filesystem.
  - `WorkspaceRegistry.create(path)` accepts any existing directory via `realpathNormalize(path)` without checking approved roots.
  - `ancestryCrumbs()` walks all the way to filesystem root `/`.

### 2.5 LLM provider abstraction and model discovery

- **Files**: `packages/llm/llm/src/index.ts` (`LlmRuntime`), `packages/llm/llm-pi-ai/src/config.ts`, `catalog.ts`, `discovery.ts`, `adapter.ts`, `stream.ts`, `context.ts`.
- **Capabilities already present**:
  - Custom `openai-completions` routes with `baseURL`, `apiKeyEnv`, `defaultInput: ['text', 'image']`, `defaultContextWindow`, `defaultMaxTokens`.
  - Compatibility switches: `compat.supportsDeveloperRole`, `compat.maxTokensField` (`'max_tokens' | 'max_completion_tokens'`).
  - Host-side `GET {baseURL}/models` discovery (`discovery.ts`) using `Authorization: Bearer <apiKey>` with 4 MiB response bound and `AbortSignal`.
  - SSE streaming with `AbortSignal.any` + `idleWatchdog` (`streamIdleTimeoutMs`) and clean iterator teardown on abort.

### 2.6 Tool authoring (`defineTool`) and durable attachments

- **Files**: `packages/core/tools/src/schema.ts`, `contracts.ts`, `pipeline.ts`; `packages/attachment/attachment/src/index.ts`, `types.ts`.
- **Verified APIs**:
  - `defineTool({ name, description, parameters, output: { schema, render, presentationMeta? }, execute(args, exec) })` with `exec.signal` fused into pipeline cancellation.
  - `ctx.attachments.saveImage({ data: Uint8Array, mediaType, name? })` returns `ImageAttachmentRef` (`{ attachmentId, mediaType, bytes, width, height, name? }`).

### 2.7 Host HTTP routes and Client rendering slots

- **Host routes**: `ctx.connection.fetch.register({ path: '/api/<segment>', methods, requestBody, fetch })` automatically enforces `auth: 'required'`, `isTrustedApiRequest`, cookie auth, and `peer.principal.kind === 'owner'` (403 otherwise).
- **Client tool views**:
  - `'tool.call.toolview'` (`kind: 'keyed'`, `scope: 'session'`) in `packages/client/ui-tool/src/client/contract/slots.ts`.
  - `GenericToolCard` flattens `ImageBlock`s into plain text. Only `ReadImageRow` renders images, and `'tool.call.images'` is a `single` child slot owned exclusively by `readImageToolview`.
  - A custom `nexus_generate_image` toolview must register on `'tool.call.toolview'` and render images via `loadImage(ref)` / `'conversation.images.content'` without re-declaring `'tool.call.images'`.
- **Client runtime progress panel**:
  - `'conversation.session.panel.section'` (`kind: 'list'`, `scope: 'session'`) and `'conversation.session.panel.indicator'` in `packages/client/ui-session-panel/src/client/slots.ts`.

## 3. Smallest recommended patch plan

1. **Minimal core patches** (upstream-rebaseable):
   - Opt-in `trustForwardedProto` (loopback peer only) in `packages/host/authentication-token`.
   - Opt-in `allowedRoots` + canonical `realpath` containment + breadcrumb clamping in `packages/host/directory-picker-browse` and `packages/workspace/workspace`.
2. **Additive Nexus packages** (zero edits to `agent-loop` or session format):
   - `nexus-gateway/`: Python FastAPI gateway (auth, single-active-model lock, capability registry, Qwen native `<tool_call>` streaming parser/validator, progress API, FLUX.2 Diffusers adapters).
   - `packages/experimental/nexus-image-tool/`: Host `defineTool` (`nexus_generate_image`) + `ctx.attachments.saveImage`.
   - `packages/experimental/nexus-runtime-proxy/`: Host same-origin `/api/nexus.runtime-progress` proxy.
   - `packages/experimental/nexus-client/`: Client `nexus_generate_image` toolview + Session panel runtime progress section.
   - `apps/cli/config/examples/colab.overlay.yml` + `colab/supervisor.mjs` + Colab notebook runbook.
