/**
 * Nexus Host plugin: registers the `nexus_generate_image` tool (backed by
 * `ctx.attachments.saveImage`) and the authenticated same-origin
 * `/api/nexus.runtime-progress` proxy to the loopback Nexus gateway.
 * @module @deepseek-ai/dsh-nexus-host
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type {} from '@deepseek-ai/dsh-client-connection'
import type {} from '@deepseek-ai/dsh-host-webserver'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolDefinition } from '@deepseek-ai/dsh-tools'

/** Stable plugin name. */
export const name = 'nexus-host'
/** Required Host services: tool registry, attachment store, and connection fetch router. */
export const inject = ['tools', 'attachments', 'connection']

/** Maximum image response size accepted from the loopback Nexus gateway (24 MiB). */
const MAX_IMAGE_RESPONSE_BYTES = 24 * 1024 * 1024
/** Maximum progress snapshot size relayed to the browser (256 KiB). */
const MAX_PROGRESS_RESPONSE_BYTES = 256 * 1024

/** Validated configuration for the Nexus Host plugin. */
export interface Config {
  /** Loopback base URL of the Nexus gateway (e.g. `http://127.0.0.1:7861`). */
  gatewayUrl: string
  /** Environment variable name holding the Nexus bearer token. */
  apiKeyEnv: string
  /** Timeout in milliseconds for image generation calls. */
  imageTimeoutMs: number
  /** Timeout in milliseconds for runtime progress proxy calls. */
  progressTimeoutMs: number
}

/** Configuration schema with safe loopback defaults. */
export const Config: z<Config> = z.object({
  gatewayUrl: z.string().default('http://127.0.0.1:7861'),
  apiKeyEnv: z.string().default('NEXUS_API_KEY'),
  imageTimeoutMs: z.natural().min(1000).default(300_000),
  progressTimeoutMs: z.natural().min(500).default(5000),
})

/** Verify that the configured gateway URL targets loopback only. */
export function assertLoopbackGatewayUrl(rawUrl: string): URL {
  const parsed = new URL(rawUrl)
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Nexus gateway URL must use http/https, got ${parsed.protocol}`)
  }
  const host = parsed.hostname.toLowerCase()
  if (host !== '127.0.0.1' && host !== '::1' && host !== '[::1]' && host !== 'localhost') {
    throw new Error(`Nexus gateway URL must be loopback-only, got ${parsed.hostname}`)
  }
  return parsed
}

interface GatewayImagePayload {
  operation_id?: string
  artifact_id?: string
  model?: string
  quality?: string
  filename?: string
  master_path?: string
  media_type?: string
  width?: number
  height?: number
  steps?: number
  seed?: number | null
  b64_png?: string
}

/**
 * Create the `nexus_generate_image` tool definition bound to the Host's
 * attachment store and loopback Nexus gateway.
 */
export function createNexusGenerateImageTool(
  ctx: Context,
  config: Config,
  fetchImpl: typeof fetch = fetch,
): ToolDefinition {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  return defineTool({
    name: 'nexus_generate_image',
    description:
      'Generate an image using a local Nexus Diffusers (FLUX.2) pipeline independent of the active chat model '
      + 'and attach it durably to the conversation while saving a PNG master copy to Google Drive.',
    parameters: {
      prompt: { type: 'string', description: 'Text prompt describing the image to generate.', required: true },
      model: { type: 'string', description: 'Optional Nexus image model ID from /v1/image-models.' },
      quality: {
        type: 'string',
        enum: ['normal', 'detailed', 'maximum'],
        description: 'Explicit image quality profile independent of chat reasoning Effort.',
      },
      width: { type: 'integer', description: 'Optional image width in pixels (256–1536, aligned to 16).' },
      height: { type: 'integer', description: 'Optional image height in pixels (256–1536, aligned to 16).' },
      steps: { type: 'integer', description: 'Optional denoising step count (model-specific default when omitted).' },
      seed: { type: 'integer', description: 'Optional integer RNG seed for reproducibility.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          attachmentId: { type: 'string' },
          model: { type: 'string' },
          quality: { type: 'string' },
          filename: { type: 'string' },
          width: { type: 'integer' },
          height: { type: 'integer' },
          steps: { type: 'integer' },
          operationId: { type: 'string' },
          savedToDrive: { type: 'boolean' },
          attachment: { type: 'json' },
        },
      },
      render(_args, value) {
        const blocks: Array<{ type: 'text'; text: string } | { type: 'image'; attachment: ImageAttachmentRef }> = [
          {
            type: 'text',
            text:
              `Generated image ${value.filename} (${value.width}x${value.height}, `
              + `${value.steps} steps, quality ${value.quality ?? 'detailed'}, `
              + `model ${value.model}, attachment ${value.attachmentId}).`,
          },
        ]
        if (isImageAttachmentRef(value.attachment)) {
          blocks.push({ type: 'image', attachment: value.attachment })
        }
        return blocks
      },
      presentationMeta(_args, value) {
        return {
          attachmentId: value.attachmentId ?? '',
          model: value.model ?? '',
          quality: value.quality ?? 'detailed',
          filename: value.filename ?? '',
          width: value.width ?? 0,
          height: value.height ?? 0,
          steps: value.steps ?? 0,
          operationId: value.operationId ?? '',
          savedToDrive: value.savedToDrive ?? false,
        }
      },
    },
    async execute(args, exec) {
      const apiKey = process.env[config.apiKeyEnv] ?? ''
      if (!apiKey) {
        throw new Error(`Missing Nexus gateway bearer token in env var ${config.apiKeyEnv}`)
      }
      const endpoint = new URL('/v1/images/generations', baseUrl)
      const timeoutSignal = AbortSignal.timeout(config.imageTimeoutMs)
      const signal = AbortSignal.any([exec.signal, timeoutSignal])
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          prompt: args.prompt,
          model: args.model,
          quality: args.quality,
          width: args.width,
          height: args.height,
          steps: args.steps,
          seed: args.seed,
        }),
        signal,
      })
      if (!response.ok) {
        const text = (await response.text()).slice(0, 500)
        throw new Error(`Nexus image generation failed (HTTP ${response.status}): ${text}`)
      }
      const rawText = await response.text()
      if (rawText.length > MAX_IMAGE_RESPONSE_BYTES) {
        throw new Error('Nexus image response exceeded maximum allowed size')
      }
      const payload = JSON.parse(rawText) as GatewayImagePayload
      if (!payload.b64_png || !payload.filename) {
        throw new Error('Nexus image response missing required image payload or filename')
      }
      const bytes = Uint8Array.from(Buffer.from(payload.b64_png, 'base64'))
      const ref: ImageAttachmentRef = await ctx.attachments.saveImage({
        data: bytes,
        mediaType: 'image/png',
        name: payload.filename,
      })
      const attachmentJson: Record<string, string | number> = {
        attachmentId: String(ref.attachmentId),
        mediaType: ref.mediaType,
        bytes: ref.bytes,
        width: ref.width,
        height: ref.height,
        ...(ref.name === undefined ? {} : { name: ref.name }),
      }
      return {
        attachmentId: String(ref.attachmentId),
        model: payload.model ?? args.model ?? 'flux',
        quality: payload.quality ?? args.quality ?? 'detailed',
        filename: payload.filename,
        width: payload.width ?? ref.width,
        height: payload.height ?? ref.height,
        steps: payload.steps ?? args.steps ?? 0,
        operationId: payload.operation_id ?? '',
        savedToDrive: Boolean(payload.master_path),
        attachment: attachmentJson,
      }
    },
  })
}

function isImageAttachmentRef(value: unknown): value is ImageAttachmentRef {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const rec = value as Record<string, unknown>
  return typeof rec.attachmentId === 'string'
    && typeof rec.mediaType === 'string'
    && typeof rec.bytes === 'number'
    && typeof rec.width === 'number'
    && typeof rec.height === 'number'
}

/** Compact expandable client-side progress pill script injected when webServer is active. */
const COMPACT_PROGRESS_PILL_SCRIPT = `(function(){
  if (globalThis.__NEXUS_PROGRESS_PILL__) return;
  globalThis.__NEXUS_PROGRESS_PILL__ = true;
  var root = document.createElement('div');
  root.id = 'nexus-runtime-pill';
  root.style.cssText = 'position:fixed;bottom:14px;right:14px;z-index:9999;font:12px/1.35 system-ui,sans-serif;color:#e5e7eb;display:none;';
  var btn = document.createElement('button');
  btn.type = 'button';
  btn.style.cssText = 'background:#18181b;border:1px solid #3f3f46;color:#e5e7eb;border-radius:999px;padding:5px 11px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.35);';
  var panel = document.createElement('div');
  panel.style.cssText = 'display:none;margin-bottom:6px;background:#18181b;border:1px solid #3f3f46;border-radius:10px;padding:10px 12px;width:290px;box-shadow:0 8px 24px rgba(0,0,0,.45);white-space:pre-wrap;';
  btn.onclick = function(){ panel.style.display = panel.style.display === 'none' ? 'block' : 'none'; };
  root.appendChild(panel);
  root.appendChild(btn);
  document.addEventListener('DOMContentLoaded', function(){ document.body.appendChild(root); });
  if (document.body) document.body.appendChild(root);
  async function poll(){
    if (document.hidden) return;
    try {
      var r = await fetch('/api/nexus.runtime-progress', { credentials: 'same-origin' });
      if (!r.ok) return;
      var d = await r.json();
      var ops = Array.isArray(d.operations) ? d.operations : [];
      var active = ops.slice().reverse().find(function(o){ return o && (o.status === 'running' || o.status === 'queued'); });
      if (!active) { root.style.display = 'none'; return; }
      root.style.display = 'block';
      var pct = active.indeterminate ? '' : (' — ' + Math.round(active.progress || 0) + '%');
      btn.textContent = (active.phase || active.kind) + ' ' + (active.model || '') + pct;
      panel.textContent =
        'Operation: ' + (active.operation_id || '') + '\\n' +
        'Model: ' + (active.model || '') + '\\n' +
        'Stage: ' + (active.phase || '') + ' (' + (active.status || '') + ')\\n' +
        (active.total_bytes ? ('Bytes: ' + active.bytes_copied + ' / ' + active.total_bytes + '\\n') : '') +
        (active.steps_total ? ('Steps: ' + active.step + ' / ' + active.steps_total + '\\n') : '') +
        (active.message ? ('Status: ' + active.message + '\\n') : '') +
        (active.error ? ('Error: ' + active.error) : '');
    } catch (_) {}
  }
  setInterval(poll, 3000);
})();`

function registerGatewayProxy(
  ctx: Context,
  config: Config,
  baseUrl: URL,
  publicPath: string,
  upstreamPath: string,
): () => void {
  return ctx.connection.fetch.register({
    path: publicPath,
    methods: ['GET'],
    requestBody: 'buffered',
    async fetch(request) {
      const apiKey = process.env[config.apiKeyEnv] ?? ''
      if (!apiKey) {
        return new Response(
          JSON.stringify({ error: 'Nexus gateway token not configured on Host' }),
          { status: 503, headers: { 'content-type': 'application/json' } },
        )
      }
      const endpoint = new URL(upstreamPath, baseUrl)
      const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
      try {
        const upstream = await fetch(endpoint, {
          method: 'GET',
          headers: { authorization: `Bearer ${apiKey}` },
          signal,
        })
        const body = await upstream.text()
        if (body.length > MAX_PROGRESS_RESPONSE_BYTES) {
          return new Response(
            JSON.stringify({ error: 'Upstream payload too large' }),
            { status: 502, headers: { 'content-type': 'application/json' } },
          )
        }
        return new Response(body, {
          status: upstream.status,
          headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
        })
      } catch {
        return new Response(
          JSON.stringify({ error: 'Nexus gateway unavailable' }),
          { status: 502, headers: { 'content-type': 'application/json' } },
        )
      }
    },
  })
}

/**
 * Register the `nexus_generate_image` tool and same-origin Host proxies
 * for runtime progress, hardware/worker status, and independent image models.
 */
export function apply(ctx: Context, config: Config): void {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  const tool = createNexusGenerateImageTool(ctx, config)
  ctx.effect(() => ctx.tools.register(tool), 'nexus-host: nexus_generate_image tool')

  ctx.effect(
    () => registerGatewayProxy(ctx, config, baseUrl, '/api/nexus.runtime-progress', '/v1/progress'),
    'nexus-host: /api/nexus.runtime-progress proxy',
  )
  ctx.effect(
    () => registerGatewayProxy(ctx, config, baseUrl, '/api/nexus.runtime', '/v1/runtime'),
    'nexus-host: /api/nexus.runtime proxy',
  )
  ctx.effect(
    () => registerGatewayProxy(ctx, config, baseUrl, '/api/nexus.image-models', '/v1/image-models'),
    'nexus-host: /api/nexus.image-models proxy',
  )
  ctx.inject(['webServer'], (webCtx) => {
    webCtx.on('webserver/index-inject', (table) => {
      table.push({ kind: 'script', placement: 'head', text: COMPACT_PROGRESS_PILL_SCRIPT })
    })
  })
}
