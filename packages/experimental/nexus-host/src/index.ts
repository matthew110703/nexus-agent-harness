/**
 * Nexus Host plugin: registers the `nexus_generate_image` tool (backed by
 * `ctx.attachments.saveImage`), profile-level image-model persistence, and
 * authenticated same-origin Host proxies to the loopback Nexus gateway.
 * @module @deepseek-ai/dsh-nexus-host
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type {} from '@deepseek-ai/dsh-client-connection'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolDefinition } from '@deepseek-ai/dsh-tools'

/** Stable plugin name. */
export const name = 'nexus-host'
/** Required Host services: tool registry, attachment store, and connection fetch router. */
export const inject = ['tools', 'attachments', 'connection']

/** Maximum image response size accepted from the loopback Nexus gateway (24 MiB). */
const MAX_IMAGE_RESPONSE_BYTES = 24 * 1024 * 1024
/** Maximum JSON proxy snapshot size relayed to the browser (256 KiB). */
const MAX_PROXY_RESPONSE_BYTES = 256 * 1024
const IMAGE_SELECTION_FILENAME = 'nexus-image-selection.json'

/** Validated configuration for the Nexus Host plugin. */
export interface Config {
  /** Loopback base URL of the Nexus gateway (e.g. `http://127.0.0.1:7861`). */
  gatewayUrl: string
  /** Environment variable name holding the Nexus bearer token. */
  apiKeyEnv: string
  /** Timeout in milliseconds for image generation calls. */
  imageTimeoutMs: number
  /** Timeout in milliseconds for runtime/progress proxy calls. */
  progressTimeoutMs: number
}

/** Configuration schema with safe loopback defaults. */
export const Config: z<Config> = z.object({
  gatewayUrl: z.string().default('http://127.0.0.1:7861'),
  apiKeyEnv: z.string().default('NEXUS_API_KEY'),
  imageTimeoutMs: z.natural().min(1000).default(900_000),
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

interface ImageSelectionRecord {
  selectedModelId: string | null
  updatedAt: string
}

/** Resolve the profile-level persistence file path (profileDir -> DSH_HOME -> cwd). */
export function resolveImageSelectionPath(ctx: Context): string {
  const profileCtx = ctx.get('profileContext') as { readonly profileDir?: string } | undefined
  const baseDir = profileCtx?.profileDir || process.env.DSH_HOME || process.cwd()
  return join(baseDir, IMAGE_SELECTION_FILENAME)
}

/** Read the persisted profile-level selected image model ID, or null when unset. */
export function readSelectedImageModel(selectionPath: string): string | null {
  try {
    const raw = readFileSync(selectionPath, 'utf8')
    const parsed = JSON.parse(raw) as Partial<ImageSelectionRecord>
    return typeof parsed.selectedModelId === 'string' && parsed.selectedModelId.trim().length > 0
      ? parsed.selectedModelId.trim()
      : null
  } catch {
    return null
  }
}

/** Atomically persist the profile-level selected image model ID. */
export function writeSelectedImageModel(selectionPath: string, modelId: string | null): void {
  const dir = dirname(selectionPath)
  mkdirSync(dir, { recursive: true })
  const record: ImageSelectionRecord = {
    selectedModelId: modelId,
    updatedAt: new Date().toISOString(),
  }
  const tmpPath = `${selectionPath}.${process.pid}.tmp`
  writeFileSync(tmpPath, `${JSON.stringify(record, null, 2)}\n`, 'utf8')
  renameSync(tmpPath, selectionPath)
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

function isImageAttachmentRef(value: unknown): value is ImageAttachmentRef {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const rec = value as Record<string, unknown>
  return typeof rec.attachmentId === 'string'
    && typeof rec.mediaType === 'string'
    && typeof rec.bytes === 'number'
    && typeof rec.width === 'number'
    && typeof rec.height === 'number'
}

/**
 * Create the `nexus_generate_image` tool definition bound to the Host's
 * attachment store, profile-persisted image model selection, and loopback gateway.
 */
export function createNexusGenerateImageTool(
  ctx: Context,
  config: Config,
  fetchImpl: typeof fetch = fetch,
  selectionPath: string = resolveImageSelectionPath(ctx),
): ToolDefinition {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  return defineTool({
    name: 'nexus_generate_image',
    description:
      'Generate 1 image using the selected local Nexus Diffusers (FLUX.2) pipeline independent of the active chat model '
      + 'and attach it durably to the conversation while saving a PNG master copy to Google Drive.',
    parameters: {
      prompt: { type: 'string', description: 'Text prompt describing the image to generate.', required: true },
      model: { type: 'string', description: 'Optional per-call image model override; defaults to the profile-selected image model.' },
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
      const chosenModel = (typeof args.model === 'string' && args.model.trim().length > 0)
        ? args.model.trim()
        : readSelectedImageModel(selectionPath)
      if (!chosenModel) {
        throw new Error(
          'No image model selected. Choose an image model in the Nexus runtime selector or pass the model argument explicitly.',
        )
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
          model: chosenModel,
          quality: args.quality,
          width: args.width,
          height: args.height,
          steps: args.steps,
          seed: args.seed,
          num_images: 1,
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
        model: payload.model ?? chosenModel,
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

const OPERATION_ID_RE = /^[A-Za-z0-9_.-]{1,128}$/

function redactSecrets(text: string, apiKey: string): string {
  let out = text
  if (apiKey) {
    out = out.split(apiKey).join('[REDACTED]')
  }
  return out.replace(/Bearer\s+[a-z0-9._~+/=-]{8,}/gi, 'Bearer [REDACTED]')
}

function normalizeUpstreamError(status: number, rawBody: string, apiKey: string): string {
  const safeRaw = redactSecrets(rawBody, apiKey)
  try {
    const parsed = JSON.parse(safeRaw) as Record<string, unknown>
    const detail = parsed.detail
    let message = typeof parsed.error === 'string' ? parsed.error : ''
    if (!message && typeof detail === 'string') {
      message = detail
    } else if (!message && detail && typeof detail === 'object') {
      const nested = detail as Record<string, unknown>
      const errObj = nested.error as Record<string, unknown> | undefined
      if (typeof errObj?.message === 'string') {
        message = errObj.message
      } else if (typeof nested.message === 'string') {
        message = nested.message
      }
    }
    if (!message) {
      message = `Gateway request failed (HTTP ${status})`
    }
    const payload: Record<string, unknown> = {
      error: redactSecrets(message, apiKey),
      status,
    }
    if (typeof parsed.operation_id === 'string') {
      payload.operation_id = parsed.operation_id
    }
    if (safeRaw.length > 0 && safeRaw.length <= 2048) {
      payload.details = safeRaw
    }
    return JSON.stringify(payload)
  } catch {
    return JSON.stringify({
      error: safeRaw.trim().slice(0, 300) || `Gateway request failed (HTTP ${status})`,
      status,
    })
  }
}

async function fetchGatewayJson(
  config: Config,
  baseUrl: URL,
  upstreamPath: string,
  method: 'GET' | 'POST',
  signal: AbortSignal,
  body?: unknown,
  fetchImpl: typeof fetch = fetch,
): Promise<{ status: number; body: string }> {
  const apiKey = process.env[config.apiKeyEnv] ?? ''
  if (!apiKey) {
    return { status: 503, body: JSON.stringify({ error: 'Nexus gateway token not configured on Host' }) }
  }
  const endpoint = new URL(upstreamPath, baseUrl)
  const upstream = await fetchImpl(endpoint, {
    method,
    headers: {
      authorization: `Bearer ${apiKey}`,
      ...(body === undefined ? {} : { 'content-type': 'application/json' }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    signal,
  })
  const text = await upstream.text()
  if (text.length > MAX_PROXY_RESPONSE_BYTES) {
    return { status: 502, body: JSON.stringify({ error: 'Upstream payload too large' }) }
  }
  if (!upstream.ok) {
    return { status: upstream.status, body: normalizeUpstreamError(upstream.status, text, apiKey) }
  }
  return { status: upstream.status, body: redactSecrets(text, apiKey) }
}

function registerGatewayGetProxy(
  ctx: Context,
  config: Config,
  baseUrl: URL,
  publicPath: string,
  upstreamPath: string,
  fetchImpl: typeof fetch = fetch,
): () => void {
  return ctx.connection.fetch.register({
    path: publicPath,
    methods: ['GET'],
    requestBody: 'buffered',
    async fetch(request) {
      const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
      try {
        const res = await fetchGatewayJson(config, baseUrl, upstreamPath, 'GET', signal, undefined, fetchImpl)
        return new Response(res.body, {
          status: res.status,
          headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
        })
      } catch (err) {
        const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
        return new Response(
          JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
          { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
        )
      }
    },
  })
}

/**
 * Register the `nexus_generate_image` tool and authenticated same-origin Host
 * endpoints for progress, runtime status, load/unload, and profile image selection.
 */
export function apply(
  ctx: Context,
  config: Config,
  fetchImpl: typeof fetch = fetch,
  selectionPath: string = resolveImageSelectionPath(ctx),
): void {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  const tool = createNexusGenerateImageTool(ctx, config, fetchImpl, selectionPath)
  ctx.effect(() => ctx.tools.register(tool), 'nexus-host: nexus_generate_image tool')

  // GET /api/nexus.runtime-progress (supports optional ?operation_id=<op_id>)
  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.runtime-progress',
        methods: ['GET'],
        requestBody: 'buffered',
        async fetch(request) {
          const reqUrl = new URL(request.url, 'http://localhost')
          const opId = reqUrl.searchParams.get('operation_id')?.trim() ?? ''
          let upstreamPath = '/v1/progress'
          if (opId.length > 0) {
            if (!OPERATION_ID_RE.test(opId)) {
              return new Response(
                JSON.stringify({ error: 'Invalid operation_id format' }),
                { status: 400, headers: { 'content-type': 'application/json' } },
              )
            }
            upstreamPath = `/v1/progress/${encodeURIComponent(opId)}`
          }
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
          try {
            const res = await fetchGatewayJson(config, baseUrl, upstreamPath, 'GET', signal, undefined, fetchImpl)
            return new Response(res.body, {
              status: res.status,
              headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
            })
          } catch (err) {
            const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
            return new Response(
              JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
              { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
            )
          }
        },
      }),
    'nexus-host: /api/nexus.runtime-progress proxy',
  )

  ctx.effect(
    () => registerGatewayGetProxy(ctx, config, baseUrl, '/api/nexus.runtime', '/v1/runtime', fetchImpl),
    'nexus-host: /api/nexus.runtime proxy',
  )

  // GET /api/nexus.image-models: combines live Gateway image catalog with profile selection.
  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.image-models',
        methods: ['GET'],
        requestBody: 'buffered',
        async fetch(request) {
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
          try {
            const res = await fetchGatewayJson(config, baseUrl, '/v1/image-models', 'GET', signal, undefined, fetchImpl)
            if (res.status !== 200) {
              return new Response(res.body, { status: res.status, headers: { 'content-type': 'application/json' } })
            }
            const parsed = JSON.parse(res.body) as { data?: Array<{ id: string }>; quality_profiles?: string[] }
            const selectedModelId = readSelectedImageModel(selectionPath)
            return new Response(
              JSON.stringify({
                models: parsed.data ?? [],
                qualityProfiles: parsed.quality_profiles ?? ['normal', 'detailed', 'maximum'],
                selectedModelId,
              }),
              { status: 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } },
            )
          } catch (err) {
            const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
            return new Response(
              JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
              { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
            )
          }
        },
      }),
    'nexus-host: /api/nexus.image-models proxy',
  )

  // POST /api/nexus.image-model: validates modelId against live Gateway catalog and persists to profile.
  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.image-model',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request) {
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
          let body: { modelId?: string | null }
          try {
            body = JSON.parse(await request.text()) as { modelId?: string | null }
          } catch {
            return new Response(JSON.stringify({ error: 'Invalid JSON body' }), { status: 400, headers: { 'content-type': 'application/json' } })
          }
          const rawId = body.modelId
          let models: Array<{ id: string }> = []
          let qualityProfiles: string[] = ['normal', 'detailed', 'maximum']
          try {
            const catalogRes = await fetchGatewayJson(config, baseUrl, '/v1/image-models', 'GET', signal, undefined, fetchImpl)
            if (catalogRes.status !== 200) {
              return new Response(catalogRes.body, { status: catalogRes.status, headers: { 'content-type': 'application/json' } })
            }
            const parsed = JSON.parse(catalogRes.body) as { data?: Array<{ id: string }>; quality_profiles?: string[] }
            models = parsed.data ?? []
            qualityProfiles = parsed.quality_profiles ?? qualityProfiles
          } catch (err) {
            const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
            return new Response(
              JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
              { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
            )
          }
          if (rawId === null || rawId === '') {
            writeSelectedImageModel(selectionPath, null)
            return new Response(
              JSON.stringify({ selectedModelId: null, models, qualityProfiles }),
              { status: 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } },
            )
          }
          if (typeof rawId !== 'string') {
            return new Response(JSON.stringify({ error: 'modelId must be a string or null' }), { status: 400, headers: { 'content-type': 'application/json' } })
          }
          const valid = models.some(m => m.id === rawId)
          if (!valid) {
            return new Response(
              JSON.stringify({ error: `Image model "${rawId}" is not available in the Nexus catalog` }),
              { status: 400, headers: { 'content-type': 'application/json' } },
            )
          }
          writeSelectedImageModel(selectionPath, rawId)
          return new Response(
            JSON.stringify({ selectedModelId: rawId, models, qualityProfiles }),
            { status: 200, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } },
          )
        },
      }),
    'nexus-host: /api/nexus.image-model endpoint',
  )

  // POST /api/nexus.runtime-load & /api/nexus.runtime-unload
  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.runtime-load',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request) {
          let body: { model?: string; kind?: 'chat' | 'diffusion'; async?: boolean }
          try {
            body = JSON.parse(await request.text()) as { model?: string; kind?: 'chat' | 'diffusion'; async?: boolean }
          } catch {
            return new Response(JSON.stringify({ error: 'Invalid JSON body' }), { status: 400, headers: { 'content-type': 'application/json' } })
          }
          if (typeof body.model !== 'string' || body.model.trim().length === 0) {
            return new Response(JSON.stringify({ error: 'model is required' }), { status: 400, headers: { 'content-type': 'application/json' } })
          }
          const kind = body.kind === 'diffusion' ? 'diffusion' : 'chat'
          const asyncMode = Boolean(body.async)
          const timeoutMs = asyncMode ? Math.max(config.progressTimeoutMs, 15_000) : config.imageTimeoutMs
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(timeoutMs)])
          try {
            const res = await fetchGatewayJson(
              config,
              baseUrl,
              '/v1/runtime/load',
              'POST',
              signal,
              { model: body.model.trim(), kind, async: asyncMode },
              fetchImpl,
            )
            return new Response(res.body, {
              status: res.status,
              headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
            })
          } catch (err) {
            const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
            return new Response(
              JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
              { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
            )
          }
        },
      }),
    'nexus-host: /api/nexus.runtime-load proxy',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.runtime-unload',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request) {
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(30_000)])
          try {
            const res = await fetchGatewayJson(config, baseUrl, '/v1/runtime/unload', 'POST', signal, {}, fetchImpl)
            return new Response(res.body, {
              status: res.status,
              headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
            })
          } catch (err) {
            const isTimeout = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError')
            return new Response(
              JSON.stringify({ error: isTimeout ? 'Gateway request timed out' : 'Nexus gateway unavailable' }),
              { status: isTimeout ? 504 : 502, headers: { 'content-type': 'application/json' } },
            )
          }
        },
      }),
    'nexus-host: /api/nexus.runtime-unload proxy',
  )
}
