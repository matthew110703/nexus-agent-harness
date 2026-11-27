/**
 * Nexus Host plugin (v1.1.0 — Single-Model Lock):
 * Registers the `nexus_generate_image` tool (backed by `ctx.attachments.saveImage`)
 * and read-only authenticated same-origin Host proxies for live Gateway runtime
 * status and operation progress. Model selection, staging, and loading are owned
 * exclusively by Colab bootstrap.
 * @module @deepseek-ai/dsh-nexus-host
 */

import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type {} from '@deepseek-ai/dsh-client-connection'
import type {} from '@deepseek-ai/dsh-working-directory'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolDefinition } from '@deepseek-ai/dsh-tools'
import { registerNexusPurgeRoutes } from './purge.ts'
import {
  createNexusInspectRuntimeTool,
  createNexusWebResearchTool,
  registerNexusWebProvider,
} from './web-provider.ts'

/** Stable plugin name. */
export const name = 'nexus-host'
/** Required Host services: tool registry, attachment store, and connection fetch router. */
export const inject = ['tools', 'attachments', 'connection']

/** Maximum image response size accepted from the loopback Nexus gateway (24 MiB). */
const MAX_IMAGE_RESPONSE_BYTES = 24 * 1024 * 1024
/** Maximum JSON proxy snapshot size relayed to the browser (256 KiB). */
const MAX_PROXY_RESPONSE_BYTES = 256 * 1024
const OPERATION_ID_RE = /^[A-Za-z0-9_.-]{1,128}$/

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

interface GatewayImagePayload {
  operation_id?: string
  artifact_id?: string
  model?: string
  quality?: string
  filename?: string
  master_path?: string
  drive_path?: string
  workspace_path?: string
  media_type?: string
  width?: number
  height?: number
  steps?: number
  seed?: number | null
  b64_png?: string
}

async function resolveLastUploadedImageB64(
  ctx: Context,
  sessionId: string,
  signal: AbortSignal,
): Promise<string | undefined> {
  if (!sessionId) return undefined
  const dshHome = process.env.DSH_HOME?.trim() || path.join(os.homedir(), '.dsh')
  const candidates = [
    path.join(dshHome, 'sessions', sessionId, 'events.jsonl'),
    path.join(dshHome, 'sessions', `${sessionId}.jsonl`),
  ]
  for (const jsonlPath of candidates) {
    let raw: string
    try {
      raw = await fs.readFile(jsonlPath, 'utf8')
    } catch {
      continue
    }
    const lines = raw.trim().split('\n')
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i]?.trim()
      if (!line) continue
      try {
        const ev = JSON.parse(line) as {
          type?: string
          role?: string
          content?: Array<{ type?: string; attachment?: unknown }>
        }
        if (!Array.isArray(ev.content)) continue
        for (let j = ev.content.length - 1; j >= 0; j--) {
          const block = ev.content[j]
          if (block !== undefined && block.type === 'image') {
            const att = block.attachment ?? block
            if (isImageAttachmentRef(att)) {
              const stored = await ctx.attachments.readImage(att, signal)
              return Buffer.from(stored.data).toString('base64')
            }
          }
        }
      } catch {
        // Ignore malformed JSONL lines while scanning backwards
      }
    }
  }
  return undefined
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
 * Create the `nexus_generate_image` tool bound to the single Colab-configured
 * image model on the loopback Nexus gateway and the Host attachment store.
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
      'Generate 1 image using the Colab-selected local Nexus Diffusers pipeline, '
      + 'attach it durably to the conversation (displayed inline automatically in the chat UI), '
      + 'and save a PNG master copy to Google Drive. Do not emit a Markdown image link after calling this tool.',
    parameters: {
      prompt: { type: 'string', description: 'Text prompt describing the image to generate.', required: true },
      quality: {
        type: 'string',
        enum: ['normal', 'detailed', 'maximum'],
        description: 'Explicit image quality profile independent of chat reasoning Effort.',
      },
      width: { type: 'integer', description: 'Optional image width in pixels (256–1536, aligned to 16).' },
      height: { type: 'integer', description: 'Optional image height in pixels (256–1536, aligned to 16).' },
      steps: { type: 'integer', description: 'Optional denoising step count (model-specific default when omitted).' },
      seed: { type: 'integer', description: 'Optional integer RNG seed for reproducibility.' },
      use_uploaded_image: {
        type: 'boolean',
        description: 'Use the most recent image uploaded by the user in this chat as the img2img reference.',
      },
      reference_image_path: {
        type: 'string',
        description: 'Optional workspace or Drive file path to a reference image for img2img editing.',
      },
      strength: {
        type: 'number',
        description: 'Img2img denoising strength from 0.1 (subtle edit) to 1.0 (full re-imagining), default 0.75.',
      },
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
          drivePath: { type: 'string' },
          workspacePath: { type: 'string' },
          masterPath: { type: 'string' },
          attachment: { type: 'json' },
        },
      },
      render(_args, value) {
        const pathNote = value.drivePath
          ? ` Saved to Drive: ${value.drivePath}` + (value.workspacePath ? ` and Workspace: ${value.workspacePath}.` : '.')
          : ''
        const blocks: Array<{ type: 'text'; text: string } | { type: 'image'; attachment: ImageAttachmentRef }> = [
          {
            type: 'text',
            text:
              `Generated image ${value.filename} (${value.width}x${value.height}, `
              + `${value.steps} steps, quality ${value.quality ?? 'detailed'}, `
              + `model ${value.model}, attachment ${value.attachmentId}).${pathNote} `
              + 'The image is already displayed inline in the UI; do not include a Markdown ![]() image link.',
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
          drivePath: value.drivePath ?? '',
          workspacePath: value.workspacePath ?? '',
          masterPath: value.masterPath ?? '',
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
      let rawWorkspace = ''
      const workingDir = ctx.get('workingDirectory')
      if (exec.agent !== undefined && workingDir !== undefined) {
        try {
          rawWorkspace = await workingDir.ensure(exec.agent, exec.signal)
        } catch {
          rawWorkspace = ''
        }
      }

      const onAbort = () => {
        const cancelUrl = new URL('/v1/runtime/cancel', baseUrl)
        void fetchImpl(cancelUrl, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: `Bearer ${apiKey}`,
          },
          body: '{}',
          signal: AbortSignal.timeout(3000),
        }).catch(() => {
          // Best-effort cancellation; gateway restores the language model when swap unwinds.
        })
      }
      exec.signal.addEventListener('abort', onAbort, { once: true })

      let referenceImageB64: string | undefined
      if (typeof args.reference_image_path === 'string' && args.reference_image_path.trim()) {
        const resolvedRef = path.resolve(rawWorkspace || process.cwd(), args.reference_image_path.trim())
        const refBytes = await fs.readFile(resolvedRef)
        referenceImageB64 = refBytes.toString('base64')
      } else if (args.use_uploaded_image) {
        const sid = 'sessionId' in exec && typeof exec.sessionId === 'string' ? exec.sessionId : ''
        referenceImageB64 = await resolveLastUploadedImageB64(ctx, sid, signal)
      }
      const sessionId = 'sessionId' in exec && typeof exec.sessionId === 'string' ? exec.sessionId : ''

      let response: Response
      try {
        response = await fetchImpl(endpoint, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            prompt: args.prompt,
            quality: args.quality,
            width: args.width,
            height: args.height,
            steps: args.steps,
            seed: args.seed,
            num_images: 1,
            workspace: rawWorkspace || undefined,
            session_id: sessionId || undefined,
            reference_image_b64: referenceImageB64,
            strength: typeof args.strength === 'number' ? args.strength : undefined,
          }),
          signal,
        })
      } finally {
        exec.signal.removeEventListener('abort', onAbort)
      }
      if (!response.ok) {
        const errMsg = normalizeUpstreamError(response.status, await response.text(), apiKey)
        throw new Error(`Nexus image generation failed (HTTP ${response.status}): ${errMsg}`)
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
        model: payload.model ?? 'configured-image-model',
        quality: payload.quality ?? args.quality ?? 'detailed',
        filename: payload.filename,
        width: payload.width ?? ref.width,
        height: payload.height ?? ref.height,
        steps: payload.steps ?? args.steps ?? 0,
        operationId: payload.operation_id ?? '',
        savedToDrive: Boolean(payload.drive_path || payload.master_path),
        drivePath: payload.drive_path ?? payload.master_path ?? '',
        workspacePath: payload.workspace_path ?? '',
        masterPath: payload.master_path ?? '',
        attachment: attachmentJson,
      }
    },
  })
}

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
  signal: AbortSignal,
  fetchImpl: typeof fetch = fetch,
): Promise<{ status: number; body: string }> {
  const apiKey = process.env[config.apiKeyEnv] ?? ''
  if (!apiKey) {
    return { status: 503, body: JSON.stringify({ error: 'Nexus gateway token not configured on Host' }) }
  }
  const endpoint = new URL(upstreamPath, baseUrl)
  const upstream = await fetchImpl(endpoint, {
    method: 'GET',
    headers: { authorization: `Bearer ${apiKey}` },
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

/**
 * Register the `nexus_generate_image` tool and read-only same-origin Host
 * endpoints for live Gateway runtime status and operation progress.
 */
export function apply(
  ctx: Context,
  config: Config,
  fetchImpl: typeof fetch = fetch,
): void {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  const tool = createNexusGenerateImageTool(ctx, config, fetchImpl)
  ctx.effect(() => ctx.tools.register(tool), 'nexus-host: nexus_generate_image tool')

  const researchTool = createNexusWebResearchTool(baseUrl.toString(), process.env[config.apiKeyEnv] ?? '', fetchImpl)
  ctx.effect(() => ctx.tools.register(researchTool), 'nexus-host: nexus_web_research tool')

  const runtimeTool = createNexusInspectRuntimeTool(baseUrl.toString(), process.env[config.apiKeyEnv] ?? '', fetchImpl)
  ctx.effect(() => ctx.tools.register(runtimeTool), 'nexus-host: nexus_inspect_runtime tool')

  registerNexusWebProvider(ctx, baseUrl.toString(), process.env[config.apiKeyEnv] ?? '', fetchImpl)
  registerNexusPurgeRoutes(ctx)

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
            const res = await fetchGatewayJson(config, baseUrl, upstreamPath, signal, fetchImpl)
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
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.runtime',
        methods: ['GET'],
        requestBody: 'buffered',
        async fetch(request) {
          const signal = AbortSignal.any([request.signal, AbortSignal.timeout(config.progressTimeoutMs)])
          try {
            const res = await fetchGatewayJson(config, baseUrl, '/v1/runtime', signal, fetchImpl)
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
    'nexus-host: /api/nexus.runtime proxy',
  )
}
