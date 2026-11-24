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
  model?: string
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
    description: 'Generate an image using a local Nexus Diffusers (FLUX.2) pipeline and attach it durably to the conversation.',
    parameters: {
      prompt: { type: 'string', description: 'Text prompt describing the image to generate.', required: true },
      model: { type: 'string', description: 'Optional Nexus image model ID from the Diffusers catalog.' },
      width: { type: 'integer', description: 'Optional image width in pixels (64–1536, aligned to 16).' },
      height: { type: 'integer', description: 'Optional image height in pixels (64–1536, aligned to 16).' },
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
          filename: { type: 'string' },
          width: { type: 'integer' },
          height: { type: 'integer' },
          steps: { type: 'integer' },
          operationId: { type: 'string' },
          savedToDrive: { type: 'boolean' },
        },
      },
      render(_args, value) {
        return [
          {
            type: 'text',
            text: `Generated image ${value.filename} (${value.width}x${value.height}, ${value.steps} steps, model ${value.model}, attachment ${value.attachmentId}).`,
          },
        ]
      },
      presentationMeta(_args, value) {
        return {
          attachmentId: value.attachmentId ?? '',
          model: value.model ?? '',
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
      return {
        attachmentId: String(ref.attachmentId),
        model: payload.model ?? args.model ?? 'flux',
        filename: payload.filename,
        width: payload.width ?? ref.width,
        height: payload.height ?? ref.height,
        steps: payload.steps ?? args.steps ?? 0,
        operationId: payload.operation_id ?? '',
        savedToDrive: Boolean(payload.master_path),
      }
    },
  })
}

/**
 * Register the `nexus_generate_image` tool and the same-origin
 * `/api/nexus.runtime-progress` Host proxy.
 */
export function apply(ctx: Context, config: Config): void {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  const tool = createNexusGenerateImageTool(ctx, config)
  ctx.effect(() => ctx.tools.register(tool), 'nexus-host: nexus_generate_image tool')

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.runtime-progress',
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
          const endpoint = new URL('/runtime/progress', baseUrl)
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
                JSON.stringify({ error: 'Progress payload too large' }),
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
      }),
    'nexus-host: /api/nexus.runtime-progress proxy',
  )
}
