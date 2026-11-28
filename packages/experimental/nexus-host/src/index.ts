/**
 * Nexus Host plugin (v2.3.0 — Intent-Guarded Image Generation & Workspace Sync):
 * Registers `nexus_generate_image` (with pre-GPU visual-question intent guard and
 * `fs/observed` workspace file-tree invalidations for Google Drive FUSE),
 * `nexus_web_research`, `nexus_inspect_runtime`, and read-only authenticated
 * same-origin Host proxies for live Gateway runtime status and operation progress.
 * @module @deepseek-ai/dsh-nexus-host
 */

import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type {} from '@deepseek-ai/dsh-client-connection'
import type { FileSystem } from '@deepseek-ai/dsh-fs'
import type {} from '@deepseek-ai/dsh-system-prompt'
import { defineTool, ToolArgsError } from '@deepseek-ai/dsh-tools'
import type { ToolDefinition, ToolExecution } from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-working-directory'
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

const VISUAL_QUESTION_RE = new RegExp(
  '\\b(?:who\\s+is|what\\s+is|what\'s\\s+in|identify|recognize|recognise|analyze|analyse|'
  + 'describe|explain\\s+(?:this|the)\\s+(?:image|photo|picture)|ocr|read\\s+(?:the\\s+)?text|'
  + 'tell\\s+me\\s+(?:who|what)|can\\s+you\\s+(?:see|analyze|identify))\\b',
  'i',
)
const EXPLICIT_IMAGE_GEN_RE = new RegExp(
  '\\b(?:generate|create|draw|paint|render|illustrate|make\\s+(?:an?\\s+|me\\s+)?'
  + '(?:image|picture|photo|portrait|artwork|wallpaper|logo|illustration)|edit|modify|restyle|stylize|'
  + 'transform|turn\\s+(?:this|him|her|it|them)\\s+into|change\\s+(?:the\\s+)?'
  + '(?:background|style|outfit|hair|lighting|color)|add\\s+.*\\s+to\\s+(?:the\\s+)?(?:image|photo)|'
  + 'remove\\s+.*\\s+from\\s+(?:the\\s+)?(?:image|photo)|upscale|remix|img2img)\\b',
  'i',
)

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

/**
 * Verify that the configured gateway URL targets loopback only.
 * @param rawUrl - Candidate gateway base URL.
 * @returns Validated loopback URL instance.
 */
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

/**
 * Check whether an image generation call is actually a hallucinated attempt to identify
 * or analyze an uploaded image when the user asked a visual question without requesting generation.
 * @param userText - Latest human prompt text in the session.
 * @param promptArg - Tool `prompt` argument passed by the model.
 * @param hasUploadedImage - Whether the latest user turn attached an image or `use_uploaded_image` is set.
 * @returns `true` when the call should be blocked before touching the GPU.
 */
export function shouldBlockUnsolicitedImageGen(
  userText: string,
  promptArg: string,
  hasUploadedImage: boolean,
): boolean {
  const cleanUser = userText.trim()
  const cleanPrompt = promptArg.trim()
  if (VISUAL_QUESTION_RE.test(cleanPrompt) && !EXPLICIT_IMAGE_GEN_RE.test(cleanPrompt)) {
    return true
  }
  if (!cleanUser) return false
  if (EXPLICIT_IMAGE_GEN_RE.test(cleanUser)) {
    return false
  }
  if (VISUAL_QUESTION_RE.test(cleanUser)) {
    return true
  }
  if (hasUploadedImage && cleanUser.endsWith('?')) {
    return true
  }
  return false
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

interface LatestUserTurnSnapshot {
  text: string
  hasImage: boolean
  imageAttachment?: ImageAttachmentRef
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function extractMessageContentFromEvent(parsed: Record<string, unknown>): {
  role: string
  content: unknown[]
  isRuntimeNotice: boolean
} | undefined {
  // Handle both SessionEvent envelope (`{ type: 'user/message', data: { message: { role, content, source } } }`)
  // and flat legacy JSONL test fixtures (`{ role: 'user', content: [...] }`).
  const evType = typeof parsed.type === 'string' ? parsed.type : ''
  const data = isRecord(parsed.data) ? parsed.data : undefined
  const msg = data && isRecord(data.message) ? data.message : parsed
  const role = typeof msg.role === 'string' ? msg.role : (evType === 'user/message' ? 'user' : '')
  const content = Array.isArray(msg.content) ? msg.content : undefined
  if (!content) return undefined
  const source = isRecord(msg.source) ? msg.source : undefined
  const sourceKind = typeof source?.kind === 'string' ? source.kind : ''
  const isRuntimeNotice = sourceKind !== '' && sourceKind !== 'user' && sourceKind !== 'prompt'
  return { role, content, isRuntimeNotice }
}

async function inspectLatestUserTurn(
  sessionId: string,
): Promise<LatestUserTurnSnapshot> {
  if (!sessionId) return { text: '', hasImage: false }
  const dshHome = process.env.DSH_HOME?.trim() || path.join(os.homedir(), '.dsh')
  const candidates = [
    path.join(dshHome, 'sessions', sessionId, 'events.jsonl'),
    path.join(dshHome, 'sessions', `${sessionId}.jsonl`),
  ]
  for (const jsonlPath of candidates) {
    let raw: string
    try {
      raw = await fs.readFile(jsonlPath, 'utf8')
    } catch (_readErr) {
      // Try next candidate session log path when absent.
      continue
    }
    const lines = raw.trim().split('\n')
    let latestUserText = ''
    let latestUserHasImg = false
    let latestImgRef: ImageAttachmentRef | undefined

    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i]?.trim()
      if (!line) continue
      let parsed: unknown
      try {
        parsed = JSON.parse(line)
      } catch (_jsonErr) {
        // Skip malformed trailing JSONL line.
        continue
      }
      if (!isRecord(parsed)) continue
      const extracted = extractMessageContentFromEvent(parsed)
      if (!extracted || extracted.role !== 'user' || extracted.isRuntimeNotice) continue

      const textParts: string[] = []
      let foundImgInTurn = false
      for (let j = extracted.content.length - 1; j >= 0; j--) {
        const block = extracted.content[j]
        if (!isRecord(block)) continue
        if (block.type === 'text' && typeof block.text === 'string') {
          textParts.unshift(block.text)
        } else if (block.type === 'image') {
          foundImgInTurn = true
          const att = block.attachment ?? block
          if (latestImgRef === undefined && isImageAttachmentRef(att)) {
            latestImgRef = att
          }
        }
      }
      if (!latestUserText && (textParts.length > 0 || foundImgInTurn)) {
        latestUserText = textParts.join('\n').trim()
        latestUserHasImg = foundImgInTurn
        if (latestImgRef !== undefined) break
      }
    }
    return {
      text: latestUserText,
      hasImage: latestUserHasImg,
      ...(latestImgRef === undefined ? {} : { imageAttachment: latestImgRef }),
    }
  }
  return { text: '', hasImage: false }
}

async function resolveLastUploadedImageB64(
  ctx: Context,
  sessionId: string,
  signal: AbortSignal,
): Promise<string | undefined> {
  const turnInfo = await inspectLatestUserTurn(sessionId)
  if (turnInfo.imageAttachment !== undefined) {
    const stored = await ctx.attachments.readImage(turnInfo.imageAttachment, signal)
    return Buffer.from(stored.data).toString('base64')
  }
  return undefined
}

function isImageAttachmentRef(value: unknown): value is ImageAttachmentRef {
  if (!isRecord(value)) return false
  return typeof value.attachmentId === 'string'
    && typeof value.mediaType === 'string'
    && typeof value.bytes === 'number'
    && typeof value.width === 'number'
    && typeof value.height === 'number'
}

async function observeSinglePath(
  ctx: Context,
  hostFs: FileSystem,
  candidatePath: string,
  exec: ToolExecution,
): Promise<void> {
  const target = await hostFs.resolve(candidatePath, { signal: exec.signal })
  const info = await hostFs.stat(target, exec.signal)
  if (info !== undefined) {
    ctx.emit('fs/observed', target, { kind: 'present', version: info.version }, exec)
  }
}

async function emitWorkspaceImageObserved(
  ctx: Context,
  workspaceRoot: string,
  filename: string,
  pngBytes: Uint8Array,
  exec: ToolExecution,
): Promise<string> {
  if (!workspaceRoot.trim() || !filename.trim()) return ''
  const genDir = path.join(workspaceRoot, 'generated_images')
  const fullFilePath = path.join(genDir, filename)
  try {
    await fs.mkdir(genDir, { recursive: true })
  } catch (_mkdirErr) {
    // Best-effort workspace mirror; Drive master copy is already saved.
    return ''
  }
  let existsOnDisk = true
  try {
    await fs.access(fullFilePath)
  } catch (_missingErr) {
    // Gateway had not mirrored into workspaceRoot yet; write PNG bytes from Host.
    existsOnDisk = false
  }
  if (!existsOnDisk) {
    try {
      await fs.writeFile(fullFilePath, pngBytes)
    } catch (_writeErr) {
      // Best-effort workspace mirror; Drive master copy is already saved.
      return ''
    }
  }

  const hostFs: FileSystem | undefined = ctx.get('fs')
  if (hostFs !== undefined) {
    for (const candidatePath of [workspaceRoot, genDir, fullFilePath]) {
      try {
        await observeSinglePath(ctx, hostFs, candidatePath, exec)
      } catch (_obsErr) {
        // Ignore non-fatal observation errors outside sandboxed roots.
      }
    }
  }
  return fullFilePath
}

function extractSessionId(exec: ToolExecution): string {
  if (exec.agent !== undefined && exec.agent.session.id) {
    return String(exec.agent.session.id)
  }
  if ('sessionId' in exec && typeof exec.sessionId === 'string') {
    return exec.sessionId
  }
  return ''
}

/**
 * Create the `nexus_generate_image` tool bound to the single Colab-configured
 * image model on the loopback Nexus gateway and the Host attachment store.
 * @param ctx - Host Cordis context.
 * @param config - Validated Nexus Host configuration.
 * @param fetchImpl - Fetch implementation (injectable for unit tests).
 * @returns Registered `ToolDefinition` for `nexus_generate_image`.
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
      'Synthesize 1 NEW or EDITED image using the local Diffusers image generator, attach it inline to the chat, '
      + 'and save a PNG to the Workspace (`generated_images/`) and Google Drive. '
      + 'CRITICAL: This tool is a blind image synthesizer — it CANNOT see, inspect, OCR, describe, or identify people/objects in uploaded images. '
      + 'ONLY call this tool when the user explicitly asks to generate, draw, create, edit, or restyle an image. '
      + 'Do not emit a Markdown ![]() image link after calling this tool.',
    parameters: {
      prompt: { type: 'string', description: 'Visual prompt describing the new or edited image to synthesize.', required: true },
      quality: {
        type: 'string',
        enum: ['normal', 'detailed', 'maximum'],
        description: 'Image quality profile (`normal`, `detailed`, or `maximum`).',
      },
      width: { type: 'integer', description: 'Optional width in pixels (256–1536, multiple of 16).' },
      height: { type: 'integer', description: 'Optional height in pixels (256–1536, multiple of 16).' },
      steps: { type: 'integer', description: 'Optional denoising step count.' },
      seed: { type: 'integer', description: 'Optional RNG seed.' },
      use_uploaded_image: {
        type: 'boolean',
        description: 'Set true ONLY when editing/restyling an image uploaded by the user on explicit user request.',
      },
      reference_image_path: {
        type: 'string',
        description: 'Optional file path to a reference image for explicit img2img editing.',
      },
      strength: {
        type: 'number',
        description: 'Img2img denoising strength from 0.1 (subtle edit) to 1.0 (full restyle), default 0.75.',
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
        const wsNote = value.workspacePath ? ` Workspace file: generated_images/${value.filename}.` : ''
        const blocks: Array<{ type: 'text'; text: string } | { type: 'image'; attachment: ImageAttachmentRef }> = [
          {
            type: 'text',
            text:
              `Generated image ${value.filename} (${value.width}x${value.height}, `
              + `${value.steps} steps, quality ${value.quality ?? 'detailed'}, `
              + `model ${value.model}, attachment ${value.attachmentId}).${wsNote} `
              + 'The image preview is already rendered beneath your reply; do not include a Markdown ![]() image link.',
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
      const sessionId = extractSessionId(exec)
      const latestTurn = await inspectLatestUserTurn(sessionId)
      const hasRefFlag = args.use_uploaded_image === true
        || (typeof args.reference_image_path === 'string' && args.reference_image_path.trim().length > 0)
        || latestTurn.hasImage
      if (shouldBlockUnsolicitedImageGen(latestTurn.text, args.prompt, hasRefFlag)) {
        throw new ToolArgsError([
          'Aborted: nexus_generate_image is an image synthesizer and CANNOT see, identify, describe, or analyze images. '
          + 'The user did not ask to generate or edit an image. Answer the user directly in plain text without calling nexus_generate_image.',
        ])
      }

      const endpoint = new URL('/v1/images/generations', baseUrl)
      const timeoutSignal = AbortSignal.timeout(config.imageTimeoutMs)
      const signal = AbortSignal.any([exec.signal, timeoutSignal])
      let rawWorkspace = process.env.NEXUS_DEFAULT_WORKSPACE?.trim() ?? ''
      const workingDir = ctx.get('workingDirectory')
      if (exec.agent !== undefined && workingDir !== undefined) {
        try {
          rawWorkspace = await workingDir.ensure(exec.agent, exec.signal)
        } catch (_wdErr) {
          // Fall back to NEXUS_DEFAULT_WORKSPACE when session working directory is unavailable.
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
        referenceImageB64 = await resolveLastUploadedImageB64(ctx, sessionId, signal)
      }

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

      const syncedWorkspaceFile = rawWorkspace
        ? await emitWorkspaceImageObserved(ctx, rawWorkspace, payload.filename, bytes, exec)
        : ''
      const finalWorkspacePath = payload.workspace_path || syncedWorkspaceFile

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
        workspacePath: finalWorkspacePath,
        masterPath: finalWorkspacePath || payload.master_path || payload.drive_path || '',
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
    } else if (!message && isRecord(detail)) {
      const errObj = isRecord(detail.error) ? detail.error : undefined
      if (typeof errObj?.message === 'string') {
        message = errObj.message
      } else if (typeof detail.message === 'string') {
        message = detail.message
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
  } catch (_jsonErr) {
    // Upstream returned non-JSON text; truncate and wrap safely.
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
 * Register `nexus_generate_image` (when an image model is enabled), `nexus_web_research`,
 * `nexus_inspect_runtime`, and read-only same-origin Host endpoints for live Gateway status.
 * @param ctx - Host Cordis context.
 * @param config - Validated Nexus Host configuration.
 * @param fetchImpl - Optional fetch override for tests.
 */
export function apply(
  ctx: Context,
  config: Config,
  fetchImpl: typeof fetch = fetch,
): void {
  const baseUrl = assertLoopbackGatewayUrl(config.gatewayUrl)

  if (process.env.NEXUS_IMAGE_MODEL_ENABLED !== '0') {
    const tool = createNexusGenerateImageTool(ctx, config, fetchImpl)
    ctx.effect(() => ctx.tools.register(tool), 'nexus-host: nexus_generate_image tool')

    const sysPrompt = ctx.get('systemPrompt')
    if (sysPrompt !== undefined) {
      sysPrompt.section({
        name: 'tool:nexus_generate_image',
        order: 2950,
        text:
          'Only call `nexus_generate_image` when the user explicitly asks to create, draw, generate, edit, or restyle an image. '
          + '`nexus_generate_image` cannot see, OCR, describe, or identify uploaded images; never call it to analyze an attached image.',
        retirement: 'The nexus_generate_image tool is no longer available.',
      })
    }
  }

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
