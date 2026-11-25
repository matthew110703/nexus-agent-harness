import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import {
  apply,
  assertLoopbackGatewayUrl,
  createNexusGenerateImageTool,
  readSelectedImageModel,
  writeSelectedImageModel,
} from '../src/index.ts'

describe('nexus-host', () => {
  it('enforces loopback-only gateway URLs', () => {
    expect(assertLoopbackGatewayUrl('http://127.0.0.1:7861').hostname).toBe('127.0.0.1')
    expect(assertLoopbackGatewayUrl('http://localhost:7861').hostname).toBe('localhost')
    expect(() => assertLoopbackGatewayUrl('https://evil.example.com')).toThrow(/loopback-only/)
  })

  it('persists profile image-model selection, rejects when unset, and attaches PNG durably', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'nexus-host-'))
    const selPath = join(dir, 'nexus-image-selection.json')
    try {
      process.env.TEST_NEXUS_KEY = 'secret-token-123'
      const saveImage = vi.fn(async () => ({
        attachmentId: 'att_01',
        mediaType: 'image/png' as const,
        bytes: 12,
        width: 256,
        height: 256,
        name: 'nexus_abc.png',
      }))
      const ctx = new Context()
      Object.assign(ctx, { attachments: { saveImage } })

      const fakeFetch = vi.fn(async (_url: URL | RequestInfo, init?: RequestInit) => {
        expect((init?.headers as Record<string, string>).authorization).toBe('Bearer secret-token-123')
        const sent = JSON.parse(String(init?.body)) as { model: string; num_images: number }
        expect(sent.model).toBe('FLUX.2-klein-9B')
        expect(sent.num_images).toBe(1)
        return new Response(
          JSON.stringify({
            operation_id: 'op_1',
            model: 'FLUX.2-klein-9B',
            filename: 'nexus_abc.png',
            master_path: '/content/drive/MyDrive/Nexus/outputs/generated_images/nexus_abc.png',
            width: 256,
            height: 256,
            steps: 4,
            b64_png: Buffer.from('FAKEPNG').toString('base64'),
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        )
      })

      const tool = createNexusGenerateImageTool(
        ctx,
        {
          gatewayUrl: 'http://127.0.0.1:7861',
          apiKeyEnv: 'TEST_NEXUS_KEY',
          imageTimeoutMs: 5000,
          progressTimeoutMs: 2000,
        },
        fakeFetch as unknown as typeof fetch,
        selPath,
      )

      // No saved selection and no explicit args.model -> MUST reject!
      await expect(
        tool.execute({ prompt: 'a glowing moon' }, { signal: new AbortController().signal } as never),
      ).rejects.toThrow(/No image model selected/)

      // Persist profile-level selection -> tool picks it up automatically.
      writeSelectedImageModel(selPath, 'FLUX.2-klein-9B')
      expect(readSelectedImageModel(selPath)).toBe('FLUX.2-klein-9B')

      const result = await tool.execute(
        { prompt: 'a glowing moon', quality: 'maximum', steps: 4 },
        { signal: new AbortController().signal } as never,
      )
      expect(saveImage).toHaveBeenCalledTimes(1)
      const expected = {
        attachmentId: 'att_01',
        model: 'FLUX.2-klein-9B',
        quality: 'maximum',
        filename: 'nexus_abc.png',
        width: 256,
        height: 256,
        steps: 4,
        operationId: 'op_1',
        savedToDrive: true,
        attachment: {
          attachmentId: 'att_01',
          mediaType: 'image/png',
          bytes: 12,
          width: 256,
          height: 256,
          name: 'nexus_abc.png',
        },
      }
      expect(result).toEqual(expected)
      const rendered = tool.output.render({ prompt: 'a glowing moon', quality: 'maximum', steps: 4 }, expected)
      expect(rendered).toHaveLength(2)
      expect(rendered[1]).toMatchObject({ type: 'image', attachment: { attachmentId: 'att_01' } })
    } finally {
      delete process.env.TEST_NEXUS_KEY
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('proxies runtime-load, operation-scoped progress, image-model persistence, and redacts secrets on error', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'nexus-host-routes-'))
    const selPath = join(dir, 'nexus-image-selection.json')
    try {
      process.env.TEST_NEXUS_KEY = 'super-secret-key-999'
      const handlers = new Map<string, (req: Request) => Promise<Response>>()
      const ctx = new Context()
      Object.assign(ctx, {
        tools: { register: vi.fn(() => () => {}) },
        attachments: { saveImage: vi.fn() },
        connection: {
          fetch: {
            register: vi.fn((def: { path: string; fetch: (req: Request) => Promise<Response> }) => {
              handlers.set(def.path, def.fetch)
              return () => { handlers.delete(def.path) }
            }),
          },
        },
      })

      const calls: Array<{ url: string; method: string; body?: string | undefined }> = []
      const fakeFetch: typeof fetch = async (input, init) => {
        const url = String(input)
        const method = init?.method ?? 'GET'
        const body = typeof init?.body === 'string' ? init.body : undefined
        calls.push({ url, method, body })

        if (url.endsWith('/v1/runtime/load')) {
          const parsed = JSON.parse(body ?? '{}') as { model?: string; kind?: string; async?: boolean }
          if (parsed.model === 'fail-secret') {
            return new Response(
              JSON.stringify({ detail: 'Worker failed with Bearer super-secret-key-999' }),
              { status: 503, headers: { 'content-type': 'application/json' } },
            )
          }
          return new Response(
            JSON.stringify({ status: 'loading', operation_id: 'op_abc123', model: parsed.model, kind: parsed.kind }),
            { status: 202, headers: { 'content-type': 'application/json' } },
          )
        }
        if (url.endsWith('/v1/progress/op_abc123')) {
          return new Response(
            JSON.stringify({ operation_id: 'op_abc123', status: 'running', phase: 'weight_loading', indeterminate: true }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          )
        }
        if (url.endsWith('/v1/image-models')) {
          return new Response(
            JSON.stringify({ data: [{ id: 'FLUX.2-klein-9B', display_name: 'FLUX.2 Klein 9B' }], quality_profiles: ['normal', 'detailed', 'maximum'] }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          )
        }
        return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'content-type': 'application/json' } })
      }

      apply(
        ctx,
        {
          gatewayUrl: 'http://127.0.0.1:7861',
          apiKeyEnv: 'TEST_NEXUS_KEY',
          imageTimeoutMs: 5000,
          progressTimeoutMs: 2000,
        },
        fakeFetch,
        selPath,
      )

      // 1. Missing model returns HTTP 400 with actionable error before touching Gateway.
      const loadHandler = handlers.get('/api/nexus.runtime-load')!
      const badLoad = await loadHandler(new Request('http://localhost/api/nexus.runtime-load', {
        method: 'POST',
        body: JSON.stringify({}),
      }))
      expect(badLoad.status).toBe(400)
      expect((await badLoad.json()).error).toBe('model is required')
      expect(calls).toHaveLength(0)

      // 2. Valid canonical model forwards { model, kind: 'chat', async: true } and preserves 202 + operation_id.
      const okLoad = await loadHandler(new Request('http://localhost/api/nexus.runtime-load', {
        method: 'POST',
        body: JSON.stringify({ model: 'Qwen-3-VL-8B-Instruct-heretic', kind: 'chat', async: true }),
      }))
      expect(okLoad.status).toBe(202)
      expect(await okLoad.json()).toMatchObject({ status: 'loading', operation_id: 'op_abc123' })
      expect(JSON.parse(calls[0]?.body ?? '{}')).toEqual({
        model: 'Qwen-3-VL-8B-Instruct-heretic',
        kind: 'chat',
        async: true,
      })

      // 3. Upstream error preserves HTTP status and redacts secrets.
      const secretErr = await loadHandler(new Request('http://localhost/api/nexus.runtime-load', {
        method: 'POST',
        body: JSON.stringify({ model: 'fail-secret', kind: 'chat', async: true }),
      }))
      expect(secretErr.status).toBe(503)
      const errBody = await secretErr.json() as { error: string }
      expect(errBody.error).not.toContain('super-secret-key-999')
      expect(errBody.error).toContain('[REDACTED]')

      // 4. Operation-scoped progress proxy forwards ?operation_id=op_abc123 -> /v1/progress/op_abc123.
      const progHandler = handlers.get('/api/nexus.runtime-progress')!
      const progRes = await progHandler(new Request('http://localhost/api/nexus.runtime-progress?operation_id=op_abc123'))
      expect(progRes.status).toBe(200)
      expect(await progRes.json()).toMatchObject({ operation_id: 'op_abc123', phase: 'weight_loading' })

      // 5. POST /api/nexus.image-model persists selection and returns full models list + selectedModelId.
      const imgSaveHandler = handlers.get('/api/nexus.image-model')!
      const imgSaveRes = await imgSaveHandler(new Request('http://localhost/api/nexus.image-model', {
        method: 'POST',
        body: JSON.stringify({ modelId: 'FLUX.2-klein-9B' }),
      }))
      expect(imgSaveRes.status).toBe(200)
      expect(await imgSaveRes.json()).toMatchObject({
        selectedModelId: 'FLUX.2-klein-9B',
        models: [{ id: 'FLUX.2-klein-9B' }],
      })
    } finally {
      delete process.env.TEST_NEXUS_KEY
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
