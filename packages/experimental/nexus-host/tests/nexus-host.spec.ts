import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import {
  apply,
  assertLoopbackGatewayUrl,
  createNexusGenerateImageTool,
} from '../src/index.ts'

describe('nexus-host (v1.1.0 single-model)', () => {
  it('enforces loopback-only gateway URLs', () => {
    expect(assertLoopbackGatewayUrl('http://127.0.0.1:7861').hostname).toBe('127.0.0.1')
    expect(assertLoopbackGatewayUrl('http://localhost:7861').hostname).toBe('localhost')
    expect(() => assertLoopbackGatewayUrl('https://evil.example.com')).toThrow(/loopback-only/)
  })

  it('calls single-model /v1/images/generations without model override and attaches PNG durably', async () => {
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

      const fakeFetch: typeof fetch = vi.fn<typeof fetch>(async (_url, init) => {
        expect((init?.headers as Record<string, string>).authorization).toBe('Bearer secret-token-123')
        const rawBody = typeof init?.body === 'string' ? init.body : ''
        const sent = JSON.parse(rawBody) as Record<string, unknown>
        expect(sent.model).toBeUndefined()
        expect(sent.num_images).toBe(1)
        expect(sent.prompt).toBe('a glowing moon')
        return new Response(
          JSON.stringify({
            operation_id: 'op_1',
            model: 'FLUX.2-klein-4B',
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
        fakeFetch,
      )

      const result = await tool.execute(
        { prompt: 'a glowing moon', quality: 'maximum', steps: 4 },
        { signal: new AbortController().signal } as never,
      )
      expect(saveImage).toHaveBeenCalledTimes(1)
      expect(result).toMatchObject({
        attachmentId: 'att_01',
        model: 'FLUX.2-klein-4B',
        quality: 'maximum',
        filename: 'nexus_abc.png',
        savedToDrive: true,
      })
      const rendered = tool.output.render(
        { prompt: 'a glowing moon', quality: 'maximum', steps: 4 },
        result as Parameters<typeof tool.output.render>[1],
      )
      expect(rendered).toHaveLength(2)
      expect(rendered[1]).toMatchObject({ type: 'image', attachment: { attachmentId: 'att_01' } })
    } finally {
      delete process.env.TEST_NEXUS_KEY
    }
  })

  it('registers only read-only runtime/progress routes and redacts secrets on upstream error', async () => {
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

      const fakeFetch: typeof fetch = async (input) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
        if (url.endsWith('/v1/runtime')) {
          return new Response(
            JSON.stringify({ detail: 'Worker crashed with Bearer super-secret-key-999' }),
            { status: 503, headers: { 'content-type': 'application/json' } },
          )
        }
        if (url.endsWith('/v1/progress/op_swap_1')) {
          return new Response(
            JSON.stringify({ operation_id: 'op_swap_1', status: 'running', phase: 'denoising', step: 2, steps_total: 4 }),
            { status: 200, headers: { 'content-type': 'application/json' } },
          )
        }
        return new Response(JSON.stringify({ chat: null, diffusion: null, operations: [] }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        })
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
      )

      // Obsolete v1.0.1 mutation & image-selection routes MUST NOT be registered.
      expect(handlers.has('/api/nexus.runtime-load')).toBe(false)
      expect(handlers.has('/api/nexus.runtime-unload')).toBe(false)
      expect(handlers.has('/api/nexus.image-models')).toBe(false)
      expect(handlers.has('/api/nexus.image-model')).toBe(false)
      expect(handlers.has('/api/nexus.runtime')).toBe(true)
      expect(handlers.has('/api/nexus.runtime-progress')).toBe(true)

      const rtRes = await handlers.get('/api/nexus.runtime')!(new Request('http://localhost/api/nexus.runtime'))
      expect(rtRes.status).toBe(503)
      const errBody = await rtRes.json() as { error: string }
      expect(errBody.error).not.toContain('super-secret-key-999')
      expect(errBody.error).toContain('[REDACTED]')

      const progRes = await handlers.get('/api/nexus.runtime-progress')!(
        new Request('http://localhost/api/nexus.runtime-progress?operation_id=op_swap_1'),
      )
      expect(progRes.status).toBe(200)
      expect(await progRes.json()).toMatchObject({ operation_id: 'op_swap_1', phase: 'denoising', step: 2 })
    } finally {
      delete process.env.TEST_NEXUS_KEY
    }
  })
})
