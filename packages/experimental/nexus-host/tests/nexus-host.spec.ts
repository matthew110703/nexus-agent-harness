import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { assertLoopbackGatewayUrl, createNexusGenerateImageTool } from '../src/index.ts'

describe('nexus-host', () => {
  it('enforces loopback-only gateway URLs', () => {
    expect(assertLoopbackGatewayUrl('http://127.0.0.1:7861').hostname).toBe('127.0.0.1')
    expect(assertLoopbackGatewayUrl('http://localhost:7861').hostname).toBe('localhost')
    expect(() => assertLoopbackGatewayUrl('https://evil.example.com')).toThrow(/loopback-only/)
  })

  it('calls Nexus gateway with bearer token and persists image via ctx.attachments.saveImage', async () => {
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
      return new Response(
        JSON.stringify({
          operation_id: 'op_1',
          model: 'FLUX.2-klein-kv',
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
    )

    const result = await tool.execute(
      { prompt: 'a glowing moon', steps: 4 },
      { signal: new AbortController().signal } as never,
    )
    expect(saveImage).toHaveBeenCalledTimes(1)
    expect(result).toEqual({
      attachmentId: 'att_01',
      model: 'FLUX.2-klein-kv',
      filename: 'nexus_abc.png',
      width: 256,
      height: 256,
      steps: 4,
      operationId: 'op_1',
      savedToDrive: true,
    })
    delete process.env.TEST_NEXUS_KEY
  })
})
