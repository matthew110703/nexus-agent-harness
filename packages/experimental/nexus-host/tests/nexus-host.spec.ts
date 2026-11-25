import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import {
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
})
