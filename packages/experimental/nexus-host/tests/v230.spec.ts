import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'
import { Context } from '@deepseek-ai/cordis'
import { ToolArgsError } from '@deepseek-ai/dsh-tools'
import { describe, expect, it, vi } from 'vitest'
import {
  createNexusGenerateImageTool,
  shouldBlockUnsolicitedImageGen,
} from '../src/index.ts'

describe('nexus-host v2.3.0 — intent guard, fs/observed workspace sync & overlay contract', () => {
  it('blocks visual identification questions on uploaded images while allowing explicit generation/edits', () => {
    expect(shouldBlockUnsolicitedImageGen('who is in this image', 'identify person in image', true)).toBe(true)
    expect(shouldBlockUnsolicitedImageGen('can you analyze images?', 'analyze image', false)).toBe(true)
    expect(shouldBlockUnsolicitedImageGen('What is this?', 'a person standing', true)).toBe(true)
    expect(shouldBlockUnsolicitedImageGen('Turn this photo into a cyberpunk anime portrait', 'cyberpunk anime portrait', true)).toBe(false)
    expect(shouldBlockUnsolicitedImageGen('Generate a neon skyline at dusk', 'neon skyline at dusk', false)).toBe(false)
  })

  it('aborts nexus_generate_image with ToolArgsError before calling GPU when user asks to identify an uploaded image', async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'nexus-intent-'))
    const prevHome = process.env.DSH_HOME
    const prevKey = process.env.TEST_NEXUS_KEY
    try {
      const dshHome = path.join(tmp, 'dsh-home')
      const sessDir = path.join(dshHome, 'sessions', 'sess_identify')
      await fs.mkdir(sessDir, { recursive: true })
      const eventLine = JSON.stringify({
        type: 'user/message',
        data: {
          message: {
            role: 'user',
            source: { kind: 'user' },
            content: [
              { type: 'text', text: 'Who is in this photo?' },
              {
                type: 'image',
                attachment: {
                  attachmentId: 'att_u1',
                  mediaType: 'image/png',
                  bytes: 100,
                  width: 512,
                  height: 512,
                },
              },
            ],
          },
        },
      })
      await fs.writeFile(path.join(sessDir, 'events.jsonl'), `${eventLine}\n`, 'utf8')

      process.env.DSH_HOME = dshHome
      process.env.TEST_NEXUS_KEY = 'secret-123'

      const fetchSpy = vi.fn<typeof fetch>()
      const ctx = new Context()
      Object.assign(ctx, { attachments: { saveImage: vi.fn(), readImage: vi.fn() } })

      const tool = createNexusGenerateImageTool(
        ctx,
        {
          gatewayUrl: 'http://127.0.0.1:7861',
          apiKeyEnv: 'TEST_NEXUS_KEY',
          imageTimeoutMs: 5000,
          progressTimeoutMs: 2000,
        },
        fetchSpy,
      )

      await expect(
        tool.execute(
          { prompt: 'person in photo', use_uploaded_image: true },
          { signal: new AbortController().signal, sessionId: 'sess_identify' } as never,
        ),
      ).rejects.toBeInstanceOf(ToolArgsError)
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      if (prevHome === undefined) delete process.env.DSH_HOME
      else process.env.DSH_HOME = prevHome
      if (prevKey === undefined) delete process.env.TEST_NEXUS_KEY
      else process.env.TEST_NEXUS_KEY = prevKey
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  it('writes generated PNG to workspace/generated_images and emits fs/observed on workspace, dir, and file targets', async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'nexus-ws-obs-'))
    const prevKey = process.env.TEST_NEXUS_KEY
    const prevWs = process.env.NEXUS_DEFAULT_WORKSPACE
    try {
      const wsRoot = path.join(tmp, 'default-workspace')
      await fs.mkdir(wsRoot, { recursive: true })
      process.env.TEST_NEXUS_KEY = 'secret-123'
      process.env.NEXUS_DEFAULT_WORKSPACE = wsRoot

      const observedPaths: string[] = []
      const ctx = new Context()
      ctx.on('fs/observed', (target) => {
        observedPaths.push((target as { targetKey?: string }).targetKey ?? '')
      })
      const fakeFs = {
        resolve: vi.fn(async (p: string) => ({ targetKey: p })),
        stat: vi.fn(async () => ({ type: 'file', version: 'v1' })),
      }
      ctx.provide('fs')
      ctx.set('fs', fakeFs as never)
      Object.assign(ctx, {
        attachments: {
          saveImage: vi.fn(async () => ({
            attachmentId: 'att_gen_1',
            mediaType: 'image/png' as const,
            bytes: 8,
            width: 1024,
            height: 1024,
            name: 'gen_1.png',
          })),
        },
      })

      const fakeFetch: typeof fetch = vi.fn<typeof fetch>(async () => new Response(
        JSON.stringify({
          operation_id: 'op_1',
          model: 'LUSTIFY-v2.0',
          filename: 'gen_1.png',
          drive_path: path.join(tmp, 'drive_out', 'gen_1.png'),
          width: 1024,
          height: 1024,
          steps: 25,
          b64_png: Buffer.from('PNGDATA!').toString('base64'),
        }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      ))

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

      const res = await tool.execute(
        { prompt: 'dramatic mountain sunrise' },
        { signal: new AbortController().signal } as never,
      ) as { workspacePath: string }

      const expectedFile = path.join(wsRoot, 'generated_images', 'gen_1.png')
      expect(res.workspacePath).toBe(expectedFile)
      await expect(fs.stat(expectedFile)).resolves.toBeDefined()
      expect(observedPaths).toEqual([
        wsRoot,
        path.join(wsRoot, 'generated_images'),
        expectedFile,
      ])
    } finally {
      if (prevKey === undefined) delete process.env.TEST_NEXUS_KEY
      else process.env.TEST_NEXUS_KEY = prevKey
      if (prevWs === undefined) delete process.env.NEXUS_DEFAULT_WORKSPACE
      else process.env.NEXUS_DEFAULT_WORKSPACE = prevWs
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })

  it('validates colab.overlay.yml preserves web.fetchProvider, ui-workspace.recentViewLimit, working-directory, and disables cloud/multi-agent conflicts', async () => {
    const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..')
    const overlayText = await fs.readFile(path.join(repoRoot, 'apps/cli/config/examples/colab.overlay.yml'), 'utf8')

    expect(overlayText).toContain('searchProvider: nexus-search')
    expect(overlayText).toContain('fetchProvider: http')
    expect(overlayText).toContain('- id: web-search-deepseek\n  disabled: true')
    expect(overlayText).toContain('- id: web-deepseek-api\n  disabled: true')
    expect(overlayText).toContain('- id: session-title-llm\n  disabled: true')
    expect(overlayText).toContain('- id: agent-team\n  disabled: true')
    expect(overlayText).toContain('- id: tool-agent-team\n  disabled: true')
    expect(overlayText).toContain('recentViewLimit: 10')
    expect(overlayText).toContain('- id: working-directory')
    expect(overlayText).toContain('requiresToolResultName: true')
    expect(overlayText).toContain("readImage: !!js process.env.NEXUS_VISION_INPUT_ENABLED === '1'")
  })
})
