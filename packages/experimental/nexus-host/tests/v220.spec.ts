import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { purgeSessionArtifacts } from '../src/purge.ts'
import { createNexusInspectRuntimeTool, createNexusWebResearchTool } from '../src/web-provider.ts'

describe('nexus-host v2.2.0', () => {
  it('nexus_web_research posts queries to /v1/web/research', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      queries: ['llama 4 scout gguf'],
      results: [{ title: 'HF', url: 'https://huggingface.co', snippet: 'weights' }],
      pages: [],
      citations: [],
      warnings: [],
    }), { status: 200 }))
    const tool = createNexusWebResearchTool('http://127.0.0.1:7861', 'secret', fetchMock)
    const out = await tool.execute({ queries: ['llama 4 scout gguf'], max_pages: 2 }, {
      signal: new AbortController().signal,
    } as Parameters<typeof tool.execute>[1])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect((out as { queries: string[] }).queries).toEqual(['llama 4 scout gguf'])
  })

  it('nexus_inspect_runtime fetches /v1/runtime', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      gateway_version: '2.2.0',
      language_model: { id: 'Qwen3.5-4B', ready: true },
    }), { status: 200 }))
    const tool = createNexusInspectRuntimeTool('http://127.0.0.1:7861', 'secret', fetchMock)
    const out = await tool.execute({ include_models: false }, {
      signal: new AbortController().signal,
    } as Parameters<typeof tool.execute>[1])
    expect((out as { gateway_version: string }).gateway_version).toBe('2.2.0')
  })

  it('purgeSessionArtifacts deletes session dir and only matching session generated images', async () => {
    const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'nexus-purge-'))
    const prevHome = process.env.DSH_HOME
    const prevImgDir = process.env.NEXUS_OUTPUT_IMAGE_DIR
    try {
      const dshHome = path.join(tmp, 'dsh-home')
      const imgDir = path.join(tmp, 'images')
      const sessDir = path.join(dshHome, 'sessions', 'sess_target')
      await fs.mkdir(sessDir, { recursive: true })
      await fs.writeFile(path.join(sessDir, 'events.jsonl'), '{}\n', 'utf8')
      await fs.mkdir(imgDir, { recursive: true })
      await fs.writeFile(path.join(imgDir, 'a.png'), 'PNG_A')
      await fs.writeFile(path.join(imgDir, 'a.png.json'), JSON.stringify({ session_id: 'sess_target', filename: 'a.png' }))
      await fs.writeFile(path.join(imgDir, 'b.png'), 'PNG_B')
      await fs.writeFile(path.join(imgDir, 'b.png.json'), JSON.stringify({ session_id: 'sess_other', filename: 'b.png' }))

      process.env.DSH_HOME = dshHome
      process.env.NEXUS_OUTPUT_IMAGE_DIR = imgDir
      const summary = await purgeSessionArtifacts('sess_target')
      expect(summary.deletedSessionDir).toBe(true)
      expect(summary.deletedImages).toEqual([path.join(imgDir, 'a.png')])
      await expect(fs.stat(sessDir)).rejects.toThrow()
      await expect(fs.stat(path.join(imgDir, 'a.png'))).rejects.toThrow()
      await expect(fs.stat(path.join(imgDir, 'b.png'))).resolves.toBeDefined()
    } finally {
      if (prevHome === undefined) delete process.env.DSH_HOME
      else process.env.DSH_HOME = prevHome
      if (prevImgDir === undefined) delete process.env.NEXUS_OUTPUT_IMAGE_DIR
      else process.env.NEXUS_OUTPUT_IMAGE_DIR = prevImgDir
      await fs.rm(tmp, { recursive: true, force: true })
    }
  })
})
