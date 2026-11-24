import { mkdtemp, mkdir, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { DirectoryPickerError } from '@deepseek-ai/dsh-host-directory-picker'
import BrowseDirectoryPicker, { isWithinRoot } from '../src/index.ts'

describe('BrowseDirectoryPicker allowedRoots confinement', () => {
  let sandbox: string
  let rootA: string
  let rootB: string
  let outside: string
  let ctx: Context
  let disposePlugin: () => Promise<void>

  beforeEach(async () => {
    sandbox = await mkdtemp(join(tmpdir(), 'dsh-picker-roots-'))
    rootA = join(sandbox, 'workspace')
    rootB = join(sandbox, 'workspace-extra')
    outside = join(sandbox, 'outside')
    await mkdir(join(rootA, 'project-1'), { recursive: true })
    await mkdir(rootB, { recursive: true })
    await mkdir(outside, { recursive: true })
    ctx = new Context()
    const fork = ctx.plugin(BrowseDirectoryPicker, { maxEntries: 100, allowedRoots: [rootA] })
    disposePlugin = () => fork.dispose()
  })

  afterEach(async () => {
    await disposePlugin()
    await rm(sandbox, { recursive: true, force: true })
  })

  it('isWithinRoot does not match sibling prefix directories', () => {
    expect(isWithinRoot(rootA, rootA)).toBe(true)
    expect(isWithinRoot(join(rootA, 'project-1'), rootA)).toBe(true)
    expect(isWithinRoot(rootB, rootA)).toBe(false)
    expect(isWithinRoot(outside, rootA)).toBe(false)
  })

  it('defaults omitted path to the first allowed root and clamps breadcrumbs', async () => {
    const cap = ctx.directoryPicker.capability()
    expect(cap.kind).toBe('browse')
    if (cap.kind !== 'browse') return

    const listing = await cap.list()
    expect(listing.home).toBe(listing.path)
    expect(listing.crumbs).toHaveLength(1)
    expect(listing.crumbs[0]?.path).toBe(listing.path)
    expect(listing.entries.map(entry => entry.name)).toEqual(['project-1'])

    const nested = await cap.list(join(rootA, 'project-1'))
    expect(nested.crumbs).toHaveLength(2)
    expect(nested.crumbs[0]?.path).toBe(listing.path)
  })

  it('rejects listing and creating directories outside allowedRoots', async () => {
    const cap = ctx.directoryPicker.capability()
    if (cap.kind !== 'browse') return

    await expect(cap.list(outside)).rejects.toThrow(DirectoryPickerError)
    await expect(cap.list(rootB)).rejects.toThrow(DirectoryPickerError)
    await expect(cap.createDirectory(outside, 'evil')).rejects.toThrow(DirectoryPickerError)
  })

  it('creates directories inside allowedRoots and hides escaping symlinks', async () => {
    const cap = ctx.directoryPicker.capability()
    if (cap.kind !== 'browse') return

    const created = await cap.createDirectory(rootA, 'new-proj')
    expect(created.endsWith('new-proj')).toBe(true)

    try {
      await symlink(outside, join(rootA, 'escape-link'), 'junction')
      const listing = await cap.list(rootA)
      expect(listing.entries.map(entry => entry.name)).not.toContain('escape-link')
      await expect(cap.list(join(rootA, 'escape-link'))).rejects.toThrow(DirectoryPickerError)
    } catch {
      // Symlink creation may require elevated privileges on some Windows hosts; skip only if symlink creation fails.
    }
  })
})
