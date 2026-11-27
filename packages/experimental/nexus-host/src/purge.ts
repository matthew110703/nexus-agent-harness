/**
 * Permanent session and workspace purge routes for Nexus Colab deployments.
 * Deletes session JSONL logs, projection cache entries, generated image PNG/JSON sidecars,
 * and optional Drive workspace folders when explicitly requested.
 */

import * as fs from 'node:fs/promises'
import * as os from 'node:os'
import * as path from 'node:path'
import type { Context } from '@deepseek-ai/cordis'

export interface PurgeSessionSummary {
  sessionId: string
  deletedSessionDir: boolean
  deletedImages: string[]
}

export interface PurgeWorkspaceSummary {
  workspaceId: string
  deletedSessions: PurgeSessionSummary[]
  deletedWorkspaceEntry: boolean
  deletedDiskFolder: boolean
}

function resolveDshHome(): string {
  return process.env.DSH_HOME?.trim() || path.join(os.homedir(), '.dsh')
}

function resolveAllowedWorkspaceRoots(): string[] {
  const candidates = [
    process.env.NEXUS_WORKSPACE_ROOT?.trim(),
    process.env.NEXUS_SHARED_WORKSPACE_ROOT?.trim(),
    '/content/drive/MyDrive/Nexus/workspaces',
  ].filter((x): x is string => Boolean(x))
  return candidates.map(p => path.resolve(p))
}

function isStrictlyInsideAllowedRoot(targetPath: string): boolean {
  const resolved = path.resolve(targetPath)
  for (const root of resolveAllowedWorkspaceRoots()) {
    if (resolved !== root && resolved.startsWith(root + path.sep)) {
      return true
    }
  }
  return false
}

async function deleteSessionGeneratedImages(sessionId: string, extraDirs: string[]): Promise<string[]> {
  const deleted: string[] = []
  const dirs = new Set<string>([
    process.env.NEXUS_OUTPUT_IMAGE_DIR?.trim() || '/content/drive/MyDrive/Nexus/outputs/generated_images',
    ...extraDirs.filter(Boolean),
  ])
  for (const dir of dirs) {
    let entries: string[]
    try {
      entries = await fs.readdir(dir)
    } catch {
      continue
    }
    for (const entry of entries) {
      if (!entry.endsWith('.png.json')) continue
      const metaPath = path.join(dir, entry)
      try {
        const raw = await fs.readFile(metaPath, 'utf8')
        const meta = JSON.parse(raw) as { session_id?: string; filename?: string }
        if (meta.session_id === sessionId) {
          const pngName = meta.filename || entry.slice(0, -5)
          const pngPath = path.join(dir, pngName)
          await fs.rm(metaPath, { force: true })
          await fs.rm(pngPath, { force: true })
          deleted.push(pngPath)
        }
      } catch {
        // Skip unreadable or malformed sidecar files
      }
    }
  }
  return deleted
}

/**
 * Permanently delete a session's persisted JSONL directory, projection cache residue,
 * and any generated image PNG/JSON artifacts tagged with `sessionId`.
 *
 * @param sessionId - Session identifier to purge.
 * @param extraImageDirs - Optional workspace `generated_images` directories to scan.
 * @returns Summary of deleted session files and image artifacts.
 */
export async function purgeSessionArtifacts(
  sessionId: string,
  extraImageDirs: string[] = [],
): Promise<PurgeSessionSummary> {
  const cleanId = sessionId.trim()
  if (!cleanId || cleanId.includes('/') || cleanId.includes('\\') || cleanId.includes('..')) {
    throw new Error('Invalid sessionId')
  }
  const dshHome = resolveDshHome()
  const sessionDir = path.join(dshHome, 'sessions', cleanId)
  const sessionFile = path.join(dshHome, 'sessions', `${cleanId}.jsonl`)
  let deletedSessionDir = false
  try {
    await fs.rm(sessionDir, { recursive: true, force: true })
    await fs.rm(sessionFile, { force: true })
    deletedSessionDir = true
  } catch {
    deletedSessionDir = false
  }
  const deletedImages = await deleteSessionGeneratedImages(cleanId, extraImageDirs)
  return {
    sessionId: cleanId,
    deletedSessionDir,
    deletedImages,
  }
}

/**
 * Register `/api/nexus.purge-session` and `/api/nexus.purge-workspace` routes on `ctx.connection`.
 *
 * @param ctx - Cordis plugin context.
 */
export function registerNexusPurgeRoutes(ctx: Context): void {
  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.purge-session',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request) {
          try {
            const body = (await request.json()) as {
              sessionId?: string
              workspacePath?: string
            }
            const sessionId = (body.sessionId ?? '').trim()
            if (!sessionId) {
              return new Response(JSON.stringify({ error: 'sessionId is required' }), {
                status: 400,
                headers: { 'content-type': 'application/json' },
              })
            }
            const commands = ctx.get('commands') as
              | { execute?: (name: string, payload: Record<string, unknown>) => Promise<unknown> }
              | undefined
            if (commands?.execute) {
              try {
                await commands.execute('session.stop', { sessionId })
              } catch {
                // Session may already be idle
              }
              try {
                await commands.execute('workspace.removeSession', { sessionId })
              } catch {
                // Session may not belong to a workspace
              }
            }
            const extraDirs = body.workspacePath
              ? [path.join(body.workspacePath, 'generated_images')]
              : []
            const summary = await purgeSessionArtifacts(sessionId, extraDirs)
            return new Response(JSON.stringify(summary), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            })
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err)
            return new Response(JSON.stringify({ error: message }), {
              status: 400,
              headers: { 'content-type': 'application/json' },
            })
          }
        },
      }),
    'nexus-host: /api/nexus.purge-session route',
  )

  ctx.effect(
    () =>
      ctx.connection.fetch.register({
        path: '/api/nexus.purge-workspace',
        methods: ['POST'],
        requestBody: 'buffered',
        async fetch(request) {
          try {
            const body = (await request.json()) as {
              workspaceId?: string
              sessionIds?: string[]
              workspacePath?: string
              deleteChats?: boolean
              deleteDiskFolder?: boolean
            }
            const workspaceId = (body.workspaceId ?? '').trim()
            if (!workspaceId) {
              return new Response(JSON.stringify({ error: 'workspaceId is required' }), {
                status: 400,
                headers: { 'content-type': 'application/json' },
              })
            }
            const deleteChats = body.deleteChats !== false
            const deleteDiskFolder = body.deleteDiskFolder === true
            const workspacePath = (body.workspacePath ?? '').trim()
            const extraDirs = workspacePath ? [path.join(workspacePath, 'generated_images')] : []

            const deletedSessions: PurgeSessionSummary[] = []
            if (deleteChats && Array.isArray(body.sessionIds)) {
              for (const sid of body.sessionIds) {
                if (typeof sid === 'string' && sid.trim()) {
                  deletedSessions.push(await purgeSessionArtifacts(sid.trim(), extraDirs))
                }
              }
            }

            let deletedWorkspaceEntry = false
            const commands = ctx.get('commands') as
              | { execute?: (name: string, payload: Record<string, unknown>) => Promise<unknown> }
              | undefined
            if (commands?.execute) {
              try {
                await commands.execute('workspace.delete', { workspaceId })
                deletedWorkspaceEntry = true
              } catch {
                deletedWorkspaceEntry = false
              }
            }

            let deletedDiskFolder = false
            if (deleteDiskFolder && workspacePath && isStrictlyInsideAllowedRoot(workspacePath)) {
              await fs.rm(path.resolve(workspacePath), { recursive: true, force: true })
              deletedDiskFolder = true
            }

            const summary: PurgeWorkspaceSummary = {
              workspaceId,
              deletedSessions,
              deletedWorkspaceEntry,
              deletedDiskFolder,
            }
            return new Response(JSON.stringify(summary), {
              status: 200,
              headers: { 'content-type': 'application/json' },
            })
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err)
            return new Response(JSON.stringify({ error: message }), {
              status: 400,
              headers: { 'content-type': 'application/json' },
            })
          }
        },
      }),
    'nexus-host: /api/nexus.purge-workspace route',
  )
}
