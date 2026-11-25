/**
 * Browser plugin half for `@deepseek-ai/dsh-client-nexus-runtime-ui`.
 * Mounts `NexusRuntimeControl` into the composer's `conversation.input.right`
 * slot, binding the live per-session selected chat model from
 * `ModelDirectoryResolver` (`ctx.modelDirectories`).
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-model-selection/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-slots'
import {
  NexusRuntimeControl,
  type NexusImageModelsPayload,
  type NexusProgressPayload,
  type NexusRuntimePayload,
} from './NexusRuntimeControl.tsx'

export { NexusRuntimeControl } from './NexusRuntimeControl.tsx'
export type {
  NexusImageModelOption,
  NexusImageModelsPayload,
  NexusProgressPayload,
  NexusRuntimeControlProps,
  NexusRuntimePayload,
} from './NexusRuntimeControl.tsx'

/** Required client services: the slot registry and model directory resolver. */
export const inject = ['slots', 'modelDirectories']

async function readJsonOrThrow<T>(response: Response): Promise<T> {
  const text = await response.text()
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const parsed = JSON.parse(text) as { error?: string }
      if (parsed && typeof parsed.error === 'string' && parsed.error) {
        message = parsed.error
      }
    } catch (_parseErr) {
      // Non-JSON error body; keep HTTP status summary.
    }
    throw new Error(message)
  }
  return JSON.parse(text) as T
}

async function fetchJsonOrNull<T>(url: string): Promise<T | null> {
  const response = await fetch(url, { credentials: 'same-origin' })
  if (!response.ok) return null
  return (await response.json()) as T
}

/**
 * Mount the Nexus runtime control into the composer's right-side action slot.
 * @param ctx - browser Cordis root context.
 */
export function apply(ctx: ClientContext): void {
  ctx.inject(['slots', 'modelDirectories'], (scope: ClientContext) => {
    const models = scope.modelDirectories
    scope.slots.inject('conversation.input.right', () => scope.slots.register({
      name: 'conversation.input.right',
      id: 'nexus-runtime-control',
      order: 10,
      inject: (sessionId: SessionId) => {
        const directory = models.directoryFor(sessionId)
        return {
          getSelectedChatModel: () => directory.store.getSnapshot().current?.model ?? null,
          subscribeSelectedChatModel: (listener: () => void) => directory.store.subscribe(listener),
          fetchProgress: () => fetchJsonOrNull<NexusProgressPayload>('/api/nexus.runtime-progress'),
          fetchRuntime: () => fetchJsonOrNull<NexusRuntimePayload>('/api/nexus.runtime'),
          fetchImageModels: () => fetchJsonOrNull<NexusImageModelsPayload>('/api/nexus.image-models'),
          saveImageModel: async (modelId: string | null) => {
            const response = await fetch('/api/nexus.image-model', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ model_id: modelId }),
            })
            return readJsonOrThrow<NexusImageModelsPayload>(response)
          },
          loadModel: async (modelId: string) => {
            const response = await fetch('/api/nexus.runtime-load', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ model_id: modelId }),
            })
            await readJsonOrThrow<Record<string, unknown>>(response)
          },
          unloadModel: async () => {
            const response = await fetch('/api/nexus.runtime-unload', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({}),
            })
            await readJsonOrThrow<Record<string, unknown>>(response)
          },
        }
      },
    }, NexusRuntimeControl))
  })
}
