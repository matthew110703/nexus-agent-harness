/**
 * Browser plugin half for `@deepseek-ai/dsh-client-nexus-runtime-ui`.
 * Mounts `NexusRuntimeControl` into the composer's `conversation.input.right`
 * slot, binding the live per-session selected chat model (provider + canonical
 * model ID + catalog display name) from `ModelDirectoryResolver`.
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { ModelDirectoryState } from '@deepseek-ai/dsh-client-ui-model-selection/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-slots'
import {
  NexusApiError,
  NexusRuntimeControl,
  type NexusImageModelsPayload,
  type NexusLoadResponse,
  type NexusOperationRecord,
  type NexusProgressSnapshot,
  type NexusRuntimePayload,
  type SelectedChatModelInfo,
} from './NexusRuntimeControl.tsx'

export { NexusApiError, NexusRuntimeControl } from './NexusRuntimeControl.tsx'
export { resolveCompactModelName, resolveModelDisplayName } from './displayName.ts'
export type {
  NexusHardwarePayload,
  NexusImageModelOption,
  NexusImageModelsPayload,
  NexusLoadResponse,
  NexusOperationRecord,
  NexusProgressSnapshot,
  NexusRuntimeControlProps,
  NexusRuntimePayload,
  NexusWorkerPayload,
  SelectedChatModelInfo,
} from './NexusRuntimeControl.tsx'

/** Required client services: the slot registry and model directory resolver. */
export const inject = ['slots', 'modelDirectories']

function extractSelectedChatModel(state: ModelDirectoryState): SelectedChatModelInfo | null {
  const current = state.current
  if (!current || !current.model) return null
  let displayName: string | undefined
  for (const group of state.groups) {
    if (group.id !== current.provider) continue
    const match = group.models.find(m => m.id === current.model)
    if (match?.name) {
      displayName = match.name
    }
    break
  }
  return {
    provider: current.provider,
    modelId: current.model,
    ...(displayName !== undefined ? { displayName } : {}),
  }
}

async function readJsonOrThrow<T>(response: Response): Promise<T> {
  const text = await response.text()
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status})`
    let operationId: string | undefined
    let details: string | undefined
    try {
      const parsed = JSON.parse(text) as { error?: string; detail?: string; operation_id?: string; details?: string }
      if (typeof parsed.error === 'string' && parsed.error) {
        message = parsed.error
      } else if (typeof parsed.detail === 'string' && parsed.detail) {
        message = parsed.detail
      }
      if (typeof parsed.operation_id === 'string') {
        operationId = parsed.operation_id
      }
      if (typeof parsed.details === 'string') {
        details = parsed.details
      }
    } catch (_parseErr) {
      if (text.trim().length > 0) {
        message = text.trim().slice(0, 240)
      }
    }
    throw new NexusApiError(message, response.status, operationId, details)
  }
  return JSON.parse(text) as T
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
          getSelectedChatModel: () => extractSelectedChatModel(directory.store.getSnapshot()),
          subscribeSelectedChatModel: (listener: () => void) => directory.store.subscribe(listener),
          fetchProgress: async (signal?: AbortSignal) => {
            const response = await fetch('/api/nexus.runtime-progress', {
              credentials: 'same-origin',
              ...(signal ? { signal } : {}),
            })
            return readJsonOrThrow<NexusProgressSnapshot>(response)
          },
          fetchOperationProgress: async (operationId: string, signal?: AbortSignal) => {
            const response = await fetch(
              `/api/nexus.runtime-progress?operation_id=${encodeURIComponent(operationId)}`,
              { credentials: 'same-origin', ...(signal ? { signal } : {}) },
            )
            if (response.status === 404) return null
            return readJsonOrThrow<NexusOperationRecord>(response)
          },
          fetchRuntime: async (signal?: AbortSignal) => {
            const response = await fetch('/api/nexus.runtime', {
              credentials: 'same-origin',
              ...(signal ? { signal } : {}),
            })
            return readJsonOrThrow<NexusRuntimePayload>(response)
          },
          fetchImageModels: async (signal?: AbortSignal) => {
            const response = await fetch('/api/nexus.image-models', {
              credentials: 'same-origin',
              ...(signal ? { signal } : {}),
            })
            return readJsonOrThrow<NexusImageModelsPayload>(response)
          },
          saveImageModel: async (modelId: string | null) => {
            const response = await fetch('/api/nexus.image-model', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ modelId }),
            })
            return readJsonOrThrow<NexusImageModelsPayload>(response)
          },
          loadModel: async (modelId: string) => {
            const response = await fetch('/api/nexus.runtime-load', {
              method: 'POST',
              credentials: 'same-origin',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify({ model: modelId, kind: 'chat', async: true }),
            })
            return readJsonOrThrow<NexusLoadResponse>(response)
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
