/**
 * Browser plugin half for `@deepseek-ai/dsh-client-nexus-runtime-ui` (v1.1.0).
 * Mounts the read-only `NexusRuntimeControl` status & progress badge into the
 * composer's `conversation.input.right` slot.
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-slots'
import type {} from '@deepseek-ai/dsh-client-ui-tool/client'
import { tNexus } from './locales.ts'
import { NexusImageToolRow } from './NexusImageToolRow.tsx'
import {
  NexusApiError,
  NexusRuntimeControl,
  type NexusProgressSnapshot,
  type NexusRuntimePayload,
} from './NexusRuntimeControl.tsx'

export { NexusApiError, NexusRuntimeControl } from './NexusRuntimeControl.tsx'
export { NexusImageToolRow } from './NexusImageToolRow.tsx'
export { resolveCompactModelName, resolveModelDisplayName } from './displayName.ts'
export {
  getNotificationPermission,
  isTabUnfocused,
  notifyUnfocusedCompletion,
  playCompletionChime,
  readNotifyPrefs,
  requestNotificationPermission,
  writeNotifyPrefs,
  type NexusNotifyPrefs,
} from './notifications.ts'
export type {
  NexusHardwarePayload,
  NexusLockedModelInfo,
  NexusOperationRecord,
  NexusProgressSnapshot,
  NexusRuntimeControlProps,
  NexusRuntimePayload,
  NexusWorkerPayload,
} from './NexusRuntimeControl.tsx'

/** Required client service: the slot registry. */
export const inject = ['slots']

async function readJsonOrThrow<T>(response: Response): Promise<T> {
  const text = await response.text()
  if (!response.ok) {
    let message = tNexus('error.requestFailed', { status: response.status })
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
 * Mount the read-only Nexus runtime status control into the composer's right-side action slot.
 * @param ctx - browser Cordis root context.
 */
export function apply(ctx: ClientContext): void {
  ctx.inject(['slots'], (scope: ClientContext) => {
    scope.slots.inject('conversation.input.right', () => scope.slots.register({
      name: 'conversation.input.right',
      id: 'nexus-runtime-control',
      order: 10,
      inject: () => ({
        fetchProgress: async (signal?: AbortSignal) => {
          const response = await fetch('/api/nexus.runtime-progress', {
            credentials: 'same-origin',
            ...(signal ? { signal } : {}),
          })
          return readJsonOrThrow<NexusProgressSnapshot>(response)
        },
        fetchRuntime: async (signal?: AbortSignal) => {
          const response = await fetch('/api/nexus.runtime', {
            credentials: 'same-origin',
            ...(signal ? { signal } : {}),
          })
          return readJsonOrThrow<NexusRuntimePayload>(response)
        },
      }),
    }, NexusRuntimeControl))

    scope.slots.inject('tool.call.toolview', () => scope.slots.register({
      name: 'tool.call.toolview',
      key: 'nexus_generate_image',
    }, NexusImageToolRow))
  })
}
