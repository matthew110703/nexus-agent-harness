/**
 * Browser plugin half for `@deepseek-ai/dsh-client-nexus-runtime-ui` (v1.1.0).
 * Mounts the read-only `NexusRuntimeControl` status & progress badge into the
 * composer's `conversation.input.right` slot.
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type {} from '@deepseek-ai/dsh-client-ui-chat/client'
import type {} from '@deepseek-ai/dsh-client-ui-context-menu/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-session/client'
import type {} from '@deepseek-ai/dsh-client-ui-slots'
import type {} from '@deepseek-ai/dsh-client-ui-tool/client'
import type {} from '@deepseek-ai/dsh-client-ui-workspace/client'
import { tNexus } from './locales.ts'
import { NexusImageToolRow } from './NexusImageToolRow.tsx'
import { NexusTurnImagesTail, type NexusTurnImagesTailInjected } from './NexusTurnImagesTail.tsx'
import { nexusTurnImagesDefinition } from './turn-images.ts'
import {
  NexusApiError,
  NexusRuntimeControl,
  type NexusProgressSnapshot,
  type NexusRuntimePayload,
} from './NexusRuntimeControl.tsx'

export { NexusApiError, NexusRuntimeControl } from './NexusRuntimeControl.tsx'
export { NexusImageToolRow } from './NexusImageToolRow.tsx'
export { NexusTurnImagesTail, type NexusTurnImagesTailInjected, type NexusTurnImagesTailProps } from './NexusTurnImagesTail.tsx'
export { nexusTurnImagesDefinition, type NexusTurnImage } from './turn-images.ts'
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

  ctx.inject(['uiConversation', 'sessions', 'slots'], (convScope: ClientContext) => {
    convScope.effect(
      () => convScope.uiConversation.events.register(nexusTurnImagesDefinition),
      'nexus-runtime-ui: turn images conversation definition',
    )
    convScope.slots.inject('conversation.chat.turnTail', () => convScope.slots.register({
      name: 'conversation.chat.turnTail',
      id: '@deepseek-ai/dsh-client-nexus-runtime-ui/turn-images',
      inject: (sessionId: SessionId): NexusTurnImagesTailInjected => {
        const binding = convScope.sessions.binding(sessionId)
        if (binding === undefined) {
          return {
            keyedHooks: {
              images: () => undefined,
            },
          }
        }
        const chat = convScope.uiConversation.binding(binding).target('chat')
        return {
          keyedHooks: {
            images: (turn: string) => {
              const snapshot = chat.getSnapshot()
              if (snapshot === undefined) return undefined
              const turnNumber = Number(turn)
              if (!Number.isFinite(turnNumber)) return undefined
              return snapshot.nodes.turnDataSource(turnNumber, 'nexus-generated-images')
            },
          },
        }
      },
    }, NexusTurnImagesTail))
  })

  ctx.inject(['contextMenus'], (menuScope: ClientContext) => {
    menuScope.effect(
      () =>
        menuScope.contextMenus.register({
          id: 'session.nexus-delete',
          target: 'session',
          order: 600,
          resolve: target => [
            {
              id: 'nexus-delete',
              label: tNexus('menu.deleteSession'),
              separatorBefore: true,
              run: () => {
                if (typeof window !== 'undefined' && !window.confirm(tNexus('menu.deleteSessionConfirm'))) {
                  return
                }
                void fetch('/api/nexus.purge-session', {
                  method: 'POST',
                  credentials: 'same-origin',
                  headers: { 'content-type': 'application/json' },
                  body: JSON.stringify({ sessionId: target.sessionId }),
                })
              },
            },
          ],
        }),
      'nexus-runtime-ui: session delete menu',
    )
  })
}
