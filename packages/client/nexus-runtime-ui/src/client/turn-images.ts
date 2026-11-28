/**
 * Conversation node projection extracting `nexus_generate_image` tool results per turn
 * so generated images render directly in `conversation.chat.turnTail` outside the collapsed
 * `Called tools` accordion.
 */

import type { ChatNode } from '@deepseek-ai/dsh-client-ui-chat/client'
import type { ConversationNodeDefinition } from '@deepseek-ai/dsh-client-ui-conversation/client'

/** Projected generated-image record associated with one `nexus_generate_image` tool call. */
export interface NexusTurnImage {
  callId: string
  filename: string
  attachmentId: string
  width: number
  height: number
  steps: number
  model: string
  quality: string
  drivePath: string
  workspacePath: string
  masterPath: string
  prompt: string
}

declare module '@deepseek-ai/dsh-client-ui-chat/client' {
  interface ChatNodeDataMap {
    'nexus-generated-images': NexusTurnImage
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  return value as Record<string, unknown>
}

function strField(rec: Record<string, unknown> | undefined, key: string): string {
  const val = rec?.[key]
  return typeof val === 'string' ? val : ''
}

function numField(rec: Record<string, unknown> | undefined, key: string): number {
  const val = rec?.[key]
  return typeof val === 'number' && Number.isFinite(val) ? val : 0
}

function parseArgsPrompt(rawArgs: string): string {
  try {
    const parsed = JSON.parse(rawArgs) as Record<string, unknown>
    return strField(parsed, 'prompt')
  } catch {
    return ''
  }
}

/**
 * Conversation event definition that projects completed `nexus_generate_image` tool calls
 * into turn-scoped `nexus-generated-images` items.
 */
export const nexusTurnImagesDefinition: ConversationNodeDefinition<NexusTurnImage> = {
  kind: 'nexus-generated-images',
  target: 'chat',
  match: (event) => {
    if (event.type === 'tool/call' && event.data.name === 'nexus_generate_image') {
      return { id: `nexus-img-${event.data.callId}`, role: 'start' }
    }
    if (event.type === 'tool/result') {
      const meta = asRecord(event.data.meta)
      const callId = event.data.message.toolCallId
      if (callId && meta && (strField(meta, 'attachmentId') || strField(meta, 'masterPath') || strField(meta, 'drivePath'))) {
        return { id: `nexus-img-${callId}`, role: 'update' }
      }
    }
    return null
  },
  start: (context, match) => {
    const ev = match.event
    const prompt = ev.type === 'tool/call' ? parseArgsPrompt(ev.data.arguments) : ''
    const callId = ev.type === 'tool/call' ? ev.data.callId : context.id
    return {
      callId,
      filename: '',
      attachmentId: '',
      width: 0,
      height: 0,
      steps: 0,
      model: '',
      quality: 'detailed',
      drivePath: '',
      workspacePath: '',
      masterPath: '',
      prompt,
    }
  },
  update: (context, match) => {
    const prev = context.state
    const ev = match.event
    if (ev.type !== 'tool/result' || ev.data.error !== undefined) {
      return prev
    }
    const meta = asRecord(ev.data.meta)
    if (!meta) return prev
    const drivePath = strField(meta, 'drivePath')
    const workspacePath = strField(meta, 'workspacePath')
    const masterPath = strField(meta, 'masterPath') || drivePath || workspacePath
    return {
      ...prev,
      filename: strField(meta, 'filename') || prev.filename,
      attachmentId: strField(meta, 'attachmentId') || prev.attachmentId,
      width: numField(meta, 'width') || prev.width,
      height: numField(meta, 'height') || prev.height,
      steps: numField(meta, 'steps') || prev.steps,
      model: strField(meta, 'model') || prev.model,
      quality: strField(meta, 'quality') || prev.quality,
      drivePath: drivePath || prev.drivePath,
      workspacePath: workspacePath || prev.workspacePath,
      masterPath: masterPath || prev.masterPath,
    }
  },
  buildViewNode: (context) => {
    const start = context.start ?? context.matches[0]
    const data = context.state
    if (start === undefined || data === undefined) return null
    if (!data.attachmentId && !data.masterPath && !data.drivePath) return null
    return {
      key: context.key,
      kind: 'nexus-generated-images',
      id: context.id,
      target: 'chat',
      anchorSeq: start.event.seq,
      location: start.location,
      visibility: 'hidden',
      data,
    } satisfies ChatNode<'nexus-generated-images'>
  },
}
