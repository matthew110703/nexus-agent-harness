/**
 * Inline generated-image tool card for `nexus_generate_image`.
 * Registered into the keyed `tool.call.toolview` slot without claiming any child slot,
 * using the owner-provided `loadImage` callback to render durable attachments inline.
 */
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { ToolCallViewProps } from '@deepseek-ai/dsh-client-ui-tool/client'
import { tNexus } from './locales.ts'

const ALLOWED_IMAGE_MEDIA_TYPES: ReadonlySet<ImageMediaType> = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
])

interface NexusImageCardInfo {
  readonly attachment: ImageAttachmentRef | null
  readonly model: string
  readonly width: number
  readonly height: number
  readonly steps: number
  readonly quality: string
  readonly savedToDrive: boolean
  readonly prompt: string
}

function isImageAttachmentRef(value: unknown): value is ImageAttachmentRef {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const rec = value as Record<string, unknown>
  return typeof rec.attachmentId === 'string'
    && rec.attachmentId.length > 0
    && typeof rec.mediaType === 'string'
    && ALLOWED_IMAGE_MEDIA_TYPES.has(rec.mediaType as ImageMediaType)
    && typeof rec.bytes === 'number'
    && typeof rec.width === 'number'
    && typeof rec.height === 'number'
}

function extractPromptFromArgs(argsRaw: string | undefined): string {
  if (!argsRaw) return ''
  try {
    const parsed = JSON.parse(argsRaw) as Record<string, unknown>
    return typeof parsed.prompt === 'string' ? parsed.prompt.trim() : ''
  } catch {
    return ''
  }
}

function extractCardInfo(props: ToolCallViewProps): NexusImageCardInfo {
  const prompt = props.phase === 'preparing'
    ? ''
    : props.phase === 'start'
      ? extractPromptFromArgs(props.block.argsRaw)
      : extractPromptFromArgs(props.block.call?.argsRaw)

  if (props.phase !== 'result') {
    return {
      attachment: null,
      model: '',
      width: 0,
      height: 0,
      steps: 0,
      quality: '',
      savedToDrive: false,
      prompt,
    }
  }

  let attachment: ImageAttachmentRef | null = null
  const content = props.block.content
  if (Array.isArray(content)) {
    for (const part of content) {
      if (typeof part === 'object' && part !== null && 'type' in part && part.type === 'image' && 'attachment' in part) {
        if (isImageAttachmentRef(part.attachment)) {
          attachment = part.attachment
          break
        }
      }
    }
  }

  const meta = (typeof props.block.meta === 'object' && props.block.meta !== null && !Array.isArray(props.block.meta))
    ? props.block.meta as Record<string, unknown>
    : {}

  return {
    attachment,
    model: typeof meta.model === 'string' ? meta.model : '',
    width: typeof meta.width === 'number' ? meta.width : (attachment?.width ?? 0),
    height: typeof meta.height === 'number' ? meta.height : (attachment?.height ?? 0),
    steps: typeof meta.steps === 'number' ? meta.steps : 0,
    quality: typeof meta.quality === 'string' ? meta.quality : '',
    savedToDrive: meta.savedToDrive === true,
    prompt,
  }
}

/**
 * Inline `nexus_generate_image` tool view displaying the generated image directly in chat
 * with click-to-zoom lightbox and generation metadata.
 * @param props - Standard atomic toolview props supplied by `ui-tool`.
 */
export function NexusImageToolRow(props: ToolCallViewProps) {
  const info = extractCardInfo(props)
  const attachment = info.attachment
  const attachmentId = attachment ? String(attachment.attachmentId) : ''
  const [imageUrl, setImageUrl] = useState<string | null>(() => (
    attachment && typeof props.loadImage.peek === 'function' ? (props.loadImage.peek(attachment) ?? null) : null
  ))
  const [loadError, setLoadError] = useState(false)
  const [lightboxOpen, setLightboxOpen] = useState(false)

  useEffect(() => {
    if (!attachment) {
      setImageUrl(null)
      setLoadError(false)
      return
    }
    const peeked = typeof props.loadImage.peek === 'function' ? props.loadImage.peek(attachment) : undefined
    if (peeked) {
      setImageUrl(peeked)
      setLoadError(false)
      return
    }
    let cancelled = false
    setLoadError(false)
    void props.loadImage(attachment).then(
      (url) => {
        if (!cancelled) setImageUrl(url)
      },
      () => {
        if (!cancelled) setLoadError(true)
      },
    )
    return () => {
      cancelled = true
    }
  }, [attachmentId, props.loadImage])

  const isError = props.phase === 'result' && props.block.isError === true
  const errorText = isError && Array.isArray(props.block.content)
    ? props.block.content
      .map(p => (typeof p === 'object' && p !== null && 'type' in p && p.type === 'text' && 'text' in p && typeof p.text === 'string' ? p.text : ''))
      .filter(Boolean)
      .join('\n')
    : ''

  const metaBits: string[] = []
  if (info.model) metaBits.push(info.model)
  if (info.width > 0 && info.height > 0) metaBits.push(`${info.width}×${info.height}`)
  if (info.steps > 0) metaBits.push(tNexus('imageCard.stepsLabel', { steps: info.steps }))
  if (info.savedToDrive) metaBits.push(tNexus('imageCard.savedToDrive'))

  return (
    <div
      style={{
        border: '1px solid var(--border-subtle, rgba(255,255,255,0.12))',
        borderRadius: '10px',
        padding: '10px 12px',
        background: 'var(--bg-elevated, rgba(255,255,255,0.03))',
        margin: '6px 0',
        maxWidth: '520px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', marginBottom: '6px' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, opacity: 0.9 }}>
          {props.phase !== 'result'
            ? tNexus('imageCard.generating')
            : isError
              ? tNexus('imageCard.failed')
              : tNexus('imageCard.title')}
        </span>
        {props.inspect ? (
          <button
            type="button"
            onClick={props.inspect}
            style={{
              fontSize: '11px',
              background: 'transparent',
              border: '1px solid var(--border-subtle, rgba(255,255,255,0.15))',
              borderRadius: '6px',
              padding: '2px 7px',
              cursor: 'pointer',
              color: 'inherit',
              opacity: 0.8,
            }}
          >
            {tNexus('imageCard.inspect')}
          </button>
        ) : null}
      </div>

      {info.prompt ? (
        <div
          style={{
            fontSize: '12px',
            opacity: 0.72,
            marginBottom: '8px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
          }}
        >
          {info.prompt}
        </div>
      ) : null}

      {isError && errorText ? (
        <div style={{ fontSize: '12px', color: 'var(--status-error, #f87171)', whiteSpace: 'pre-wrap' }}>
          {errorText}
        </div>
      ) : null}

      {attachment && !isError ? (
        <div>
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={info.prompt || tNexus('imageCard.altFallback')}
              onClick={() => setLightboxOpen(true)}
              style={{
                display: 'block',
                width: '100%',
                maxHeight: '420px',
                objectFit: 'contain',
                borderRadius: '8px',
                cursor: 'zoom-in',
                background: 'rgba(0,0,0,0.2)',
              }}
            />
          ) : (
            <div style={{ fontSize: '12px', opacity: 0.65, padding: '16px 0' }}>
              {loadError ? tNexus('imageCard.previewError') : tNexus('imageCard.loadingPreview')}
            </div>
          )}
        </div>
      ) : null}

      {metaBits.length > 0 ? (
        <div style={{ fontSize: '11px', opacity: 0.62, marginTop: '6px' }}>
          {metaBits.join(' · ')}
        </div>
      ) : null}

      {lightboxOpen && imageUrl && typeof document !== 'undefined'
        ? createPortal(
          <div
            role="dialog"
            aria-modal="true"
            aria-label={tNexus('imageCard.closeLightboxAria')}
            onClick={() => setLightboxOpen(false)}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9999,
              background: 'rgba(0,0,0,0.86)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '24px',
              cursor: 'zoom-out',
            }}
          >
            <img
              src={imageUrl}
              alt={info.prompt || tNexus('imageCard.altFallback')}
              style={{ maxWidth: '94vw', maxHeight: '92vh', borderRadius: '8px', objectFit: 'contain' }}
            />
          </div>,
          document.body,
        )
        : null}
    </div>
  )
}
