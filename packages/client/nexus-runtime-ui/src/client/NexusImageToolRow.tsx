/**
 * Compact 56px inline tool summary row for `nexus_generate_image` inside `Called tools`.
 * Keeps the collapsed/expanded tool accordion sleek and consistent with `PresentedFileCard`,
 * while `NexusTurnImagesTail` renders the full hero showcase beneath the assistant reply.
 */

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ImageAttachmentRef, ImageMediaType } from '@deepseek-ai/dsh-attachment'
import type { ToolCallViewProps } from '@deepseek-ai/dsh-client-ui-tool/client'
import { resolveCompactModelName } from './displayName.ts'
import { tNexus } from './locales.ts'
import css from './NexusImageCard.module.css'

const ALLOWED_IMAGE_MEDIA_TYPES: ReadonlySet<ImageMediaType> = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
])

interface NexusImageCardInfo {
  readonly attachment: ImageAttachmentRef | null
  readonly filename: string
  readonly model: string
  readonly width: number
  readonly height: number
  readonly steps: number
  readonly savedToDrive: boolean
  readonly prompt: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isImageAttachmentRef(value: unknown): value is ImageAttachmentRef {
  if (!isRecord(value)) return false
  return typeof value.attachmentId === 'string'
    && value.attachmentId.length > 0
    && typeof value.mediaType === 'string'
    && ALLOWED_IMAGE_MEDIA_TYPES.has(value.mediaType as ImageMediaType)
    && typeof value.bytes === 'number'
    && typeof value.width === 'number'
    && typeof value.height === 'number'
}

function extractPromptFromArgs(argsRaw: string | undefined): string {
  if (!argsRaw) return ''
  try {
    const parsed = JSON.parse(argsRaw) as Record<string, unknown>
    return typeof parsed.prompt === 'string' ? parsed.prompt.trim() : ''
  } catch (_parseErr) {
    // Streaming partial JSON arguments may not be complete yet.
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
      filename: '',
      model: '',
      width: 0,
      height: 0,
      steps: 0,
      savedToDrive: false,
      prompt,
    }
  }

  let attachment: ImageAttachmentRef | null = null
  const content = props.block.content
  if (Array.isArray(content)) {
    for (const rawPart of content) {
      if (isRecord(rawPart) && rawPart.type === 'image' && isImageAttachmentRef(rawPart.attachment)) {
        attachment = rawPart.attachment
        break
      }
    }
  }

  const meta = isRecord(props.block.meta) ? props.block.meta : {}

  return {
    attachment,
    filename: typeof meta.filename === 'string' ? meta.filename : (attachment?.name ?? ''),
    model: typeof meta.model === 'string' ? meta.model : '',
    width: typeof meta.width === 'number' ? meta.width : (attachment?.width ?? 0),
    height: typeof meta.height === 'number' ? meta.height : (attachment?.height ?? 0),
    steps: typeof meta.steps === 'number' ? meta.steps : 0,
    savedToDrive: meta.savedToDrive === true,
    prompt,
  }
}

/**
 * Compact `nexus_generate_image` tool row inside `Called tools` with thumbnail, metadata,
 * `Inspect`, and `Escape`-closing modal zoom.
 * @param props - Standard atomic toolview props supplied by `ui-tool`.
 * @returns Compact 56px tool summary row.
 */
export function NexusImageToolRow(props: ToolCallViewProps) {
  const info = extractCardInfo(props)
  const attachment = info.attachment
  const attachmentId = attachment ? String(attachment.attachmentId) : ''
  const [imageUrl, setImageUrl] = useState<string | null>(() => (
    attachment && typeof props.loadImage.peek === 'function' ? (props.loadImage.peek(attachment) ?? null) : null
  ))
  const [lightboxOpen, setLightboxOpen] = useState(false)

  useEffect(() => {
    if (!attachment) {
      setImageUrl(null)
      return
    }
    const peeked = typeof props.loadImage.peek === 'function' ? props.loadImage.peek(attachment) : undefined
    if (peeked) {
      setImageUrl(peeked)
      return
    }
    let cancelled = false
    void props.loadImage(attachment).then(
      (url) => {
        if (!cancelled) setImageUrl(url)
      },
      () => {
        if (!cancelled) setImageUrl(null)
      },
    )
    return () => {
      cancelled = true
    }
  }, [attachment, attachmentId, props.loadImage])

  useEffect(() => {
    if (!lightboxOpen || typeof window === 'undefined') return
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        ev.stopPropagation()
        setLightboxOpen(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [lightboxOpen])

  const isError = props.phase === 'result' && props.block.isError
  const errorText = isError && Array.isArray(props.block.content)
    ? props.block.content
      .map(p => (isRecord(p) && p.type === 'text' && typeof p.text === 'string' ? p.text : ''))
      .filter(Boolean)
      .join('\n')
    : ''

  const metaBits: string[] = []
  if (info.width > 0 && info.height > 0) metaBits.push(`${info.width}×${info.height}`)
  if (info.model) metaBits.push(resolveCompactModelName(info.model))
  if (info.steps > 0) metaBits.push(tNexus('imageCard.stepsLabel', { steps: info.steps }))
  if (info.savedToDrive) metaBits.push(tNexus('imageCard.savedToDrive'))

  const titleText = props.phase !== 'result'
    ? tNexus('imageCard.generating')
    : isError
      ? tNexus('imageCard.failed')
      : (info.filename || tNexus('imageCard.title'))

  const subText = metaBits.length > 0 ? metaBits.join(' · ') : info.prompt

  return (
    <div className={css.toolRow}>
      {imageUrl && !isError ? (
        <button
          type="button"
          className={css.toolThumbBtn}
          aria-label={tNexus('imageCard.zoom')}
          onClick={() => {
            setLightboxOpen(true)
          }}
        >
          <img
            src={imageUrl}
            alt={info.prompt || tNexus('imageCard.altFallback')}
            className={css.toolThumbImg}
          />
        </button>
      ) : (
        <div className={css.toolThumbBtn}>
          <span className={css.toolThumbPlaceholder} />
        </div>
      )}

      <div className={css.toolBody}>
        <span className={css.toolTitle}>{titleText}</span>
        {isError && errorText ? (
          <span className={css.toolError}>{errorText}</span>
        ) : subText ? (
          <span className={css.toolSub}>{subText}</span>
        ) : null}
      </div>

      {props.inspect ? (
        <button type="button" className={css.actionBtn} onClick={props.inspect}>
          {tNexus('imageCard.inspect')}
        </button>
      ) : null}

      {lightboxOpen && imageUrl && typeof document !== 'undefined'
        ? createPortal(
          <div
            role="dialog"
            aria-modal="true"
            aria-label={tNexus('imageCard.closeLightboxAria')}
            className={css.lightboxBackdrop}
          >
            <div
              className={css.lightboxMask}
              onClick={() => {
                setLightboxOpen(false)
              }}
            />
            <div
              className={css.lightboxStage}
              onClick={(ev) => {
                if (ev.target === ev.currentTarget) {
                  setLightboxOpen(false)
                }
              }}
            >
              <div className={css.lightboxToolbar}>
                <button
                  type="button"
                  className={css.glassBtn}
                  aria-label={tNexus('imageCard.closeLightboxAria')}
                  onClick={() => {
                    setLightboxOpen(false)
                  }}
                >
                  {tNexus('imageCard.closeLightbox')}
                </button>
              </div>
              <img
                src={imageUrl}
                alt={info.prompt || tNexus('imageCard.altFallback')}
                className={css.lightboxImage}
              />
            </div>
          </div>,
          document.body,
        )
        : null}
    </div>
  )
}
