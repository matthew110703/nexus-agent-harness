/**
 * Turn-tail image gallery rendered directly beneath the assistant reply (outside `Called tools`).
 * Displays full-bleed PNG previews via `/api/file?path=...`, click-to-zoom lightbox,
 * Download PNG, Open in Preview, and exact Drive/Workspace save paths.
 */

import { useState } from 'react'
import { createPortal } from 'react-dom'
import type { PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { tNexus } from './locales.ts'
import type { NexusTurnImage } from './turn-images.ts'

export interface NexusTurnImagesTailInjected {
  useImages: (turn: PropsRuntime<'conversation.chat.turnTail'>['turn']) => readonly NexusTurnImage[]
}

export type NexusTurnImagesTailProps = PropsRuntime<'conversation.chat.turnTail'> & NexusTurnImagesTailInjected

function resolveImageUrl(item: NexusTurnImage): string {
  const diskPath = item.masterPath || item.drivePath || item.workspacePath
  if (diskPath) {
    return `/api/file?path=${encodeURIComponent(diskPath)}`
  }
  if (item.attachmentId) {
    return `/api/attachments/${encodeURIComponent(item.attachmentId)}`
  }
  return ''
}

/**
 * Render generated images for the current assistant turn directly in the main chat flow.
 * @param props - Turn-tail slot props plus `useImages` projection hook.
 * @returns Gallery JSX or `null` when the turn generated no images.
 */
export function NexusTurnImagesTail(props: NexusTurnImagesTailProps) {
  const images = props.useImages(props.turn).filter(img => Boolean(img.attachmentId || img.masterPath || img.drivePath))
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null)
  const [lightboxAlt, setLightboxAlt] = useState<string>('')
  const [copiedId, setCopiedId] = useState<string | null>(null)

  if (images.length === 0) {
    return null
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px', maxWidth: '580px' }}>
      {images.map((item) => {
        const src = resolveImageUrl(item)
        const primaryPath = item.drivePath || item.masterPath || item.workspacePath
        const metaParts = [
          item.width > 0 && item.height > 0 ? `${item.width}x${item.height}` : '',
          item.steps > 0 ? tNexus('imageCard.stepsLabel', { steps: item.steps }) : '',
          item.model,
        ].filter(Boolean)

        return (
          <div
            key={item.callId}
            data-nexus-turn-image={item.callId}
            style={{
              border: '1px solid var(--border-subtle, rgba(255,255,255,0.14))',
              borderRadius: '12px',
              padding: '10px',
              background: 'var(--bg-elevated, rgba(255,255,255,0.03))',
            }}
          >
            {src ? (
              <img
                src={src}
                alt={item.prompt || item.filename || tNexus('imageCard.altFallback')}
                onClick={() => {
                  setLightboxUrl(src)
                  setLightboxAlt(item.prompt || item.filename || tNexus('imageCard.altFallback'))
                }}
                style={{
                  display: 'block',
                  width: '100%',
                  maxHeight: '480px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  cursor: 'zoom-in',
                  background: 'rgba(0,0,0,0.18)',
                }}
              />
            ) : null}

            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
                marginTop: '8px',
                fontSize: '11px',
              }}
            >
              <span style={{ opacity: 0.72 }}>{metaParts.join(' · ')}</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {src ? (
                  <a
                    href={src}
                    download={item.filename || 'generated.png'}
                    style={{
                      padding: '3px 8px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-subtle, rgba(255,255,255,0.16))',
                      color: 'inherit',
                      textDecoration: 'none',
                    }}
                  >
                    {tNexus('imageCard.download')}
                  </a>
                ) : null}
                {primaryPath ? (
                  <button
                    type="button"
                    onClick={() => {
                      props.openFile(primaryPath)
                    }}
                    style={{
                      padding: '3px 8px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-subtle, rgba(255,255,255,0.16))',
                      background: 'transparent',
                      color: 'inherit',
                      cursor: 'pointer',
                    }}
                  >
                    {tNexus('imageCard.openPreview')}
                  </button>
                ) : null}
                {primaryPath ? (
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard.writeText(primaryPath).then(() => {
                        setCopiedId(item.callId)
                        setTimeout(() => { setCopiedId(cur => (cur === item.callId ? null : cur)) }, 1500)
                      })
                    }}
                    style={{
                      padding: '3px 8px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-subtle, rgba(255,255,255,0.16))',
                      background: 'transparent',
                      color: 'inherit',
                      cursor: 'pointer',
                    }}
                  >
                    {copiedId === item.callId ? tNexus('imageCard.copied') : tNexus('imageCard.copyPath')}
                  </button>
                ) : null}
              </div>
            </div>

            {primaryPath ? (
              <div style={{ marginTop: '6px', fontSize: '11px', opacity: 0.6, wordBreak: 'break-all' }}>
                {item.drivePath ? `${tNexus('imageCard.drivePath')}: ${item.drivePath}` : primaryPath}
              </div>
            ) : null}
          </div>
        )
      })}

      {lightboxUrl && typeof document !== 'undefined'
        ? createPortal(
          <div
            role="dialog"
            aria-modal="true"
            aria-label={tNexus('imageCard.closeLightboxAria')}
            onClick={() => { setLightboxUrl(null) }}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9999,
              background: 'rgba(0,0,0,0.88)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '24px',
              cursor: 'zoom-out',
            }}
          >
            <img
              src={lightboxUrl}
              alt={lightboxAlt}
              style={{ maxWidth: '94vw', maxHeight: '92vh', borderRadius: '8px', objectFit: 'contain' }}
            />
          </div>,
          document.body,
        )
        : null}
    </div>
  )
}
