/**
 * Turn-tail image showcase rendered directly beneath the assistant reply (outside `Called tools`).
 * Uses reactive `keyedHooks` store subscription, native `--dsw-alias-*` tokens, one-click
 * Right-Sidebar zoom/pan preview (`openFile`), and an `Escape`-enabled blurred modal lightbox.
 */

import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import type { ObservableSnapshot } from '@deepseek-ai/dsh-client-store'
import type { InjectFace, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import { resolveCompactModelName } from './displayName.ts'
import { tNexus } from './locales.ts'
import css from './NexusImageCard.module.css'
import type { NexusTurnImage } from './turn-images.ts'

/** Turn-keyed reactive image source injected from `conversation.chat.turnTail`. */
export interface NexusTurnImagesTailInjected {
  keyedHooks: {
    images: (turn: string) => ObservableSnapshot<readonly NexusTurnImage[]> | undefined
  }
}

/** Complete props received by `NexusTurnImagesTail`. */
export type NexusTurnImagesTailProps = PropsRuntime<'conversation.chat.turnTail'> & InjectFace<NexusTurnImagesTailInjected>

function resolvePrimaryImageUrl(item: NexusTurnImage): string {
  const diskPath = item.workspacePath || item.masterPath || item.drivePath
  if (diskPath) {
    return `/api/file?path=${encodeURIComponent(diskPath)}`
  }
  if (item.attachmentId) {
    return `/api/attachments/${encodeURIComponent(item.attachmentId)}`
  }
  return ''
}

function resolveFallbackImageUrl(item: NexusTurnImage): string {
  const altDiskPath = item.drivePath && item.drivePath !== item.workspacePath ? item.drivePath : item.masterPath
  if (altDiskPath) {
    return `/api/file?path=${encodeURIComponent(altDiskPath)}`
  }
  return ''
}

function formatShortBadgeLabel(item: NexusTurnImage): string {
  const name = item.filename || 'generated.png'
  if (item.workspacePath) {
    return tNexus('imageCard.workspaceFileBadge', { name })
  }
  return name
}

interface LightboxState {
  url: string
  alt: string
  filename: string
  sidebarPath: string
}

/**
 * Render generated images for the current assistant turn directly in the main chat flow.
 * @param props - Turn-tail slot props plus reactive `useImages` keyed selector hook.
 * @returns Showcase gallery JSX or `null` when the turn generated no images.
 */
export function NexusTurnImagesTail(props: NexusTurnImagesTailProps) {
  const turnKey = String(props.turn.turn)
  const rawImages = props.useImages(turnKey) ?? []
  const images = rawImages.filter(img => Boolean(img.attachmentId || img.masterPath || img.drivePath || img.workspacePath))

  const [loadedMap, setLoadedMap] = useState<Record<string, boolean>>({})
  const [fallbackMap, setFallbackMap] = useState<Record<string, boolean>>({})
  const [errorMap, setErrorMap] = useState<Record<string, boolean>>({})
  const [lightbox, setLightbox] = useState<LightboxState | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  useEffect(() => {
    if (lightbox === null || typeof window === 'undefined') return
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        ev.stopPropagation()
        setLightbox(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [lightbox])

  if (images.length === 0) {
    return null
  }

  return (
    <div className={css.gallery} data-nexus-turn-gallery={turnKey}>
      {images.map((item) => {
        const useFallback = fallbackMap[item.callId] === true
        const primaryUrl = resolvePrimaryImageUrl(item)
        const backupUrl = resolveFallbackImageUrl(item)
        const src = useFallback && backupUrl ? backupUrl : primaryUrl
        const sidebarTargetPath = item.workspacePath || item.masterPath || item.drivePath
        const copyTargetPath = item.workspacePath || item.drivePath || item.masterPath
        const altText = item.prompt || item.filename || tNexus('imageCard.altFallback')
        const isLoaded = loadedMap[item.callId] === true
        const hasError = errorMap[item.callId] === true

        const metaParts = [
          item.width > 0 && item.height > 0 ? `${item.width}×${item.height}` : '',
          item.model ? resolveCompactModelName(item.model) : '',
          item.steps > 0 ? tNexus('imageCard.stepsLabel', { steps: item.steps }) : '',
        ].filter(Boolean)

        const openLightbox = () => {
          if (!src) return
          setLightbox({
            url: src,
            alt: altText,
            filename: item.filename || 'generated.png',
            sidebarPath: sidebarTargetPath,
          })
        }

        return (
          <div key={item.callId} className={css.heroCard} data-nexus-turn-image={item.callId}>
            <div className={css.stage}>
              {!isLoaded ? (
                <div className={css.skeleton}>
                  {hasError ? tNexus('imageCard.previewError') : tNexus('imageCard.loadingPreview')}
                </div>
              ) : null}

              {src && !hasError ? (
                <button
                  type="button"
                  className={css.stageButton}
                  aria-label={tNexus('imageCard.zoom')}
                  onClick={openLightbox}
                >
                  <img
                    src={src}
                    alt={altText}
                    className={css.heroImg}
                    data-loading={isLoaded ? 'false' : 'true'}
                    onLoad={() => {
                      setLoadedMap(prev => ({ ...prev, [item.callId]: true }))
                    }}
                    onError={() => {
                      if (!useFallback && backupUrl && backupUrl !== primaryUrl) {
                        setFallbackMap(prev => ({ ...prev, [item.callId]: true }))
                      } else {
                        setErrorMap(prev => ({ ...prev, [item.callId]: true }))
                      }
                    }}
                  />
                </button>
              ) : null}

              <div className={css.floatingActions}>
                {src && !hasError ? (
                  <button type="button" className={css.glassBtn} onClick={openLightbox}>
                    {tNexus('imageCard.zoom')}
                  </button>
                ) : null}
                {src ? (
                  <a
                    href={src}
                    download={item.filename || 'generated.png'}
                    className={css.glassBtn}
                  >
                    {tNexus('imageCard.download')}
                  </a>
                ) : null}
              </div>
            </div>

            <div className={css.footer}>
              <div className={css.footerMeta}>
                {sidebarTargetPath ? (
                  <button
                    type="button"
                    className={css.workspaceBadge}
                    title={sidebarTargetPath}
                    onClick={() => {
                      props.openFile(sidebarTargetPath)
                    }}
                  >
                    {formatShortBadgeLabel(item)}
                  </button>
                ) : (
                  <span className={css.workspaceBadge}>{formatShortBadgeLabel(item)}</span>
                )}
                {metaParts.length > 0 ? (
                  <span className={css.metaSubline}>{metaParts.join(' · ')}</span>
                ) : null}
              </div>

              <div className={css.footerActions}>
                {sidebarTargetPath ? (
                  <button
                    type="button"
                    className={css.actionBtn}
                    onClick={() => {
                      props.openFile(sidebarTargetPath)
                    }}
                  >
                    {tNexus('imageCard.openPreview')}
                  </button>
                ) : null}
                {copyTargetPath ? (
                  <button
                    type="button"
                    className={css.actionBtn}
                    onClick={() => {
                      void navigator.clipboard.writeText(copyTargetPath).then(() => {
                        setCopiedId(item.callId)
                        setTimeout(() => {
                          setCopiedId(cur => (cur === item.callId ? null : cur))
                        }, 1500)
                      })
                    }}
                  >
                    {copiedId === item.callId ? tNexus('imageCard.copied') : tNexus('imageCard.copyPath')}
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        )
      })}

      {lightbox !== null && typeof document !== 'undefined'
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
                setLightbox(null)
              }}
            />
            <div
              className={css.lightboxStage}
              onClick={(ev) => {
                if (ev.target === ev.currentTarget) {
                  setLightbox(null)
                }
              }}
            >
              <div className={css.lightboxToolbar}>
                {lightbox.sidebarPath ? (
                  <button
                    type="button"
                    className={css.glassBtn}
                    onClick={() => {
                      const target = lightbox.sidebarPath
                      setLightbox(null)
                      props.openFile(target)
                    }}
                  >
                    {tNexus('imageCard.openPreview')}
                  </button>
                ) : null}
                <a
                  href={lightbox.url}
                  download={lightbox.filename}
                  className={css.glassBtn}
                >
                  {tNexus('imageCard.download')}
                </a>
                <button
                  type="button"
                  className={css.glassBtn}
                  aria-label={tNexus('imageCard.closeLightboxAria')}
                  onClick={() => {
                    setLightbox(null)
                  }}
                >
                  {tNexus('imageCard.closeLightbox')}
                </button>
              </div>
              <img src={lightbox.url} alt={lightbox.alt} className={css.lightboxImage} />
            </div>
          </div>,
          document.body,
        )
        : null}
    </div>
  )
}
