/**
 * Read-only Nexus single-model runtime status, VRAM, and live progress control
 * mounted in the composer's `conversation.input.right` slot (v1.1.0).
 * Model selection, staging, and loading are owned strictly by Colab startup.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { resolveCompactModelName, resolveModelDisplayName } from './displayName.ts'
import { tNexus, type NexusCopyKey } from './locales.ts'
import {
  getNotificationPermission,
  notifyUnfocusedCompletion,
  readNotifyPrefs,
  requestNotificationPermission,
  writeNotifyPrefs,
  type NexusNotifyPrefs,
} from './notifications.ts'
import css from './NexusRuntimeControl.module.css'

/** Structured error thrown by Host proxy calls. */
export class NexusApiError extends Error {
  readonly status: number
  readonly operationId: string | undefined
  readonly details: string | undefined

  constructor(message: string, status: number, operationId?: string, details?: string) {
    super(message)
    this.name = 'NexusApiError'
    this.status = status
    this.operationId = operationId
    this.details = details
  }
}

/** Single Gateway progress record returned by `/v1/progress` or `/v1/progress/{id}`. */
export interface NexusOperationRecord {
  readonly operation_id: string
  readonly kind: string
  readonly model: string
  readonly status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  readonly phase: string
  readonly progress: number
  readonly indeterminate: boolean
  readonly step?: number
  readonly steps_total?: number
  readonly message?: string
  readonly error?: string | null
}

/** Worker status block embedded in `/v1/runtime` and `/v1/progress`. */
export interface NexusWorkerPayload {
  readonly pid: number
  readonly alive: boolean
  readonly active_kind?: 'language' | 'image' | null
  readonly active_model_id?: string | null
  readonly cuda_allocated_bytes?: number
  readonly cuda_reserved_bytes?: number
  readonly last_error?: string | null
}

/** Hardware block returned by `/v1/runtime`. */
export interface NexusHardwarePayload {
  readonly cuda_available?: boolean
  readonly total_vram_bytes?: number
  readonly free_vram_bytes?: number
  readonly gpu_count?: number
  readonly tier?: string
  readonly gpus?: ReadonlyArray<{ readonly name?: string; readonly total_vram_bytes?: number; readonly free_vram_bytes?: number }>
}

/** Locked model descriptor returned by `/v1/runtime`. */
export interface NexusLockedModelInfo {
  readonly id: string
  readonly display_name?: string | null
  readonly implementation?: string | null
}

/** Progress snapshot returned by `GET /api/nexus.runtime-progress`. */
export interface NexusProgressSnapshot {
  readonly chat?: NexusOperationRecord | null
  readonly diffusion?: NexusOperationRecord | null
  readonly operations?: readonly NexusOperationRecord[]
  readonly worker?: NexusWorkerPayload | null
}

/** Runtime status snapshot returned by `GET /api/nexus.runtime`. */
export interface NexusRuntimePayload {
  readonly version?: string
  readonly ready?: boolean
  readonly state?: string
  readonly busy?: boolean
  readonly swapping?: boolean
  readonly hardware?: NexusHardwarePayload | null
  readonly worker?: NexusWorkerPayload | null
  readonly language_model?: NexusLockedModelInfo | null
  readonly image_model?: NexusLockedModelInfo | null
}

/** Injected read-only HTTP callbacks for NexusRuntimeControl. */
export interface NexusRuntimeControlProps {
  readonly fetchProgress: (signal?: AbortSignal) => Promise<NexusProgressSnapshot>
  readonly fetchRuntime: (signal?: AbortSignal) => Promise<NexusRuntimePayload>
}

interface PopoverCoords {
  readonly bottom: number
  readonly right: number
}

const PHASE_COPY_KEYS: Record<string, NexusCopyKey> = {
  resolving: 'phase.resolving',
  processor_init: 'phase.tokenizer',
  weight_loading: 'phase.weightLoading',
  placement: 'phase.placement',
  ready: 'phase.ready',
  generating: 'phase.generating',
  swapping_to_image: 'phase.swappingToImage',
  denoising: 'phase.denoising',
  saving: 'phase.saving',
  restoring_language_model: 'phase.restoringLanguage',
}

function formatPhaseLabel(op: NexusOperationRecord): string {
  if (op.phase === 'denoising' && op.steps_total && op.steps_total > 0) {
    return tNexus('phase.denoisingStep', { step: op.step ?? 0, total: op.steps_total })
  }
  const key = PHASE_COPY_KEYS[op.phase]
  return key ? tNexus(key) : op.phase
}

function formatGb(bytes: number): string {
  return `${(bytes / (1024 ** 3)).toFixed(1)} GB`
}

function formatGpuMemory(runtime: NexusRuntimePayload | null): string {
  const hw = runtime?.hardware
  if (!hw) return tNexus('state.checking')
  if (!hw.cuda_available || !hw.total_vram_bytes) return tNexus('state.cpuOnly')
  const total = hw.total_vram_bytes
  const allocated = runtime.worker?.alive ? (runtime.worker.cuda_allocated_bytes ?? 0) : 0
  const used = allocated > 0
    ? allocated
    : Math.max(0, total - (hw.free_vram_bytes ?? total))
  const gpuName = hw.gpus?.[0]?.name
  const memText = `${formatGb(used)} / ${formatGb(total)}`
  return gpuName ? `${gpuName} · ${memText}` : memText
}

function pickActiveOperation(snapshot: NexusProgressSnapshot | null): NexusOperationRecord | null {
  const diffOp = snapshot?.diffusion ?? null
  if (diffOp && (diffOp.status === 'queued' || diffOp.status === 'running')) return diffOp
  const chatOp = snapshot?.chat ?? null
  if (chatOp && (chatOp.status === 'queued' || chatOp.status === 'running')) return chatOp
  return null
}

/**
 * Render the read-only Nexus single-model status badge and portaled diagnostics popover.
 * @param props - read-only Host API callbacks.
 * @returns the composer trigger button and, when open, a portaled popover dialog.
 */
export function NexusRuntimeControl(props: NexusRuntimeControlProps) {
  const { fetchProgress, fetchRuntime } = props

  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<PopoverCoords | null>(null)
  const [runtime, setRuntime] = useState<NexusRuntimePayload | null>(null)
  const [progressSnap, setProgressSnap] = useState<NexusProgressSnapshot | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [notifyPrefs, setNotifyPrefs] = useState<NexusNotifyPrefs>(() => readNotifyPrefs())
  const [notifyPerm, setNotifyPerm] = useState<NotificationPermission | 'unsupported'>(() => getNotificationPermission())

  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const popoverRef = useRef<HTMLDivElement | null>(null)
  const pollInFlightRef = useRef(false)
  const mountedRef = useRef(true)
  const activeAbortRef = useRef<AbortController | null>(null)
  const prevChatOpRef = useRef<{ id: string; status: string } | null>(null)
  const prevDiffOpRef = useRef<{ id: string; status: string } | null>(null)
  const notifyPrefsRef = useRef<NexusNotifyPrefs>(notifyPrefs)
  notifyPrefsRef.current = notifyPrefs

  const handleToggleNotify = useCallback(async () => {
    const nextEnabled = !notifyPrefs.enabled
    if (nextEnabled) {
      const perm = await requestNotificationPermission()
      setNotifyPerm(perm)
    }
    const next: NexusNotifyPrefs = { ...notifyPrefs, enabled: nextEnabled }
    setNotifyPrefs(next)
    writeNotifyPrefs(next)
  }, [notifyPrefs])

  const handleToggleSound = useCallback(() => {
    const next: NexusNotifyPrefs = { ...notifyPrefs, sound: !notifyPrefs.sound }
    setNotifyPrefs(next)
    writeNotifyPrefs(next)
  }, [notifyPrefs])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      activeAbortRef.current?.abort()
    }
  }, [])

  const closePopover = useCallback((restoreFocus: boolean) => {
    setOpen(false)
    if (restoreFocus) {
      triggerRef.current?.focus()
    }
  }, [])

  const pollNow = useCallback(async () => {
    if (pollInFlightRef.current || !mountedRef.current) return
    pollInFlightRef.current = true
    const controller = new AbortController()
    activeAbortRef.current = controller
    try {
      const [rtResult, progResult] = await Promise.allSettled([
        fetchRuntime(controller.signal),
        fetchProgress(controller.signal),
      ])
      if (controller.signal.aborted) return

      if (rtResult.status === 'fulfilled') {
        setRuntime(rtResult.value)
        setStatusError(null)
      } else {
        const err: unknown = rtResult.reason
        setStatusError(err instanceof Error ? err.message : String(err))
      }

      if (progResult.status === 'fulfilled') {
        const snap = progResult.value
        setProgressSnap(snap)

        const chatOp = snap.chat ?? null
        if (chatOp) {
          const prev = prevChatOpRef.current
          if (
            prev !== null
            && prev.id === chatOp.operation_id
            && (prev.status === 'running' || prev.status === 'queued')
            && (chatOp.status === 'completed' || chatOp.status === 'failed')
          ) {
            const label = resolveCompactModelName(chatOp.model, null)
            notifyUnfocusedCompletion({
              title: chatOp.status === 'completed'
                ? tNexus('notify.chatReadyTitle', { model: label })
                : tNexus('notify.chatFailedTitle', { model: label }),
              body: tNexus('notify.chatReadyBody'),
              prefs: notifyPrefsRef.current,
            })
          }
          prevChatOpRef.current = { id: chatOp.operation_id, status: chatOp.status }
        }

        const diffOp = snap.diffusion ?? null
        if (diffOp) {
          const prev = prevDiffOpRef.current
          if (
            prev !== null
            && prev.id === diffOp.operation_id
            && (prev.status === 'running' || prev.status === 'queued')
            && diffOp.status === 'completed'
          ) {
            const label = resolveCompactModelName(diffOp.model, null)
            notifyUnfocusedCompletion({
              title: tNexus('notify.imageReadyTitle', { model: label }),
              body: tNexus('notify.imageReadyBody'),
              prefs: notifyPrefsRef.current,
            })
          }
          prevDiffOpRef.current = { id: diffOp.operation_id, status: diffOp.status }
        }
      }
    } finally {
      pollInFlightRef.current = false
    }
  }, [fetchProgress, fetchRuntime])

  const activeOp = pickActiveOperation(progressSnap)
  const isOpRunning = Boolean(activeOp) || Boolean(runtime?.swapping)

  useEffect(() => {
    void pollNow()
    const intervalMs = isOpRunning ? 1000 : open ? 2500 : 6000
    const timer = window.setInterval(() => {
      void pollNow()
    }, intervalMs)
    return () => {
      window.clearInterval(timer)
    }
  }, [isOpRunning, open, pollNow])

  useEffect(() => {
    if (!open) return
    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect()
      if (!rect) return
      setCoords({
        bottom: Math.max(12, window.innerHeight - rect.top + 8),
        right: Math.max(12, Math.min(window.innerWidth - 360, window.innerWidth - rect.right)),
      })
    }
    updatePosition()

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null
      if (!target) return
      if (popoverRef.current?.contains(target)) return
      if (triggerRef.current?.contains(target)) return
      closePopover(false)
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closePopover(true)
      }
    }

    window.addEventListener('resize', updatePosition)
    document.addEventListener('pointerdown', onPointerDown)
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('resize', updatePosition)
      document.removeEventListener('pointerdown', onPointerDown)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [open, closePopover])

  const langInfo = runtime?.language_model ?? null
  const imgInfo = runtime?.image_model ?? null
  const langFullName = langInfo ? resolveModelDisplayName(langInfo.id, langInfo.display_name) : null
  const langCompactName = langInfo ? resolveCompactModelName(langInfo.id, langInfo.display_name) : null
  const imgFullName = imgInfo ? resolveModelDisplayName(imgInfo.id, imgInfo.display_name) : null

  const isReady = Boolean(runtime?.ready && runtime.worker?.alive)
  const isSwapping = Boolean(runtime?.swapping)

  let badgeText = tNexus('state.runtimeOff')
  let dotClass = ''
  if (activeOp) {
    dotClass = css.dotRunning ?? ''
    if (activeOp.phase === 'restoring_language_model') {
      badgeText = langCompactName
        ? tNexus('badge.restoringNamed', { model: langCompactName })
        : tNexus('badge.restoringModel')
    } else if (activeOp.phase === 'swapping_to_image') {
      badgeText = tNexus('state.swappingGpu')
    } else if (activeOp.kind === 'image_generation' || activeOp.phase === 'denoising') {
      badgeText = !activeOp.indeterminate && activeOp.progress > 0
        ? tNexus('badge.generatingPct', { pct: Math.round(activeOp.progress) })
        : tNexus('badge.generating')
    } else if (activeOp.kind === 'chat_generation') {
      badgeText = tNexus('badge.generating')
    } else {
      badgeText = langCompactName
        ? tNexus('badge.loadingNamed', { model: langCompactName })
        : tNexus('badge.loading')
    }
  } else if (isSwapping) {
    dotClass = css.dotRunning ?? ''
    badgeText = tNexus('state.swappingGpu')
  } else if (statusError || runtime?.state === 'degraded') {
    dotClass = css.dotError ?? ''
    badgeText = tNexus('badge.runtimeError')
  } else if (isReady && langCompactName) {
    dotClass = css.dotReady ?? ''
    badgeText = tNexus('badge.readyNamed', { model: langCompactName })
  }

  const runtimeStateText = isSwapping
    ? tNexus('state.swappingGpu')
    : activeOp
      ? formatPhaseLabel(activeOp)
      : isReady
        ? tNexus('state.ready')
        : runtime?.state === 'degraded'
          ? tNexus('state.degraded')
          : tNexus('state.notLoaded')

  const gpuMemoryText = formatGpuMemory(runtime)

  const popoverNode = open && coords && typeof document !== 'undefined'
    ? createPortal(
      <div
        ref={popoverRef}
        className={css.popover}
        role="dialog"
        aria-label={tNexus('panel.ariaLabel')}
        style={{ bottom: `${coords.bottom}px`, right: `${coords.right}px` }}
      >
        <div className={css.header}>
          <div className={css.headerLeft}>
            <span className={css.title}>{tNexus('panel.title')}</span>
            <span className={css.statePill}>
              <span className={`${css.dot} ${dotClass}`} />
              {runtimeStateText}
            </span>
          </div>
          <button
            type="button"
            className={css.closeBtn}
            onClick={() => { closePopover(true) }}
            aria-label={tNexus('panel.closeAria')}
          >
            ✕
          </button>
        </div>

        <div className={css.section}>
          <div className={css.row}>
            <span className={css.label}>{tNexus('panel.languageModel')}</span>
            <span className={css.value} title={langInfo?.id ?? tNexus('state.unavailable')}>
              {langFullName ?? tNexus('state.unavailable')}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>{tNexus('panel.imageGenerator')}</span>
            <span className={css.value} title={imgInfo?.id ?? tNexus('state.disabled')}>
              {imgFullName ?? tNexus('state.disabled')}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>{tNexus('panel.gpuMemory')}</span>
            <span className={css.value} title={gpuMemoryText}>
              {gpuMemoryText}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>{tNexus('notify.section')}</span>
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <button
                type="button"
                className={css.statePill}
                onClick={() => { void handleToggleNotify() }}
                style={{ cursor: 'pointer', border: 'none' }}
              >
                {notifyPerm === 'denied'
                  ? tNexus('notify.denied')
                  : notifyPrefs.enabled
                    ? tNexus('notify.enabled')
                    : tNexus('notify.enable')}
              </button>
              {notifyPrefs.enabled ? (
                <button
                  type="button"
                  className={css.statePill}
                  onClick={handleToggleSound}
                  style={{ cursor: 'pointer', border: 'none', opacity: notifyPrefs.sound ? 1 : 0.5 }}
                >
                  {tNexus('notify.sound')}
                </button>
              ) : null}
            </div>
          </div>
          <div className={css.helper}>
            {tNexus('panel.lockedNote')}
          </div>
        </div>

        {activeOp && (
          <div className={css.section} data-section="active-operation">
            <div className={css.row}>
              <span className={css.label}>{tNexus('panel.progress')}</span>
              <span className={css.value}>
                {formatPhaseLabel(activeOp)}
                {!activeOp.indeterminate && activeOp.progress > 0 ? ` · ${Math.round(activeOp.progress)}%` : ''}
              </span>
            </div>
            <div className={css.progressTrack}>
              {activeOp.indeterminate ? (
                <div className={css.progressIndeterminate} />
              ) : (
                <div
                  className={css.progressFill}
                  style={{ width: `${Math.max(0, Math.min(100, activeOp.progress))}%` }}
                />
              )}
            </div>
            {activeOp.message && <div className={css.helper}>{activeOp.message}</div>}
          </div>
        )}

        {statusError && (
          <div className={css.errorBox} role="alert" data-error-scope="runtime-status">
            {statusError}
          </div>
        )}
      </div>,
      document.body,
    )
    : null

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={css.trigger}
        onClick={() => { setOpen(prev => !prev) }}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={langInfo ? tNexus('panel.lockedTooltip', { model: langInfo.id }) : tNexus('panel.title')}
      >
        <span className={`${css.dot} ${dotClass}`} />
        <span className={css.triggerLabel}>{badgeText}</span>
      </button>
      {popoverNode}
    </>
  )
}
