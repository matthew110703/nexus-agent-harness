/**
 * Compact Nexus runtime & independent image-model control mounted in the
 * composer's `conversation.input.right` slot.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { resolveCompactModelName, resolveModelDisplayName } from './displayName.ts'
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

/** Per-session selected chat model info resolved from ModelDirectoryResolver. */
export interface SelectedChatModelInfo {
  readonly provider: string
  readonly modelId: string
  readonly displayName?: string
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
  readonly bytes_copied?: number
  readonly total_bytes?: number
  readonly step?: number
  readonly steps_total?: number
  readonly message?: string
  readonly error?: string | null
}

/** Worker status block embedded in `/v1/runtime` and `/v1/progress`. */
export interface NexusWorkerPayload {
  readonly pid: number
  readonly alive: boolean
  readonly active_kind?: 'chat' | 'diffusion' | null
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

/** Progress snapshot returned by `GET /api/nexus.runtime-progress`. */
export interface NexusProgressSnapshot {
  readonly chat?: NexusOperationRecord | null
  readonly diffusion?: NexusOperationRecord | null
  readonly operations?: readonly NexusOperationRecord[]
  readonly worker?: NexusWorkerPayload | null
}

/** Runtime status snapshot returned by `GET /api/nexus.runtime`. */
export interface NexusRuntimePayload {
  readonly hardware?: NexusHardwarePayload | null
  readonly worker?: NexusWorkerPayload | null
  readonly busy?: boolean
  readonly active_model?: string | null
  readonly active_kind?: 'chat' | 'diffusion' | null
}

/** Response from `POST /api/nexus.runtime-load`. */
export interface NexusLoadResponse {
  readonly status: 'loading' | 'loaded'
  readonly operation_id?: string | null
  readonly model: string
  readonly kind: 'chat' | 'diffusion'
  readonly worker?: NexusWorkerPayload | null
}

/** Image model catalog item returned by `/api/nexus.image-models`. */
export interface NexusImageModelOption {
  readonly id: string
  readonly display_name?: string | null
  readonly adapter_family?: string | null
}

/** Image model catalog + profile selection returned by `/api/nexus.image-models` & `POST /api/nexus.image-model`. */
export interface NexusImageModelsPayload {
  readonly selectedModelId: string | null
  readonly models: readonly NexusImageModelOption[]
  readonly qualityProfiles?: readonly string[]
}

/** Injected data and HTTP callbacks for NexusRuntimeControl. */
export interface NexusRuntimeControlProps {
  readonly getSelectedChatModel: () => SelectedChatModelInfo | null
  readonly subscribeSelectedChatModel: (listener: () => void) => () => void
  readonly fetchProgress: (signal?: AbortSignal) => Promise<NexusProgressSnapshot>
  readonly fetchOperationProgress: (operationId: string, signal?: AbortSignal) => Promise<NexusOperationRecord | null>
  readonly fetchRuntime: (signal?: AbortSignal) => Promise<NexusRuntimePayload>
  readonly fetchImageModels: (signal?: AbortSignal) => Promise<NexusImageModelsPayload>
  readonly saveImageModel: (modelId: string | null) => Promise<NexusImageModelsPayload>
  readonly loadModel: (modelId: string) => Promise<NexusLoadResponse>
  readonly unloadModel: () => Promise<void>
}

interface ScopedErrorState {
  readonly message: string
  readonly diagnostic?: string | undefined
}

interface PopoverCoords {
  readonly bottom: number
  readonly right: number
}

const PHASE_LABELS: Record<string, string> = {
  resolving: 'Resolving model',
  discovery: 'Scanning model files',
  checkpoint_validation: 'Checking model files',
  staging: 'Staging model files',
  processor_init: 'Loading tokenizer & processor',
  weight_loading: 'Loading model weights',
  placement: 'Preparing GPU placement',
  ready: 'Ready',
  generating: 'Generating response',
  denoising: 'Generating image',
  saving: 'Saving output',
}

function formatPhaseLabel(op: NexusOperationRecord): string {
  if (op.phase === 'denoising' && op.steps_total && op.steps_total > 0) {
    return `Denoising step ${op.step ?? 0}/${op.steps_total}`
  }
  return PHASE_LABELS[op.phase] ?? op.phase
}

function formatGb(bytes: number): string {
  return `${(bytes / (1024 ** 3)).toFixed(1)} GB`
}

function formatGpuMemory(runtime: NexusRuntimePayload | null): string {
  const hw = runtime?.hardware
  if (!hw) return 'Checking…'
  if (!hw.cuda_available || !hw.total_vram_bytes) return 'CPU only'
  const total = hw.total_vram_bytes
  const allocated = runtime?.worker?.alive ? (runtime.worker.cuda_allocated_bytes ?? 0) : 0
  const used = allocated > 0
    ? allocated
    : Math.max(0, total - (hw.free_vram_bytes ?? total))
  const gpuName = hw.gpus?.[0]?.name
  const memText = `${formatGb(used)} / ${formatGb(total)}`
  return gpuName ? `${gpuName} · ${memText}` : memText
}

function formatScopedError(err: unknown, context?: { modelId?: string; operationId?: string }): ScopedErrorState {
  if (err instanceof NexusApiError) {
    const diagParts = [
      err.status ? `HTTP ${err.status}` : null,
      err.operationId || context?.operationId ? `op=${err.operationId ?? context?.operationId}` : null,
      context?.modelId ? `model=${context.modelId}` : null,
      err.details ? err.details : null,
    ].filter(Boolean)
    return {
      message: err.message,
      diagnostic: diagParts.length > 0 ? diagParts.join('\n') : undefined,
    }
  }
  const message = err instanceof Error ? err.message : String(err)
  return { message }
}

function pickActiveOperation(
  trackedOp: NexusOperationRecord | null,
  snapshot: NexusProgressSnapshot | null,
): NexusOperationRecord | null {
  if (trackedOp && (trackedOp.status === 'queued' || trackedOp.status === 'running')) {
    return trackedOp
  }
  const chatOp = snapshot?.chat ?? null
  if (chatOp && (chatOp.status === 'queued' || chatOp.status === 'running')) return chatOp
  const diffOp = snapshot?.diffusion ?? null
  if (diffOp && (diffOp.status === 'queued' || diffOp.status === 'running')) return diffOp
  return null
}

/**
 * Render the compact Nexus runtime badge and portaled status/image-model popover.
 * @param props - reactive model-directory selector and Host API callbacks.
 * @returns the composer trigger button and, when open, a portaled popover dialog.
 */
export function NexusRuntimeControl(props: NexusRuntimeControlProps) {
  const {
    getSelectedChatModel,
    subscribeSelectedChatModel,
    fetchProgress,
    fetchOperationProgress,
    fetchRuntime,
    fetchImageModels,
    saveImageModel,
    loadModel,
    unloadModel,
  } = props

  const [selectedChat, setSelectedChat] = useState<SelectedChatModelInfo | null>(() => getSelectedChatModel())
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<PopoverCoords | null>(null)
  const [runtime, setRuntime] = useState<NexusRuntimePayload | null>(null)
  const [progressSnap, setProgressSnap] = useState<NexusProgressSnapshot | null>(null)
  const [trackedOpId, setTrackedOpId] = useState<string | null>(null)
  const [trackedOp, setTrackedOp] = useState<NexusOperationRecord | null>(null)
  const [imageState, setImageState] = useState<NexusImageModelsPayload | null>(null)

  const [submittingLoad, setSubmittingLoad] = useState(false)
  const [unloading, setUnloading] = useState(false)
  const [savingImage, setSavingImage] = useState(false)

  const [chatLoadError, setChatLoadError] = useState<ScopedErrorState | null>(null)
  const [imageError, setImageError] = useState<ScopedErrorState | null>(null)
  const [statusError, setStatusError] = useState<ScopedErrorState | null>(null)
  const [showChatDiag, setShowChatDiag] = useState(false)

  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const popoverRef = useRef<HTMLDivElement | null>(null)
  const pollInFlightRef = useRef(false)
  const mountedRef = useRef(true)
  const activeAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      activeAbortRef.current?.abort()
    }
  }, [])

  useEffect(() => {
    setSelectedChat(getSelectedChatModel())
    return subscribeSelectedChatModel(() => {
      setSelectedChat(getSelectedChatModel())
    })
  }, [getSelectedChatModel, subscribeSelectedChatModel])

  const closePopover = useCallback((restoreFocus: boolean) => {
    setOpen(false)
    if (restoreFocus) {
      triggerRef.current?.focus()
    }
  }, [])

  const pollNow = useCallback(async (explicitOpId?: string | null) => {
    if (pollInFlightRef.current || !mountedRef.current) return
    pollInFlightRef.current = true
    const controller = new AbortController()
    activeAbortRef.current = controller
    const opIdToCheck = explicitOpId !== undefined ? explicitOpId : trackedOpId
    try {
      const [rtResult, progResult, opResult] = await Promise.allSettled([
        fetchRuntime(controller.signal),
        fetchProgress(controller.signal),
        opIdToCheck ? fetchOperationProgress(opIdToCheck, controller.signal) : Promise.resolve(null),
      ])
      if (!mountedRef.current || controller.signal.aborted) return

      if (rtResult.status === 'fulfilled') {
        setRuntime(rtResult.value)
        setStatusError(null)
      } else {
        setStatusError(formatScopedError(rtResult.reason))
      }

      if (progResult.status === 'fulfilled') {
        setProgressSnap(progResult.value)
      }

      if (opIdToCheck && opResult.status === 'fulfilled') {
        const rec = opResult.value
        if (rec === null) {
          // Operation expired on Gateway; clear tracking and rely on live runtime snapshot.
          setTrackedOpId(null)
          setTrackedOp(null)
        } else {
          setTrackedOp(rec)
          if (rec.status === 'completed') {
            setTrackedOpId(null)
            setChatLoadError(null)
            // Refresh runtime immediately after terminal completion so Ready reflects worker truth.
            const freshRt = await fetchRuntime(controller.signal).catch(() => null)
            if (mountedRef.current && freshRt) setRuntime(freshRt)
          } else if (rec.status === 'failed' || rec.status === 'cancelled') {
            setTrackedOpId(null)
            if (rec.status === 'failed') {
              setChatLoadError({
                message: rec.error || rec.message || 'Model loading failed',
                diagnostic: `op=${rec.operation_id}\nmodel=${rec.model}\nphase=${rec.phase}`,
              })
            }
          }
        }
      }
    } finally {
      pollInFlightRef.current = false
    }
  }, [fetchOperationProgress, fetchProgress, fetchRuntime, trackedOpId])

  const activeOp = pickActiveOperation(trackedOp, progressSnap)
  const isOpRunning = Boolean(activeOp) || submittingLoad

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
    const controller = new AbortController()
    void fetchImageModels(controller.signal)
      .then((data) => {
        if (mountedRef.current) {
          setImageState(data)
          setImageError(null)
        }
      })
      .catch((err) => {
        if (mountedRef.current && !controller.signal.aborted) {
          setImageError(formatScopedError(err))
        }
      })
    return () => controller.abort()
  }, [open, fetchImageModels])

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

  const selectedModelId = selectedChat?.modelId ?? null
  const isNexusProvider = selectedChat !== null && selectedChat.provider === 'nexus'
  const selectedFullName = selectedModelId ? resolveModelDisplayName(selectedModelId, selectedChat?.displayName) : null
  const selectedCompactName = selectedModelId ? resolveCompactModelName(selectedModelId, selectedChat?.displayName) : null

  const workerAlive = Boolean(runtime?.worker?.alive)
  const loadedModelId = workerAlive ? (runtime?.worker?.active_model_id ?? null) : null
  const loadedFullName = loadedModelId ? resolveModelDisplayName(loadedModelId) : null
  const loadedCompactName = loadedModelId ? resolveCompactModelName(loadedModelId) : null
  const isSelectedAlreadyLoaded = Boolean(selectedModelId && loadedModelId && selectedModelId === loadedModelId)

  const handleLoad = useCallback(async () => {
    if (!selectedModelId || !isNexusProvider || isOpRunning || unloading) return
    setSubmittingLoad(true)
    setChatLoadError(null)
    setShowChatDiag(false)
    try {
      const res = await loadModel(selectedModelId)
      if (!mountedRef.current) return
      if (res.status === 'loading' && res.operation_id) {
        setTrackedOpId(res.operation_id)
        setTrackedOp({
          operation_id: res.operation_id,
          kind: 'chat_model_load',
          model: selectedModelId,
          status: 'running',
          phase: 'resolving',
          progress: 0,
          indeterminate: true,
          message: `Starting load for ${selectedCompactName ?? selectedModelId}`,
        })
        void pollNow(res.operation_id)
      } else {
        setTrackedOpId(null)
        setTrackedOp(null)
        void pollNow(null)
      }
    } catch (err) {
      if (!mountedRef.current) return
      setChatLoadError(formatScopedError(err, { modelId: selectedModelId }))
    } finally {
      if (mountedRef.current) setSubmittingLoad(false)
    }
  }, [isNexusProvider, isOpRunning, loadModel, pollNow, selectedCompactName, selectedModelId, unloading])

  const handleUnload = useCallback(async () => {
    if (!loadedModelId || isOpRunning || unloading) return
    setUnloading(true)
    setChatLoadError(null)
    try {
      await unloadModel()
      if (!mountedRef.current) return
      setTrackedOpId(null)
      setTrackedOp(null)
      void pollNow(null)
    } catch (err) {
      if (!mountedRef.current) return
      setChatLoadError(formatScopedError(err, { modelId: loadedModelId }))
    } finally {
      if (mountedRef.current) setUnloading(false)
    }
  }, [isOpRunning, loadedModelId, pollNow, unloadModel, unloading])

  const handleImageModelChange = useCallback(async (nextValue: string) => {
    setSavingImage(true)
    setImageError(null)
    try {
      const updated = await saveImageModel(nextValue === '' ? null : nextValue)
      if (mountedRef.current) setImageState(updated)
    } catch (err) {
      if (mountedRef.current) setImageError(formatScopedError(err))
    } finally {
      if (mountedRef.current) setSavingImage(false)
    }
  }, [saveImageModel])

  // Derive state-aware composer badge (truthful to worker/operation state, never raw technical ID).
  let badgeText = 'Runtime off'
  let dotClass = ''
  if (activeOp) {
    dotClass = css.dotRunning ?? ''
    if (activeOp.kind === 'image_generation' || activeOp.phase === 'denoising') {
      const pct = !activeOp.indeterminate && activeOp.progress > 0 ? ` · ${Math.round(activeOp.progress)}%` : ''
      badgeText = `Generating${pct}`
    } else if (activeOp.kind === 'chat_generation') {
      badgeText = 'Generating'
    } else {
      const targetName = resolveCompactModelName(activeOp.model) || activeOp.model
      badgeText = `Loading · ${targetName}`
    }
  } else if (submittingLoad && selectedCompactName) {
    dotClass = css.dotRunning ?? ''
    badgeText = `Loading · ${selectedCompactName}`
  } else if (chatLoadError || statusError) {
    dotClass = css.dotError ?? ''
    badgeText = 'Runtime error'
  } else if (loadedCompactName) {
    dotClass = css.dotReady ?? ''
    badgeText = `Ready · ${loadedCompactName}`
  }

  // Dynamic model-aware primary action button label.
  const compactTarget = selectedCompactName ?? selectedModelId ?? ''
  let loadButtonLabel = 'Select a Nexus model'
  if (!selectedModelId) {
    loadButtonLabel = 'Select a chat model'
  } else if (!isNexusProvider) {
    loadButtonLabel = 'External provider'
  } else if (submittingLoad || (activeOp && activeOp.kind === 'chat_model_load')) {
    loadButtonLabel = `Loading ${compactTarget}…`
  } else if (isSelectedAlreadyLoaded) {
    loadButtonLabel = 'Loaded'
  } else if (loadedModelId) {
    loadButtonLabel = `Switch to ${compactTarget}`
  } else {
    loadButtonLabel = `Load ${compactTarget}`
  }

  const canClickLoad = Boolean(
    selectedModelId
    && isNexusProvider
    && !isSelectedAlreadyLoaded
    && !isOpRunning
    && !unloading,
  )
  const canClickUnload = Boolean(loadedModelId && !isOpRunning && !unloading)
  const runtimeStateText = isOpRunning ? 'Loading' : loadedModelId ? 'Ready' : 'Not loaded'
  const gpuMemoryText = formatGpuMemory(runtime)

  const popoverNode = open && coords && typeof document !== 'undefined'
    ? createPortal(
      <div
        ref={popoverRef}
        className={css.popover}
        role="dialog"
        aria-label="Nexus runtime"
        style={{ bottom: `${coords.bottom}px`, right: `${coords.right}px` }}
      >
        <div className={css.header}>
          <div className={css.headerLeft}>
            <span className={css.title}>Nexus runtime</span>
            <span className={css.statePill}>
              <span className={`${css.dot} ${dotClass}`} />
              {runtimeStateText}
            </span>
          </div>
          <button
            type="button"
            className={css.closeBtn}
            onClick={() => closePopover(true)}
            aria-label="Close Nexus runtime panel"
          >
            ✕
          </button>
        </div>

        <div className={css.section}>
          <div className={css.row}>
            <span className={css.label}>Chat model</span>
            <span className={css.value} title={selectedModelId ?? 'None selected'}>
              {selectedFullName ?? 'None selected'}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>Runtime</span>
            <span className={css.value} title={loadedModelId ?? 'Not loaded'}>
              {loadedFullName ?? 'Not loaded'}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>GPU memory</span>
            <span className={css.value} title={gpuMemoryText}>
              {gpuMemoryText}
            </span>
          </div>

          {!isNexusProvider && selectedModelId && (
            <div className={css.helper}>
              Selected model uses an external provider; Nexus GPU loading applies to Nexus models only.
            </div>
          )}

          <div className={css.actions}>
            <button
              type="button"
              className={css.primaryBtn}
              disabled={!canClickLoad}
              onClick={() => void handleLoad()}
            >
              {loadButtonLabel}
            </button>
            <button
              type="button"
              className={css.secondaryBtn}
              disabled={!canClickUnload}
              onClick={() => void handleUnload()}
            >
              {unloading ? 'Unloading…' : 'Unload'}
            </button>
          </div>

          {chatLoadError && (
            <div className={css.errorBox} role="alert" data-error-scope="chat-load">
              <div>{chatLoadError.message}</div>
              {chatLoadError.diagnostic && (
                <>
                  <button
                    type="button"
                    className={css.detailsToggle}
                    onClick={() => setShowChatDiag(prev => !prev)}
                  >
                    {showChatDiag ? 'Hide details' : 'Technical details'}
                  </button>
                  {showChatDiag && <pre className={css.detailsPre}>{chatLoadError.diagnostic}</pre>}
                </>
              )}
            </div>
          )}
        </div>

        {activeOp && (
          <div className={css.section} data-section="active-operation">
            <div className={css.row}>
              <span className={css.label}>Progress</span>
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

        <div className={css.section} data-section="image-generator">
          <label className={css.label} htmlFor="nexus-image-model-select">
            Image generator
          </label>
          <select
            id="nexus-image-model-select"
            className={css.select}
            value={imageState?.selectedModelId ?? ''}
            disabled={savingImage}
            onChange={event => void handleImageModelChange(event.target.value)}
          >
            <option value="">Choose an image model</option>
            {(imageState?.models ?? []).map(item => (
              <option key={item.id} value={item.id}>
                {resolveModelDisplayName(item.id, item.display_name)}
              </option>
            ))}
          </select>
          <div className={css.helper}>Required for image generation only.</div>
          {imageError && (
            <div className={css.errorBox} role="alert" data-error-scope="image-selection">
              {imageError.message}
            </div>
          )}
        </div>

        {statusError && (
          <div className={css.errorBox} role="alert" data-error-scope="runtime-status">
            {statusError.message}
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
        onClick={() => setOpen(prev => !prev)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title={loadedModelId ? `Loaded: ${loadedModelId}` : selectedModelId ? `Selected: ${selectedModelId}` : 'Nexus runtime'}
      >
        <span className={`${css.dot} ${dotClass}`} />
        <span className={css.triggerLabel}>{badgeText}</span>
      </button>
      {popoverNode}
    </>
  )
}
