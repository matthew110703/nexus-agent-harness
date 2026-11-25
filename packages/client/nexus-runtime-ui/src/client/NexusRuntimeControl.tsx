/**
 * Compact Nexus runtime & independent image-model control mounted in the
 * composer's `conversation.input.right` slot.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import css from './NexusRuntimeControl.module.css'

/** Active gateway operation payload returned by `/api/nexus.runtime-progress`. */
export interface NexusProgressPayload {
  readonly status?: string
  readonly kind?: string
  readonly stage?: string
  readonly model_id?: string | null
  readonly percent?: number | null
  readonly current?: number | null
  readonly total?: number | null
  readonly unit?: string | null
  readonly message?: string | null
}

/** Active runtime status payload returned by `/api/nexus.runtime`. */
export interface NexusRuntimePayload {
  readonly loaded_model_id?: string | null
  readonly loaded_model_type?: string | null
  readonly loaded_adapter_family?: string | null
  readonly worker_alive?: boolean
  readonly vram_allocated_mb?: number | null
  readonly vram_total_mb?: number | null
  readonly vram_used_mb?: number | null
}

/** Image model catalog item returned by `/api/nexus.image-models`. */
export interface NexusImageModelOption {
  readonly id: string
  readonly adapter_family?: string | null
}

/** Image model catalog + profile selection returned by `/api/nexus.image-models`. */
export interface NexusImageModelsPayload {
  readonly selected_image_model: string | null
  readonly models: readonly NexusImageModelOption[]
}

/** Injected data and HTTP callbacks for NexusRuntimeControl. */
export interface NexusRuntimeControlProps {
  /** Read the current session's selected chat model id from ModelDirectoryResolver. */
  readonly getSelectedChatModel: () => string | null
  /** Subscribe to selected chat model changes for this session. */
  readonly subscribeSelectedChatModel: (listener: () => void) => () => void
  /** Fetch live gateway progress from the Host endpoint. */
  readonly fetchProgress: () => Promise<NexusProgressPayload | null>
  /** Fetch live worker/GPU runtime status from the Host endpoint. */
  readonly fetchRuntime: () => Promise<NexusRuntimePayload | null>
  /** Fetch discovered image models and profile-persisted selection. */
  readonly fetchImageModels: () => Promise<NexusImageModelsPayload | null>
  /** Persist profile-level image model selection (`null` clears selection). */
  readonly saveImageModel: (modelId: string | null) => Promise<NexusImageModelsPayload>
  /** Explicitly load a model into the GPU worker. */
  readonly loadModel: (modelId: string) => Promise<void>
  /** Explicitly unload the active GPU worker model. */
  readonly unloadModel: () => Promise<void>
}

interface PopoverCoords {
  readonly bottom: number
  readonly right: number
}

function formatTriggerLabel(
  progress: NexusProgressPayload | null,
  runtime: NexusRuntimePayload | null,
  selectedChatModel: string | null,
): string {
  if (progress && progress.status === 'running') {
    const pct = typeof progress.percent === 'number' ? ` ${Math.round(progress.percent)}%` : ''
    const stage = progress.stage || progress.kind || 'Working'
    return `${stage}${pct}`
  }
  if (runtime?.loaded_model_id) {
    return `GPU: ${runtime.loaded_model_id}`
  }
  if (selectedChatModel) {
    return `Idle (${selectedChatModel})`
  }
  return 'Nexus GPU'
}

/**
 * Render the compact Nexus runtime trigger and portaled status/image-model popover.
 * @param props - reactive model-directory selector and Host API callbacks.
 * @returns the trigger button and, when open, a document.body-portaled popover.
 */
export function NexusRuntimeControl(props: NexusRuntimeControlProps) {
  const {
    getSelectedChatModel,
    subscribeSelectedChatModel,
    fetchProgress,
    fetchRuntime,
    fetchImageModels,
    saveImageModel,
    loadModel,
    unloadModel,
  } = props

  const [selectedChatModel, setSelectedChatModel] = useState<string | null>(() => getSelectedChatModel())

  useEffect(() => {
    setSelectedChatModel(getSelectedChatModel())
    return subscribeSelectedChatModel(() => {
      setSelectedChatModel(getSelectedChatModel())
    })
  }, [getSelectedChatModel, subscribeSelectedChatModel])

  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState<PopoverCoords | null>(null)
  const [progress, setProgress] = useState<NexusProgressPayload | null>(null)
  const [runtime, setRuntime] = useState<NexusRuntimePayload | null>(null)
  const [imageState, setImageState] = useState<NexusImageModelsPayload | null>(null)
  const [busyAction, setBusyAction] = useState<'load' | 'unload' | 'image' | null>(null)
  const [errorText, setErrorText] = useState<string | null>(null)

  const triggerRef = useRef<HTMLButtonElement | null>(null)

  const refreshStatus = useCallback(async () => {
    const [nextProgress, nextRuntime] = await Promise.all([
      fetchProgress().catch(() => null),
      fetchRuntime().catch(() => null),
    ])
    if (nextProgress !== null) setProgress(nextProgress)
    if (nextRuntime !== null) setRuntime(nextRuntime)
  }, [fetchProgress, fetchRuntime])

  const refreshImageModels = useCallback(async () => {
    const next = await fetchImageModels().catch(() => null)
    if (next !== null) setImageState(next)
  }, [fetchImageModels])

  useEffect(() => {
    let active = true
    const run = async () => {
      if (!active) return
      await refreshStatus()
    }
    void run()
    const intervalMs = open || progress?.status === 'running' ? 1500 : 5000
    const timer = window.setInterval(() => {
      void run()
    }, intervalMs)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [open, progress?.status, refreshStatus])

  useEffect(() => {
    if (!open) return
    void refreshImageModels()
    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect()
      if (!rect) return
      setCoords({
        bottom: Math.max(12, window.innerHeight - rect.top + 8),
        right: Math.max(12, window.innerWidth - rect.right),
      })
    }
    updatePosition()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('resize', updatePosition)
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('resize', updatePosition)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [open, refreshImageModels])

  const handleLoadSelected = useCallback(async () => {
    if (!selectedChatModel) return
    setBusyAction('load')
    setErrorText(null)
    try {
      await loadModel(selectedChatModel)
      await refreshStatus()
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error))
    } finally {
      setBusyAction(null)
    }
  }, [loadModel, refreshStatus, selectedChatModel])

  const handleUnload = useCallback(async () => {
    setBusyAction('unload')
    setErrorText(null)
    try {
      await unloadModel()
      await refreshStatus()
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error))
    } finally {
      setBusyAction(null)
    }
  }, [unloadModel, refreshStatus])

  const handleImageModelChange = useCallback(async (nextValue: string) => {
    setBusyAction('image')
    setErrorText(null)
    try {
      const updated = await saveImageModel(nextValue === '' ? null : nextValue)
      setImageState(updated)
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : String(error))
    } finally {
      setBusyAction(null)
    }
  }, [saveImageModel])

  const isRunning = progress?.status === 'running'
  const isError = progress?.status === 'error'
  const isLoaded = Boolean(runtime?.loaded_model_id)
  const dotClass = isError
    ? css.dotError
    : isRunning
      ? css.dotRunning
      : isLoaded
        ? css.dotReady
        : ''

  const triggerLabel = formatTriggerLabel(progress, runtime, selectedChatModel)
  const loadedModelId = runtime?.loaded_model_id ?? null
  const canLoadSelected = Boolean(selectedChatModel) && selectedChatModel !== loadedModelId && busyAction === null
  const canUnload = Boolean(loadedModelId) && busyAction === null

  const vramText =
    typeof runtime?.vram_allocated_mb === 'number' && typeof runtime?.vram_total_mb === 'number'
      ? `${Math.round(runtime.vram_allocated_mb)} / ${Math.round(runtime.vram_total_mb)} MB`
      : typeof runtime?.vram_used_mb === 'number' && typeof runtime?.vram_total_mb === 'number'
        ? `${Math.round(runtime.vram_used_mb)} / ${Math.round(runtime.vram_total_mb)} MB`
        : '—'

  const popoverNode = open && coords && typeof document !== 'undefined'
    ? createPortal(
      <div
        className={css.popover}
        role="dialog"
        aria-label="Nexus runtime status"
        style={{ bottom: `${coords.bottom}px`, right: `${coords.right}px` }}
      >
        <div className={css.header}>
          <span className={css.title}>Nexus runtime</span>
          <button
            type="button"
            className={css.closeBtn}
            onClick={() => setOpen(false)}
            aria-label="Close Nexus runtime panel"
          >
            ✕
          </button>
        </div>

        <div className={css.section}>
          <div className={css.row}>
            <span className={css.label}>Selected chat model</span>
            <span className={css.value} title={selectedChatModel ?? 'None'}>
              {selectedChatModel ?? 'None'}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>Loaded worker model</span>
            <span className={css.value} title={loadedModelId ?? 'Unloaded'}>
              {loadedModelId ?? 'Unloaded'}
            </span>
          </div>
          <div className={css.row}>
            <span className={css.label}>GPU VRAM</span>
            <span className={css.value}>{vramText}</span>
          </div>
          <div className={css.actions}>
            <button
              type="button"
              className={css.actionBtn}
              disabled={!canLoadSelected}
              onClick={() => void handleLoadSelected()}
            >
              {busyAction === 'load' ? 'Loading…' : 'Load selected'}
            </button>
            <button
              type="button"
              className={css.actionBtn}
              disabled={!canUnload}
              onClick={() => void handleUnload()}
            >
              {busyAction === 'unload' ? 'Unloading…' : 'Unload'}
            </button>
          </div>
        </div>

        <div className={css.section}>
          <label className={css.label} htmlFor="nexus-image-model-select">
            Selected image model (profile)
          </label>
          <select
            id="nexus-image-model-select"
            className={css.select}
            value={imageState?.selected_image_model ?? ''}
            disabled={busyAction === 'image'}
            onChange={event => void handleImageModelChange(event.target.value)}
          >
            <option value="">Not selected (must choose before generating)</option>
            {(imageState?.models ?? []).map(item => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        </div>

        {progress && progress.status && progress.status !== 'idle' && (
          <div className={css.section}>
            <div className={css.row}>
              <span className={css.label}>Active operation</span>
              <span className={css.value}>
                {progress.stage || progress.kind || progress.status}
              </span>
            </div>
            {typeof progress.percent === 'number' && (
              <div className={css.progressTrack}>
                <div
                  className={css.progressFill}
                  style={{ width: `${Math.max(0, Math.min(100, progress.percent))}%` }}
                />
              </div>
            )}
            {progress.message && <div className={css.muted}>{progress.message}</div>}
          </div>
        )}

        {errorText && <div className={css.error} role="alert">{errorText}</div>}
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
        title={triggerLabel}
      >
        <span className={`${css.dot} ${dotClass}`} />
        <span className={css.triggerLabel}>{triggerLabel}</span>
      </button>
      {popoverNode}
    </>
  )
}
