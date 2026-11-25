// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import {
  NexusApiError,
  NexusRuntimeControl,
  type NexusOperationRecord,
  type NexusRuntimePayload,
  type SelectedChatModelInfo,
} from '../src/client/NexusRuntimeControl.tsx'
import { resolveCompactModelName, resolveModelDisplayName } from '../src/client/displayName.ts'

function makeRuntime(overrides?: Partial<NexusRuntimePayload>): NexusRuntimePayload {
  return {
    hardware: {
      cuda_available: true,
      total_vram_bytes: 24 * 1024 ** 3,
      free_vram_bytes: 20 * 1024 ** 3,
      gpus: [{ name: 'NVIDIA L4', total_vram_bytes: 24 * 1024 ** 3, free_vram_bytes: 20 * 1024 ** 3 }],
    },
    worker: {
      pid: 0,
      alive: false,
      active_kind: null,
      active_model_id: null,
      cuda_allocated_bytes: 0,
    },
    busy: false,
    active_model: null,
    active_kind: null,
    ...overrides,
  }
}

describe('displayName resolver', () => {
  it('formats long technical model IDs into readable labels without changing canonical IDs', () => {
    expect(resolveModelDisplayName('Qwen-3-VL-8B-Instruct-heretic')).toBe('Qwen3-VL 8B · Heretic')
    expect(resolveModelDisplayName('Qwen3-VL-4b-Heretic')).toBe('Qwen3-VL 4B · Heretic')
    expect(resolveCompactModelName('Qwen-3-VL-8B-Instruct-heretic')).toBe('Qwen3-VL 8B')
    expect(resolveModelDisplayName('custom-id', 'Custom Pretty Name')).toBe('Custom Pretty Name')
  })
})

describe('NexusRuntimeControl', () => {
  it('handles open/close, outside click, Escape, inside click, select interaction, display names, and model-aware Load', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    const selected: SelectedChatModelInfo = {
      provider: 'nexus',
      modelId: 'Qwen-3-VL-8B-Instruct-heretic',
    }
    const saveImageModel = vi.fn(async (modelId: string | null) => ({
      selectedModelId: modelId,
      models: [{ id: 'FLUX.2-klein-9B', display_name: 'FLUX.2 Klein 9B' }],
    }))
    const loadModel = vi.fn(async (modelId: string) => ({
      status: 'loading' as const,
      operation_id: 'op_load_1',
      model: modelId,
      kind: 'chat' as const,
    }))
    const unloadModel = vi.fn(async () => {})
    const fetchOperationProgress = vi.fn(async (): Promise<NexusOperationRecord | null> => ({
      operation_id: 'op_load_1',
      kind: 'chat_model_load',
      model: 'Qwen-3-VL-8B-Instruct-heretic',
      status: 'running',
      phase: 'weight_loading',
      progress: 0,
      indeterminate: true,
      message: 'Loading weights in isolated worker',
    }))

    await act(async () => {
      root.render(
        <NexusRuntimeControl
          getSelectedChatModel={() => selected}
          subscribeSelectedChatModel={() => () => {}}
          fetchProgress={async () => ({ chat: null, diffusion: null, operations: [] })}
          fetchOperationProgress={fetchOperationProgress}
          fetchRuntime={async () => makeRuntime()}
          fetchImageModels={async () => ({
            selectedModelId: null,
            models: [{ id: 'FLUX.2-klein-9B', display_name: 'FLUX.2 Klein 9B' }],
          })}
          saveImageModel={saveImageModel}
          loadModel={loadModel}
          unloadModel={unloadModel}
        />,
      )
    })

    // Composer badge reflects actual runtime state ("Runtime off"), not "Idle (full-model-id)".
    const trigger = container.querySelector('button')!
    expect(trigger.textContent).toBe('Runtime off')

    // Open popover via trigger click.
    await act(async () => { trigger.click() })
    const dialog = document.body.querySelector('[role="dialog"]')!
    expect(dialog).not.toBeNull()
    expect(dialog.textContent).toContain('Qwen3-VL 8B · Heretic')
    expect(dialog.textContent).toContain('Not loaded')
    expect(dialog.textContent).toContain('NVIDIA L4')
    expect(dialog.textContent).toContain('Required for image generation only.')

    // Click inside popover and interact with image select -> MUST NOT close popover.
    const select = dialog.querySelector('select')!
    await act(async () => {
      select.dispatchEvent(new Event('pointerdown', { bubbles: true }))
    })
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull()

    // Load button is model-aware and works even with no image model selected.
    const loadBtn = Array.from(dialog.querySelectorAll('button')).find(b => b.textContent === 'Load Qwen3-VL 8B')!
    expect(loadBtn).not.toBeUndefined()
    expect(loadBtn.disabled).toBe(false)

    await act(async () => { loadBtn.click() })
    expect(loadModel).toHaveBeenCalledWith('Qwen-3-VL-8B-Instruct-heretic')
    // While loading, badge and button update and duplicate Load is disabled.
    expect(trigger.textContent).toBe('Loading · Qwen3-VL 8B')
    const loadingBtn = Array.from(dialog.querySelectorAll('button')).find(b => b.textContent?.includes('Loading Qwen3-VL 8B'))!
    expect(loadingBtn.disabled).toBe(true)

    // Escape closes popover and restores focus to trigger.
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    })
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()

    // Reopen and close via outside pointerdown.
    await act(async () => { trigger.click() })
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull()
    await act(async () => {
      document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }))
    })
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()

    // Unmount cleans up cleanly.
    await act(async () => { root.unmount() })
    container.remove()
  })

  it('shows Ready state, Switch button, scoped chat error (not under image selector), and aborts on unmount', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    let capturedSignal: AbortSignal | undefined
    const fetchRuntime = vi.fn(async (signal?: AbortSignal) => {
      capturedSignal = signal
      return makeRuntime({
        worker: {
          pid: 1001,
          alive: true,
          active_kind: 'chat',
          active_model_id: 'Qwen3-VL-4b-Heretic',
          cuda_allocated_bytes: 4 * 1024 ** 3,
        },
        active_model: 'Qwen3-VL-4b-Heretic',
        active_kind: 'chat',
      })
    })
    const loadModel = vi.fn(async () => {
      throw new NexusApiError('GPU memory is insufficient', 507, 'op_oom', 'CUDA OOM')
    })

    await act(async () => {
      root.render(
        <NexusRuntimeControl
          getSelectedChatModel={() => ({ provider: 'nexus', modelId: 'Qwen-3-VL-8B-Instruct-heretic' })}
          subscribeSelectedChatModel={() => () => {}}
          fetchProgress={async () => ({ chat: null, diffusion: null, operations: [] })}
          fetchOperationProgress={async () => null}
          fetchRuntime={fetchRuntime}
          fetchImageModels={async () => ({ selectedModelId: null, models: [] })}
          saveImageModel={async () => ({ selectedModelId: null, models: [] })}
          loadModel={loadModel}
          unloadModel={async () => {}}
        />,
      )
    })

    const trigger = container.querySelector('button')!
    expect(trigger.textContent).toBe('Ready · Qwen3-VL 4B')

    await act(async () => { trigger.click() })
    const dialog = document.body.querySelector('[role="dialog"]')!
    const switchBtn = Array.from(dialog.querySelectorAll('button')).find(b => b.textContent === 'Switch to Qwen3-VL 8B')!
    expect(switchBtn).not.toBeUndefined()

    await act(async () => { switchBtn.click() })
    const chatErr = dialog.querySelector('[data-error-scope="chat-load"]')!
    expect(chatErr).not.toBeNull()
    expect(chatErr.textContent).toContain('GPU memory is insufficient')
    expect(dialog.querySelector('[data-error-scope="image-selection"]')).toBeNull()

    await act(async () => { root.unmount() })
    expect(capturedSignal?.aborted).toBe(true)
    container.remove()
  })
})
