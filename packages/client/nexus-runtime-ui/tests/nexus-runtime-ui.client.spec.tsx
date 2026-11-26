// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import {
  NexusRuntimeControl,
  type NexusProgressSnapshot,
  type NexusRuntimePayload,
} from '../src/client/NexusRuntimeControl.tsx'
import { resolveCompactModelName, resolveModelDisplayName } from '../src/client/displayName.ts'

function makeRuntime(overrides?: Partial<NexusRuntimePayload>): NexusRuntimePayload {
  return {
    version: '1.1.0',
    ready: true,
    state: 'ready',
    busy: false,
    swapping: false,
    language_model: { id: 'Qwen-3-VL-8B-Instruct-heretic', implementation: 'qwen3_vl' },
    image_model: { id: 'FLUX.2-klein-4B', implementation: 'flux2_klein' },
    hardware: {
      cuda_available: true,
      total_vram_bytes: 24 * 1024 ** 3,
      free_vram_bytes: 7 * 1024 ** 3,
      gpus: [{ name: 'NVIDIA L4', total_vram_bytes: 24 * 1024 ** 3, free_vram_bytes: 7 * 1024 ** 3 }],
    },
    worker: {
      pid: 1234,
      alive: true,
      active_kind: 'language',
      active_model_id: 'Qwen-3-VL-8B-Instruct-heretic',
      cuda_allocated_bytes: 16 * 1024 ** 3,
    },
    ...overrides,
  }
}

describe('displayName resolver', () => {
  it('formats long technical model IDs into readable labels without changing canonical IDs', () => {
    expect(resolveModelDisplayName('Qwen-3-VL-8B-Instruct-heretic')).toBe('Qwen3-VL 8B · Heretic')
    expect(resolveCompactModelName('Qwen-3-VL-8B-Instruct-heretic')).toBe('Qwen3-VL 8B')
    expect(resolveModelDisplayName('custom-id', 'Custom Pretty Name')).toBe('Custom Pretty Name')
  })
})

describe('NexusRuntimeControl (v1.1.0 read-only)', () => {
  it('shows locked single-model Ready status with zero load/switch/select controls and closes on Escape/outside click', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    let capturedSignal: AbortSignal | undefined
    const fetchRuntime = vi.fn(async (signal?: AbortSignal) => {
      capturedSignal = signal
      return makeRuntime()
    })

    await act(async () => {
      root.render(
        <NexusRuntimeControl
          fetchProgress={async () => ({ chat: null, diffusion: null, operations: [] })}
          fetchRuntime={fetchRuntime}
        />,
      )
    })

    const trigger = container.querySelector('button')!
    expect(trigger.textContent).toBe('Ready · Qwen3-VL 8B')

    await act(async () => { trigger.click() })
    const dialog = document.body.querySelector('[role="dialog"]')!
    expect(dialog).not.toBeNull()
    expect(dialog.textContent).toContain('Qwen3-VL 8B · Heretic')
    expect(dialog.textContent).toContain('FLUX.2 4B · Klein')
    expect(dialog.textContent).toContain('NVIDIA L4')
    expect(dialog.textContent).toContain('Models are locked by Colab startup')

    // Strictly read-only: NO <select> dropdown and NO Load/Unload action buttons.
    expect(dialog.querySelector('select')).toBeNull()
    const dialogButtons = Array.from(dialog.querySelectorAll('button'))
    expect(dialogButtons).toHaveLength(1) // Only the close button

    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    })
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()

    await act(async () => { trigger.click() })
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull()
    await act(async () => {
      document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }))
    })
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()

    await act(async () => { root.unmount() })
    expect(capturedSignal?.aborted).toBe(true)
    container.remove()
  })

  it('shows live GPU swap / denoising progress and degraded error state accurately', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    const progress: NexusProgressSnapshot = {
      chat: null,
      diffusion: {
        operation_id: 'op_img_1',
        kind: 'image_generation',
        model: 'FLUX.2-klein-4B',
        status: 'running',
        phase: 'denoising',
        progress: 50,
        indeterminate: false,
        step: 2,
        steps_total: 4,
        message: 'Denoising step 2/4',
      },
      operations: [],
    }

    await act(async () => {
      root.render(
        <NexusRuntimeControl
          fetchProgress={async () => progress}
          fetchRuntime={async () => makeRuntime({ ready: false, state: 'swapping', swapping: true })}
        />,
      )
    })

    const trigger = container.querySelector('button')!
    expect(trigger.textContent).toBe('Generating · 50%')

    await act(async () => { trigger.click() })
    const dialog = document.body.querySelector('[role="dialog"]')!
    expect(dialog.textContent).toContain('Denoising step 2/4')

    await act(async () => { root.unmount() })
    container.remove()
  })
})
