// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import { NexusRuntimeControl } from '../src/client/NexusRuntimeControl.tsx'

describe('NexusRuntimeControl', () => {
  it('shows selected chat model separately from loaded worker model and persists image model choice', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    const saveImageModel = vi.fn(async (modelId: string | null) => ({
      selected_image_model: modelId,
      models: [{ id: 'sdxl-base-1.0', adapter_family: 'diffusers' }],
    }))
    const loadModel = vi.fn(async () => {})
    const unloadModel = vi.fn(async () => {})

    await act(async () => {
      root.render(
        <NexusRuntimeControl
          getSelectedChatModel={() => 'Qwen3-VL-8B'}
          subscribeSelectedChatModel={() => () => {}}
          fetchProgress={async () => ({ status: 'idle' })}
          fetchRuntime={async () => ({
            loaded_model_id: 'Qwen3-VL-4B',
            vram_allocated_mb: 4096,
            vram_total_mb: 16384,
          })}
          fetchImageModels={async () => ({
            selected_image_model: null,
            models: [{ id: 'sdxl-base-1.0', adapter_family: 'diffusers' }],
          })}
          saveImageModel={saveImageModel}
          loadModel={loadModel}
          unloadModel={unloadModel}
        />,
      )
    })

    const trigger = container.querySelector('button')
    expect(trigger).not.toBeNull()
    expect(trigger?.textContent).toContain('GPU: Qwen3-VL-4B')

    await act(async () => {
      trigger?.click()
    })

    const dialog = document.body.querySelector('[role="dialog"]')
    expect(dialog).not.toBeNull()
    expect(dialog?.textContent).toContain('Qwen3-VL-8B')
    expect(dialog?.textContent).toContain('Qwen3-VL-4B')
    expect(dialog?.textContent).toContain('4096 / 16384 MB')

    const loadButton = Array.from(dialog?.querySelectorAll('button') ?? []).find(
      b => b.textContent === 'Load selected',
    )
    expect(loadButton).not.toBeUndefined()
    await act(async () => {
      loadButton?.click()
    })
    expect(loadModel).toHaveBeenCalledWith('Qwen3-VL-8B')

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})
