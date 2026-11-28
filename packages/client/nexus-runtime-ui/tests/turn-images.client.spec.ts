// @vitest-environment jsdom
import { act, createElement } from 'react'
import { createRoot } from 'react-dom/client'
import { describe, expect, it, vi } from 'vitest'
import { NexusTurnImagesTail } from '../src/client/NexusTurnImagesTail.tsx'
import { nexusTurnImagesDefinition, type NexusTurnImage } from '../src/client/turn-images.ts'

describe('nexusTurnImagesDefinition', () => {
  it('matches nexus_generate_image call and result metadata and builds view node', () => {
    const callEvent = {
      type: 'tool/call' as const,
      seq: 1,
      time: 1000,
      data: {
        turn: 1,
        step: 1,
        callId: 'call_1',
        name: 'nexus_generate_image',
        arguments: JSON.stringify({ prompt: 'golden beach' }),
      },
    }
    const callMatch = nexusTurnImagesDefinition.match(callEvent as never)
    expect(callMatch).toEqual({ id: 'nexus-img-call_1', role: 'start' })

    const location = { kind: 'turn', turn: 1 } as const
    const initialState = nexusTurnImagesDefinition.start(
      { id: 'nexus-img-call_1' } as never,
      { ...callMatch, event: callEvent, location } as never,
      {} as never,
    )

    const resultEvent = {
      type: 'tool/result' as const,
      seq: 2,
      time: 1001,
      data: {
        turn: 1,
        step: 1,
        message: { role: 'tool', toolCallId: 'call_1', content: 'ok' },
        meta: {
          attachmentId: 'att_1',
          filename: 'beach.png',
          width: 1024,
          height: 1024,
          steps: 6,
          model: 'FLUX.2-klein-4B',
          quality: 'detailed',
          drivePath: '/content/drive/MyDrive/Nexus/outputs/generated_images/beach.png',
          workspacePath: '/content/drive/MyDrive/Nexus/workspaces/projects/ws/generated_images/beach.png',
          masterPath: '/content/drive/MyDrive/Nexus/outputs/generated_images/beach.png',
        },
      },
    }
    const resultMatch = nexusTurnImagesDefinition.match(resultEvent as never)
    expect(resultMatch).toEqual({ id: 'nexus-img-call_1', role: 'update' })

    const updatedState = nexusTurnImagesDefinition.update(
      { state: initialState } as never,
      { ...resultMatch, event: resultEvent, location } as never,
    )

    const viewNode = nexusTurnImagesDefinition.buildViewNode?.({
      key: 'nexus-img-call_1',
      kind: 'nexus-generated-images',
      id: 'nexus-img-call_1',
      state: updatedState,
      start: { event: callEvent, location },
      current: { event: resultEvent, location },
      matches: [{ event: callEvent, location }],
    } as never)
    expect(viewNode?.kind).toBe('nexus-generated-images')
    const data = viewNode?.data as NexusTurnImage
    expect(data.filename).toBe('beach.png')
    expect(data.drivePath).toContain('beach.png')
    expect(data.steps).toBe(6)
  })

  it('renders NexusTurnImagesTail from TurnLocation object (turn.turn), opens sidebar preview, and closes lightbox on Escape', async () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    const sampleImage: NexusTurnImage = {
      callId: 'call_hero_1',
      filename: 'cyber_lotus.png',
      attachmentId: 'att_hero_1',
      width: 1024,
      height: 1024,
      steps: 6,
      model: 'FLUX.2-klein-4B',
      quality: 'detailed',
      drivePath: '/content/drive/MyDrive/Nexus/outputs/generated_images/cyber_lotus.png',
      workspacePath: '/content/drive/MyDrive/Nexus/workspaces/projects/default-workspace/generated_images/cyber_lotus.png',
      masterPath: '/content/drive/MyDrive/Nexus/workspaces/projects/default-workspace/generated_images/cyber_lotus.png',
      prompt: 'glowing lotus',
    }

    const openFile = vi.fn()
    const useImages = vi.fn((turnKey: string) => (turnKey === '3' ? [sampleImage] : []))

    await act(async () => {
      root.render(
        createElement(NexusTurnImagesTail, {
          turn: { kind: 'turn', turn: 3 },
          openFile,
          useImages,
        } as never),
      )
    })

    // Verifies TurnLocation `{ kind: 'turn', turn: 3 }` resolves turnKey `'3'` (not `NaN`!)
    expect(useImages).toHaveBeenCalledWith('3')
    const card = container.querySelector('[data-nexus-turn-image="call_hero_1"]')
    expect(card).not.toBeNull()
    expect(card?.textContent).toContain('generated_images/cyber_lotus.png')
    // No raw `/content/drive/MyDrive/...` path clutter in visible text
    expect(card?.textContent).not.toContain('/content/drive/MyDrive')

    const previewBtn = Array.from(container.querySelectorAll('button')).find(
      b => b.textContent === 'Preview in sidebar',
    )!
    await act(async () => {
      previewBtn.click()
    })
    expect(openFile).toHaveBeenCalledWith(sampleImage.workspacePath)

    const zoomStageBtn = container.querySelector('button[aria-label="Zoom"]') as HTMLButtonElement
    await act(async () => {
      zoomStageBtn.click()
    })
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull()

    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    })
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()

    await act(async () => {
      root.unmount()
    })
    container.remove()
  })
})
