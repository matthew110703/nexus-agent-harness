import { describe, expect, it } from 'vitest'
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
})
