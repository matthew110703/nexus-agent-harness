/**
 * Locale dictionary for `@deepseek-ai/dsh-client-nexus-runtime-ui`.
 * All user-visible UI copy lives here to satisfy `verify-client-ui-i18n`.
 */

/** English product copy for Nexus runtime status, notifications, and inline image cards. */
export const NEXUS_UI_COPY = {
  // Phase labels
  'phase.resolving': 'Resolving model',
  'phase.tokenizer': 'Loading tokenizer & processor',
  'phase.weightLoading': 'Loading model weights',
  'phase.placement': 'Verifying GPU placement',
  'phase.ready': 'Ready',
  'phase.generating': 'Generating response',
  'phase.swappingToImage': 'Swapping GPU to image model',
  'phase.denoising': 'Generating image',
  'phase.saving': 'Saving image to Drive',
  'phase.restoringLanguage': 'Restoring language model',
  'phase.denoisingStep': 'Denoising step {step}/{total}',

  // Runtime states & badges
  'state.checking': 'Checking…',
  'state.cpuOnly': 'CPU only',
  'state.runtimeOff': 'Runtime off',
  'state.swappingGpu': 'Swapping GPU',
  'state.ready': 'Ready',
  'state.degraded': 'Degraded',
  'state.notLoaded': 'Not loaded',
  'state.unavailable': 'Unavailable',
  'state.disabled': 'Disabled',
  'badge.restoringModel': 'Restoring model',
  'badge.restoringNamed': 'Restoring · {model}',
  'badge.generating': 'Generating',
  'badge.generatingPct': 'Generating · {pct}%',
  'badge.loading': 'Loading',
  'badge.loadingNamed': 'Loading · {model}',
  'badge.runtimeError': 'Runtime error',
  'badge.readyNamed': 'Ready · {model}',
  'imageCard.stepsLabel': '{steps} steps',

  // Panel & labels
  'panel.ariaLabel': 'Nexus runtime',
  'panel.title': 'Nexus runtime',
  'panel.closeAria': 'Close Nexus runtime panel',
  'panel.languageModel': 'Language model',
  'panel.imageGenerator': 'Image generator',
  'panel.gpuMemory': 'GPU memory',
  'panel.lockedNote': 'Models are locked by Colab startup. Edit Section 3 in Colab to change models.',
  'panel.progress': 'Progress',
  'panel.lockedTooltip': 'Locked model: {model}',

  // Notifications
  'notify.section': 'Tab notifications',
  'notify.enable': 'Notify when ready',
  'notify.enabled': 'Notifications on',
  'notify.denied': 'Blocked in browser',
  'notify.sound': 'Chime',
  'notify.chatReadyTitle': 'Response ready — {model}',
  'notify.chatReadyBody': 'Your Nexus model finished responding.',
  'notify.chatFailedTitle': 'Response failed — {model}',
  'notify.imageReadyTitle': 'Image ready — {model}',
  'notify.imageReadyBody': 'Your generated image is ready to view.',
  'notify.tabBadge': '(✓) Ready — ',

  // Inline generated image tool card
  'imageCard.generating': 'Generating image…',
  'imageCard.title': 'Generated image',
  'imageCard.failed': 'Image generation failed',
  'imageCard.loadingPreview': 'Loading preview…',
  'imageCard.previewError': 'Could not load image preview',
  'imageCard.savedToDrive': 'Saved to Drive',
  'imageCard.inspect': 'Inspect',
  'imageCard.closeLightboxAria': 'Close image preview',
  'imageCard.altFallback': 'Generated image',

  // Errors
  'error.requestFailed': 'Request failed (HTTP {status})',
} as const

/** Union of all Nexus UI copy keys. */
export type NexusCopyKey = keyof typeof NEXUS_UI_COPY

/**
 * Format a localized Nexus UI string with simple `{name}` placeholder substitution.
 * @param key - Dictionary key from `NEXUS_UI_COPY`.
 * @param params - Optional placeholder values.
 * @returns The formatted UI string.
 */
export function tNexus(key: NexusCopyKey, params?: Record<string, string | number>): string {
  let out: string = NEXUS_UI_COPY[key]
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      out = out.replaceAll(`{${k}}`, String(v))
    }
  }
  return out
}
