/**
 * Unfocused browser tab notification manager for Nexus runtime completions.
 * Supports native Web Notification API, tab title badge with auto-restore on focus,
 * and an optional subtle two-tone Web Audio API chime (zero external assets).
 */

import { tNexus } from './locales.ts'

const STORAGE_ENABLED_KEY = 'nexus.notify.enabled'
const STORAGE_SOUND_KEY = 'nexus.notify.sound'

/** Persisted user notification preferences. */
export interface NexusNotifyPrefs {
  readonly enabled: boolean
  readonly sound: boolean
}

/** Read notification preferences safely from localStorage. */
export function readNotifyPrefs(): NexusNotifyPrefs {
  try {
    const enabledRaw = window.localStorage.getItem(STORAGE_ENABLED_KEY)
    const soundRaw = window.localStorage.getItem(STORAGE_SOUND_KEY)
    return {
      enabled: enabledRaw === null ? true : enabledRaw === 'true',
      sound: soundRaw === null ? true : soundRaw === 'true',
    }
  } catch {
    return { enabled: true, sound: true }
  }
}

/** Persist notification preferences safely to localStorage. */
export function writeNotifyPrefs(prefs: NexusNotifyPrefs): void {
  try {
    window.localStorage.setItem(STORAGE_ENABLED_KEY, String(prefs.enabled))
    window.localStorage.setItem(STORAGE_SOUND_KEY, String(prefs.sound))
  } catch {
    // Ignore storage quota or private-browsing errors.
  }
}

/** Check whether the browser tab is currently unfocused or hidden. */
export function isTabUnfocused(): boolean {
  if (typeof document === 'undefined') return false
  return document.hidden || (typeof document.hasFocus === 'function' && !document.hasFocus())
}

/** Current Web Notification permission state, or `'unsupported'` when unavailable. */
export function getNotificationPermission(): NotificationPermission | 'unsupported' {
  if (typeof window === 'undefined' || typeof Notification === 'undefined') return 'unsupported'
  return Notification.permission
}

/** Request browser Notification permission inside a user gesture handler. */
export async function requestNotificationPermission(): Promise<NotificationPermission | 'unsupported'> {
  if (typeof window === 'undefined' || typeof Notification === 'undefined') return 'unsupported'
  if (Notification.permission === 'granted' || Notification.permission === 'denied') {
    return Notification.permission
  }
  try {
    return await Notification.requestPermission()
  } catch {
    return Notification.permission
  }
}

/** Play a soft, non-intrusive two-tone Web Audio chime. */
export function playCompletionChime(): void {
  if (typeof window === 'undefined') return
  try {
    const AudioCtx = window.AudioContext
    if (typeof AudioCtx !== 'function') return
    const ctx = new AudioCtx()
    const now = ctx.currentTime
    const notes = [587.33, 880.0] // D5 -> A5 gentle interval
    for (let i = 0; i < notes.length; i++) {
      const freq = notes[i]
      if (freq === undefined) continue
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'sine'
      osc.frequency.setValueAtTime(freq, now + i * 0.11)
      gain.gain.setValueAtTime(0.0001, now + i * 0.11)
      gain.gain.exponentialRampToValueAtTime(0.06, now + i * 0.11 + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.11 + 0.24)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start(now + i * 0.11)
      osc.stop(now + i * 0.11 + 0.26)
    }
    setTimeout(() => {
      void ctx.close().catch(() => {})
    }, 600)
  } catch {
    // Ignore audio context autoplay restrictions.
  }
}

let savedOriginalTitle: string | null = null
let restoreListenerBound = false

function restoreTabTitle(): void {
  if (typeof document === 'undefined' || savedOriginalTitle === null) return
  document.title = savedOriginalTitle
  savedOriginalTitle = null
}

function ensureRestoreListeners(): void {
  if (restoreListenerBound || typeof window === 'undefined' || typeof document === 'undefined') return
  restoreListenerBound = true
  window.addEventListener('focus', restoreTabTitle)
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) restoreTabTitle()
  })
}

/**
 * Notify the user when a chat response or image generation completes while the tab is unfocused.
 * @param options - Notification title, body, and user preferences.
 * @returns True if an unfocused notification alert was triggered.
 */
export function notifyUnfocusedCompletion(options: {
  readonly title: string
  readonly body: string
  readonly prefs: NexusNotifyPrefs
}): boolean {
  if (!options.prefs.enabled || !isTabUnfocused()) return false

  // 1. Update tab title with auto-restore on focus
  if (typeof document !== 'undefined') {
    ensureRestoreListeners()
    const badge = tNexus('notify.tabBadge')
    if (savedOriginalTitle === null) {
      savedOriginalTitle = document.title
    }
    if (!document.title.startsWith(badge)) {
      document.title = `${badge}${savedOriginalTitle}`
    }
  }

  // 2. Native browser Notification when permission is granted
  if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
    try {
      const n = new Notification(options.title, {
        body: options.body,
        tag: 'nexus-completion',
        silent: options.prefs.sound,
      })
      n.onclick = () => {
        window.focus()
        n.close()
      }
    } catch {
      // Ignore notification creation errors in restricted contexts.
    }
  }

  // 3. Optional subtle Web Audio chime
  if (options.prefs.sound) {
    playCompletionChime()
  }
  return true
}
