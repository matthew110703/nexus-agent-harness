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

function readStoragePair(): NexusNotifyPrefs {
  const enabledRaw = window.localStorage.getItem(STORAGE_ENABLED_KEY)
  const soundRaw = window.localStorage.getItem(STORAGE_SOUND_KEY)
  return {
    enabled: enabledRaw === null ? true : enabledRaw === 'true',
    sound: soundRaw === null ? true : soundRaw === 'true',
  }
}

/**
 * Read notification preferences safely from localStorage.
 * @returns Current persisted notification preferences.
 */
export function readNotifyPrefs(): NexusNotifyPrefs {
  try {
    return readStoragePair()
  } catch (_readErr) {
    // Default to enabled when localStorage is unavailable.
    return { enabled: true, sound: true }
  }
}

function writeStorageKeys(prefs: NexusNotifyPrefs): void {
  window.localStorage.setItem(STORAGE_ENABLED_KEY, String(prefs.enabled))
  window.localStorage.setItem(STORAGE_SOUND_KEY, String(prefs.sound))
}

/**
 * Persist notification preferences safely to localStorage.
 * @param prefs - Notification preferences to save.
 */
export function writeNotifyPrefs(prefs: NexusNotifyPrefs): void {
  try {
    writeStorageKeys(prefs)
  } catch (_storageErr) {
    // Ignore storage quota or private-browsing errors.
  }
}

/**
 * Check whether the browser tab is currently unfocused or hidden.
 * @returns True when the document is hidden or unfocused.
 */
export function isTabUnfocused(): boolean {
  if (typeof document === 'undefined') return false
  return document.hidden || (typeof document.hasFocus === 'function' && !document.hasFocus())
}

/**
 * Current Web Notification permission state, or `'unsupported'` when unavailable.
 * @returns Current permission string.
 */
export function getNotificationPermission(): NotificationPermission | 'unsupported' {
  if (typeof window === 'undefined' || typeof Notification === 'undefined') return 'unsupported'
  return Notification.permission
}

/**
 * Request browser Notification permission inside a user gesture handler.
 * @returns Resolved Notification permission state.
 */
export async function requestNotificationPermission(): Promise<NotificationPermission | 'unsupported'> {
  if (typeof window === 'undefined' || typeof Notification === 'undefined') return 'unsupported'
  if (Notification.permission === 'granted' || Notification.permission === 'denied') {
    return Notification.permission
  }
  try {
    return await Notification.requestPermission()
  } catch (_permErr) {
    // Return current state if browser blocks permission prompt.
    return Notification.permission
  }
}

function scheduleChimeTones(): void {
  const AudioCtx = window.AudioContext
  if (typeof AudioCtx !== 'function') return
  const ctx = new AudioCtx()
  const now = ctx.currentTime
  const notes = [587.33, 880.0]
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
}

/** Play a soft, non-intrusive two-tone Web Audio chime. */
export function playCompletionChime(): void {
  if (typeof window === 'undefined') return
  try {
    scheduleChimeTones()
  } catch (_audioErr) {
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

function dispatchBrowserNotification(title: string, body: string, silent: boolean): void {
  const n = new Notification(title, {
    body,
    tag: 'nexus-completion',
    silent,
  })
  n.onclick = () => {
    window.focus()
    n.close()
  }
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

  if (typeof Notification !== 'undefined' && Notification.permission === 'granted') {
    try {
      dispatchBrowserNotification(options.title, options.body, options.prefs.sound)
    } catch (_notifErr) {
      // Ignore notification creation errors in restricted contexts.
    }
  }

  if (options.prefs.sound) {
    playCompletionChime()
  }
  return true
}
