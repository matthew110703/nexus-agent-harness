#!/usr/bin/env node
/**
 * Nexus Harness Colab Supervisor (v1.1.0 — Single-Model Lock)
 *
 * Verifies live Gateway readiness and the single Colab-loaded language model
 * via `/v1/runtime`, locks Harness `llm-pi-ai` and `agent-default-model` to
 * that single model, launches `cloudflared` (quick tunnel -> 127.0.0.1:3080)
 * and `dsh web`, redacts `NEXUS_API_KEY`, and cleans up only managed children.
 */

import { spawn } from 'node:child_process'
import net from 'node:net'
import process from 'node:process'

const HARNESS_PORT = Number(process.env.NEXUS_HARNESS_PORT || 3080)
const GATEWAY_ROOT_URL = process.env.NEXUS_GATEWAY_ROOT_URL || 'http://127.0.0.1:7861'
const OVERLAY_PATHS = [
  process.env.NEXUS_OVERLAY || 'apps/cli/config/examples/colab.overlay.yml',
  process.env.NEXUS_EXTRA_OVERLAY || '',
].filter(Boolean)
const API_KEY = process.env.NEXUS_API_KEY || ''

const TRYCLOUDFLARE_RE = /https:\/\/([a-z0-9-]+\.trycloudflare\.com)/i
const LAUNCH_URL_RE = /(https?:\/\/[^\s]+?\?token=[A-Za-z0-9_-]+)/

/** Check compatibility with the repository's Node engine: ^24.22.3 || >=26.13.1 */
export function isNodeVersionCompatible(versionString) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)/.exec(versionString || '')
  if (!match) return false
  const version = match.slice(1, 4).map(Number)
  const [major] = version
  const compare = (a, b) => {
    for (let i = 0; i < 3; i++) {
      if (a[i] !== b[i]) return a[i] > b[i] ? 1 : -1
    }
    return 0
  }
  if (major === 24) return compare(version, [24, 22, 3]) >= 0
  if (major >= 26) return compare(version, [26, 13, 1]) >= 0
  return false
}

/** Enforce the repository's Node.js engine requirement. */
export function assertMinNodeVersion(versionString = process.versions.node) {
  if (!isNodeVersionCompatible(versionString)) {
    throw new Error(`Node.js ^24.22.3 || >=26.13.1 is required (found ${versionString})`)
  }
}

/** Redact the Nexus bearer token from any log line. */
export function redactSecrets(line, secret = API_KEY) {
  if (!secret) return line
  return line.split(secret).join('[REDACTED_NEXUS_API_KEY]')
}

/** Build source-fork launcher args with all requested Cordis overlays. */
export function buildDshArgs({ overlays, publicUrl, trustedHost, port }) {
  return [
    'dsh',
    'web',
    ...overlays.flatMap((overlay) => ['--patch', overlay]),
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--public-url',
    publicUrl,
    '--trusted-host',
    trustedHost,
    '--no-open',
  ]
}

async function isPortOpen(port, host = '127.0.0.1', timeoutMs = 2500) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ port, host })
    let settled = false
    let timer
    const finish = (isOpen) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      socket.destroy()
      resolve(isOpen)
    }
    timer = setTimeout(() => finish(false), timeoutMs)
    socket.once('connect', () => finish(true))
    socket.once('error', () => finish(false))
  })
}

async function waitForPort(port, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await isPortOpen(port)) return true
    await new Promise((r) => setTimeout(r, 300))
  }
  return false
}

/** Convert Gateway's single `language_model` record into a 1-element `llm-pi-ai` model profile. */
export function buildLockedModelProfile(item) {
  if (!item || typeof item.id !== 'string' || item.id.trim().length === 0) {
    throw new Error('Gateway /v1/runtime did not return a valid locked language_model')
  }
  const id = item.id.trim()
  const entry = {
    id,
    name: typeof item.display_name === 'string' && item.display_name.trim().length > 0
      ? item.display_name.trim()
      : id,
  }
  const rawModalities = Array.isArray(item.modalities) ? item.modalities : ['text']
  const input = rawModalities.filter((m) => m === 'text' || m === 'image')
  if (input.length > 0) entry.input = input
  if (typeof item.context_window === 'number' && Number.isInteger(item.context_window) && item.context_window >= 1) {
    entry.contextWindow = item.context_window
  }
  if (typeof item.max_output_tokens === 'number' && Number.isInteger(item.max_output_tokens) && item.max_output_tokens >= 1) {
    entry.maxTokens = item.max_output_tokens
  }
  const rawEfforts = item.reasoning_efforts
  if (rawEfforts === false) {
    entry.reasoningEfforts = false
  } else if (rawEfforts && typeof rawEfforts === 'object' && !Array.isArray(rawEfforts)) {
    const cleaned = {}
    for (const [level, wire] of Object.entries(rawEfforts)) {
      if (wire === null || typeof wire === 'string') cleaned[level] = wire
    }
    if (Object.keys(cleaned).length > 0) entry.reasoningEfforts = cleaned
  }
  return entry
}

/** Verify Gateway live readiness and return the single locked language model profile. */
async function verifyGatewaySessionReady() {
  if (!API_KEY) {
    throw new Error('NEXUS_API_KEY is not set in the environment')
  }
  const runtimeUrl = new URL('/v1/runtime', GATEWAY_ROOT_URL)
  const res = await fetch(runtimeUrl, {
    headers: { Authorization: `Bearer ${API_KEY}` },
    signal: AbortSignal.timeout(8000),
  })
  if (!res.ok) {
    throw new Error(`Nexus gateway /v1/runtime check failed (HTTP ${res.status})`)
  }
  const payload = await res.json()
  const worker = payload?.worker ?? {}
  const lang = payload?.language_model
  if (!payload?.ready || !worker.alive || worker.active_model_id !== lang?.id) {
    throw new Error(
      `Nexus gateway is not ready (ready=${payload?.ready}, state=${payload?.state}, `
      + `worker.alive=${worker.alive}, active_model=${worker.active_model_id ?? 'none'}). `
      + 'Run the Colab model bootstrap cell before starting Harness.',
    )
  }
  return {
    lockedModel: buildLockedModelProfile(lang),
    imageModelId: payload?.image_model?.id ?? null,
  }
}

async function main() {
  assertMinNodeVersion()
  console.log(`[supervisor] Node ${process.versions.node} verified.`)
  console.log('[supervisor] Verifying live Nexus gateway readiness and single loaded model...')
  const { lockedModel, imageModelId } = await verifyGatewaySessionReady()
  console.log(
    `[supervisor] Locked language model ready: ${lockedModel.id}`
    + (imageModelId ? ` | Image model: ${imageModelId}` : ' | Image model: disabled'),
  )

  if (await isPortOpen(HARNESS_PORT)) {
    throw new Error(
      `Port 127.0.0.1:${HARNESS_PORT} is already in use by another process. Refusing to kill unknown processes.`,
    )
  }

  const children = new Set()
  const shutdown = () => {
    for (const child of children) {
      if (!child.killed) {
        try {
          child.kill('SIGTERM')
        } catch {
          // Process already exited.
        }
      }
    }
  }
  process.on('SIGINT', () => {
    console.log('\n[supervisor] SIGINT received, stopping managed children...')
    shutdown()
    process.exit(0)
  })
  process.on('SIGTERM', () => {
    shutdown()
    process.exit(0)
  })

  console.log(`[supervisor] Starting cloudflared tunnel -> http://127.0.0.1:${HARNESS_PORT}...`)
  const cf = spawn('cloudflared', ['tunnel', '--url', `http://127.0.0.1:${HARNESS_PORT}`], {
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  children.add(cf)

  const tunnelHost = await new Promise((resolve, reject) => {
    let resolved = false
    const timer = setTimeout(() => {
      if (!resolved) reject(new Error('Timed out waiting for cloudflared tunnel hostname'))
    }, 30_000)

    const onChunk = (chunk) => {
      const text = redactSecrets(chunk.toString('utf8'))
      for (const line of text.split(/\r?\n/)) {
        if (line.trim()) console.log(`[cloudflared] ${line}`)
        const match = line.match(TRYCLOUDFLARE_RE)
        if (match && !resolved) {
          resolved = true
          clearTimeout(timer)
          resolve(match[1])
        }
      }
    }
    cf.stdout.on('data', onChunk)
    cf.stderr.on('data', onChunk)
    cf.once('exit', (code) => {
      if (!resolved) {
        clearTimeout(timer)
        reject(new Error(`cloudflared exited before producing a hostname (code ${code})`))
      }
    })
  })

  const publicUrl = `https://${tunnelHost}`
  const dshArgs = buildDshArgs({
    overlays: OVERLAY_PATHS,
    publicUrl,
    trustedHost: tunnelHost,
    port: HARNESS_PORT,
  })
  console.log(`[supervisor] Starting locked Nexus Harness (${publicUrl})...`)
  const harnessEnv = {
    ...process.env,
    NEXUS_DISCOVERED_MODELS: JSON.stringify([lockedModel]),
    NEXUS_SELECTED_LANGUAGE_MODEL: lockedModel.id,
  }
  const harness = spawn('pnpm', dshArgs, {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: harnessEnv,
  })
  children.add(harness)

  let printedLaunchUrl = false
  const onHarnessChunk = (chunk) => {
    const text = redactSecrets(chunk.toString('utf8'))
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue
      console.log(`[harness] ${line}`)
      const match = line.match(LAUNCH_URL_RE)
      if (match && !printedLaunchUrl) {
        printedLaunchUrl = true
        console.log(`\n========================================`)
        console.log(`Nexus Harness Ready: ${match[1]}`)
        console.log(`========================================\n`)
      }
    }
  }
  harness.stdout.on('data', onHarnessChunk)
  harness.stderr.on('data', onHarnessChunk)

  const listening = await waitForPort(HARNESS_PORT, 45_000)
  if (!listening) {
    shutdown()
    throw new Error(`Harness did not start listening on 127.0.0.1:${HARNESS_PORT} within 45s`)
  }

  harness.once('exit', (code) => {
    console.log(`[supervisor] Harness exited with code ${code}; shutting down tunnel.`)
    shutdown()
    process.exit(code ?? 1)
  })
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(`[supervisor] Fatal: ${redactSecrets(String(err))}`)
    process.exit(1)
  })
}
