#!/usr/bin/env node
/**
 * Nexus Harness Colab Supervisor
 *
 * Launches `cloudflared` (quick tunnel -> 127.0.0.1:3080) and `dsh web` with
 * exact `--public-url` and `--trusted-host` flags, streams all child logs
 * continuously, redacts `NEXUS_API_KEY`, verifies TCP/HTTP readiness, and
 * cleans up only child processes it spawned.
 */

import { spawn } from 'node:child_process'
import net from 'node:net'
import process from 'node:process'

const HARNESS_PORT = Number(process.env.NEXUS_HARNESS_PORT || 3080)
const GATEWAY_URL = process.env.NEXUS_GATEWAY_URL || 'http://127.0.0.1:7861'
const OVERLAY_PATHS = [
  process.env.NEXUS_OVERLAY || 'apps/cli/config/examples/colab.overlay.yml',
  process.env.NEXUS_EXTRA_OVERLAY || '',
].filter(Boolean)
const API_KEY = process.env.NEXUS_API_KEY || ''

const TRYCLOUDFLARE_RE = /https:\/\/([a-z0-9-]+\.trycloudflare\.com)/i
const LAUNCH_URL_RE = /(https?:\/\/[^\s]+?\?token=[A-Za-z0-9_-]+)/

/**
 * Check compatibility with the repository's Node engine:
 *   ^24.22.3 || >=26.13.1
 */
export function isNodeVersionCompatible(versionString) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)/.exec(versionString || '')

  if (!match) {
    return false
  }

  const version = match.slice(1, 4).map(Number)
  const [major] = version

  const compare = (a, b) => {
    for (let i = 0; i < 3; i++) {
      if (a[i] !== b[i]) {
        return a[i] > b[i] ? 1 : -1
      }
    }
    return 0
  }

  if (major === 24) {
    return compare(version, [24, 22, 3]) >= 0
  }

  if (major >= 26) {
    return compare(version, [26, 13, 1]) >= 0
  }

  return false
}

/** Enforce the repository's Node.js engine requirement. */
export function assertMinNodeVersion(versionString = process.versions.node) {
  if (!isNodeVersionCompatible(versionString)) {
    throw new Error(
      `Node.js ^24.22.3 || >=26.13.1 is required (found ${versionString})`,
    )
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

/** Convert an OpenAI `/v1/models` payload into `llm-pi-ai` model profile entries. */
export function parseDiscoveredModels(payload) {
  const list = Array.isArray(payload?.data) ? payload.data : []
  const seen = new Set()
  const models = []
  for (const item of list) {
    if (typeof item?.id !== 'string' || item.id.trim().length === 0) continue
    const id = item.id.trim()
    if (seen.has(id)) continue
    seen.add(id)
    const entry = {
      id,
      name: typeof item.display_name === 'string' && item.display_name.trim().length > 0
        ? item.display_name.trim()
        : typeof item.name === 'string' && item.name.trim().length > 0
          ? item.name.trim()
          : id,
    }
    const rawModalities = Array.isArray(item.input)
      ? item.input
      : Array.isArray(item.modalities)
        ? item.modalities
        : undefined
    if (rawModalities !== undefined) {
      const input = rawModalities.filter((m) => m === 'text' || m === 'image')
      if (input.length > 0) entry.input = input
    }
    const contextWindow = item.context_window ?? item.context_length ?? item.contextWindow
    if (typeof contextWindow === 'number' && Number.isInteger(contextWindow) && contextWindow >= 1) {
      entry.contextWindow = contextWindow
    }
    const maxTokens = item.max_output_tokens ?? item.max_tokens ?? item.maxTokens
    if (typeof maxTokens === 'number' && Number.isInteger(maxTokens) && maxTokens >= 1) {
      entry.maxTokens = maxTokens
    }
    // Preserve model-specific reasoning Efforts without inventing fake levels.
    const rawEfforts = item.reasoning_efforts ?? item.reasoningEfforts
    if (rawEfforts === false) {
      entry.reasoningEfforts = false
    } else if (rawEfforts && typeof rawEfforts === 'object' && !Array.isArray(rawEfforts)) {
      const cleaned = {}
      for (const [level, wire] of Object.entries(rawEfforts)) {
        if (wire === null || typeof wire === 'string') {
          cleaned[level] = wire
        }
      }
      if (Object.keys(cleaned).length > 0) {
        entry.reasoningEfforts = cleaned
      }
    }
    models.push(entry)
  }
  return models
}

async function discoverGatewayModels() {
  try {
    const modelsUrl = new URL('/v1/models', GATEWAY_URL)
    const res = await fetch(modelsUrl, {
      headers: { Authorization: `Bearer ${API_KEY}` },
      signal: AbortSignal.timeout(5000),
    })
    if (!res.ok) {
      console.warn(`[supervisor] Warning: gateway /v1/models returned HTTP ${res.status}; starting with empty model list.`)
      return []
    }
    const payload = await res.json()
    return parseDiscoveredModels(payload)
  } catch (err) {
    console.warn(`[supervisor] Warning: could not fetch /v1/models (${redactSecrets(String(err))}); starting with empty model list.`)
    return []
  }
}

async function verifyGatewayHealth() {
  if (!API_KEY) {
    throw new Error('NEXUS_API_KEY is not set in the environment')
  }
  const healthUrl = new URL('/health', GATEWAY_URL)
  const res = await fetch(healthUrl, { signal: AbortSignal.timeout(5000) })
  if (!res.ok) {
    throw new Error(`Nexus gateway health check failed (HTTP ${res.status})`)
  }
}

async function main() {
  assertMinNodeVersion()
  console.log(`[supervisor] Node ${process.versions.node} verified.`)
  console.log('[supervisor] Verifying independent Nexus gateway health (127.0.0.1:7861)...')
  await verifyGatewayHealth()

  const discoveredModels = process.env.NEXUS_DISCOVERED_MODELS
    ? undefined
    : await discoverGatewayModels()
  if (discoveredModels !== undefined) {
    const names = discoveredModels.map((m) => m.id).join(', ')
    const suffix = discoveredModels.length > 0 ? `: ${names}` : ''
    console.log(`[supervisor] Discovered ${discoveredModels.length} model(s) from Nexus gateway${suffix}.`)
  }

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
  console.log(`[supervisor] Starting Nexus Harness (${publicUrl})...`)
  const harnessEnv = {
    ...process.env,
    ...(discoveredModels !== undefined ? { NEXUS_DISCOVERED_MODELS: JSON.stringify(discoveredModels) } : {}),
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
