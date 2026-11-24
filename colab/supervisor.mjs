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
const OVERLAY_PATH = process.env.NEXUS_OVERLAY || 'apps/cli/config/examples/colab.overlay.yml'
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

/** Build the exact launcher-first `dsh` argv for Colab Web mode. */
export function buildDshArgs({ overlay, publicUrl, trustedHost, port }) {
  return [
    'dsh',
    'web',
    '--patch',
    overlay,
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

async function isPortOpen(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const socket = net.createConnection({ port, host })
    socket.once('connect', () => {
      socket.destroy()
      resolve(true)
    })
    socket.once('error', () => resolve(false))
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
    overlay: OVERLAY_PATH,
    publicUrl,
    trustedHost: tunnelHost,
    port: HARNESS_PORT,
  })
  console.log(`[supervisor] Starting Nexus Harness (${publicUrl})...`)
  const harness = spawn('pnpm', dshArgs, {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
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
