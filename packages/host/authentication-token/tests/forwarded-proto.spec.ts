import { describe, expect, it } from 'vitest'
import type { ConnectionTrustRequest } from '@deepseek-ai/dsh-client-connection'
import { BrowserAuth, isEffectiveHttps } from '../src/browser-auth.ts'
import type { ConnectionIndexResponse } from '../src/types.ts'
import { RecordCredentials } from './browser-credentials.ts'

function makeRequest(remoteAddress: string | undefined, proto?: string): ConnectionTrustRequest {
  return {
    headers: proto === undefined ? {} : { 'x-forwarded-proto': proto },
    socket: remoteAddress === undefined ? undefined : { remoteAddress, destroyed: false },
  } as unknown as ConnectionTrustRequest
}

function captureResponse() {
  const state: { status?: number; headers?: Readonly<Record<string, string | string[]>> | undefined } = {}
  const res: ConnectionIndexResponse = {
    writeHead(status, headers) {
      state.status = status
      state.headers = headers
    },
    end() {},
  }
  return { res, state }
}

describe('isEffectiveHttps', () => {
  it('returns true when the web server itself is https regardless of trustForwardedProto', () => {
    expect(isEffectiveHttps(makeRequest('10.0.0.5'), 'https:', false)).toBe(true)
  })

  it('ignores X-Forwarded-Proto when trustForwardedProto is false', () => {
    expect(isEffectiveHttps(makeRequest('127.0.0.1', 'https'), 'http:', false)).toBe(false)
  })

  it('honors X-Forwarded-Proto: https from loopback peers when trustForwardedProto is true', () => {
    expect(isEffectiveHttps(makeRequest('127.0.0.1', 'https'), 'http:', true)).toBe(true)
    expect(isEffectiveHttps(makeRequest('::1', 'https, http'), 'http:', true)).toBe(true)
    expect(isEffectiveHttps(makeRequest('::ffff:127.0.0.1', 'HTTPS'), 'http:', true)).toBe(true)
  })

  it('rejects X-Forwarded-Proto from non-loopback peers even when trustForwardedProto is true', () => {
    expect(isEffectiveHttps(makeRequest('192.168.1.25', 'https'), 'http:', true)).toBe(false)
    expect(isEffectiveHttps(makeRequest(undefined, 'https'), 'http:', true)).toBe(false)
  })

  it('returns false when X-Forwarded-Proto is http or missing', () => {
    expect(isEffectiveHttps(makeRequest('127.0.0.1', 'http'), 'http:', true)).toBe(false)
    expect(isEffectiveHttps(makeRequest('127.0.0.1'), 'http:', true)).toBe(false)
  })
})

describe('BrowserAuth launch-token exchange and cookie replay regression', () => {
  it('accepts a freshly issued cookie on subsequent GET / for the same BrowserAuth instance', async () => {
    const auth = await BrowserAuth.create({}, new RecordCredentials(), 30)
    const launchUrl = new URL(auth.authenticatedUrl('http://127.0.0.1:3080/'))
    const reply = captureResponse()

    const servedOnExchange = auth.authorizeIndex(
      { method: 'GET', url: launchUrl.pathname + launchUrl.search, headers: { host: '127.0.0.1:3080' } },
      reply.res,
      false,
    )
    expect(servedOnExchange).toBe(false)
    expect(reply.state.status).toBe(303)
    const setCookie = reply.state.headers?.['set-cookie']
    const cookiePair = (Array.isArray(setCookie) ? setCookie[0] : setCookie)?.split(';', 1)[0]
    expect(cookiePair).toMatch(/^dsh-auth-[A-Za-z0-9_-]{43}=v1\./)

    const followUp = captureResponse()
    const served = auth.authorizeIndex(
      { method: 'GET', url: '/', headers: { host: '127.0.0.1:3080', cookie: cookiePair } },
      followUp.res,
      false,
    )
    expect(served).toBe(true)
    expect(followUp.state.status).toBeUndefined()
  })

  it('preserves Secure cookie issuance and validation across Cloudflare-style HTTPS -> HTTP loopback proxy', async () => {
    const auth = await BrowserAuth.create({}, new RecordCredentials(), 30)
    const tunnelHost = 'demo-tunnel.trycloudflare.com'
    const launchUrl = new URL(auth.authenticatedUrl(`https://${tunnelHost}/`))

    const exchangeReq = {
      method: 'GET',
      url: launchUrl.pathname + launchUrl.search,
      headers: { host: tunnelHost, 'x-forwarded-proto': 'https' },
      socket: { remoteAddress: '127.0.0.1', destroyed: false },
    }
    const exchangeSecure = isEffectiveHttps(exchangeReq as unknown as ConnectionTrustRequest, 'http:', true)
    expect(exchangeSecure).toBe(true)

    const reply = captureResponse()
    auth.authorizeIndex(exchangeReq, reply.res, exchangeSecure)
    expect(reply.state.status).toBe(303)
    const setCookie = reply.state.headers?.['set-cookie']
    const rawSetCookie = Array.isArray(setCookie) ? setCookie[0] : setCookie
    expect(rawSetCookie).toContain('; HttpOnly; SameSite=Strict; Secure')
    const cookiePair = rawSetCookie?.split(';', 1)[0]

    const followUpReq = {
      method: 'GET',
      url: '/',
      headers: { host: tunnelHost, 'x-forwarded-proto': 'https', cookie: cookiePair },
      socket: { remoteAddress: '127.0.0.1', destroyed: false },
    }
    const followUpSecure = isEffectiveHttps(followUpReq as unknown as ConnectionTrustRequest, 'http:', true)
    expect(followUpSecure).toBe(true)

    const followUp = captureResponse()
    const served = auth.authorizeIndex(followUpReq, followUp.res, followUpSecure)
    expect(served).toBe(true)
    expect(followUp.state.status).toBeUndefined()
  })
})
