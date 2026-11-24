import { describe, expect, it } from 'vitest'
import type { ConnectionTrustRequest } from '@deepseek-ai/dsh-client-connection'
import { isEffectiveHttps } from '../src/index.ts'

function makeRequest(remoteAddress: string | undefined, proto?: string): ConnectionTrustRequest {
  return {
    headers: proto === undefined ? {} : { 'x-forwarded-proto': proto },
    socket: remoteAddress === undefined ? undefined : { remoteAddress, destroyed: false },
  } as unknown as ConnectionTrustRequest
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
