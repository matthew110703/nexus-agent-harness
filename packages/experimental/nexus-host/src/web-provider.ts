/**
 * Keyless web search provider, multi-page web researcher tool, and runtime inspector tool
 * backed by the local Nexus Gateway.
 */

import type { Context } from '@deepseek-ai/cordis'
import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import type {} from '@deepseek-ai/dsh-web'

interface GatewaySearchHit {
  title?: string
  url?: string
  snippet?: string
}

interface GatewaySearchResponse {
  query?: string
  results?: GatewaySearchHit[]
}

type JsonPrimitive = string | number | boolean | null
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }

/**
 * Register the keyless `nexus-search` WebSearchProvider into `ctx.web`.
 *
 * @param ctx - Cordis plugin context.
 * @param gatewayUrl - Base URL of the local Nexus Gateway.
 * @param apiKey - Bearer API key for the local Nexus Gateway.
 * @param fetchImpl - Optional fetch implementation for testing.
 */
export function registerNexusWebProvider(
  ctx: Context,
  gatewayUrl: string,
  apiKey: string,
  fetchImpl: typeof fetch = fetch,
): void {
  const baseUrl = gatewayUrl.replace(/\/+$/, '')
  ctx.inject(['web'], (injected) => {
    injected.effect(
      () =>
        injected.web.registerSearchProvider({
          id: 'nexus-search',
          title: 'Nexus Web Search',
          available: () => true,
          ready: () => Promise.resolve(true),
          async search(req, signal) {
            const maxResults = 'maxResults' in req && typeof req.maxResults === 'number' ? req.maxResults : 8
            const resp = await fetchImpl(`${baseUrl}/v1/web/search`, {
              method: 'POST',
              headers: {
                'content-type': 'application/json',
                authorization: `Bearer ${apiKey}`,
              },
              body: JSON.stringify({ query: req.query, limit: maxResults }),
              signal: signal ?? null,
            })
            if (!resp.ok) {
              throw new Error(`Nexus web search failed (${resp.status})`)
            }
            const data = (await resp.json()) as GatewaySearchResponse
            const allSources = (data.results ?? [])
              .filter((r): r is { title: string; url: string; snippet?: string } => Boolean(r.url && r.title))
              .map(r => ({
                title: r.title,
                url: r.url,
                snippet: r.snippet ?? '',
              }))
            return {
              sources: allSources.slice(0, maxResults),
              truncated: allSources.length > maxResults,
            }
          },
        }),
      'nexus-host: web search provider',
    )
  })
}

/**
 * Create the `nexus_web_research` tool for parallel multi-query search and Markdown page extraction.
 *
 * @param gatewayUrl - Base URL of the local Nexus Gateway.
 * @param apiKey - Bearer API key for the local Nexus Gateway.
 * @param fetchImpl - Optional fetch implementation for testing.
 * @returns Registered Cordis tool definition.
 */
export function createNexusWebResearchTool(
  gatewayUrl: string,
  apiKey: string,
  fetchImpl: typeof fetch = fetch,
): ToolDefinition {
  const baseUrl = gatewayUrl.replace(/\/+$/, '')
  return defineTool({
    name: 'nexus_web_research',
    description:
      'Run parallel keyless web searches for 1-3 queries and fetch top pages as clean Markdown with citations.',
    parameters: {
      queries: {
        type: 'array',
        items: { type: 'string' },
        description: '1 to 3 focused search queries.',
        required: true,
      },
      max_pages: {
        type: 'integer',
        description: 'Maximum pages to fetch and extract (1-6, default 4).',
      },
      max_chars_per_page: {
        type: 'integer',
        description: 'Maximum characters per fetched page (default 12000).',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
      },
      render(_args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute(args, exec) {
      const resp = await fetchImpl(`${baseUrl}/v1/web/research`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          queries: args.queries,
          max_pages: typeof args.max_pages === 'number' ? args.max_pages : 4,
          max_chars_per_page: typeof args.max_chars_per_page === 'number' ? args.max_chars_per_page : 12000,
        }),
        signal: exec.signal,
      })
      if (!resp.ok) {
        const msg = await resp.text().catch(() => '')
        throw new Error(`nexus_web_research failed (${resp.status}): ${msg}`)
      }
      return (await resp.json()) as Record<string, JsonValue>
    },
  })
}

/**
 * Create the `nexus_inspect_runtime` tool for inspecting live GPU/RAM telemetry and loaded models.
 *
 * @param gatewayUrl - Base URL of the local Nexus Gateway.
 * @param apiKey - Bearer API key for the local Nexus Gateway.
 * @param fetchImpl - Optional fetch implementation for testing.
 * @returns Registered Cordis tool definition.
 */
export function createNexusInspectRuntimeTool(
  gatewayUrl: string,
  apiKey: string,
  fetchImpl: typeof fetch = fetch,
): ToolDefinition {
  const baseUrl = gatewayUrl.replace(/\/+$/, '')
  return defineTool({
    name: 'nexus_inspect_runtime',
    description:
      'Inspect live Nexus Gateway runtime telemetry: loaded language/image models, VRAM/RAM usage, co-residency status, and active operations.',
    parameters: {
      include_models: {
        type: 'boolean',
        description: 'Also include the full list of loaded/available models.',
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: true,
      },
      render(_args, value) {
        return [{ type: 'text', text: JSON.stringify(value) }]
      },
    },
    async execute(args, exec) {
      const headers = { authorization: `Bearer ${apiKey}` }
      const runtimeResp = await fetchImpl(`${baseUrl}/v1/runtime`, { headers, signal: exec.signal })
      if (!runtimeResp.ok) {
        throw new Error(`nexus_inspect_runtime failed (${runtimeResp.status})`)
      }
      const runtime = (await runtimeResp.json()) as Record<string, JsonValue>
      if (args.include_models) {
        const modelsResp = await fetchImpl(`${baseUrl}/v1/models`, { headers, signal: exec.signal })
        if (modelsResp.ok) {
          runtime.models = (await modelsResp.json()) as JsonValue
        }
      }
      return runtime
    },
  })
}
