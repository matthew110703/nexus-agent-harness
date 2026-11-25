/**
 * Human-readable model display name resolver.
 * Never alters the canonical model ID sent to Host or Gateway.
 */

const VARIANT_TOKENS = new Set([
  'heretic',
  'abliterated',
  'uncensored',
  'fp8',
  'fp16',
  'bf16',
  'int8',
  'int4',
  'awq',
  'gptq',
  'gguf',
  'dev',
  'schnell',
  'klein',
  'turbo',
  'base',
])

const DROP_TOKENS = new Set(['instruct', 'chat', 'it'])

function titleCaseToken(token: string): string {
  const lower = token.toLowerCase()
  if (/^\d+(?:\.\d+)?b$/i.test(token)) return token.toUpperCase()
  if (/^a\d+(?:\.\d+)?b$/i.test(token)) return token.toUpperCase()
  if (lower === 'vl') return 'VL'
  if (lower === 'fp8' || lower === 'fp16' || lower === 'bf16' || lower === 'int4' || lower === 'int8' || lower === 'awq' || lower === 'gptq' || lower === 'gguf') {
    return token.toUpperCase()
  }
  if (lower === 'flux.2' || lower === 'flux2') return 'FLUX.2'
  if (lower === 'sdxl') return 'SDXL'
  if (lower === 'qwen') return 'Qwen'
  return token.charAt(0).toUpperCase() + token.slice(1).toLowerCase()
}

/**
 * Resolve a clean display name from model metadata when available, falling back
 * to conservative formatting of the canonical model ID.
 * @param modelId - canonical model identifier sent to the backend.
 * @param metadataDisplayName - optional display name from catalog metadata.
 * @returns human-readable label for UI display.
 */
export function resolveModelDisplayName(modelId: string, metadataDisplayName?: string | null): string {
  const trimmedId = modelId.trim()
  if (!trimmedId) return ''

  const meta = metadataDisplayName?.trim()
  if (meta && meta !== trimmedId) {
    return meta
  }

  // Strip optional org prefix (e.g. "org/model-name" -> "model-name").
  const leaf = trimmedId.includes('/') ? trimmedId.slice(trimmedId.lastIndexOf('/') + 1) : trimmedId

  // Normalize "Qwen-3-VL-..." -> "Qwen3-VL-..." before tokenizing.
  const normalized = leaf.replace(/^([A-Za-z]+)-(\d+(?:\.\d+)?)(?=-|$)/, '$1$2')
  const rawTokens = normalized.split(/[-_]+/).filter(Boolean)
  if (rawTokens.length === 0) return trimmedId

  const primary: string[] = []
  const variants: string[] = []

  for (const tok of rawTokens) {
    const lower = tok.toLowerCase()
    if (DROP_TOKENS.has(lower)) continue
    if (lower === 'vl' && primary.length > 0) {
      primary[primary.length - 1] = `${primary[primary.length - 1]}-VL`
      continue
    }
    if (VARIANT_TOKENS.has(lower) && primary.length > 0) {
      variants.push(titleCaseToken(tok))
      continue
    }
    primary.push(titleCaseToken(tok))
  }

  const base = primary.join(' ').trim() || trimmedId
  if (variants.length === 0) return base
  return `${base} · ${variants.join(' ')}`
}

/**
 * Resolve a compact label suitable for buttons and the composer badge
 * (omits secondary variant suffixes when the base name is already clear).
 * @param modelId - canonical model identifier.
 * @param metadataDisplayName - optional display name from catalog metadata.
 * @returns short label (e.g. "Qwen3-VL 8B").
 */
export function resolveCompactModelName(modelId: string, metadataDisplayName?: string | null): string {
  const full = resolveModelDisplayName(modelId, metadataDisplayName)
  const sep = full.indexOf(' · ')
  return sep > 0 ? full.slice(0, sep) : full
}
