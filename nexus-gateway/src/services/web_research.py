"""Keyless web search and page-to-Markdown researcher for Nexus AI Lab."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def _is_public_ip(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_public_http_url(url: str) -> str:
    """Validate that a URL is public http/https and not a local/private network target."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or '<empty>'}")
    host = (parsed.hostname or "").strip().lower()
    if not host or host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local"):
        raise ValueError(f"Blocked non-public host: {host or '<empty>'}")
    if not _is_public_ip(host):
        raise ValueError(f"Blocked private IP address: {host}")
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        for info in infos:
            ip_str = str(info[4][0])
            if not _is_public_ip(ip_str):
                raise ValueError(f"Host {host} resolved to non-public IP {ip_str}")
    except socket.gaierror:
        pass
    return parsed.geturl()


class _HtmlToMarkdownParser(HTMLParser):
    """Lightweight stdlib HTML-to-Markdown extractor skipping nav/script/style blocks."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "aside", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self._parts: list[str] = []
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title = True
        elif t in {"h1", "h2", "h3", "h4"}:
            level = int(t[1])
            self._parts.append(f"\n\n{'#' * level} ")
        elif t in {"p", "div", "section", "article", "tr"}:
            self._parts.append("\n\n")
        elif t == "br":
            self._parts.append("\n")
        elif t == "li":
            self._parts.append("\n- ")
        elif t == "pre":
            self._in_pre = True
            self._parts.append("\n\n```\n")
        elif t == "code" and not self._in_pre:
            self._parts.append("`")
        elif t == "a":
            attr_map = dict(attrs)
            href = (attr_map.get("href") or "").strip()
            if href.startswith(("http://", "https://")):
                self._link_href = href
                self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title = False
        elif t == "pre":
            self._in_pre = False
            self._parts.append("\n```\n\n")
        elif t == "code" and not self._in_pre:
            self._parts.append("`")
        elif t == "a" and self._link_href is not None:
            label = " ".join("".join(self._link_text).split()).strip()
            href = self._link_href
            self._link_href = None
            self._link_text = []
            if label:
                self._parts.append(f"[{label}]({href})")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth > 0:
            return
        if self._link_href is not None:
            self._link_text.append(data)
            return
        if self._in_pre:
            self._parts.append(data)
        else:
            collapsed = re.sub(r"[ \t\r\n]+", " ", data)
            if collapsed:
                self._parts.append(collapsed)

    def get_markdown(self) -> tuple[str, str]:
        title = " ".join("".join(self.title_parts).split()).strip()
        raw = "".join(self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned = "\n".join(lines).strip()
        return title, cleaned


class _DdgHtmlResultParser(HTMLParser):
    """Parse DuckDuckGo HTML search results without external dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._cur_url = ""
        self._cur_title: list[str] = []
        self._cur_snippet: list[str] = []

    @staticmethod
    def _unwrap_ddg_url(href: str) -> str:
        if "duckduckgo.com/l/?" in href or href.startswith("/l/?"):
            parsed = urlparse(href)
            uddg = parse_qs(parsed.query).get("uddg")
            if uddg and uddg[0]:
                return unquote(uddg[0])
        if href.startswith("//"):
            return "https:" + href
        return href

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        cls = (attr_map.get("class") or "").split()
        href = (attr_map.get("href") or "").strip()
        if tag == "a" and ("result__a" in cls or "result-link" in cls) and href:
            self._in_link = True
            self._cur_url = self._unwrap_ddg_url(href)
            self._cur_title = []
            self._cur_snippet = []
        elif tag in {"a", "div", "span", "td"} and ("result__snippet" in cls or "result-snippet" in cls):
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            title = " ".join("".join(self._cur_title).split()).strip()
            if title and self._cur_url.startswith(("http://", "https://")):
                self.results.append({"title": title, "url": self._cur_url, "snippet": ""})
        elif self._in_snippet and tag in {"a", "div", "span", "td"}:
            self._in_snippet = False
            snippet = " ".join("".join(self._cur_snippet).split()).strip()
            if self.results and snippet and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = snippet
            self._cur_snippet = []

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._cur_title.append(data)
        elif self._in_snippet:
            self._cur_snippet.append(data)


def _search_ddg(client: httpx.Client, query: str, limit: int) -> list[dict[str, str]]:
    for endpoint in ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"):
        try:
            resp = client.post(
                endpoint,
                data={"q": query, "kl": "wt-wt"},
                headers={"User-Agent": _USER_AGENT, "Referer": "https://duckduckgo.com/"},
            )
            if resp.status_code >= 400:
                continue
            parser = _DdgHtmlResultParser()
            parser.feed(resp.text)
            if parser.results:
                return parser.results[:limit]
        except Exception:
            continue
    return []


def _search_wikipedia(client: httpx.Client, query: str, limit: int) -> list[dict[str, str]]:
    try:
        resp = client.get(
            f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit={limit}&namespace=0&format=json",
            headers={"User-Agent": _USER_AGENT},
        )
        if resp.status_code >= 400:
            return []
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 4:
            return []
        titles, descs, urls = payload[1], payload[2], payload[3]
        out: list[dict[str, str]] = []
        for idx, title in enumerate(titles):
            url = str(urls[idx]) if idx < len(urls) else ""
            snippet = str(descs[idx]) if idx < len(descs) else ""
            if url.startswith(("http://", "https://")):
                out.append({"title": str(title), "url": url, "snippet": snippet})
        return out[:limit]
    except Exception:
        return []


def search_web(query: str, *, limit: int = 8, timeout_s: float = 15.0) -> list[dict[str, str]]:
    """Run a keyless web search via DuckDuckGo HTML/Lite with Wikipedia fallback."""
    cleaned = " ".join((query or "").split()).strip()
    if not cleaned:
        return []
    bounded = max(1, min(int(limit), 15))
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        results = _search_ddg(client, cleaned, bounded)
        if not results:
            results = _search_wikipedia(client, cleaned, bounded)
    return results


def fetch_page_markdown(
    url: str,
    *,
    max_chars: int = 16_000,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Fetch a public HTTP(S) page and extract clean Markdown text."""
    safe_url = validate_public_http_url(url)
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        resp = client.get(safe_url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        final_url = validate_public_http_url(str(resp.url))
        content_type = (resp.headers.get("content-type") or "").lower()
        body = resp.text
    if "html" in content_type or "<html" in body[:500].lower():
        parser = _HtmlToMarkdownParser()
        parser.feed(body)
        title, markdown = parser.get_markdown()
    else:
        title = ""
        markdown = body.strip()
    truncated = len(markdown) > max_chars
    if truncated:
        markdown = markdown[:max_chars].rstrip() + "\n\n[Truncated]"
    return {
        "url": final_url,
        "title": title or final_url,
        "content": markdown,
        "truncated": truncated,
    }


def research_web(
    queries: list[str],
    *,
    max_pages: int = 4,
    max_chars_per_page: int = 12_000,
) -> dict[str, Any]:
    """Run multi-query web search, deduplicate URLs, and fetch top pages in parallel."""
    clean_queries = [" ".join(str(q).split()).strip() for q in queries if str(q).strip()][:3]
    if not clean_queries:
        return {"queries": [], "results": [], "pages": [], "citations": [], "warnings": ["No queries provided"]}

    all_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    warnings: list[str] = []

    with ThreadPoolExecutor(max_workers=min(3, len(clean_queries))) as pool:
        future_map = {pool.submit(search_web, q, limit=6): q for q in clean_queries}
        for fut in as_completed(future_map):
            q = future_map[fut]
            try:
                for item in fut.result():
                    u = item.get("url", "")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        all_results.append({**item, "query": q})
            except Exception as exc:
                warnings.append(f"Search failed for {q!r}: {exc}")

    target_urls = [item["url"] for item in all_results[: max(1, min(int(max_pages), 6))]]
    pages: list[dict[str, Any]] = []
    if target_urls:
        with ThreadPoolExecutor(max_workers=min(4, len(target_urls))) as pool:
            future_map = {
                pool.submit(fetch_page_markdown, u, max_chars=max_chars_per_page): u
                for u in target_urls
            }
            for fut in as_completed(future_map):
                u = future_map[fut]
                try:
                    pages.append(fut.result())
                except Exception as exc:
                    warnings.append(f"Fetch failed for {u}: {exc}")

    citations = [{"title": p["title"], "url": p["url"]} for p in pages] or [
        {"title": r["title"], "url": r["url"]} for r in all_results[:5]
    ]
    return {
        "queries": clean_queries,
        "results": all_results,
        "pages": pages,
        "citations": citations,
        "warnings": warnings,
    }
