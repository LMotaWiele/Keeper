"""Web search — local SearXNG metasearch plus full-page extraction."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlparse, urlencode, parse_qsl, urlunparse

import httpx
import trafilatura
from langchain_core.tools import tool

from config.settings import config

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_FETCH_CONCURRENCY = 5
_EXTRACT_MAX_CHARS = 4000
_TOOL_CONTENT_CHARS = 1200
_TOOL_MAX_SOURCES = 5


@dataclass
class SearchHealth:
    queries: int = 0
    zero_result_queries: int = 0
    fetch_attempts: int = 0
    fetch_failures: int = 0
    last_error: str = ""
    last_query_at: str = ""
    consecutive_zero_results: int = 0

    @property
    def fetch_failure_rate(self) -> float:
        if self.fetch_attempts <= 0:
            return 0.0
        return self.fetch_failures / self.fetch_attempts

    def snapshot(self) -> dict:
        return {
            "queries": self.queries,
            "zero_result_queries": self.zero_result_queries,
            "fetch_attempts": self.fetch_attempts,
            "fetch_failures": self.fetch_failures,
            "fetch_failure_rate": round(self.fetch_failure_rate, 3),
            "last_error": self.last_error,
            "last_query_at": self.last_query_at,
        }


HEALTH = SearchHealth()


def _build_search_url(query: str) -> str:
    template = config.searxng_query_url
    if "<query>" in template:
        url = template.replace("<query>", quote_plus(query))
    else:
        sep = "&" if "?" in template else "?"
        url = f"{template}{sep}q={quote_plus(query)}"

    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params.setdefault("format", "json")
    params.setdefault("language", "en")
    params.setdefault("safesearch", "0")
    return urlunparse(parsed._replace(query=urlencode(params)))


def _extract_text(html: str) -> str | None:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    return text[:_EXTRACT_MAX_CHARS]


async def _fetch_and_extract(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> str | None:
    HEALTH.fetch_attempts += 1
    async with sem:
        try:
            resp = await client.get(
                url,
                follow_redirects=True,
                timeout=config.search_fetch_timeout,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            HEALTH.fetch_failures += 1
            HEALTH.last_error = f"fetch {url}: {exc}"
            log.debug("Page fetch failed for %s: %s", url, exc)
            return None

    try:
        extracted = await asyncio.to_thread(_extract_text, html)
    except Exception as exc:
        HEALTH.fetch_failures += 1
        HEALTH.last_error = f"extract {url}: {exc}"
        log.debug("Extraction failed for %s: %s", url, exc)
        return None

    if extracted is None:
        HEALTH.fetch_failures += 1
        HEALTH.last_error = f"extract {url}: empty"
        return None
    return extracted


async def run_search(query: str) -> list[dict[str, Any]]:
    """Query SearXNG, extract top pages, return structured results."""
    url = _build_search_url(query)
    HEALTH.queries += 1
    HEALTH.last_query_at = datetime.now().isoformat(timespec="seconds")

    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": _USER_AGENT}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        HEALTH.last_error = f"searxng: {exc}"
        HEALTH.zero_result_queries += 1
        HEALTH.consecutive_zero_results += 1
        log.warning("SearXNG query failed for %r: %s", query, exc)
        if HEALTH.consecutive_zero_results >= 3:
            log.error(
                "Three consecutive zero-result SearXNG queries. URL=%s error=%s",
                url, exc,
            )
        return []

    raw_results = data.get("results") or []
    if not raw_results:
        HEALTH.zero_result_queries += 1
        HEALTH.consecutive_zero_results += 1
        log.warning("SearXNG returned zero results for %r", query)
        if HEALTH.consecutive_zero_results >= 3:
            log.error(
                "Three consecutive zero-result SearXNG queries. URL=%s",
                url,
            )
        return []

    HEALTH.consecutive_zero_results = 0

    seen: set[str] = set()
    hits: list[dict[str, Any]] = []
    for item in raw_results:
        item_url = (item.get("url") or "").strip()
        if not item_url or item_url in seen:
            continue
        seen.add(item_url)
        hits.append({
            "title": item.get("title") or "Untitled",
            "url": item_url,
            "content": item.get("content") or "",
            "engine": item.get("engine") or "",
            "extracted": False,
        })
        if len(hits) >= config.search_max_results:
            break

    to_fetch = hits[: config.search_max_fetch]
    sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
    async with httpx.AsyncClient(timeout=config.search_fetch_timeout, headers={"User-Agent": _USER_AGENT}) as client:
        extracted_texts = await asyncio.gather(
            *[_fetch_and_extract(client, h["url"], sem) for h in to_fetch],
            return_exceptions=False,
        )

    for hit, text in zip(to_fetch, extracted_texts):
        if text:
            hit["content"] = text
            hit["extracted"] = True

    return hits


def _format_tool_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No search results."
    lines = []
    for i, r in enumerate(results[:_TOOL_MAX_SOURCES], start=1):
        title = r.get("title") or "Untitled"
        url = r.get("url") or ""
        content = (r.get("content") or "")[:_TOOL_CONTENT_CHARS]
        lines.append(f"[{i}] {title} ({url})\n{content}")
    return "\n\n".join(lines)


@tool
async def web_search(query: str) -> str:
    """Search the web for current information. Use for news, facts you're unsure about, or anything that may have changed since training."""
    results = await run_search(query)
    return _format_tool_results(results)


# Back-compat alias used by older imports.
web_search_tool = web_search
