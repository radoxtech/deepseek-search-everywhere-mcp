"""MCP server for the DeepSeek Search & Fetch API.

Exposes web search, page fetch, search-and-read and balance tools over stdio.
Requires DEEPSEEK_API_KEY; DEEPSEEK_BASE_URL overrides the endpoint.

The command-line entry point lives in ``deepseek_search_everywhere_mcp.cli``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from .cache import SingleFlight, TTLCache
from .errors import _fail
from .errors import _map_http_error as _map_http_error  # re-exported for callers and tests
from .ratelimit import FETCH, SEARCH, RateLimiter, RateLimitExceeded
from .redact import redact
from .settings import get_settings
from .transport import _api_key, _base_url
from .transport import request as _request  # tests monkeypatch server._request
from .urls import CODE_INVALID, UrlError, validate_url

MAX_QUERY = 128
MAX_URL = 8192
SEARCH_TIMEOUT = 60.0
FETCH_TIMEOUT = 120.0
CHAT_TIMEOUT = 8.0

SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"
REWRITE_MODEL = "deepseek-flash"

SYSTEM_PROMPT = (
    "You rewrite search queries. Output ONLY the rewritten query, no quotes, no explanation. "
    "Translate the query to English. Keep search operators verbatim: site:, -site:, lang:, after:, "
    "before:, and quoted phrases. Keep proper nouns, product names and numbers unchanged. "
    "Maximum 120 characters."
)

TRACKING_PARAMS = {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid"}
MIN_TITLE_SITE_CHARS = 20
MIN_TITLE_SITE_WORDS = 3

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

mcp = FastMCP("deepseek-search-everywhere-mcp")

# Runtime state, built lazily from the active settings on first use so nothing is bound
# to an event loop at import time. ``reset_state()`` drops it all.
_search_cache: TTLCache | None = None
_search_flight: SingleFlight | None = None
_fetch_cache: TTLCache | None = None
_fetch_flight: SingleFlight | None = None
_rewrite_cache: TTLCache | None = None
_rate_limiter: RateLimiter | None = None


def reset_state() -> None:
    """Forget the response and rewrite caches, in-flight calls and the rate limiter."""
    global _search_cache, _search_flight, _fetch_cache, _fetch_flight, _rewrite_cache, _rate_limiter
    _search_cache = _search_flight = _fetch_cache = _fetch_flight = None
    _rewrite_cache = _rate_limiter = None


def _get_search_cache() -> TTLCache | None:
    """Return the search response cache, or ``None`` when caching is disabled."""
    global _search_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _search_cache is None:
        _search_cache = TTLCache(settings.cache_search_entries, settings.cache_search_bytes,
                                 ttl=settings.cache_search_ttl)
    return _search_cache


def _get_search_flight() -> SingleFlight:
    """Return the single-flight group that collapses identical concurrent searches."""
    global _search_flight
    if _search_flight is None:
        # Cache writes are done by the leader itself so only usable responses are stored.
        _search_flight = SingleFlight(None)
    return _search_flight


def _get_fetch_cache() -> TTLCache | None:
    """Return the fetch response cache, or ``None`` when caching is disabled."""
    global _fetch_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _fetch_cache is None:
        _fetch_cache = TTLCache(settings.cache_fetch_entries, settings.cache_fetch_bytes,
                                ttl=settings.cache_fetch_ttl)
    return _fetch_cache


def _get_fetch_flight() -> SingleFlight:
    """Return the single-flight group that collapses identical concurrent fetches."""
    global _fetch_flight
    if _fetch_flight is None:
        # Cache writes are done by the leader itself so only usable pages are stored.
        _fetch_flight = SingleFlight(None)
    return _fetch_flight


def _get_rewrite_cache() -> TTLCache | None:
    """Return the query-rewrite cache, or ``None`` when caching is disabled."""
    global _rewrite_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _rewrite_cache is None:
        _rewrite_cache = TTLCache(settings.cache_rewrite_entries, settings.cache_rewrite_bytes,
                                  ttl=settings.cache_rewrite_ttl)
    return _rewrite_cache


def _get_rate_limiter() -> RateLimiter:
    """Return the process-wide per-endpoint rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter.from_settings(get_settings())
    return _rate_limiter


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _search_cache_key(payload: dict[str, Any]) -> str:
    """Key a search by endpoint and upstream payload only (no key, no shaping options)."""
    return _sha256(_canonical_json({"path": SEARCH_PATH, "base_url": _base_url(), "payload": payload}))


def _fetch_cache_key(url: str) -> str:
    """Key a fetch by endpoint and the exact URL, minus its fragment (never sent upstream).

    The URL is deliberately not canonicalised; shaping options (``clean``, ``max_chars``)
    are applied per call and stay out of the key, as does the API key.
    """
    target = url.strip().split("#", 1)[0]
    return _sha256(_canonical_json({"path": FETCH_PATH, "base_url": _base_url(), "url": target}))


async def _limited_request(endpoint: str, path: str, payload: dict[str, Any],
                           timeout: float) -> dict[str, Any]:
    """Take one ``endpoint`` token, then call upstream once (transport retries included).

    Returns the transport envelope, or a ``rate_limited_local`` error envelope when no token
    is available in time. The token is refunded if the call is cancelled.
    """
    try:
        ticket = await _get_rate_limiter().acquire(endpoint)
    except RateLimitExceeded as exc:
        return exc.to_envelope()
    try:
        return await _request("POST", path, payload=payload, timeout=timeout)
    except asyncio.CancelledError:
        ticket.refund()
        raise


async def _cached_call(cache: TTLCache | None, flight: SingleFlight, key: str,
                       leader: Callable[[], Awaitable[dict[str, Any]]],
                       *, refresh: bool) -> tuple[dict[str, Any], bool]:
    """Serve ``key`` from ``cache`` or through a single-flight ``leader`` call.

    Returns ``(record, cached)``; the record is an error envelope when the leader failed.
    """
    record = None if refresh or cache is None else cache.get(key)
    if record is not None:
        return record, True
    result = await flight.run(key, leader, refresh=refresh)
    return result.value, False


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n…[truncated]", True


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("utm_") or lowered in TRACKING_PARAMS


def _canonical_url(url: str) -> str | None:
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port is None else f"{host}:{port}"
    path = parts.path
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    params = [
        (name, value)
        for name, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    params.sort()
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), netloc, path, urllib.parse.urlencode(params), "")
    )


def _title_site_key(item: dict) -> tuple[str, str] | None:
    title = _norm_text(item.get("title"))
    site = _norm_text(item.get("site_name"))
    if not title or not site:
        return None
    if len(title) < MIN_TITLE_SITE_CHARS or len(title.split()) < MIN_TITLE_SITE_WORDS:
        return None
    return (title, site)


def _dedupe_key(item: dict) -> tuple | None:
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        canonical = _canonical_url(url)
        if canonical is not None:
            return ("url", canonical)
    title_site = _title_site_key(item)
    if title_site is None:
        return None
    return ("title", *title_site)


def _dedupe_results(items: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    seen: set[tuple] = set()
    seen_title_sites: set[tuple[str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue

        key = _dedupe_key(item)
        title_site = _title_site_key(item)

        if (key is not None and key in seen) or (
            title_site is not None and title_site in seen_title_sites
        ):
            removed += 1
            continue

        if key is not None:
            seen.add(key)
        if title_site is not None:
            seen_title_sites.add(title_site)
        kept.append(item)

    return kept, removed


def _unescape_markdown(text: str) -> str:
    text = re.sub(r"\\([,.;:!?])", r"\1", text)
    return re.sub(r"(?m)^\\#(?=\s)", "#", text)


def _with_lang(query: str, lang: str | None) -> tuple[str, bool]:
    if not lang:
        return query, False
    if re.search(r"(?:^|\s)lang:\S+", query):
        return query, False
    prefix = f"lang:{lang} "
    room = MAX_QUERY - len(prefix)
    trimmed = query[:room]
    return prefix + trimmed, len(trimmed) < len(query)


async def _rewrite_query(query: str) -> str | None:
    payload: dict[str, Any] = {
        "model": REWRITE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    try:
        result = await _request(
            "POST", "/v1/chat/completions", payload=payload, timeout=CHAT_TIMEOUT, retries=1
        )
        if not result.get("ok"):
            return None
        content = result["data"]["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None
        first = next((line.strip() for line in content.splitlines() if line.strip()), "")
        if len(first) >= 2 and first[0] in "\"'" and first[-1] in "\"'":
            first = first[1:-1]
        cleaned = re.sub(r"\s+", " ", first).strip()[:MAX_QUERY].strip()
        if not cleaned:
            return None
        return cleaned
    except Exception:
        return None


async def _cached_rewrite(query: str) -> str | None:
    """Rewrite ``query`` through the rewrite cache; failed rewrites are not remembered."""
    cache = _get_rewrite_cache()
    key = (_base_url(), REWRITE_MODEL, _sha256(SYSTEM_PROMPT), query)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    rewritten = await _rewrite_query(query)
    if cache is not None and rewritten is not None:
        cache.set(key, rewritten)
    return rewritten


async def _search_upstream(payload: dict[str, Any], query_original: str | None,
                           key: str) -> dict[str, Any]:
    """Leader-only path: take a ``search`` token, call upstream, cache a usable answer.

    Returns ``{"data": ..., "query_original": ...}`` on success or an error envelope.
    """
    result = await _limited_request(SEARCH, SEARCH_PATH, payload, SEARCH_TIMEOUT)
    if not result.get("ok"):
        return result
    record = {"data": result["data"], "query_original": query_original}
    cache = _get_search_cache()
    if cache is not None and result["data"].get("results"):
        cache.set(key, record)
    return record


async def _search(query: str, count: int, ori_question: str | None, need_content: bool,
                  lang: str | None, max_chars_per_result: int,
                  max_total_chars: int, dedupe: bool = True,
                  rewrite: bool = False, *, refresh: bool = False) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return _fail("empty_query", "query must not be empty.")
    if len(query) > MAX_QUERY:
        return _fail("query_too_long", f"query must be at most {MAX_QUERY} characters.")
    if not 1 <= count <= 50:
        return _fail("bad_count", "count must be between 1 and 50.")
    if lang and not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", lang):
        return _fail("bad_lang", "lang must look like 'es' or 'pt-BR'.")

    query_original: str | None = None
    rewrite_failed = False
    if rewrite and lang is None and _api_key():
        rewritten = await _cached_rewrite(query)
        if rewritten is None:
            rewrite_failed = True
        elif rewritten != query:
            query_original = query
            query = rewritten
            if not ori_question:
                ori_question = query_original[:MAX_QUERY]

    used, trimmed = _with_lang(query, lang)
    payload: dict[str, Any] = {"query": used, "count": count, "need_content": need_content}
    if ori_question:
        payload["ori_question"] = ori_question[:MAX_QUERY]

    key = _search_cache_key(payload)
    record, cached = await _cached_call(
        _get_search_cache(), _get_search_flight(), key,
        lambda: _search_upstream(payload, query_original, key), refresh=refresh,
    )
    if record.get("ok") is False:
        return record

    raw = record["data"].get("results") or []
    removed = 0
    if dedupe:
        raw, removed = _dedupe_results(raw)
    remaining = max_total_chars
    shaped: list[dict[str, Any]] = []
    content_truncated = False

    for item in raw:
        entry: dict[str, Any] = {
            "title": item.get("title"),
            "url": item.get("url"),
            "date": item.get("date"),
            "authority": item.get("authority"),
            "rerank_score": item.get("rerank_score"),
            "site": item.get("site_name"),
        }
        snippet = item.get("snippet") or ""
        if snippet:
            entry["snippet"] = snippet

        body = item.get("content") if need_content else None
        if body:
            cleaned = _unescape_markdown(body)
            allowed = max(0, min(max_chars_per_result, remaining))
            entry["content"], cut = _truncate(cleaned, allowed)
            remaining -= len(entry["content"])
            entry["content_chars"] = len(cleaned)
            content_truncated = content_truncated or cut
        elif item.get("summary"):
            allowed = max(0, min(max_chars_per_result, remaining))
            entry["summary"], cut = _truncate(item["summary"], allowed)
            remaining -= len(entry["summary"])
            content_truncated = content_truncated or cut
        shaped.append(entry)

        if need_content and remaining <= 0:
            break

    notes: list[str] = []
    if query_original is not None:
        notes.append(f"query rewritten (flash): '{used}'")
    elif rewrite_failed:
        notes.append("query rewrite failed; using original query")
    if removed:
        notes.append(f"deduped {removed} duplicate result(s)")
    if trimmed:
        notes.append(f"Query trimmed to fit the {MAX_QUERY}-character limit.")
    if lang:
        notes.append(f"Language operator lang:{lang} applied.")
    if content_truncated:
        notes.append("Some page content was truncated to respect the character budget.")

    envelope: dict[str, Any] = {
        "ok": True,
        "query_used": used,
        "count": len(shaped),
        "need_content": need_content,
        "deduped": removed,
        "cached": cached,
        "results": shaped,
        "notes": [redact(note) for note in notes],
    }
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope


async def _fetch_upstream(url: str, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``fetch`` token, call upstream, cache a non-empty page.

    Returns ``{"data": ...}`` with the raw upstream page on success or an error envelope.
    """
    result = await _limited_request(FETCH, FETCH_PATH, {"url": url}, FETCH_TIMEOUT)
    if not result.get("ok"):
        return result
    record = {"data": result["data"]}
    cache = _get_fetch_cache()
    if cache is not None and result["data"].get("content"):
        cache.set(key, record)
    return record


@mcp.tool()
async def deepseek_web_search(
    query: str,
    count: int = 10,
    ori_question: str | None = None,
    need_content: bool = False,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 40000,
    dedupe: bool = True,
    rewrite: bool = False,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Search the web with DeepSeek's index. Returns titles, URLs, dates, authority and rerank scores.

    Use this for discovery and ranking. Set need_content=True to get full page text inline in the
    same call (billed as one request, no per-page charge) — this is usually cheaper and more
    reliable than calling deepseek_fetch afterwards.

    For non-English queries set lang to the appropriate code (e.g. 'es', 'de', 'ja'). Without it
    results may be mixed-language; for named entities test with and without the operator.
    Each result carries date (RFC 3339) and authority (0-4, -1 unknown).

    Responses are cached in memory for a few minutes and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 8 MiB fails with response_too_large.

    Args:
        query: Search query, max 128 characters. Supports site:, -site:, "phrase", lang:, after:, before:.
        count: Number of results, 1-50 (default 10).
        ori_question: Original natural-language question, max 128 characters; improves ranking.
        need_content: Return full page text per result (recommended for reading).
        lang: Language operator, e.g. 'es', 'de' or 'ja'. For non-English queries set it to the
            appropriate code; without it results may be mixed-language. For named entities test
            with and without the operator.
        max_chars_per_result: Content cap per result when need_content is set.
        max_total_chars: Total content budget across all results.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
        refresh: Bypass the response cache and query upstream again (the envelope's cached
            flag says whether the results came from the cache).
    """
    return await _search(query, count, ori_question, need_content, lang,
                         max_chars_per_result, max_total_chars, dedupe, rewrite, refresh=refresh)


@mcp.tool()
async def deepseek_search_and_read(
    query: str,
    count: int = 5,
    ori_question: str | None = None,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 30000,
    dedupe: bool = True,
    rewrite: bool = False,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Search and return full page content in one request — the recommended retrieval path.

    This is search with need_content=True. Measured on 14 benchmark questions it returned content
    for PDFs and Reddit threads where deepseek_fetch returned HTTP 404, in about 0.7s per call.

    Responses are cached in memory for a few minutes and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 8 MiB fails with response_too_large.

    Args:
        query: Search query, max 128 characters.
        count: Number of results, 1-50 (default 5).
        ori_question: Original natural-language question, max 128 characters.
        lang: Language operator, e.g. 'es', 'de' or 'ja'. For non-English queries set it to the
            appropriate code; without it results may be mixed-language. For named entities test
            with and without the operator.
        max_chars_per_result: Content cap per result.
        max_total_chars: Total content budget across all results.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
        refresh: Bypass the response cache and query upstream again (the envelope's cached
            flag says whether the results came from the cache).
    """
    return await _search(query, count, ori_question, True, lang,
                         max_chars_per_result, max_total_chars, dedupe, rewrite, refresh=refresh)


def _reject_url(url: str) -> dict[str, Any] | None:
    """Statically validate ``url``; return an error envelope if it is rejected, else ``None``.

    Malformed URLs map to ``bad_url`` and blocked targets (internal hosts, literal private
    addresses, credentials in the URL) to ``ssrf_blocked``. Every tool that forwards a URL
    to DeepSeek calls this before touching the cache, the limiter or the network.
    """
    try:
        validate_url(url)
    except UrlError as exc:
        return _fail("bad_url" if exc.code == CODE_INVALID else exc.code, exc.message)
    return None


@mcp.tool()
async def deepseek_fetch(
    url: str,
    clean: bool = True,
    max_chars: int = 20000,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Fetch one known URL and return its title and Markdown content.

    Caveat measured on 2026-10-20: this endpoint returns HTTP 404 for PDFs and for Reddit, and
    times out on some sites. When the URL is a document or a blocked site, call
    deepseek_search_and_read with the page title or a site:-scoped query instead.

    The response contains only title and content — no final URL, status code or publication date,
    so keep the provenance on your side.

    Pages are cached in memory for about half an hour and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 32 MiB fails with response_too_large.

    URLs that point at internal hosts or literal private addresses, or that embed credentials
    (user:pass@host), are rejected locally with ssrf_blocked before any request is made.

    Args:
        url: Absolute http(s) URL, max 8192 characters.
        clean: Remove DeepSeek's spurious Markdown escaping (\\# at line start, \\, \\, etc.).
        max_chars: Content cap, applied after cleaning.
        refresh: Bypass the response cache and fetch the page again (the envelope's cached
            flag says whether the page came from the cache).
    """
    url = (url or "").strip()
    if not url:
        return _fail("empty_url", "url must not be empty.")
    if len(url) > MAX_URL:
        return _fail("url_too_long", f"url must be at most {MAX_URL} characters.")
    if not re.match(r"^https?://", url, re.I):
        return _fail("bad_url", "url must start with http:// or https://.")

    rejection = _reject_url(url)
    if rejection is not None:
        return rejection

    key = _fetch_cache_key(url)
    record, cached = await _cached_call(
        _get_fetch_cache(), _get_fetch_flight(), key, lambda: _fetch_upstream(url, key),
        refresh=refresh,
    )
    if record.get("ok") is False:
        return record

    data = record["data"]
    content = data.get("content") or ""
    if clean:
        content = _unescape_markdown(content)
    content, truncated = _truncate(content, max_chars)

    return {
        "ok": True,
        "url": url,
        "title": data.get("title"),
        "content": content,
        "content_chars": len(content),
        "truncated": truncated,
        "cached": cached,
        "notes": ["Empty content: the page may be a PDF, JS-only or blocked. "
                  "Try deepseek_search_and_read instead."] if not content else [],
    }


@mcp.tool()
async def deepseek_balance() -> dict[str, Any]:
    """Check the DeepSeek account balance (granted and topped-up, in CNY and USD)."""
    result = await _request("GET", "/user/balance", timeout=20.0)
    if not result.get("ok"):
        return result
    data = result["data"]
    return {
        "ok": True,
        "is_available": data.get("is_available"),
        "balances": data.get("balance_infos") or data.get("balances") or data,
    }


async def _check() -> int:
    key = _api_key()
    print(f"base_url: {_base_url()}")
    print(f"DeepSeek key: {'set' if key else 'MISSING'} (len={len(key)})")
    if not key:
        print("Set DEEPSEEK_API_KEY or point DEEPSEEK_AUTH_JSON at a JSON file.")
        return 1
    result = await deepseek_balance()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1

