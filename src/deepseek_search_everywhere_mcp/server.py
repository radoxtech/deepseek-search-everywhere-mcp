"""MCP server for the DeepSeek Search & Fetch API.

Exposes web search, page fetch (single and batch), search-and-read, research and balance tools.
``create_server()`` builds a FastMCP instance with every tool registered; the module-level
``mcp`` serves stdio, and ``deepseek_search_everywhere_mcp.http_app`` builds fresh ones for HTTP.
Requires DEEPSEEK_API_KEY; DEEPSEEK_BASE_URL overrides the endpoint.

The command-line entry point lives in ``deepseek_search_everywhere_mcp.cli``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import logging
import math
import re
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, ParamSpec

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ._version import __version__
from .boilerplate import remove_boilerplate
from .budget import DailyBudgetTracker
from .cache import SingleFlight, TTLCache
from .deep_research import (
    build_decomposition_messages,
    build_synthesis_messages,
    parse_subqueries,
    sanitize_citations,
    select_synthesis_context,
)
from .errors import _fail
from .errors import _map_http_error as _map_http_error  # re-exported for callers and tests
from .extract import (
    build_extract_prompt,
    sanitize_json,
    validate_fields,
    validate_schema,
)
from .filters import compile_filters
from .ratelimit import FETCH, SEARCH, RateLimiter, RateLimitExceeded
from .redact import redact
from .settings import DEFAULT_LOG_LEVEL, get_settings
from .snippets import extract_snippets, split_sections
from .transport import _api_key, _base_url
from .transport import request as _request  # tests monkeypatch server._request
from .urls import CODE_INVALID, UrlError, validate_url

MAX_QUERY = 128
MAX_URL = 8192
MAX_TOPIC = 500
SEARCH_TIMEOUT = 45.0
FETCH_TIMEOUT = 60.0
BALANCE_TIMEOUT = 20.0
CHAT_TIMEOUT = 8.0
SYNTHESIS_TIMEOUT = 25.0
EXTRACT_TIMEOUT = 20.0
RESEARCH_DEADLINE = 35.0

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
MAX_COUNT = 50  # largest ``count`` accepted from callers and by the upstream search API
# With dedupe on, the upstream is asked for up to max(OVERFETCH_MIN, ceil(count * OVERFETCH_RATIO))
# extra results (capped at MAX_COUNT) so that removing duplicates still leaves ``count`` results.
OVERFETCH_MIN = 2
OVERFETCH_RATIO = 0.5
# Default summary cap per result when the search returns no page content (need_content=False).
SEARCH_SUMMARY_CHARS = 400
# Note added by every tool whose content was cut to fit a character budget.
TRUNCATED_NOTE = "Some page content was truncated to respect the character budget."
# Upstream pricing per billable Search/Fetch request: $0.75 per 1,000 requests.
COST_PER_REQUEST = 0.00075


def _envelope_meta(cost_usd: float, untrusted_content: bool = True) -> dict[str, Any]:
    """Build the standard ``_meta`` payload reporting estimated costs and security classification."""
    return {
        "cost_usd": cost_usd,
        "untrusted_content": untrusted_content,
    }


logger = logging.getLogger(__name__)

# Monotonic clock behind every envelope's ``elapsed_ms``; tests may replace it.
_clock: Callable[[], float] = time.monotonic

_P = ParamSpec("_P")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SERVER_NAME = "deepseek-search-everywhere-mcp"


@dataclass
class SessionMetrics:
    """In-memory telemetry for requests, cache hits, and estimated costs in the current session."""

    start_time: float
    requests_upstream: int = 0
    cache_hits: int = 0
    searches_upstream: int = 0
    fetches_upstream: int = 0


# Runtime state, built lazily from the active settings on first use so nothing is bound
# to an event loop at import time. ``reset_state()`` drops it all.
_search_cache: TTLCache | None = None
_search_flight: SingleFlight | None = None
_fetch_cache: TTLCache | None = None
_fetch_flight: SingleFlight | None = None
_rewrite_cache: TTLCache | None = None
_rewrite_flight: SingleFlight | None = None
_rate_limiter: RateLimiter | None = None
_session_metrics: SessionMetrics | None = None
_budget_tracker: DailyBudgetTracker | None = None


def reset_state() -> None:
    """Forget the response and rewrite caches, in-flight calls, session metrics and the rate limiter."""
    global _search_cache, _search_flight, _fetch_cache, _fetch_flight
    global _rewrite_cache, _rewrite_flight, _rate_limiter, _session_metrics, _budget_tracker
    _search_cache = _search_flight = _fetch_cache = _fetch_flight = None
    _rewrite_cache = _rewrite_flight = _rate_limiter = _session_metrics = _budget_tracker = None


def _get_budget_tracker() -> DailyBudgetTracker | None:
    """Return the active daily budget tracker, or ``None`` when daily budget is disabled."""
    global _budget_tracker
    settings = get_settings()
    if settings.daily_budget_usd is None:
        return None
    if _budget_tracker is None:
        _budget_tracker = DailyBudgetTracker(settings.daily_budget_usd)
    return _budget_tracker


def _get_session_metrics() -> SessionMetrics:
    """Return the active session metrics tracker, creating it lazily."""
    global _session_metrics
    if _session_metrics is None:
        _session_metrics = SessionMetrics(start_time=_clock())
    return _session_metrics


def _get_search_cache() -> TTLCache | None:
    """Return the search response cache, or ``None`` when caching is disabled."""
    global _search_cache
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    if _search_cache is None:
        _search_cache = TTLCache(
            settings.cache_search_entries, settings.cache_search_bytes, ttl=settings.cache_search_ttl
        )
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
        _fetch_cache = TTLCache(
            settings.cache_fetch_entries, settings.cache_fetch_bytes, ttl=settings.cache_fetch_ttl
        )
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
        _rewrite_cache = TTLCache(
            settings.cache_rewrite_entries, settings.cache_rewrite_bytes, ttl=settings.cache_rewrite_ttl
        )
    return _rewrite_cache


def _get_rewrite_flight() -> SingleFlight:
    """Return the single-flight group that collapses identical concurrent query rewrites."""
    global _rewrite_flight
    if _rewrite_flight is None:
        # Cache writes are done by the leader itself so failed rewrites are not stored.
        _rewrite_flight = SingleFlight(None)
    return _rewrite_flight


def _get_rate_limiter() -> RateLimiter:
    """Return the process-wide per-endpoint rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter.from_settings(get_settings())
    return _rate_limiter


def _clear_caches() -> dict[str, int]:
    """Purge in-memory response caches (search, fetch, rewrite) and return evicted counts."""
    cleared = {
        "search_entries": len(_search_cache) if _search_cache is not None else 0,
        "fetch_entries": len(_fetch_cache) if _fetch_cache is not None else 0,
        "rewrite_entries": len(_rewrite_cache) if _rewrite_cache is not None else 0,
    }
    if _search_cache is not None:
        _search_cache.clear()
    if _fetch_cache is not None:
        _fetch_cache.clear()
    if _rewrite_cache is not None:
        _rewrite_cache.clear()
    return cleared


def _get_cache_stats() -> dict[str, Any]:
    """Return current occupancy and capacity stats across active caches."""
    return {
        "search": _search_cache.stats() if _search_cache is not None else None,
        "fetch": _fetch_cache.stats() if _fetch_cache is not None else None,
        "rewrite": _rewrite_cache.stats() if _rewrite_cache is not None else None,
    }


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


async def _limited_request(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Take one ``endpoint`` token, then call upstream once (transport retries included).

    Returns the transport envelope, or a ``rate_limited_local`` error envelope when no token
    is available in time. The token is refunded if the call is cancelled.
    """
    try:
        ticket = await _get_rate_limiter().acquire(endpoint)
    except RateLimitExceeded as exc:
        return exc.to_envelope()
    try:
        metrics = _get_session_metrics()
        metrics.requests_upstream += 1
        if endpoint == SEARCH:
            metrics.searches_upstream += 1
        elif endpoint == FETCH:
            metrics.fetches_upstream += 1
        return await _request("POST", path, payload=payload, timeout=timeout, deadline=deadline)
    except asyncio.CancelledError:
        ticket.refund()
        raise


async def _cached_call(
    cache: TTLCache | None,
    flight: SingleFlight,
    key: str,
    leader: Callable[[], Awaitable[dict[str, Any]]],
    *,
    refresh: bool,
) -> tuple[dict[str, Any], bool]:
    """Serve ``key`` from ``cache`` or through a single-flight ``leader`` call.

    Returns ``(record, cached)``; the record is an error envelope when the leader failed.
    """
    record = None if refresh or cache is None else cache.get(key)
    if record is not None:
        _get_session_metrics().cache_hits += 1
        return record, True
    result = await flight.run(key, leader, refresh=refresh)
    return result.value, False


def _timed(
    tool: Callable[_P, Awaitable[dict[str, Any]]],
) -> Callable[_P, Awaitable[dict[str, Any]]]:
    """Add ``elapsed_ms``, the call's wall time on the monotonic clock, to every envelope ``tool`` returns.

    Success and error envelopes alike get it, on a copy; a cache hit reports its own (small) time.
    ``functools.wraps`` keeps the signature and docstring that FastMCP turns into the tool schema.
    """

    @functools.wraps(tool)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> dict[str, Any]:
        started = _clock()
        envelope = await tool(*args, **kwargs)
        # A copy: the envelope may be shared (single-flight followers, the upstream's error dict).
        return {**envelope, "elapsed_ms": max(0, round((_clock() - started) * 1000))}

    return wrapper


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
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc, path, urllib.parse.urlencode(params), ""))


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

        if (key is not None and key in seen) or (title_site is not None and title_site in seen_title_sites):
            removed += 1
            continue

        if key is not None:
            seen.add(key)
        if title_site is not None:
            seen_title_sites.add(title_site)
        kept.append(item)

    return kept, removed


def _upstream_count(count: int, dedupe: bool) -> int:
    """Number of results to request upstream: ``count`` plus a dedupe margin, capped at the API max."""
    if not dedupe:
        return count
    extra = max(OVERFETCH_MIN, math.ceil(count * OVERFETCH_RATIO))
    return min(count + extra, MAX_COUNT)


def _url_host(url: object) -> str | None:
    """Return the URL host without ``www.``, or None when the URL has none."""
    if not isinstance(url, str):
        return None
    try:
        host = urllib.parse.urlsplit(url.strip()).hostname
    except ValueError:
        return None
    return host.removeprefix("www.") if host else None


TITLE_SEPARATOR = " - "


def _collapse_repeated_title(title: Any) -> Any:
    """Collapse a title made of two identical halves joined by `` - `` (``"X - X"`` becomes ``"X"``).

    Some pages (Reddit threads, for one) repeat the title around the separator. Any other title,
    including ``"A - B"`` and non-string values, is returned unchanged.
    """
    if not isinstance(title, str):
        return title
    stripped = title.strip()
    half, remainder = divmod(len(stripped) - len(TITLE_SEPARATOR), 2)
    if remainder or half < 1 or stripped[half : half + len(TITLE_SEPARATOR)] != TITLE_SEPARATOR:
        return title
    first, second = stripped[:half], stripped[half + len(TITLE_SEPARATOR) :]
    return first if first == second else title


def _title_or_host(item: dict) -> str | None:
    """Return the cleaned result title, or the URL host without ``www.`` when the title is blank."""
    title = _collapse_repeated_title(item.get("title"))
    if isinstance(title, str) and title.strip():
        return title
    return _url_host(item.get("url")) or title


# One alternation, scanned left to right, so a backslash escape is consumed before any backtick
# it escapes can open an inline code span. Code (fenced blocks, then inline spans) is matched
# only to be skipped; a fence without a closing line runs to the end of the text.
_MARKDOWN_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\(?P<char>[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])",
    re.DOTALL | re.MULTILINE,
)


def _unescape_markdown(text: str) -> str:
    """Remove CommonMark backslash escapes from ``text``, leaving code untouched.

    Any ASCII punctuation character may be backslash-escaped in CommonMark, and the fetch
    backend escapes many of them (``\\(``, ``\\-``, ``\\[``, ``\\/``...), which only costs tokens
    and hurts readability. Fenced code blocks and inline code spans are skipped because a
    backslash there is literal. An escaped backslash (``\\\\``) is reduced to a single one.
    A ``\\#`` at the start of a line is unescaped like any other, so an escaped heading marker
    becomes a heading again.
    """
    return _MARKDOWN_TOKEN.sub(lambda match: match["char"] or match[0], text)


# Code first, so escapes inside fenced blocks and inline spans are matched as part of the code and kept.
_JSON_ESCAPE_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\u(?P<high>[dD][89abAB][0-9a-fA-F]{2})\\u(?P<low>[dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u(?P<unit>[0-9a-fA-F]{4})"
    r"|(?P<slash>\\/)",
    re.DOTALL | re.MULTILINE,
)
_JSON_ESCAPE_HINT = re.compile(r"\\u[0-9a-fA-F]{4}|\\/")


def _decode_json_escape(match: re.Match[str]) -> str:
    if match["slash"]:
        return "/"
    if match["high"]:
        high, low = int(match["high"], 16), int(match["low"], 16)
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    if match["unit"]:
        code = int(match["unit"], 16)
        # Control characters and lone surrogates stay escaped: they are not text worth restoring.
        if code < 0x20 or 0xD800 <= code <= 0xDFFF:
            return match[0]
        return chr(code)
    return match[0]


def _decode_json_escapes(text: str) -> str:
    """Decode JSON string escapes (``\\uXXXX``, surrogate pairs, ``\\/``) left in search-index text.

    The search index sometimes returns a page's text still JSON-escaped, so ``ó`` arrives as
    the six characters ``\\u00f3`` and ``</p>`` as ``<\\/p>``. Fenced code blocks and inline code
    spans are skipped, so a literal escape shown in code is kept; text without such sequences
    is returned unchanged.
    """
    if not _JSON_ESCAPE_HINT.search(text):
        return text
    return _JSON_ESCAPE_TOKEN.sub(_decode_json_escape, text)


# Simple attribute-less block and inline tags that leak into index content, snippets and summaries.
_HTML_TAG = re.compile(r"</?(?:p|br|h[1-6]|div|span|li|ul|ol|strong|em|b|i)\s*/?>", re.IGNORECASE)
_TAG_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|(?P<tags>(?:[ \t]*" + _HTML_TAG.pattern + r"[ \t]*)+)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _strip_html_tags(text: str) -> str:
    """Replace runs of simple HTML tags (``<p>``, ``<br>``, ``</h2>``...) with a single space.

    Only attribute-less tags from a short list are touched, outside code, so text that merely
    mentions angle brackets (``a < b``, ``<T>``) stays as it is.
    """
    if "<" not in text:
        return text
    stripped = _TAG_TOKEN.sub(lambda match: " " if match["tags"] else match[0], text)
    return stripped.strip() if stripped != text else text


def _clean_index_text(text: str) -> str:
    """Decode JSON escapes in a search snippet or summary, then drop leftover simple HTML tags."""
    return _strip_html_tags(_decode_json_escapes(text))


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
            "POST", "/v1/chat/completions", payload=payload, timeout=CHAT_TIMEOUT, attempts=1
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
    except Exception as exc:  # broad on purpose: rewriting is best-effort and must never fail a search
        logger.debug("Query rewrite failed: %s", redact(f"{type(exc).__name__}: {exc}"))
        return None


async def _cached_rewrite(query: str) -> str | None:
    """Rewrite ``query`` through the rewrite cache and single-flight group.

    Identical concurrent rewrites share one model call; failed rewrites are not remembered.
    """
    cache = _get_rewrite_cache()
    key = (_base_url(), REWRITE_MODEL, _sha256(SYSTEM_PROMPT), query)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    async def leader() -> str | None:
        rewritten = await _rewrite_query(query)
        if cache is not None and rewritten is not None:
            cache.set(key, rewritten)
        return rewritten

    return (await _get_rewrite_flight().run(key, leader)).value


async def _search_upstream(payload: dict[str, Any], query_original: str | None, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``search`` token, call upstream, cache a usable answer.

    Returns ``{"data": ..., "query_original": ...}`` on success or an error envelope.
    """
    budget = _get_budget_tracker()
    if budget is not None:
        allowed, spent, limit = await budget.check_budget(COST_PER_REQUEST)
        if not allowed:
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${limit:.2f} exceeded (spent ${spent:.4f}).",
                402,
            )
    result = await _limited_request(SEARCH, SEARCH_PATH, payload, SEARCH_TIMEOUT)
    if not result.get("ok"):
        return result
    if budget is not None:
        await budget.record_spend(COST_PER_REQUEST)
    record = {"data": result["data"], "query_original": query_original}
    cache = _get_search_cache()
    if cache is not None and result["data"].get("results"):
        cache.set(key, record)
    return record


def _check_search_args(
    query: str, count: int, lang: str | None, filter_ops: str | None = None
) -> dict[str, Any] | None:
    """Validate query, count, lang and filter budget; return an error envelope or None."""
    if not query:
        return _fail("empty_query", "query must not be empty.")
    ops_len = len(filter_ops) if filter_ops else 0
    budget = MAX_QUERY - (ops_len + 1 if ops_len else 0)
    if len(query) > budget:
        if ops_len:
            return _fail(
                "query_too_long",
                f"query must be at most {budget} characters when filters ({ops_len} chars) are applied.",
            )
        return _fail("query_too_long", f"query must be at most {MAX_QUERY} characters.")
    if not 1 <= count <= MAX_COUNT:
        return _fail("bad_count", f"count must be between 1 and {MAX_COUNT}.")
    if lang and not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", lang):
        return _fail("bad_lang", "lang must look like 'es' or 'pt-BR'.")
    return None


async def _maybe_rewrite(
    query: str, ori_question: str | None, rewrite: bool, lang: str | None
) -> tuple[str, str | None, str | None, bool]:
    """Apply the optional query rewrite.

    Returns ``(query, ori_question, query_original, rewrite_failed)``; ``query_original`` is
    set only when the rewrite changed the query, and then also seeds a missing ``ori_question``.
    """
    if not (rewrite and lang is None and _api_key()):
        return query, ori_question, None, False
    rewritten = await _cached_rewrite(query)
    if rewritten is None:
        return query, ori_question, None, True
    if rewritten == query:
        return query, ori_question, None, False
    return rewritten, ori_question or query[:MAX_QUERY], query, False


def _build_search_payload(
    query: str,
    count: int,
    dedupe: bool,
    need_content: bool,
    ori_question: str | None,
    *,
    localization: str | None = None,
) -> dict[str, Any]:
    """Build the upstream search payload (``query`` already carries any ``lang:`` operator)."""
    payload: dict[str, Any] = {
        "query": query,
        "count": _upstream_count(count, dedupe),
        "need_content": need_content,
    }
    if ori_question:
        payload["ori_question"] = ori_question[:MAX_QUERY]
    if localization and localization.strip():
        payload["localization"] = localization.strip()
    return payload


# A calendar date written as YYYY-MM-DD on its own (optionally followed by an ISO time), not part
# of a longer number, identifier or version string.
_ISO_DATE = re.compile(r"(?<![\w./-])(\d{4})-(\d{2})-(\d{2})(?![0-9]|[-./][0-9]|[A-SU-Za-z])")
# How far into a page's text a publication date (ISO or month name) is looked for, and how much raw
# text is cleaned first. Journal pages often open with a few hundred characters of site navigation,
# authors and affiliations before the issue line, so the window reaches past them.
CONTENT_DATE_WINDOW = 600
_CONTENT_DATE_RAW = 2000
MIN_CONTENT_YEAR = 1990


def _date_in(text: str, today: date | None = None) -> str | None:
    """Return the first valid ``YYYY-MM-DD`` date in ``text`` (year 1990 to next year), or None."""
    max_year = (today or datetime.now(UTC).date()).year + 1
    for match in _ISO_DATE.finditer(text):
        year, month, day = (int(part) for part in match.groups())
        if not MIN_CONTENT_YEAR <= year <= max_year:
            continue
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


# Full month names, English and Polish (nominative "wrzesień" and genitive "września", the form used
# after a day number), mapped to the month number. Abbreviations are not accepted.
_MONTH_NAMES: dict[str, int] = {
    name: number
    for number, names in enumerate(
        (
            ("january", "styczeń", "stycznia"),
            ("february", "luty", "lutego"),
            ("march", "marzec", "marca"),
            ("april", "kwiecień", "kwietnia"),
            ("may", "maj", "maja"),
            ("june", "czerwiec", "czerwca"),
            ("july", "lipiec", "lipca"),
            ("august", "sierpień", "sierpnia"),
            ("september", "wrzesień", "września"),
            ("october", "październik", "października"),
            ("november", "listopad", "listopada"),
            ("december", "grudzień", "grudnia"),
        ),
        start=1,
    )
    for name in names
}
# "September 2025", "wrzesień 2025", with an optional day: "15 września 2025", "September 15, 2025".
_MONTH_DATE = re.compile(
    r"(?<![\w.,/-])(?:(?P<day>\d{1,2})\s+)?"
    r"(?P<month>" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")"
    r"(?:\s+(?P<day_after>\d{1,2}),)?\s+(?P<year>\d{4})(?![\w-]|[.,/]\d)",
    re.IGNORECASE,
)
# The year-first order used in PubMed-style citations: "2025 September".
_YEAR_MONTH_DATE = re.compile(
    r"(?<![\w.,/-])(?P<year>\d{4})\s+"
    r"(?P<month>" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)


def _month_date_in(text: str, today: date | None = None) -> str | None:
    """Return the first valid month-name date in ``text`` as ``YYYY-MM-DD``, or None.

    A full English or Polish month name next to a year (1990 to next year), as "September 2025" or
    "2025 September", gives the first of that month; a day written with a month-first date
    ("15 września 2025", "September 15, 2025") gives that day. The earliest match wins.
    """
    max_year = (today or datetime.now(UTC).date()).year + 1
    matches = [*_MONTH_DATE.finditer(text), *_YEAR_MONTH_DATE.finditer(text)]
    for match in sorted(matches, key=lambda found: found.start()):
        year = int(match["year"])
        if not MIN_CONTENT_YEAR <= year <= max_year:
            continue
        groups = match.groupdict()
        day = groups.get("day") or groups.get("day_after")
        try:
            return date(year, _MONTH_NAMES[match["month"].lower()], int(day) if day else 1).isoformat()
        except ValueError:
            continue
    return None


def _content_date(title: Any, text: Any, today: date | None = None) -> str | None:
    """Find a publication date in a result's title or the first characters of its text.

    Used only when the upstream gave no date. The text is cleaned the way content is before the
    first ``CONTENT_DATE_WINDOW`` characters are searched. An ISO ``YYYY-MM-DD`` date anywhere in
    the title or that window is preferred; otherwise the first month-name date is used. A bare
    year is never taken as a date.
    """
    candidates: list[str] = []
    if isinstance(title, str):
        candidates.append(title)
    if isinstance(text, str) and text:
        head = _decode_json_escapes(_unescape_markdown(text[:_CONTENT_DATE_RAW]))
        candidates.append(head[:CONTENT_DATE_WINDOW])
    for find in (_date_in, _month_date_in):
        if found := next((hit for hit in (find(c, today) for c in candidates) if hit), None):
            return found
    return None


def _dated(upstream: Any, title: Any, text: Any) -> tuple[Any, str | None]:
    """Return ``(date, date_source)``: the upstream date, else one found in the content, else None."""
    if upstream:
        return upstream, "upstream"
    found = _content_date(title, text)
    return (found, "content") if found else (None, None)


def _result_entry(item: dict[str, Any]) -> dict[str, Any]:
    """Map one upstream result to its metadata fields, with the host as a blank-title fallback."""
    when, date_source = _dated(
        item.get("date"), item.get("title"), item.get("content") or item.get("summary")
    )
    entry: dict[str, Any] = {
        "title": _title_or_host(item),
        "url": item.get("url"),
        "date": when,
        "date_source": date_source,
        "authority": item.get("authority"),
        "rerank_score": item.get("rerank_score"),
        "site": item.get("site_name"),
    }
    snippet = _clean_index_text(item.get("snippet") or "")
    if snippet:
        entry["snippet"] = snippet
    return entry


def _summary_cap(summary_chars: int | None, need_content: bool, per_result: int) -> int:
    """Return the per-result summary cap: ``summary_chars`` (or its default) bounded by ``per_result``.

    The default is ``SEARCH_SUMMARY_CHARS`` without page content, and ``per_result`` with it (the
    summary is then only a fallback for results that came back without content). 0 omits summaries.
    """
    if summary_chars is None:
        summary_chars = per_result if need_content else SEARCH_SUMMARY_CHARS
    return min(summary_chars, per_result)


def _apply_char_budget(
    entry: dict[str, Any],
    item: dict[str, Any],
    need_content: bool,
    per_result: int,
    remaining: int,
    summary_cap: int,
    strip_boilerplate: bool = True,
) -> tuple[int, bool]:
    """Add the result's content (or summary) to ``entry``, cut to the per-result and total budgets.

    Content is unescaped (Markdown escapes, then any JSON escapes left by the search index), rid of
    leftover simple HTML tags outside code and, with ``strip_boilerplate``, of known site-UI lines before it
    is capped at ``per_result``; a summary, used when there is no content, is capped at
    ``summary_cap`` and is left out entirely when that cap is 0. With content, ``chars`` is the
    length returned.
    Returns ``(remaining, truncated)``: the budget left afterwards and whether any text was cut.
    """
    body = item.get("content") if need_content else None
    if body:
        cleaned = _strip_html_tags(_decode_json_escapes(_unescape_markdown(body)))
        if strip_boilerplate:
            cleaned = remove_boilerplate(cleaned, item.get("url"))
        entry["content"], cut = _truncate(cleaned, max(0, min(per_result, remaining)))
        entry["chars"] = len(entry["content"])
        return remaining - len(entry["content"]), cut
    summary = _clean_index_text(item.get("summary") or "")
    if summary and summary_cap > 0:
        entry["summary"], cut = _truncate(summary, max(0, min(summary_cap, remaining)))
        return remaining - len(entry["summary"]), cut
    return remaining, False


def _shape_results(
    raw: list[dict[str, Any]],
    need_content: bool,
    per_result: int,
    total: int,
    summary_cap: int,
    strip_boilerplate: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Shape upstream results under the character budgets; return ``(results, truncated)``.

    With ``need_content`` set, results stop once the total budget is spent.
    """
    remaining = total
    shaped: list[dict[str, Any]] = []
    truncated = False
    for item in raw:
        entry = _result_entry(item)
        remaining, cut = _apply_char_budget(
            entry, item, need_content, per_result, remaining, summary_cap, strip_boilerplate
        )
        truncated = truncated or cut
        shaped.append(entry)
        if need_content and remaining <= 0:
            break
    return shaped, truncated


def _search_notes(
    used: str,
    query_original: str | None,
    rewrite_failed: bool,
    removed: int,
    trimmed: bool,
    lang: str | None,
    truncated: bool,
    filter_ops: str | None = None,
) -> list[str]:
    """Assemble the (redacted) notes of a search envelope, in their fixed order."""
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
    if filter_ops:
        notes.append(f"Filters applied: {filter_ops}")
    if truncated:
        notes.append(TRUNCATED_NOTE)
    return [redact(note) for note in notes]


async def _search(
    query: str,
    count: int,
    ori_question: str | None,
    need_content: bool,
    lang: str | None,
    max_chars_per_result: int,
    max_total_chars: int,
    dedupe: bool = True,
    rewrite: bool = False,
    *,
    localization: str | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Run one validated, cached search and shape the envelope shared by the search tools.

    ``summary_chars`` caps the summary of results without content (see ``_summary_cap``).
    ``strip_boilerplate`` removes known site-UI lines from page content; it only shapes the
    response, so the cache key and the cached upstream data are the same either way.
    """
    if localization is not None:
        if not isinstance(localization, str):
            return _fail("bad_request", "localization must be a string.")
        if len(localization.strip()) > 64:
            return _fail("bad_request", "localization exceeds maximum length of 64 characters.")

    query = (query or "").strip()
    ops_string, filter_code, filter_err = compile_filters(
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        exclude_terms=exclude_terms,
        exact_phrases=exact_phrases,
        after=after,
        before=before,
    )
    if filter_code:
        return _fail(filter_code, filter_err or "Invalid search filters.")

    rejection = _check_search_args(query, count, lang, filter_ops=ops_string)
    if rejection is not None:
        return rejection
    if summary_chars is not None and summary_chars < 0:
        return _fail("bad_request", "summary_chars must be at least 0.")

    query, ori_question, query_original, rewrite_failed = await _maybe_rewrite(
        query, ori_question, rewrite, lang
    )
    ops_len = len(ops_string) if ops_string else 0
    budget = MAX_QUERY - (ops_len + 1 if ops_len else 0)
    if len(query) > budget:
        query = query[:budget]

    used, trimmed = _with_lang(query, lang)
    if ops_string:
        if used:
            combined = f"{used} {ops_string}"
            if len(combined) > MAX_QUERY:
                room = MAX_QUERY - len(ops_string) - 1
                used = f"{used[:room]} {ops_string}"
                trimmed = True
            else:
                used = combined
        else:
            used = ops_string

    payload = _build_search_payload(
        used, count, dedupe, need_content, ori_question, localization=localization
    )
    key = _search_cache_key(payload)
    record, cached = await _cached_call(
        _get_search_cache(),
        _get_search_flight(),
        key,
        lambda: _search_upstream(payload, query_original, key),
        refresh=refresh,
    )
    if record.get("ok") is False:
        return record

    raw = record["data"].get("results") or []
    removed = 0
    if dedupe:
        raw, removed = _dedupe_results(raw)
        raw = raw[:count]
    summary_cap = _summary_cap(summary_chars, need_content, max_chars_per_result)
    shaped, truncated = _shape_results(
        raw, need_content, max_chars_per_result, max_total_chars, summary_cap, strip_boilerplate
    )

    cost = 0.0 if cached else COST_PER_REQUEST
    envelope: dict[str, Any] = {
        "ok": True,
        "query_used": used,
        "count": len(shaped),
        "need_content": need_content,
        "deduped": removed,
        "cached": cached,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": shaped,
        "notes": _search_notes(
            used, query_original, rewrite_failed, removed, trimmed, lang, truncated, filter_ops=ops_string
        ),
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope


async def _fetch_upstream(url: str, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``fetch`` token, call upstream, cache a non-empty page.

    Returns ``{"data": ...}`` with the raw upstream page on success or an error envelope.
    """
    budget = _get_budget_tracker()
    if budget is not None:
        allowed, spent, limit = await budget.check_budget(COST_PER_REQUEST)
        if not allowed:
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${limit:.2f} exceeded (spent ${spent:.4f}).",
                402,
            )
    result = await _limited_request(FETCH, FETCH_PATH, {"url": url}, FETCH_TIMEOUT)
    if not result.get("ok"):
        return result
    if budget is not None:
        await budget.record_spend(COST_PER_REQUEST)
    record = {"data": result["data"]}
    cache = _get_fetch_cache()
    if cache is not None and result["data"].get("content"):
        cache.set(key, record)
    return record


@_timed
async def deepseek_search(
    query: str,
    count: int = 5,
    ori_question: str | None = None,
    *,
    need_content: bool = True,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 30000,
    dedupe: bool = True,
    rewrite: bool = False,
    localization: str | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Search the web with DeepSeek index and return page text, snippets, and metadata.

    Primary search and retrieval tool. Returns snippets (need_content=False) or full text.

    Args:
        query: Query, max 128 chars (site:, -site:, "phrase", after:, before:).
        count: Results count, 1-50 (default 5).
        ori_question: Natural question improving ranking (max 128 chars).
        need_content: Inline full text (default True; False for snippets).
        lang: Language filter (e.g. 'en', 'es').
        max_chars_per_result: Cap per result (default 4000).
        max_total_chars: Total content cap (default 30000).
        dedupe: Collapse duplicate URLs (default True).
        rewrite: Query rewrite via deepseek-flash (default False).
        localization: Geo-targeting IP.
        exclude_terms: Exclude terms list.
        exact_phrases: Exact phrases list.
        after: Date on/after (YYYY-MM-DD).
        before: Date on/before (YYYY-MM-DD).
        include_domains: Restrict domains list.
        exclude_domains: Exclude domains list.
        refresh: Bypass cache (default False).
        summary_chars: Summary cap without content.
        strip_boilerplate: Remove site UI lines (default True).
    """
    return await _search(
        query,
        count,
        ori_question,
        need_content,
        lang,
        max_chars_per_result,
        max_total_chars,
        dedupe,
        rewrite,
        localization=localization,
        exclude_terms=exclude_terms,
        exact_phrases=exact_phrases,
        after=after,
        before=before,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        refresh=refresh,
        summary_chars=summary_chars,
        strip_boilerplate=strip_boilerplate,
    )


@_timed
async def deepseek_search_many(
    queries: list[str],
    *,
    count: int = 5,
    need_content: bool = False,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 30000,
    dedupe: bool = True,
    rewrite: bool = False,
    localization: str | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Search the web with multiple queries concurrently and return ranked results per query.

    Batch multi-query search across 1-5 queries in parallel. Returns snippets (default)
    or full text. Deduplicates identical queries and preserves query order.

    Args:
        queries: 1-5 search queries, max 128 chars each.
        count: Results count per query, 1-50 (default 5).
        need_content: Inline full text (default False for snippets).
        lang: Language filter (e.g. 'en', 'es').
        max_chars_per_result: Cap per result (default 4000).
        max_total_chars: Total content cap across queries (default 30000).
        dedupe: Collapse duplicate URLs (default True).
        rewrite: Query rewrite via deepseek-flash (default False).
        localization: Geo-targeting IP.
        exclude_terms: Exclude terms list.
        exact_phrases: Exact phrases list.
        after: Date on/after (YYYY-MM-DD).
        before: Date on/before (YYYY-MM-DD).
        include_domains: Restrict domains list.
        exclude_domains: Exclude domains list.
        refresh: Bypass cache (default False).
        summary_chars: Summary cap without content.
        strip_boilerplate: Remove site UI lines (default True).
        deadline: Timeout in seconds for all queries.
    """
    if not isinstance(queries, list) or not queries:
        return _fail("bad_request", "queries must be a non-empty list of strings.")
    if len(queries) > 5:
        return _fail("bad_request", "queries must contain between 1 and 5 entries.")
    if not all(isinstance(q, str) for q in queries):
        return _fail("bad_request", "all items in queries must be strings.")
    if count < 1 or count > MAX_COUNT:
        return _fail("bad_count", f"count must be between 1 and {MAX_COUNT}.")
    if max_chars_per_result < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars_per_result and max_total_chars must be at least 1.")

    unique_queries: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for q in queries:
        trimmed = q.strip()
        if not trimmed:
            continue
        if trimmed in seen:
            duplicates += 1
            continue
        seen.add(trimmed)
        unique_queries.append(trimmed)

    if not unique_queries:
        return _fail("bad_request", "queries must contain at least one non-empty query.")

    per_query_chars = max(1000, max_total_chars // len(unique_queries))

    async def _run_one(q: str) -> dict[str, Any]:
        return await _search(
            q,
            count=count,
            ori_question=None,
            need_content=need_content,
            lang=lang,
            max_chars_per_result=max_chars_per_result,
            max_total_chars=per_query_chars,
            dedupe=dedupe,
            rewrite=rewrite,
            localization=localization,
            exclude_terms=exclude_terms,
            exact_phrases=exact_phrases,
            after=after,
            before=before,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            refresh=refresh,
            summary_chars=summary_chars,
            strip_boilerplate=strip_boilerplate,
        )

    tasks = [_run_one(q) for q in unique_queries]
    timeout = deadline if deadline is not None else SEARCH_TIMEOUT
    try:
        raw_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout,
        )
    except TimeoutError:
        return _fail("timeout", f"Search batch timed out after {timeout} seconds.")

    results: list[dict[str, Any]] = []
    requests_used = 0
    cached_count = 0
    failed_count = 0
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate query/queries")

    for q, res in zip(unique_queries, raw_results, strict=True):
        if isinstance(res, Exception):
            failed_count += 1
            results.append(
                {
                    "query": q,
                    "ok": False,
                    "count": 0,
                    "results": [],
                    "cached": False,
                    "error": _fail("network_error", str(res))["error"],
                    "notes": [],
                }
            )
        elif isinstance(res, dict) and res.get("ok"):
            is_cached = res.get("cached", False)
            if is_cached:
                cached_count += 1
            else:
                requests_used += 1
            results.append(
                {
                    "query": q,
                    "ok": True,
                    "count": len(res.get("results") or []),
                    "results": res.get("results") or [],
                    "cached": is_cached,
                    "notes": res.get("notes") or [],
                }
            )
            for n in res.get("notes") or []:
                if n not in notes:
                    notes.append(n)
        else:
            failed_count += 1
            err = res.get("error") if isinstance(res, dict) else None
            results.append(
                {
                    "query": q,
                    "ok": False,
                    "count": 0,
                    "results": [],
                    "cached": False,
                    "error": err,
                    "notes": res.get("notes", []) if isinstance(res, dict) else [],
                }
            )

    cost = round(requests_used * COST_PER_REQUEST, 5)
    return {
        "ok": True,
        "count": len(results),
        "total_results": sum(r["count"] for r in results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": results,
        "searches": results,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }


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
    localization: str | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Deprecated alias for deepseek_search(..., need_content=False)."""
    return await deepseek_search(
        query,
        count=count,
        ori_question=ori_question,
        need_content=need_content,
        lang=lang,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        dedupe=dedupe,
        rewrite=rewrite,
        localization=localization,
        exclude_terms=exclude_terms,
        exact_phrases=exact_phrases,
        after=after,
        before=before,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        refresh=refresh,
        summary_chars=summary_chars,
        strip_boilerplate=strip_boilerplate,
    )


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
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Deprecated alias for deepseek_search(..., need_content=True)."""
    return await deepseek_search(
        query,
        count=count,
        ori_question=ori_question,
        need_content=True,
        lang=lang,
        max_chars_per_result=max_chars_per_result,
        max_total_chars=max_total_chars,
        dedupe=dedupe,
        rewrite=rewrite,
        after=after,
        before=before,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        refresh=refresh,
        summary_chars=summary_chars,
        strip_boilerplate=strip_boilerplate,
    )


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


def _check_fetch_url(url: str) -> dict[str, Any] | None:
    """Validate a stripped fetch URL; return an error envelope if it is unusable, else ``None``."""
    if not url:
        return _fail("empty_url", "url must not be empty.")
    if len(url) > MAX_URL:
        return _fail("url_too_long", f"url must be at most {MAX_URL} characters.")
    if not re.match(r"^https?://", url, re.I):
        return _fail("bad_url", "url must start with http:// or https://.")
    return _reject_url(url)


@_timed
async def deepseek_fetch(
    url: str,
    clean: bool = True,
    max_chars: int = 20000,
    *,
    offset: int | None = None,
    limit: int | None = None,
    find_text: str | None = None,
    query: str | None = None,
    section: int | None = None,
    max_snippets: int = 3,
    refresh: bool = False,
    strip_boilerplate: bool = True,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Fetch one known URL and return its title and Markdown content.

    Retrieves Markdown for a known URL. For multiple URLs use deepseek_fetch_many;
    for structured data use deepseek_extract. Auto-resolves on 404 via search index.

    Args:
        url: Absolute http(s) URL, max 8192 chars.
        clean: Remove Markdown backslash escapes (default True).
        max_chars: Content cap (default 20000).
        offset: Start reading offset (0-based).
        limit: Max characters from offset.
        find_text: Find substring and start offset there.
        query: Query to extract ranked snippets.
        section: 1-based document section index.
        max_snippets: Max snippets when query is set (default 3).
        refresh: Bypass cache (default False).
        strip_boilerplate: Remove site UI lines (default True).
        fallback_to_search: Auto-search fallback on 404 (default True).
    """
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
    return await _fetch(
        url,
        clean,
        max_chars,
        offset=offset,
        limit=limit,
        find_text=find_text,
        query=query,
        section=section,
        max_snippets=max_snippets,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
        fallback_to_search=fallback_to_search,
    )


_SEARCH_ADVICE = (
    "Direct fetch is unavailable for this document: use deepseek_search instead, "
    "which returns page content via the search index."
)


# A PDF served without an extension under a /pdf/<id> path, as arXiv, bioRxiv and medRxiv do.
_PDF_ID_PATH = re.compile(r"/pdf/[^/]+/?$")


def _needs_search_advice(url: str) -> bool:
    """True for a PDF path (``.pdf`` or ``/pdf/<id>``; query and fragment ignored) or a Reddit host."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    return (
        parts.path.lower().endswith(".pdf")
        or _PDF_ID_PATH.search(parts.path.lower()) is not None
        or host in ("reddit.com", "redd.it")
        or host.endswith((".reddit.com", ".redd.it"))
    )


def _with_search_advice(error: dict[str, Any], url: str) -> dict[str, Any]:
    """Append the search-and-read advice to a 404 for a PDF or Reddit URL; code and status stay."""
    detail = error.get("error")
    if not isinstance(detail, dict) or detail.get("status") != 404 or not _needs_search_advice(url):
        return error
    message = redact(f"{detail.get('message', '')} — {_SEARCH_ADVICE}")
    return {**error, "error": {**detail, "message": message}}


async def _search_fallback_fetch(
    url: str,
    *,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any] | None:
    """Attempt to retrieve page content via DeepSeek search when fetch returns 404 for PDF/Reddit.

    Returns ``{"data": {"title": ..., "content": ...}, "fallback_from_search": True}`` on success,
    or ``None`` if search yielded no matching results for the target domain or had empty content.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        target_host = (parts.hostname or "").lower()
    except ValueError:
        return None

    host_for_site = target_host[4:] if target_host.startswith("www.") else target_host
    path_segments = [s for s in parts.path.split("/") if s and s not in ("pdf", "comments", "r")]
    query_terms = " ".join(path_segments[-3:]) if path_segments else ""
    search_query = f"site:{host_for_site} {query_terms}".strip()[:MAX_QUERY]

    search_res = await _search(
        search_query,
        count=2,
        ori_question=None,
        need_content=True,
        lang=None,
        max_chars_per_result=40000,
        max_total_chars=40000,
        dedupe=True,
        rewrite=False,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
    )
    if not search_res.get("ok"):
        return None

    for res in search_res.get("results") or []:
        res_url = res.get("url") or ""
        try:
            res_parts = urllib.parse.urlsplit(res_url)
            res_host = (res_parts.hostname or "").lower()
        except ValueError:
            continue
        if not (
            res_host == target_host
            or res_host.endswith(f".{target_host}")
            or target_host.endswith(f".{res_host}")
        ):
            continue

        raw_content = res.get("content") or ""
        if raw_content.strip():
            return {
                "data": {
                    "title": res.get("title") or "",
                    "content": raw_content,
                },
                "fallback_from_search": True,
            }
    return None


async def _fetch(
    url: str,
    clean: bool,
    max_chars: int,
    *,
    offset: int | None = None,
    limit: int | None = None,
    find_text: str | None = None,
    query: str | None = None,
    section: int | None = None,
    max_snippets: int = 3,
    refresh: bool,
    strip_boilerplate: bool = True,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Fetch an already validated ``url`` through the cache, single-flight and limiter.

    Cleaning, snippets, sectional reading, pagination and ``strip_boilerplate`` shape the cached page and
    never change the cache key.

    Returns the ``deepseek_fetch`` envelope, or an error envelope from upstream or the limiter.
    """
    if offset is not None and offset < 0:
        return _fail("bad_request", "offset must be at least 0.")
    if limit is not None and limit <= 0:
        return _fail("bad_request", "limit must be greater than 0.")

    key = _fetch_cache_key(url)
    record, cached = await _cached_call(
        _get_fetch_cache(),
        _get_fetch_flight(),
        key,
        lambda: _fetch_upstream(url, key),
        refresh=refresh,
    )
    fallback_used = False
    if record.get("ok") is False:
        if fallback_to_search and record.get("error", {}).get("status") == 404 and _needs_search_advice(url):
            fallback_record = await _search_fallback_fetch(
                url,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
            )
            if fallback_record is not None:
                record = fallback_record
                cached = False
                fallback_used = True
                fetch_cache = _get_fetch_cache()
                if fetch_cache is not None:
                    fetch_cache.set(key, record)
            else:
                return _with_search_advice(record, url)
        else:
            return _with_search_advice(record, url)

    data = record["data"]
    content = data.get("content") or ""
    if clean:
        content = _unescape_markdown(content)
    only_boilerplate = False
    if strip_boilerplate:
        stripped = remove_boilerplate(content, url)
        only_boilerplate = bool(content.strip()) and not stripped
        content = stripped

    total_sections: int | None = None
    section_has_more: bool | None = None
    section_index: int | None = None
    snippets_count: int | None = None

    if section is not None:
        if section < 1:
            return _fail("bad_request", "section must be at least 1.")
        sections = split_sections(content)
        total_sections = len(sections)
        if total_sections == 0:
            content = ""
        elif section > total_sections:
            return _fail(
                "bad_request", f"section {section} out of bounds; document has {total_sections} section(s)."
            )
        else:
            section_index = section
            content = sections[section - 1]["content"]
            section_has_more = section < total_sections
    elif query:
        content, snippets_count = extract_snippets(content, query, max_snippets=max_snippets)

    total_chars = len(content)
    notes = _fetch_notes(content, only_boilerplate)

    if find_text:
        target = find_text.strip()
        if target:
            search_start = offset if offset is not None else 0
            idx = content.lower().find(target.lower(), search_start)
            if idx >= 0:
                offset = idx
            else:
                notes.append(f"find_text '{target}' not found in page content.")
                content = ""

    is_windowed = offset is not None or limit is not None or find_text is not None
    if is_windowed:
        start = offset if offset is not None else 0
        effective_limit = min(limit if limit is not None else max_chars, max_chars)
        content = content[start : start + effective_limit]
        window_has_more = (start + len(content)) < total_chars
        truncated = window_has_more
    else:
        start = 0
        content, truncated = _truncate(content, max_chars)
        window_has_more = None

    if fallback_used:
        cost = 2 * COST_PER_REQUEST
    else:
        cost = 0.0 if cached else COST_PER_REQUEST
    if fallback_used or record.get("fallback_from_search"):
        notes.append("Content retrieved via DeepSeek search index (direct fetch unavailable).")
    envelope: dict[str, Any] = {
        "ok": True,
        "url": url,
        "title": _collapse_repeated_title(data.get("title")),
        "content": content,
        "chars": len(content),
        "total_chars": total_chars,
        "truncated": truncated,
        "cached": cached,
        "cost_usd": cost,
        "untrusted_content": True,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if section is not None:
        envelope["section"] = section_index
        envelope["total_sections"] = total_sections
        envelope["has_more"] = section_has_more
    elif is_windowed:
        envelope["offset"] = start
        envelope["has_more"] = window_has_more
        if find_text:
            envelope["find_text"] = find_text
    if query:
        envelope["query"] = query
        envelope["snippets_count"] = snippets_count
    return envelope


def _fetch_notes(content: str, only_boilerplate: bool) -> list[str]:
    """Explain an empty fetch: a page that was nothing but boilerplate, or one with no content at all."""
    if content:
        return []
    if only_boilerplate:
        return [
            "Empty content: the page rendered only a cookie-consent banner or site interface, which was "
            "removed. Try deepseek_search instead, or strip_boilerplate=False to see it."
        ]
    return ["Empty content: the page may be a PDF, JS-only or blocked. Try deepseek_search instead."]


def _all_cached(requests_used: int, cached_count: int) -> bool:
    """Top-level ``cached`` of a multi-response tool: something came from the cache and nothing upstream."""
    return cached_count > 0 and requests_used == 0


# Error codes that mean no upstream request was made for a batch item.
_LOCAL_ERROR_CODES = frozenset({"rate_limited_local", "deadline_exceeded", "internal_error"})


def _batch_item(url: str, envelope: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``deepseek_fetch`` envelope to one ``deepseek_fetch_many`` result."""
    if envelope.get("ok") is False:
        return {"url": url, "ok": False, "error": envelope["error"], "chars": 0, "cached": False}
    return {
        "url": url,
        "ok": True,
        "title": envelope["title"],
        "content": envelope["content"],
        "chars": len(envelope["content"]),
        "cached": envelope["cached"],
        "untrusted_content": True,
    }


async def _fetch_isolated(
    url: str,
    clean: bool,
    max_chars: int,
    refresh: bool,
    semaphore: asyncio.Semaphore,
    strip_boilerplate: bool = True,
    query: str | None = None,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Run one batch fetch under ``semaphore``; unexpected exceptions become error envelopes."""
    async with semaphore:
        try:
            return await _fetch(
                url,
                clean,
                max_chars,
                query=query,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
                fallback_to_search=fallback_to_search,
            )
        except Exception as exc:  # broad on purpose: one failing URL must not fail the whole batch
            message = redact(f"{type(exc).__name__}: {exc}")
            logger.debug("Batch fetch of a URL failed: %s", message)
            return _fail("internal_error", message)


# A batch entry: the stripped URL and its rejection envelope (``None`` when it is fetchable).
_BatchItem = tuple[str, dict[str, Any] | None]


def _dedupe_urls(urls: list[str]) -> tuple[list[_BatchItem], int]:
    """Strip and validate ``urls``, skip empty entries and collapse canonical duplicates.

    Returns ``(items, duplicates)``. Rejected URLs are kept (keyed verbatim) so they can be
    reported in place; fetchable ones are keyed by their canonical form.
    """
    items: list[_BatchItem] = []
    seen: set[str] = set()
    duplicates = 0
    for raw in urls:
        url = raw.strip() if isinstance(raw, str) else ""
        if not url:
            continue
        rejection = _check_fetch_url(url)
        key = url if rejection is not None else (_canonical_url(url) or url)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        items.append((url, rejection))
    return items, duplicates


async def _run_batch(
    items: list[_BatchItem],
    clean: bool,
    per_url: int,
    refresh: bool,
    strip_boilerplate: bool = True,
    query: str | None = None,
    deadline: float | None = None,
    fallback_to_search: bool = True,
) -> dict[int, asyncio.Task[dict[str, Any]]]:
    """Fetch every non-rejected item concurrently under the batch concurrency and deadline.

    Returns the tasks by item index; those still pending at the deadline are cancelled.
    """
    settings = get_settings()
    timeout = deadline if deadline is not None else settings.batch_deadline
    semaphore = asyncio.Semaphore(settings.batch_concurrency)
    tasks = {
        index: asyncio.create_task(
            _fetch_isolated(
                url,
                clean,
                per_url,
                refresh,
                semaphore,
                strip_boilerplate,
                query=query,
                fallback_to_search=fallback_to_search,
            )
        )
        for index, (url, rejection) in enumerate(items)
        if rejection is None
    }
    try:
        if tasks:
            await asyncio.wait(tasks.values(), timeout=timeout)
    finally:
        pending = [task for task in tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return tasks


def _batch_results(
    items: list[_BatchItem],
    tasks: dict[int, asyncio.Task[dict[str, Any]]],
    max_total_chars: int,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Collect batch results in input order, trimming content to ``max_total_chars`` overall.

    Returns ``(results, trimmed, requests_used)``.
    """
    actual_deadline = deadline if deadline is not None else get_settings().batch_deadline
    deadline_error = _fail(
        "deadline_exceeded", f"Not fetched within the {actual_deadline:g} s batch deadline."
    )
    results: list[dict[str, Any]] = []
    remaining = max_total_chars
    trimmed = False
    requests_used = 0
    for index, (url, rejection) in enumerate(items):
        if rejection is not None:
            envelope = rejection
        elif tasks[index].cancelled():
            envelope = deadline_error
        else:
            envelope = tasks[index].result()
        item = _batch_item(url, envelope)
        if item["ok"]:
            trimmed = trimmed or envelope["truncated"]
            if item["chars"] > remaining:
                item["content"] = item["content"][:remaining]
                item["chars"] = remaining
                trimmed = True
            remaining -= item["chars"]
            requests_used += not item["cached"]
        elif rejection is None and item["error"].get("code") not in _LOCAL_ERROR_CODES:
            requests_used += 1
        results.append(item)
    return results, trimmed, requests_used


def _assemble_batch(
    items: list[_BatchItem],
    tasks: dict[int, asyncio.Task[dict[str, Any]]],
    duplicates: int,
    max_total_chars: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Build the ``deepseek_fetch_many`` envelope from the finished batch."""
    results, trimmed, requests_used = _batch_results(items, tasks, max_total_chars, deadline=deadline)
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")
    if trimmed:
        notes.append(TRUNCATED_NOTE)
    failed_count = sum(not item["ok"] for item in results)
    cached_count = sum(item["cached"] for item in results)
    cost = round(requests_used * COST_PER_REQUEST, 5)
    return {
        "ok": True,
        "count": len(results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": results,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }


@_timed
async def deepseek_fetch_many(
    urls: list[str],
    max_chars_per_url: int = 4000,
    max_total_chars: int = 20000,
    clean: bool = True,
    *,
    query: str | None = None,
    refresh: bool = False,
    strip_boilerplate: bool = True,
    deadline: float | None = None,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Fetch up to 10 known URLs concurrently and return each page's title and Markdown content.

    Fetch multiple known URLs in parallel. Use when target URLs are known.
    Skips blanks, deduplicates, and preserves order.

    Args:
        urls: 1-10 absolute http(s) URLs, max 8192 chars each.
        max_chars_per_url: Content cap per page (default 4000).
        max_total_chars: Total content cap across pages (default 20000).
        clean: Remove Markdown backslash escapes (default True).
        query: Query to extract ranked snippets across pages.
        refresh: Bypass cache (default False).
        strip_boilerplate: Remove site UI lines (default True).
        deadline: Max seconds before cancelling fetches (default 120s).
        fallback_to_search: Auto-search fallback on 404 (default True).
    """
    settings = get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail("bad_request", f"urls must contain 1-{settings.batch_max_urls} entries.")
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars_per_url and max_total_chars must be at least 1.")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "urls must contain at least one non-empty URL.")
    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    tasks = await _run_batch(
        items,
        clean,
        per_url,
        refresh,
        strip_boilerplate,
        query=query,
        deadline=deadline,
        fallback_to_search=fallback_to_search,
    )
    return _assemble_batch(items, tasks, duplicates, max_total_chars, deadline=deadline)


# Search results whose content is shorter than this are treated as gaps and fetched directly.
RESEARCH_GAP_CHARS = 200
# Default content budgets for deepseek_research: per source and across all sources.
RESEARCH_PAGE_CHARS = 6000
RESEARCH_TOTAL_CHARS = 30000


def _research_source(index: int, item: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Turn one shaped search result into a research source (``origin`` = ``search``)."""
    return {
        "id": index,
        "title": item.get("title"),
        "url": item.get("url"),
        "date": item.get("date"),
        "date_source": item.get("date_source"),
        "site": item.get("site"),
        "authority": item.get("authority"),
        "rerank_score": item.get("rerank_score"),
        "content": item.get("content") or "",
        "chars": 0,
        "origin": "search",
        "error": None,
        "cached": cached,
    }


def _gap_check(source: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Classify a source: ``(url, None)`` to fetch, ``(None, error)`` if unfetchable, else no gap."""
    if len(source["content"]) >= RESEARCH_GAP_CHARS:
        return None, None
    url = source["url"].strip() if isinstance(source["url"], str) else ""
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return None, rejection["error"]
    return url, None


@dataclass
class _GapFill:
    """Outcome of fetching the research gaps (counts only cover the fetch phase)."""

    requests_used: int = 0
    cached_count: int = 0
    fetched: int = 0
    failed: int = 0
    trimmed: bool = False


def _select_gaps(sources: list[dict[str, Any]], fill: _GapFill) -> dict[str, dict[str, Any]]:
    """Pick the sources to fetch, keyed by canonical URL; mark rejected ones in place.

    At most ``research_max_pages`` distinct URLs are selected; rejections count as failures.
    """
    gaps: dict[str, dict[str, Any]] = {}
    for source in sources:
        url, rejection = _gap_check(source)
        if rejection is not None:
            source["error"] = rejection
            fill.failed += 1
        key = None if url is None else (_canonical_url(url) or url)
        if key is not None and key not in gaps and len(gaps) < get_settings().research_max_pages:
            gaps[key] = source
    return gaps


async def _fill_gaps(
    sources: list[dict[str, Any]],
    per_page_chars: int,
    total_chars: int,
    refresh: bool,
    strip_boilerplate: bool = True,
) -> _GapFill:
    """Fetch sources with missing or short content and merge the fetched pages in place."""
    fill = _GapFill()
    gaps = _select_gaps(sources, fill)
    if not gaps:
        return fill
    batch = await deepseek_fetch_many(
        [source["url"].strip() for source in gaps.values()],
        max_chars_per_url=per_page_chars,
        max_total_chars=total_chars,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
        deadline=get_settings().research_gap_deadline,
    )
    fill.requests_used = batch["requests_used"]
    fill.cached_count = batch["cached_count"]
    fill.trimmed = TRUNCATED_NOTE in batch["notes"]
    for source, item in zip(gaps.values(), batch["results"], strict=True):
        if not item["ok"]:
            source["error"] = item["error"]
            fill.failed += 1
        elif item["content"]:
            source["content"] = item["content"]
            if item["title"] and (not source["title"] or source["title"] == _url_host(source["url"])):
                source["title"] = item["title"]  # a fetched title beats a blank or host-fallback one
            source["origin"] = "fetch"
            source["cached"] = item["cached"]
            if source["date"] is None and (found := _content_date(None, item["content"])):
                source["date"], source["date_source"] = found, "content"
            fill.fetched += 1
    return fill


def _budget_sources(sources: list[dict[str, Any]], per_page_chars: int, total_chars: int) -> bool:
    """Cut source content to the per-source and total budgets in order; return whether any was cut."""
    trimmed = False
    remaining = total_chars
    for source in sources:
        allowed = min(per_page_chars, remaining)
        if len(source["content"]) > allowed:
            source["content"] = source["content"][:allowed]
            trimmed = True
        source["chars"] = len(source["content"])
        remaining -= source["chars"]
    return trimmed


def _research_envelope(
    question: str, sources: list[dict[str, Any]], search: dict[str, Any], fill: _GapFill, trimmed: bool
) -> dict[str, Any]:
    """Build the ``deepseek_research`` envelope from the search, the gap fill and the budget pass."""
    notes: list[str] = list(search["notes"])
    if fill.fetched:
        notes.append(f"fetched {fill.fetched} page(s) with missing or short search content")
    if fill.failed:
        notes.append(f"{fill.failed} fetch(es) failed; see the source's error")
    if trimmed and TRUNCATED_NOTE not in notes:
        notes.append(TRUNCATED_NOTE)
    requests_used = (0 if search["cached"] else 1) + fill.requests_used
    cached_count = (1 if search["cached"] else 0) + fill.cached_count
    cost = round(requests_used * COST_PER_REQUEST, 5)
    envelope: dict[str, Any] = {
        "ok": True,
        "question": question,
        "query_used": search["query_used"],
        "sources": sources,
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "cost_usd": cost,
        "untrusted_content": True,
        "notes": notes,
        "failed_count": fill.failed,
        "partial": fill.failed > 0,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if "query_original" in search:
        envelope["query_original"] = search["query_original"]
    return envelope


@_timed
async def deepseek_research(
    question: str,
    max_pages: int = 5,
    per_page_chars: int = RESEARCH_PAGE_CHARS,
    total_chars: int = RESEARCH_TOTAL_CHARS,
    *,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    lang: str | None = None,
    rewrite: bool = False,
    dedupe: bool = True,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Single-pass source gathering: search with page content and auto-fetch stub pages.

    Gathers candidate sources for research. It does not iterate, refine the query or summarise:
    one search returning raw content. Auto-fetches stub results in parallel.

    Args:
        question: Query, max 128 chars.
        max_pages: Results, 1-50 (default 5).
        per_page_chars: Cap per source (default 6000).
        total_chars: Total content cap (default 30000).
        after: Date on or after (YYYY-MM-DD).
        before: Date on or before (YYYY-MM-DD).
        include_domains: Restrict domains (e.g. ['arxiv.org']).
        exclude_domains: Exclude domains (e.g. ['wikipedia.org']).
        lang: Language filter (e.g. 'pl', 'es').
        rewrite: Translate query via deepseek-flash (default False).
        dedupe: Collapse duplicate results (default True).
        refresh: Bypass cache (default False).
        strip_boilerplate: Remove site UI lines (default True).
    """
    if per_page_chars < 1 or total_chars < 1:
        return _fail("bad_request", "per_page_chars and total_chars must be at least 1.")
    search = await _search(
        question,
        max_pages,
        None,
        True,
        lang,
        per_page_chars,
        per_page_chars * max_pages,
        dedupe,
        rewrite,
        after=after,
        before=before,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
    )
    if search.get("ok") is False:
        return search

    sources = [
        _research_source(index, item, search["cached"]) for index, item in enumerate(search["results"], 1)
    ]
    gaps = await _fill_gaps(sources, per_page_chars, total_chars, refresh, strip_boilerplate)
    trimmed = _budget_sources(sources, per_page_chars, total_chars) or gaps.trimmed
    return _research_envelope(question, sources, search, gaps, trimmed)


@_timed
async def deepseek_deep_research(
    topic: str,
    *,
    max_queries: int = 3,
    max_pages_per_query: int = 3,
    lang: str | None = None,
    dedupe: bool = True,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Execute multi-angle research on a topic and generate a cited briefing report.

    Decomposes topic into subqueries, retrieves sources concurrently, and synthesizes
    a cited report [1], [2]. Use for deep topic synthesis. Do not use for simple lookups
    (use deepseek_search/deepseek_research) or reading known URLs (use deepseek_fetch).

    Args:
        topic: Research subject, max 500 chars.
        max_queries: Subqueries to run (2-4, default 3).
        max_pages_per_query: Sources per subquery (1-5, default 3).
        lang: Language filter (e.g. 'en', 'pl').
        dedupe: Collapse duplicates across subqueries (default True).
        refresh: Bypass response cache (default False).
        strip_boilerplate: Strip site UI text (default True).
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.")

    clamped_queries = max(2, min(4, max_queries))
    clamped_pages = max(1, min(5, max_pages_per_query))
    t0 = _clock()

    # Step 1: Decompose topic into subqueries
    dec_messages = build_decomposition_messages(clean_topic, max_queries=clamped_queries)
    dec_res = await _request(
        "POST",
        "/v1/chat/completions",
        payload={
            "model": REWRITE_MODEL,
            "messages": dec_messages,
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
            "thinking": {"type": "disabled"},
        },
        timeout=CHAT_TIMEOUT,
        attempts=1,
    )
    if dec_res.get("ok") and dec_res.get("data", {}).get("choices"):
        raw_dec = dec_res["data"]["choices"][0]["message"]["content"]
        subqueries = parse_subqueries(raw_dec, clean_topic, max_queries=clamped_queries)
    else:
        subqueries = [clean_topic[:MAX_QUERY].strip()]

    # Step 2: Concurrently execute search across subqueries
    search_coros = [
        _search(
            subq,
            clamped_pages,
            None,
            True,
            lang,
            RESEARCH_PAGE_CHARS,
            RESEARCH_PAGE_CHARS * clamped_pages,
            dedupe,
            False,
            refresh=refresh,
            strip_boilerplate=strip_boilerplate,
        )
        for subq in subqueries
    ]
    search_results = await asyncio.gather(*search_coros)

    successful_searches = [r for r in search_results if r.get("ok")]
    if not successful_searches:
        return search_results[0]

    # Step 3: Merge, deduplicate, and convert to research sources
    seen_urls: set[str] = set()
    merged_items: list[dict[str, Any]] = []
    cached_count = sum(s.get("cached_count", 0) for s in successful_searches)
    requests_used = sum(s.get("requests_used", 0) for s in successful_searches)

    for s in successful_searches:
        for item in s.get("results", []):
            url = item.get("url") or ""
            canon = _canonical_url(url) or url
            if canon not in seen_urls:
                seen_urls.add(canon)
                merged_items.append(item)

    sources = [
        _research_source(idx, item, item.get("cached", False)) for idx, item in enumerate(merged_items, 1)
    ]

    # Step 4: Gap-fill stub pages
    fill = await _fill_gaps(sources, RESEARCH_PAGE_CHARS, 30000, refresh, strip_boilerplate)
    requests_used += fill.requests_used
    cached_count += fill.cached_count

    notes: list[str] = []
    if fill.fetched:
        notes.append(f"fetched {fill.fetched} page(s) with missing or short search content")
    if fill.failed:
        notes.append(f"{fill.failed} fetch(es) failed; see the source's error")

    # Step 5: Select synthesis context (capped at 6 sources, 24k chars)
    synthesis_context = select_synthesis_context(sources)
    if not synthesis_context:
        cost = round(requests_used * COST_PER_REQUEST, 5)
        return {
            "ok": True,
            "topic": clean_topic,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": sources,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, cached_count),
            "cached_count": cached_count,
            "cost_usd": cost,
            "untrusted_content": True,
            "notes": notes + ["No readable source content available for synthesis."],
            "_meta": _envelope_meta(cost, untrusted_content=True),
        }

    # Soft deadline check before synthesis
    elapsed = _clock() - t0
    if elapsed > RESEARCH_DEADLINE - 5.0:
        cost = round(requests_used * COST_PER_REQUEST, 5)
        return {
            "ok": True,
            "topic": clean_topic,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": sources,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, cached_count),
            "cached_count": cached_count,
            "cost_usd": cost,
            "untrusted_content": True,
            "notes": notes + ["Synthesis deadline reached; returning gathered sources."],
            "_meta": _envelope_meta(cost, untrusted_content=True),
        }

    # Step 6: Synthesize cited briefing report
    synthesis_messages = build_synthesis_messages(clean_topic, synthesis_context)
    synth_res = await _request(
        "POST",
        "/v1/chat/completions",
        payload={
            "model": REWRITE_MODEL,
            "messages": synthesis_messages,
            "temperature": 0.2,
            "max_tokens": 2048,
            "stream": False,
            "thinking": {"type": "disabled"},
        },
        timeout=SYNTHESIS_TIMEOUT,
        attempts=1,
    )

    if not synth_res.get("ok"):
        cost = round((requests_used + 1) * COST_PER_REQUEST, 5)
        return {
            "ok": True,
            "topic": clean_topic,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": sources,
            "requests_used": requests_used + 1,
            "cached": _all_cached(requests_used + 1, cached_count),
            "cached_count": cached_count,
            "cost_usd": cost,
            "untrusted_content": True,
            "notes": notes
            + [
                f"Synthesis model call failed: {synth_res.get('error', {}).get('message', 'unknown')}; "
                "returning gathered sources."
            ],
            "_meta": _envelope_meta(cost, untrusted_content=True),
        }

    report_text = synth_res["data"]["choices"][0]["message"]["content"]
    clean_report = sanitize_citations(report_text, max_id=len(synthesis_context))
    cost = round((requests_used + 1) * COST_PER_REQUEST, 5)

    return {
        "ok": True,
        "topic": clean_topic,
        "subqueries": subqueries,
        "report": clean_report,
        "synthesized": True,
        "sources": [
            {
                "id": s["id"],
                "title": s["title"],
                "url": s["url"],
            }
            for s in synthesis_context
        ],
        "raw_sources": sources,
        "requests_used": requests_used + 1,
        "cached": _all_cached(requests_used + 1, cached_count),
        "cached_count": cached_count,
        "cost_usd": cost,
        "untrusted_content": True,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }


def _validate_extract_spec(fields: list[str] | None, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate that either 'fields' or 'schema' is provided and structurally sound."""
    if not fields and not schema:
        return _fail("bad_request", "Either 'fields' or 'schema' must be provided.")
    if fields is not None and schema is not None:
        return _fail("bad_request", "Provide either 'fields' or 'schema', not both.")
    if fields is not None:
        valid_fields = (
            isinstance(fields, list)
            and bool(fields)
            and all(isinstance(f, str) and f.strip() for f in fields)
        )
        if not valid_fields:
            return _fail("bad_request", "'fields' must be a non-empty list of non-empty strings.")
    if schema is not None and not isinstance(schema, dict):
        return _fail("bad_request", "'schema' must be a dictionary.")
    return None


async def _extract_page(
    clean_url: str,
    fetch_res: dict[str, Any],
    system_prompt: str,
    fields: list[str] | None,
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    """Perform LLM extraction for one successfully fetched page.

    Returns ``(item_result, upstream_llm_requests)``.
    """
    content = fetch_res.get("content") or ""
    if not content.strip():
        notes = fetch_res.get("notes") or []
        msg = notes[0] if notes else "The fetched page has no readable content to extract from."
        return {
            "ok": False,
            "url": clean_url,
            "error": {"code": "empty_page", "message": msg},
        }, 0

    page_title = fetch_res.get("title") or "Untitled"
    user_content = f"URL: {clean_url}\nTitle: {page_title}\n\nPage Content:\n{content}"

    payload: dict[str, Any] = {
        "model": REWRITE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "stream": False,
        "thinking": {"type": "disabled"},
    }

    result = await _request(
        "POST", "/v1/chat/completions", payload=payload, timeout=EXTRACT_TIMEOUT, attempts=1
    )
    if not result.get("ok"):
        return {
            "ok": False,
            "url": clean_url,
            "error": result.get("error")
            or {"code": "extraction_failed", "message": "Extraction model request failed."},
        }, 1

    choices = result.get("data", {}).get("choices") or []
    raw_text = choices[0]["message"]["content"] if choices else None
    if not isinstance(raw_text, str):
        return {
            "ok": False,
            "url": clean_url,
            "error": {
                "code": "extraction_failed",
                "message": "Invalid response payload from extraction model.",
            },
        }, 1

    parsed, err = sanitize_json(raw_text)
    if err or parsed is None:
        return {
            "ok": False,
            "url": clean_url,
            "error": {
                "code": "extraction_failed",
                "message": err or "Failed to parse extraction response as JSON.",
            },
        }, 1

    notes: list[str] = []
    if fetch_res.get("cached"):
        notes.append("Page fetched from local cache.")

    if fields is not None:
        parsed = validate_fields(parsed, fields)
    elif schema is not None:
        schema_errors = validate_schema(parsed, schema)
        if schema_errors:
            notes.extend(schema_errors)

    return {
        "ok": True,
        "url": clean_url,
        "title": fetch_res.get("title"),
        "data": parsed,
        "cached": fetch_res.get("cached", False),
        "notes": notes,
    }, 1


@_timed
async def deepseek_extract(
    url: str,
    *,
    fields: list[str] | None = None,
    schema: dict[str, Any] | None = None,
    instruction: str | None = None,
    clean: bool = True,
    max_chars: int = 40000,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Extract structured JSON from a web page conforming to target fields or a schema.

    Extracts specific entities or key-value data without returning full page text.
    Returns extracted JSON under 'data'. For batch use deepseek_extract_many.

    Args:
        url: Absolute web page URL to extract from.
        fields: Target attribute names (e.g. ['price']), exclusive with schema.
        schema: JSON Schema defining target structure, exclusive with fields.
        instruction: Optional extraction guidance.
        clean: Remove Markdown backslash escapes (default True).
        max_chars: Content cap before extraction (default 40000).
        refresh: Bypass cache (default False).
        strip_boilerplate: Remove site UI lines (default True).
    """
    spec_err = _validate_extract_spec(fields, schema)
    if spec_err is not None:
        return spec_err

    clean_url = url.strip() if isinstance(url, str) else ""
    rejection = _check_fetch_url(clean_url)
    if rejection is not None:
        return rejection

    fetch_res = await _fetch(
        clean_url,
        clean=clean,
        max_chars=max_chars,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
    )
    if fetch_res.get("ok") is False:
        return fetch_res

    system_prompt = build_extract_prompt(fields, schema, instruction)
    item, llm_reqs = await _extract_page(clean_url, fetch_res, system_prompt, fields, schema)
    if not item.get("ok"):
        return {"ok": False, "error": item["error"]}

    fetch_reqs = round(fetch_res.get("cost_usd", 0.0) / COST_PER_REQUEST)
    requests_used = fetch_reqs + llm_reqs
    cost = round(requests_used * COST_PER_REQUEST, 5)
    return {
        "ok": True,
        "url": clean_url,
        "title": item.get("title"),
        "data": item.get("data"),
        "cached": fetch_res.get("cached", False),
        "cost_usd": cost,
        "untrusted_content": True,
        "notes": item.get("notes", []),
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }


@_timed
async def deepseek_extract_many(
    urls: list[str],
    *,
    fields: list[str] | None = None,
    schema: dict[str, Any] | None = None,
    instruction: str | None = None,
    clean: bool = True,
    max_chars_per_url: int = 20000,
    max_total_chars: int = 60000,
    refresh: bool = False,
    strip_boilerplate: bool = True,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Extract structured JSON from multiple web pages concurrently conforming to target fields or a schema.

    Batch extraction across 1-10 URLs. Extracts entities or key-value data in parallel
    without returning full page text. Returns extracted JSON per URL under 'data'.

    Args:
        urls: 1-10 absolute web page URLs to extract from.
        fields: Target attribute names (e.g. ['price']), exclusive with schema.
        schema: JSON Schema defining target structure, exclusive with fields.
        instruction: Optional extraction guidance across pages.
        clean: Remove Markdown backslash escapes (default True).
        max_chars_per_url: Content cap per page (default 20000).
        max_total_chars: Total content cap (default 60000).
        refresh: Bypass cache (default False).
        strip_boilerplate: Remove site UI lines (default True).
        deadline: Max seconds before cancelling fetches (default 120s).
    """
    spec_err = _validate_extract_spec(fields, schema)
    if spec_err is not None:
        return spec_err

    settings = get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail("bad_request", f"urls must contain 1-{settings.batch_max_urls} entries.")
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars_per_url and max_total_chars must be at least 1.")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "urls must contain at least one non-empty URL.")

    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    fetch_tasks = await _run_batch(items, clean, per_url, refresh, strip_boilerplate, deadline=deadline)

    actual_deadline = deadline if deadline is not None else settings.batch_deadline
    deadline_error = _fail(
        "deadline_exceeded", f"Not fetched within the {actual_deadline:g} s batch deadline."
    )

    system_prompt = build_extract_prompt(fields, schema, instruction)

    extract_coros: list[Awaitable[tuple[dict[str, Any], int]]] = []
    extract_indices: dict[int, int] = {}
    fetch_envelopes: dict[int, dict[str, Any]] = {}

    requests_used = 0
    for index, (url, rejection) in enumerate(items):
        if rejection is not None:
            fetch_envelopes[index] = rejection
        elif fetch_tasks[index].cancelled():
            fetch_envelopes[index] = deadline_error
        else:
            fetch_envelopes[index] = fetch_tasks[index].result()

        env = fetch_envelopes[index]
        if env.get("ok"):
            fetch_reqs = round(env.get("cost_usd", 0.0) / COST_PER_REQUEST)
            requests_used += fetch_reqs
            extract_indices[index] = len(extract_coros)
            extract_coros.append(_extract_page(url, env, system_prompt, fields, schema))
        elif rejection is None and env.get("error", {}).get("code") not in _LOCAL_ERROR_CODES:
            requests_used += 1

    extracted_results: list[tuple[dict[str, Any], int]] = []
    if extract_coros:
        extracted_results = await asyncio.gather(*extract_coros)

    results: list[dict[str, Any]] = []
    for index, (url, _) in enumerate(items):
        if index in extract_indices:
            item_result, llm_reqs = extracted_results[extract_indices[index]]
            requests_used += llm_reqs
            results.append(item_result)
        else:
            env = fetch_envelopes[index]
            results.append(
                {
                    "ok": False,
                    "url": url,
                    "error": env.get("error", {"code": "fetch_failed", "message": "Failed to fetch page."}),
                }
            )

    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")

    failed_count = sum(not item.get("ok", False) for item in results)
    cached_count = sum(item.get("cached", False) for item in results)
    cost = round(requests_used * COST_PER_REQUEST, 5)

    return {
        "ok": True,
        "count": len(results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": results,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }


@_timed
async def deepseek_check_balance(*, clear_cache: bool = False) -> dict[str, Any]:
    """Check account balance (CNY/USD), session telemetry, and manage response cache.

    Verify funds, monitor session usage, or reset session caches (search, fetch, rewrite).
    Use clear_cache=True to purge in-memory caches.

    Args:
        clear_cache: Clear all in-memory response caches when True (default False).
    """
    cleared_info: dict[str, int] | None = None
    if clear_cache:
        cleared_info = _clear_caches()

    result = await _request("GET", "/user/balance", timeout=BALANCE_TIMEOUT)
    if not result.get("ok"):
        if clear_cache:
            result["cache_cleared"] = True
            result["cache_evicted"] = cleared_info
        return result
    data = result["data"]
    metrics = _get_session_metrics()
    uptime = round(_clock() - metrics.start_time, 1)
    total_handled = metrics.requests_upstream + metrics.cache_hits
    hit_rate = round(metrics.cache_hits / total_handled, 3) if total_handled > 0 else 0.0
    cost_usd = round(metrics.requests_upstream * 0.00075, 5)
    savings_usd = round(metrics.cache_hits * 0.00075, 5)

    session_data: dict[str, Any] = {
        "uptime_seconds": uptime,
        "requests_upstream": metrics.requests_upstream,
        "cache_hits": metrics.cache_hits,
        "total_queries_handled": total_handled,
        "cache_hit_rate": hit_rate,
        "estimated_upstream_cost_usd": cost_usd,
        "estimated_cache_savings_usd": savings_usd,
        "cache": _get_cache_stats(),
    }
    if clear_cache:
        session_data["cache_cleared"] = True
        session_data["cache_evicted"] = cleared_info

    resp: dict[str, Any] = {
        "ok": True,
        "is_available": data.get("is_available"),
        "balances": data.get("balance_infos") or data.get("balances") or data,
        "session": session_data,
    }
    if clear_cache:
        resp["cache_cleared"] = True
        resp["cache_evicted"] = cleared_info
    return resp


# Backward compatibility alias for v1.x
deepseek_balance = deepseek_check_balance


TOOLS: tuple[Callable[..., Awaitable[dict[str, Any]]], ...] = (
    deepseek_search,
    deepseek_search_many,
    deepseek_fetch,
    deepseek_fetch_many,
    deepseek_research,
    deepseek_deep_research,
    deepseek_extract,
    deepseek_extract_many,
    deepseek_check_balance,
)

TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    "deepseek_search": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_search_many": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_fetch": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_fetch_many": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_research": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_deep_research": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_extract": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_extract_many": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
    ),
    "deepseek_check_balance": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    ),
}


def create_server(**options: Any) -> FastMCP:
    """Build a new FastMCP instance with every tool registered.

    ``options`` are passed to :class:`FastMCP` (host, port, HTTP path, transport flags,
    ``log_level``). ``log_level`` defaults to WARNING so FastMCP's one-time root logging
    setup stays quiet; the CLI applies ``DEEPSEEK_MCP_LOG_LEVEL`` to the loggers themselves.
    Each call returns an independent server, so each HTTP application gets its own
    session manager. The server reports the package version in the ``initialize`` handshake.
    """
    options.setdefault("log_level", DEFAULT_LOG_LEVEL)
    server = FastMCP(SERVER_NAME, **options)
    # FastMCP has no ``version`` argument, so the handshake would report the MCP SDK's version.
    # The low-level server's ``version`` is what ``initialize`` returns as ``serverInfo.version``.
    server._mcp_server.version = __version__
    for tool in TOOLS:
        description = inspect.cleandoc(tool.__doc__) if tool.__doc__ else None
        server.add_tool(
            tool,
            description=description,
            annotations=TOOL_ANNOTATIONS.get(tool.__name__),
        )
    return server


mcp = create_server()
