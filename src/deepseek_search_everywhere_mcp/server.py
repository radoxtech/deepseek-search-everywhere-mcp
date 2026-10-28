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

from ._version import __version__
from .boilerplate import remove_boilerplate
from .cache import SingleFlight, TTLCache
from .errors import _fail
from .errors import _map_http_error as _map_http_error  # re-exported for callers and tests
from .ratelimit import FETCH, SEARCH, RateLimiter, RateLimitExceeded
from .redact import redact
from .settings import DEFAULT_LOG_LEVEL, get_settings
from .transport import _api_key, _base_url
from .transport import request as _request  # tests monkeypatch server._request
from .urls import CODE_INVALID, UrlError, validate_url

MAX_QUERY = 128
MAX_URL = 8192
SEARCH_TIMEOUT = 60.0
FETCH_TIMEOUT = 120.0
BALANCE_TIMEOUT = 20.0
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
MAX_COUNT = 50  # largest ``count`` accepted from callers and by the upstream search API
# With dedupe on, the upstream is asked for up to max(OVERFETCH_MIN, ceil(count * OVERFETCH_RATIO))
# extra results (capped at MAX_COUNT) so that removing duplicates still leaves ``count`` results.
OVERFETCH_MIN = 2
OVERFETCH_RATIO = 0.5
# Default summary cap per result when the search returns no page content (need_content=False).
SEARCH_SUMMARY_CHARS = 400
# Note added by every tool whose content was cut to fit a character budget.
TRUNCATED_NOTE = "Some page content was truncated to respect the character budget."

logger = logging.getLogger(__name__)

# Monotonic clock behind every envelope's ``elapsed_ms``; tests may replace it.
_clock: Callable[[], float] = time.monotonic

_P = ParamSpec("_P")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SERVER_NAME = "deepseek-search-everywhere-mcp"

# Runtime state, built lazily from the active settings on first use so nothing is bound
# to an event loop at import time. ``reset_state()`` drops it all.
_search_cache: TTLCache | None = None
_search_flight: SingleFlight | None = None
_fetch_cache: TTLCache | None = None
_fetch_flight: SingleFlight | None = None
_rewrite_cache: TTLCache | None = None
_rewrite_flight: SingleFlight | None = None
_rate_limiter: RateLimiter | None = None


def reset_state() -> None:
    """Forget the response and rewrite caches, in-flight calls and the rate limiter."""
    global _search_cache, _search_flight, _fetch_cache, _fetch_flight
    global _rewrite_cache, _rewrite_flight, _rate_limiter
    _search_cache = _search_flight = _fetch_cache = _fetch_flight = None
    _rewrite_cache = _rewrite_flight = _rate_limiter = None


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
    endpoint: str, path: str, payload: dict[str, Any], timeout: float
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
        return await _request("POST", path, payload=payload, timeout=timeout)
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
    result = await _limited_request(SEARCH, SEARCH_PATH, payload, SEARCH_TIMEOUT)
    if not result.get("ok"):
        return result
    record = {"data": result["data"], "query_original": query_original}
    cache = _get_search_cache()
    if cache is not None and result["data"].get("results"):
        cache.set(key, record)
    return record


def _check_search_args(query: str, count: int, lang: str | None) -> dict[str, Any] | None:
    """Validate a stripped query, ``count`` and ``lang``; return an error envelope or ``None``."""
    if not query:
        return _fail("empty_query", "query must not be empty.")
    if len(query) > MAX_QUERY:
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
    query: str, count: int, dedupe: bool, need_content: bool, ori_question: str | None
) -> dict[str, Any]:
    """Build the upstream search payload (``query`` already carries any ``lang:`` operator)."""
    payload: dict[str, Any] = {
        "query": query,
        "count": _upstream_count(count, dedupe),
        "need_content": need_content,
    }
    if ori_question:
        payload["ori_question"] = ori_question[:MAX_QUERY]
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
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Run one validated, cached search and shape the envelope shared by the search tools.

    ``summary_chars`` caps the summary of results without content (see ``_summary_cap``).
    ``strip_boilerplate`` removes known site-UI lines from page content; it only shapes the
    response, so the cache key and the cached upstream data are the same either way.
    """
    query = (query or "").strip()
    rejection = _check_search_args(query, count, lang)
    if rejection is not None:
        return rejection
    if summary_chars is not None and summary_chars < 0:
        return _fail("bad_request", "summary_chars must be at least 0.")

    query, ori_question, query_original, rewrite_failed = await _maybe_rewrite(
        query, ori_question, rewrite, lang
    )
    used, trimmed = _with_lang(query, lang)
    payload = _build_search_payload(used, count, dedupe, need_content, ori_question)
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

    envelope: dict[str, Any] = {
        "ok": True,
        "query_used": used,
        "count": len(shaped),
        "need_content": need_content,
        "deduped": removed,
        "cached": cached,
        "results": shaped,
        "notes": _search_notes(used, query_original, rewrite_failed, removed, trimmed, lang, truncated),
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


@_timed
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
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Search the web with DeepSeek's index. Returns titles, URLs, dates, authority and rerank scores.

    Use this for discovery and ranking. Set need_content=True to get full page text inline in the
    same call (billed as one request, no per-page charge) — this is usually cheaper and more
    reliable than calling deepseek_fetch afterwards.

    With content, each result's chars is the length of the content returned.
    Without need_content each result carries a short summary instead, capped at summary_chars
    (default 400) and never longer than max_chars_per_result; summary_chars=0 leaves it out, which
    keeps a 10-result response small.

    For non-English queries set lang to the appropriate code (e.g. 'es', 'de', 'ja'). Without it
    results may be mixed-language; for named entities test with and without the operator.
    Each result carries date and authority (0-4, -1 unknown). date is the upstream RFC 3339 date
    or, when the upstream has none, a date found at the start of the result's text (YYYY-MM-DD, or a
    full English or Polish month name next to a year, read as the first of that month);
    date_source says which ('upstream', 'content' or null when there is no date).

    Responses are cached in memory for a few minutes and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 8 MiB fails with response_too_large.
    Every envelope, success or error, carries elapsed_ms: the call's wall time in milliseconds.

    Args:
        query: Search query, max 128 characters. Supports site:, -site:, "phrase", lang:, after:, before:.
        count: Number of results to return, 1-50 (default 10). With dedupe on, extras are
            requested upstream so duplicates do not leave you short.
        ori_question: Original natural-language question, max 128 characters; improves ranking.
        need_content: Return full page text per result (recommended for reading).
        lang: Language operator, e.g. 'es', 'de' or 'ja'. For non-English queries set it to the
            appropriate code; without it results may be mixed-language. For named entities test
            with and without the operator.
        max_chars_per_result: Cap per result on the content (when need_content is set) and on
            the summary.
        max_total_chars: Total budget across all results, for content and summaries alike.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
        refresh: Bypass the response cache and query upstream again (the envelope's cached
            flag says whether the results came from the cache).
        summary_chars: Summary cap per result, at most max_chars_per_result. Defaults to 400
            without need_content; with it, a summary only stands in for missing content and
            defaults to max_chars_per_result. 0 omits summaries.
        strip_boilerplate: Remove known site-interface lines from page content (default on),
            such as the Chinese-localised comment-sort menu and sign-up footer DeepSeek returns
            for Reddit threads, and a cookie-consent banner at the very start of the page. Only
            whole lines matching a known label of that site, or a leading run of consent-dialog
            lines, are removed; False returns the content as fetched.
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
        refresh=refresh,
        summary_chars=summary_chars,
        strip_boilerplate=strip_boilerplate,
    )


@_timed
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
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Search and return full page content in one request — the recommended retrieval path.

    This is search with need_content=True. Measured on 14 benchmark questions it returned content
    for PDFs and Reddit threads where deepseek_fetch returned HTTP 404, in about 0.7s per call.

    Responses are cached in memory for a few minutes and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 8 MiB fails with response_too_large.
    Every envelope, success or error, carries elapsed_ms: the call's wall time in milliseconds.

    Args:
        query: Search query, max 128 characters.
        count: Number of results to return, 1-50 (default 5). With dedupe on, extras are
            requested upstream so duplicates do not leave you short.
        ori_question: Original natural-language question, max 128 characters.
        lang: Language operator, e.g. 'es', 'de' or 'ja'. For non-English queries set it to the
            appropriate code; without it results may be mixed-language. For named entities test
            with and without the operator.
        max_chars_per_result: Cap per result on the content (and on a fallback summary).
        max_total_chars: Total content budget across all results.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
        refresh: Bypass the response cache and query upstream again (the envelope's cached
            flag says whether the results came from the cache).
        summary_chars: Cap on the summary given to a result that came back without content, at
            most max_chars_per_result (the default). 0 omits such summaries.
        strip_boilerplate: Remove known site-interface lines from page content (default on),
            such as the Chinese-localised comment-sort menu and sign-up footer DeepSeek returns
            for Reddit threads, and a cookie-consent banner at the very start of the page. Only
            whole lines matching a known label of that site, or a leading run of consent-dialog
            lines, are removed; False returns the content as fetched.
    """
    return await _search(
        query,
        count,
        ori_question,
        True,
        lang,
        max_chars_per_result,
        max_total_chars,
        dedupe,
        rewrite,
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
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Fetch one known URL and return its title and Markdown content.

    Caveat measured on 2026-10-20: this endpoint returns HTTP 404 for PDFs and for Reddit, and
    times out on some sites. When the URL is a document or a blocked site, call
    deepseek_search_and_read with the page title or a site:-scoped query instead.

    The response contains only title and content (with chars, its length) — no final URL, status
    code or publication date, so keep the provenance on your side.

    Pages are cached in memory for about half an hour and identical concurrent calls share one
    upstream request; the envelope's cached flag tells which happened. Calls are rate-limited
    locally per endpoint: if no slot frees up within a short wait the error code is
    rate_limited_local. A response larger than 32 MiB fails with response_too_large.
    Every envelope, success or error, carries elapsed_ms: the call's wall time in milliseconds.

    URLs that point at internal hosts or literal private addresses, or that embed credentials
    (user:pass@host), are rejected locally with ssrf_blocked before any request is made.

    Args:
        url: Absolute http(s) URL, max 8192 characters.
        clean: Remove DeepSeek's spurious Markdown backslash escapes (any escaped ASCII punctuation,
            such as \\( \\- \\[ \\/ \\#); fenced code blocks and inline code are left as they are.
        max_chars: Content cap, applied after cleaning.
        refresh: Bypass the response cache and fetch the page again (the envelope's cached
            flag says whether the page came from the cache).
        strip_boilerplate: Remove known site-interface lines from page content (default on),
            such as the Chinese-localised comment-sort menu and sign-up footer DeepSeek returns
            for Reddit threads, and a cookie-consent banner at the very start of the page. Only
            whole lines matching a known label of that site, or a leading run of consent-dialog
            lines, are removed; False returns the content as fetched.
    """
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
    return await _fetch(url, clean, max_chars, refresh=refresh, strip_boilerplate=strip_boilerplate)


_SEARCH_ADVICE = (
    "PDFs and Reddit are not extractable by fetch: use deepseek_search_and_read instead, "
    "which returns their content via the search index."
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


async def _fetch(
    url: str, clean: bool, max_chars: int, *, refresh: bool, strip_boilerplate: bool = True
) -> dict[str, Any]:
    """Fetch an already validated ``url`` through the cache, single-flight and limiter.

    Cleaning and ``strip_boilerplate`` shape the cached page and never change the cache key.

    Returns the ``deepseek_fetch`` envelope, or an error envelope from upstream or the limiter.
    """
    key = _fetch_cache_key(url)
    record, cached = await _cached_call(
        _get_fetch_cache(),
        _get_fetch_flight(),
        key,
        lambda: _fetch_upstream(url, key),
        refresh=refresh,
    )
    if record.get("ok") is False:
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
    content, truncated = _truncate(content, max_chars)

    return {
        "ok": True,
        "url": url,
        "title": _collapse_repeated_title(data.get("title")),
        "content": content,
        "chars": len(content),
        "truncated": truncated,
        "cached": cached,
        "notes": _fetch_notes(content, only_boilerplate),
    }


def _fetch_notes(content: str, only_boilerplate: bool) -> list[str]:
    """Explain an empty fetch: a page that was nothing but boilerplate, or one with no content at all."""
    if content:
        return []
    if only_boilerplate:
        return [
            "Empty content: the page rendered only a cookie-consent banner or site interface, which was "
            "removed. Try deepseek_search_and_read instead, or strip_boilerplate=False to see it."
        ]
    return ["Empty content: the page may be a PDF, JS-only or blocked. Try deepseek_search_and_read instead."]


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
    }


async def _fetch_isolated(
    url: str,
    clean: bool,
    max_chars: int,
    refresh: bool,
    semaphore: asyncio.Semaphore,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Run one batch fetch under ``semaphore``; unexpected exceptions become error envelopes."""
    async with semaphore:
        try:
            return await _fetch(url, clean, max_chars, refresh=refresh, strip_boilerplate=strip_boilerplate)
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
    items: list[_BatchItem], clean: bool, per_url: int, refresh: bool, strip_boilerplate: bool = True
) -> dict[int, asyncio.Task[dict[str, Any]]]:
    """Fetch every non-rejected item concurrently under the batch concurrency and deadline.

    Returns the tasks by item index; those still pending at the deadline are cancelled.
    """
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.batch_concurrency)
    tasks = {
        index: asyncio.create_task(
            _fetch_isolated(url, clean, per_url, refresh, semaphore, strip_boilerplate)
        )
        for index, (url, rejection) in enumerate(items)
        if rejection is None
    }
    try:
        if tasks:
            await asyncio.wait(tasks.values(), timeout=settings.batch_deadline)
    finally:
        pending = [task for task in tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return tasks


def _batch_results(
    items: list[_BatchItem], tasks: dict[int, asyncio.Task[dict[str, Any]]], max_total_chars: int
) -> tuple[list[dict[str, Any]], bool, int]:
    """Collect batch results in input order, trimming content to ``max_total_chars`` overall.

    Returns ``(results, trimmed, requests_used)``.
    """
    deadline_error = _fail(
        "deadline_exceeded", f"Not fetched within the {get_settings().batch_deadline:g} s batch deadline."
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
) -> dict[str, Any]:
    """Build the ``deepseek_fetch_many`` envelope from the finished batch."""
    results, trimmed, requests_used = _batch_results(items, tasks, max_total_chars)
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")
    if trimmed:
        notes.append(TRUNCATED_NOTE)
    failed_count = sum(not item["ok"] for item in results)
    cached_count = sum(item["cached"] for item in results)
    return {
        "ok": True,
        "count": len(results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "results": results,
        "notes": notes,
    }


@_timed
async def deepseek_fetch_many(
    urls: list[str],
    max_chars_per_url: int = 4000,
    max_total_chars: int = 20000,
    clean: bool = True,
    *,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Fetch up to 10 known URLs concurrently and return each page's title and Markdown content.

    Each URL goes through the same validation, cache, rate limiter and SSRF checks as
    deepseek_fetch. Empty entries are skipped and duplicates (same canonical URL) are fetched
    once. Results keep the input order; a failing URL yields an item with ok=false and an
    error instead of failing the batch. All fetches share one deadline (120 s by default);
    URLs still pending then are cancelled and reported with error code deadline_exceeded.

    Each page gets min(max_chars_per_url, max_total_chars // count) characters and the total
    content never exceeds max_total_chars; a note says when any content was truncated.
    The envelope reports count (unique URLs), requests_used (upstream fetches attempted),
    cached (true when no upstream request was made and at least one page came from the cache),
    cached_count (items served from the cache), failed_count (items with ok=false), partial
    (true when any item failed), results, notes and elapsed_ms (the call's wall time in
    milliseconds, also on error envelopes). The top-level ok only says the batch ran; check
    partial or each item's ok for per-URL failures. Each result carries its own cached flag.

    Args:
        urls: 1-10 absolute http(s) URLs, max 8192 characters each.
        max_chars_per_url: Content cap per page, applied after cleaning.
        max_total_chars: Total content budget across all pages.
        clean: Remove DeepSeek's spurious Markdown backslash escapes (any escaped ASCII punctuation);
            fenced code blocks and inline code are left as they are.
        refresh: Bypass the response cache and fetch every page again.
        strip_boilerplate: Remove known site-interface lines from page content (default on),
            such as the Chinese-localised comment-sort menu and sign-up footer DeepSeek returns
            for Reddit threads, and a cookie-consent banner at the very start of the page. Only
            whole lines matching a known label of that site, or a leading run of consent-dialog
            lines, are removed; False returns the content as fetched.
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
    tasks = await _run_batch(items, clean, per_url, refresh, strip_boilerplate)
    return _assemble_batch(items, tasks, duplicates, max_total_chars)


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
    envelope: dict[str, Any] = {
        "ok": True,
        "question": question,
        "query_used": search["query_used"],
        "sources": sources,
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "notes": notes,
        "failed_count": fill.failed,
        "partial": fill.failed > 0,
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
    lang: str | None = None,
    rewrite: bool = False,
    dedupe: bool = True,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Single-pass source gathering: one search with page content, then direct fetches for the gaps.

    It does not iterate, refine the query or summarise: there is exactly one search, no follow-up
    searches, and the sources come back as raw content for the caller to read and synthesise.

    Use it for reconnaissance: finding candidate sources. There is no quality filter, so results can
    include low-quality or secondary pages (student papers, aggregators, clickbait); verify facts in
    the primary source before citing them. For primary sources, ask in English for the primary study
    and/or use site: for local sources. authority is an upstream field, not a credibility score.

    The search runs with need_content=True. By default it makes no model call;
    rewrite=True translates the question to English first, as in deepseek_web_search.
    Results whose content is missing or shorter than 200 characters and whose URL passes
    validation are fetched directly with deepseek_fetch_many (at most 5 per call); a fetched page
    replaces the search content of that source. A URL that fails validation (for example an
    internal address) is not fetched. Failed or rejected sources keep the search content and set the
    source's error; the call still succeeds with partial=true.

    Each source has id, title, url, date, date_source, site, authority, rerank_score, content, chars,
    origin ('search' or 'fetch'), error (null or {code, message}) and cached (whether its content came
    from the cache). date_source is 'upstream', 'content' (a YYYY-MM-DD or "Month YYYY" date found at the
    start of the title or content because the upstream gave none) or null.
    query_used is the query sent upstream (with any lang: operator); query_original is set only when
    rewrite changed the question.
    Content is capped at per_page_chars per source and total_chars overall, in source order; a
    note says when any content was truncated.
    requests_used counts upstream requests made (the search plus each fetch; cache hits are free)
    and cached_count counts responses served from the cache; cached is true when no upstream request
    was made because everything came from the cache. failed_count counts sources whose
    error is set and partial is true when it is non-zero. elapsed_ms is the call's wall time in
    milliseconds, on error envelopes too. Synthesis is left to the caller.

    Args:
        question: Search query, max 128 characters. Supports site:, -site:, "phrase", lang:.
        max_pages: Number of search results to use, 1-50 (default 5).
        per_page_chars: Content cap per source.
        total_chars: Total content budget across all sources.
        lang: Explicit language operator such as 'pl', 'es' or 'pt-BR', added to the question as
            lang:<code> unless it already has one. query_used shows the query that was sent.
        rewrite: Translate the question to English with deepseek-flash before searching (default
            False). Skipped when lang is set; query_original then holds the question as given.
        dedupe: Remove duplicate results (default True).
        refresh: Bypass the response caches for the search and the fetches.
        strip_boilerplate: Remove known site-interface lines from page content (default on),
            such as the Chinese-localised comment-sort menu and sign-up footer DeepSeek returns
            for Reddit threads, and a cookie-consent banner at the very start of the page. Only
            whole lines matching a known label of that site, or a leading run of consent-dialog
            lines, are removed; False returns the content as fetched.
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
async def deepseek_balance() -> dict[str, Any]:
    """Check the DeepSeek account balance (granted and topped-up, in CNY and USD).

    The envelope, success or error, carries elapsed_ms: the call's wall time in milliseconds.
    """
    result = await _request("GET", "/user/balance", timeout=BALANCE_TIMEOUT)
    if not result.get("ok"):
        return result
    data = result["data"]
    return {
        "ok": True,
        "is_available": data.get("is_available"),
        "balances": data.get("balance_infos") or data.get("balances") or data,
    }


TOOLS: tuple[Callable[..., Awaitable[dict[str, Any]]], ...] = (
    deepseek_web_search,
    deepseek_search_and_read,
    deepseek_fetch,
    deepseek_fetch_many,
    deepseek_research,
    deepseek_balance,
)


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
        server.add_tool(tool)
    return server


mcp = create_server()
