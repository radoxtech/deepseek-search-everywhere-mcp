"""MCP server for the DeepSeek Search & Fetch API.

Exposes web search, page fetch (single and batch), search-and-read, research and balance tools.
``create_server()`` builds a FastMCP instance with every tool registered; the module-level
``mcp`` serves stdio, and ``deepseek_search_everywhere_mcp.http_app`` builds fresh ones for HTTP.
Requires DEEPSEEK_API_KEY; DEEPSEEK_BASE_URL overrides the endpoint.

The command-line entry point lives in ``deepseek_search_everywhere_mcp.cli``.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import re
import time
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal, ParamSpec

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from ._version import __version__
from .boilerplate import remove_boilerplate
from .budget import DailyBudgetTracker, Reservation
from .cache import SingleFlight, TTLCache
from .deep_research import (
    DEPTH_CONFIG,
    ResearchJob,
    apply_token_guard,
    build_condense_messages,
    build_decomposition_messages,
    build_r2_gap_messages,
    build_synthesis_messages,
    citation_ids,
    clean_source_title,
    clean_source_url,
    filter_synthesis_output,
    get_job_store,
    normalize_citations,
    parse_subqueries,
    rank_and_diversify_sources,
    report_body_words,
    select_synthesis_context,
    source_access,
    source_identifiers,
    validate_r2_queries,
)
from .deep_research import (
    reset_state as reset_deep_research_state,
)
from .egress import check_fetch_egress, check_search_egress
from .errors import _fail
from .errors import _map_http_error as _map_http_error  # re-exported for callers and tests
from .extract import (
    build_extract_prompt,
    sanitize_json,
    validate_fields,
    validate_schema,
    verify_evidence,
)
from .filters import MAX_QUERY, compile_filters, filter_results_by_domain, fit_query
from .ratelimit import CHAT, FETCH, SEARCH, RateLimiter, RateLimitExceeded
from .redact import redact
from .sanitize import (
    sanitize_structure,
    sanitize_untrusted_text,
    strip_hidden_unicode,
)
from .settings import (
    DEFAULT_LOG_LEVEL,
    get_pricing_period,
    get_settings,
)
from .snippets import MAX_SNIPPETS, extract_snippets, split_sections
from .transport import _api_key, _base_url
from .transport import request as _request  # tests monkeypatch server._request
from .urls import (
    CODE_INVALID,
    UrlError,
    _canonical_url,
    _url_etld1,
    validate_url,
)

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

MIN_TITLE_SITE_CHARS = 20
MIN_TITLE_SITE_WORDS = 3
MAX_COUNT = 50  # largest ``count`` accepted from callers and by the upstream search API
MAX_QUERIES = 5  # most queries accepted by deepseek_search_many
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
    reset_deep_research_state()


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


async def _chat_request(
    messages: list[dict[str, Any]],
    *,
    thinking: dict[str, Any],
    model: str = REWRITE_MODEL,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
    timeout: float = CHAT_TIMEOUT,
    deadline: float | None = None,
    attempts: int = 2,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Execute a chat completion request through rate limiting and budget reservation.

    'thinking' is a required keyword-only argument with no default.
    Surfaces finish_reason, usage-based cost, and pricing_period.
    """
    if not isinstance(thinking, dict):
        raise ValueError("'thinking' must be a dictionary.")

    try:
        ticket = await _get_rate_limiter().acquire(CHAT)
    except RateLimitExceeded as exc:
        return exc.to_envelope()

    budget = _get_budget_tracker()
    reservation: Reservation | None = None
    settings = get_settings()
    est_cost = 0.001
    if budget is not None:
        est_input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 2
        est_output_tokens = max_tokens if max_tokens is not None else 2048
        peak_rates = settings.llm_pricing.peak
        est_cost = max(
            0.0001,
            (est_input_tokens / 1_000_000) * peak_rates.cache_miss_per_1m
            + (est_output_tokens / 1_000_000) * peak_rates.output_per_1m,
        )
        reservation = await budget.reserve(est_cost)
        if reservation is None:
            ticket.refund()
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${budget.limit_usd:.2f} exceeded.",
                402,
            )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "thinking": thinking,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    period = get_pricing_period()
    rates = settings.llm_pricing.peak if period == "peak" else settings.llm_pricing.off_peak

    try:
        result = await _request(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            timeout=timeout,
            deadline=deadline,
            attempts=attempts,
        )
    except BaseException:
        ticket.refund()
        if reservation is not None:
            await reservation.release()
        raise

    if not result.get("ok"):
        if reservation is not None:
            await reservation.release()
        return result

    data = result.get("data") or {}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning_content = msg.get("reasoning_content")
    finish_reason = choice.get("finish_reason")
    usage = data.get("usage") or {}

    notes: list[str] = []

    if usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        comp_tokens = int(usage.get("completion_tokens", 0))
        hit_tokens = int(usage.get("prompt_cache_hit_tokens", 0))
        miss_tokens = int(usage.get("prompt_cache_miss_tokens", max(0, prompt_tokens - hit_tokens)))
        cost_usd = (
            (hit_tokens / 1_000_000) * rates.cache_hit_per_1m
            + (miss_tokens / 1_000_000) * rates.cache_miss_per_1m
            + (comp_tokens / 1_000_000) * rates.output_per_1m
        )
    else:
        cost_usd = est_cost
        notes.append("Usage missing from upstream response; cost estimated.")

    cost_usd = round(cost_usd, 6)
    if reservation is not None:
        await reservation.commit(cost_usd)

    return {
        "ok": True,
        "content": content,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "usage": usage,
        "cost_usd": cost_usd,
        "pricing_period": period,
        "notes": notes,
        "raw": data,
    }


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


def _is_https_url(item: dict[str, Any]) -> bool:
    url = item.get("url")
    return isinstance(url, str) and url.lower().startswith("https://")


def _dedupe_results(items: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    seen_keys: dict[tuple, int] = {}
    seen_title_sites: dict[tuple[str, str], int] = {}

    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue

        key = _dedupe_key(item)
        title_site = _title_site_key(item)

        target_idx: int | None = None
        if key is not None and key in seen_keys:
            target_idx = seen_keys[key]
        elif title_site is not None and title_site in seen_title_sites:
            target_idx = seen_title_sites[title_site]

        if target_idx is not None:
            removed += 1
            if _is_https_url(item) and not _is_https_url(kept[target_idx]):
                kept[target_idx] = item
            continue

        idx = len(kept)
        if key is not None:
            seen_keys[key] = idx
        if title_site is not None:
            seen_title_sites[title_site] = idx
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
        cleaned, _ = strip_hidden_unicode(title)
        return cleaned
    fallback = _url_host(item.get("url")) or title
    if isinstance(fallback, str):
        cleaned_fb, _ = strip_hidden_unicode(fallback)
        return cleaned_fb
    return fallback


MAX_PRE_PROCESSING_CHARS = 2 * 1024 * 1024  # 2 MB ceiling


def _cap_raw_content(text: str, max_chars: int) -> str:
    """Cap untrusted text before expensive regex unescaping and boilerplate passes.

    Protects against ReDoS and memory exhaustion on huge pages while
    preserving enough buffer for boilerplate removal and Markdown unescaping
    to produce the requested ``max_chars``.
    """
    if not text:
        return ""
    pre_cap = min(len(text), max(4 * max_chars, MAX_PRE_PROCESSING_CHARS))
    return text[:pre_cap]


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
    """Decode JSON escapes in a search snippet or summary, drop HTML, and sanitize untrusted text."""
    cleaned = _strip_html_tags(_decode_json_escapes(text))
    cleaned, _ = sanitize_untrusted_text(cleaned)
    return cleaned


def _with_lang(query: str, lang: str | None) -> tuple[str, bool]:
    return fit_query(query, lang=lang)


async def _rewrite_query(query: str) -> str | None:
    try:
        result = await _chat_request(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            thinking={"type": "disabled"},
            model=REWRITE_MODEL,
            max_tokens=64,
            temperature=0,
            timeout=CHAT_TIMEOUT,
            attempts=1,
        )
        if not result.get("ok"):
            return None
        content = result.get("content")
        if not isinstance(content, str):
            return None
        first = next((line.strip() for line in content.splitlines() if line.strip()), "")
        if len(first) >= 2 and first[0] in "\"'" and first[-1] in "\"'":
            first = first[1:-1]
        cleaned, _ = fit_query(first)
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
    reservation = None
    if budget is not None:
        reservation = await budget.reserve(COST_PER_REQUEST)
        if reservation is None:
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${budget.limit_usd:.2f} exceeded.",
                402,
            )
    try:
        result = await _limited_request(SEARCH, SEARCH_PATH, payload, SEARCH_TIMEOUT)
    except BaseException:
        if reservation is not None:
            await reservation.release()
        raise

    if not result.get("ok"):
        if reservation is not None:
            await reservation.release()
        return result

    if reservation is not None:
        await reservation.commit(COST_PER_REQUEST)

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
    egress_err = check_search_egress(query, get_settings())
    if egress_err is not None:
        return _fail("egress_blocked", egress_err)
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
    ``summary_cap`` and is left out entirely when that cap is 0. Without content (snippets mode), a
    summary is only a fallback for results that have no snippet. With content, ``chars`` is the
    length returned.
    Returns ``(remaining, truncated)``: the budget left afterwards and whether any text was cut.
    """
    body = item.get("content") if need_content else None
    if body:
        budget_target = max(0, min(per_result, remaining))
        capped_body = _cap_raw_content(body, budget_target)
        cleaned = _strip_html_tags(_decode_json_escapes(_unescape_markdown(capped_body)))
        if strip_boilerplate:
            cleaned = remove_boilerplate(cleaned, item.get("url"))
        cleaned, _ = sanitize_untrusted_text(cleaned)
        entry["content"], cut = _truncate(cleaned, budget_target)
        entry["chars"] = len(entry["content"])
        return remaining - len(entry["content"]), cut
    if not need_content and entry.get("snippet"):
        # Snippets mode: the snippet is the result text, so the upstream summary is left out.
        return remaining, False
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
    domain_dropped: int = 0,
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
    if domain_dropped:
        notes.append(
            f"Dropped {domain_dropped} result(s) not matching the include_domains/exclude_domains "
            "domain filter (the search ignored it)."
        )
    if truncated:
        notes.append(TRUNCATED_NOTE)
    return [redact(note) for note in notes]


async def _search(
    query: str,
    count: int,
    ori_question: str | None,
    content: Literal["snippets", "full"] | bool,
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

    if isinstance(content, bool):
        content = "full" if content else "snippets"
    if content not in ("snippets", "full"):
        return _fail("bad_request", "content must be 'snippets' or 'full'.")
    need_content = content == "full"

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
    try:
        used, trimmed = fit_query(query, lang=lang, filter_ops=ops_string)
    except ValueError:
        return _fail("query_too_long", f"Filters and language exceed the {MAX_QUERY}-character limit.")

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
    raw, domain_dropped = filter_results_by_domain(raw, include_domains, exclude_domains)
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
        "content": content,
        "deduped": removed,
        "cached": cached,
        "requests_used": 0 if cached else 1,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": shaped,
        "notes": _search_notes(
            used,
            query_original,
            rewrite_failed,
            removed,
            trimmed,
            lang,
            truncated and need_content,
            filter_ops=ops_string,
            domain_dropped=domain_dropped,
        ),
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope


async def _fetch_upstream(url: str, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``fetch`` token, call upstream, cache a non-empty page.

    Returns ``{"data": ...}`` with the raw upstream page on success or an error envelope.
    """
    budget = _get_budget_tracker()
    reservation = None
    if budget is not None:
        reservation = await budget.reserve(COST_PER_REQUEST)
        if reservation is None:
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${budget.limit_usd:.2f} exceeded.",
                402,
            )
    try:
        result = await _limited_request(FETCH, FETCH_PATH, {"url": url}, FETCH_TIMEOUT)
    except BaseException:
        if reservation is not None:
            await reservation.release()
        raise

    if not result.get("ok"):
        if reservation is not None:
            await reservation.release()
        return result

    if reservation is not None:
        await reservation.commit(COST_PER_REQUEST)

    record = {"data": result["data"]}
    cache = _get_fetch_cache()
    if cache is not None and result["data"].get("content"):
        cache.set(key, record)
    return record


@_timed
async def deepseek_search(
    query: Annotated[
        str,
        Field(description="Search terms or question; operators like site: or lang: supported (<= 128 chars)"),
    ],
    *,
    count: Annotated[
        int, Field(description=f"Number of ranked results to return (1-{MAX_COUNT}, default 5)")
    ] = 5,
    content: Annotated[
        Literal["snippets", "full"],
        Field(
            description="Result format: 'snippets' (default: snippet, else summary capped at 400 chars) "
            "or 'full' (gap-filled text)"
        ),
    ] = "snippets",
    lang: Annotated[
        str | None,
        Field(description="Restrict results to ISO language code like 'en', 'de', 'pl' (default None)"),
    ] = None,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters of page text across results (default 30000)"),
    ] = 30000,
    rewrite: Annotated[
        bool,
        Field(description="Rewrite search query via deepseek-flash for better retrieval (default False)"),
    ] = False,
    exclude_terms: Annotated[
        list[str] | None,
        Field(description="Terms that must not appear in search results (default None)"),
    ] = None,
    exact_phrases: Annotated[
        list[str] | None,
        Field(description="Phrases that must appear verbatim in search results (default None)"),
    ] = None,
    after: Annotated[
        str | None,
        Field(description="Filter results published on or after YYYY-MM-DD date (default None)"),
    ] = None,
    before: Annotated[
        str | None,
        Field(description="Filter results published on or before YYYY-MM-DD date (default None)"),
    ] = None,
    include_domains: Annotated[
        list[str] | None,
        Field(description="Restrict search to specific domains or sites (default None)"),
    ] = None,
    exclude_domains: Annotated[
        list[str] | None,
        Field(description="Exclude specific domains from search results (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh results from API (default False)"),
    ] = False,
) -> dict[str, Any]:
    """Search the web with DeepSeek index and return ranked results, snippets, and page text.

    Primary search and retrieval tool. Use for finding online facts, web pages, and sources
    for a single query. Returns snippets (default) or full page text via gap-fill. Untrusted
    content is sanitized.

    When NOT to use: Do not use for multiple parallel queries (use deepseek_search_many),
    for reading known URLs (use deepseek_fetch), for structured schema extraction (use
    deepseek_extract), or for deep multi-query topic reports (use deepseek_research).

    Returns: {"ok": True, "results": [...], "count": int, "cost_usd": float, "notes": [...]}.
    """
    if max_total_chars < 1:
        return _fail("bad_request", "max_total_chars must be at least 1.")
    per_result = max(500, max_total_chars // max(1, count))
    res = await _search(
        query,
        count,
        None,
        content,
        lang,
        per_result,
        max_total_chars,
        dedupe=True,
        rewrite=rewrite,
        exclude_terms=exclude_terms,
        exact_phrases=exact_phrases,
        after=after,
        before=before,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        refresh=refresh,
        summary_chars=None,
        strip_boilerplate=True,
    )
    if res.get("ok") and (content == "full" or content is True):
        fill = await _fill_gaps(
            res.get("results") or [], per_result, max_total_chars, refresh, strip_boilerplate=True
        )
        if fill.requests_used:
            res["requests_used"] = (0 if res.get("cached") else 1) + fill.requests_used
            res["cost_usd"] = round(
                (0.0 if res.get("cached") else COST_PER_REQUEST) + fill.requests_used * COST_PER_REQUEST, 6
            )
            if fill.fetched:
                res["notes"].append(f"fetched {fill.fetched} page(s) with missing or short search content")
            if fill.failed:
                res["notes"].append(f"{fill.failed} fetch(es) failed; see the source's error")
    return res


@_timed
async def deepseek_search_many(
    queries: Annotated[
        list[str],
        Field(
            description=f"Search queries to execute concurrently (1-{MAX_QUERIES} queries, each <= 128 chars)"
        ),
    ],
    *,
    count: Annotated[
        int, Field(description=f"Number of ranked results to return (1-{MAX_COUNT}, default 5)")
    ] = 5,
    content: Annotated[
        Literal["snippets", "full"],
        Field(
            description="Result format: 'snippets' (default: snippet, else summary capped at 400 chars) "
            "or 'full' (gap-filled text)"
        ),
    ] = "snippets",
    lang: Annotated[
        str | None,
        Field(description="Restrict results to ISO language code like 'en', 'de', 'pl' (default None)"),
    ] = None,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters of page text across results (default 30000)"),
    ] = 30000,
    rewrite: Annotated[
        bool,
        Field(description="Rewrite search query via deepseek-flash for better retrieval (default False)"),
    ] = False,
    exclude_terms: Annotated[
        list[str] | None,
        Field(description="Terms that must not appear in search results (default None)"),
    ] = None,
    exact_phrases: Annotated[
        list[str] | None,
        Field(description="Phrases that must appear verbatim in search results (default None)"),
    ] = None,
    after: Annotated[
        str | None,
        Field(description="Filter results published on or after YYYY-MM-DD date (default None)"),
    ] = None,
    before: Annotated[
        str | None,
        Field(description="Filter results published on or before YYYY-MM-DD date (default None)"),
    ] = None,
    include_domains: Annotated[
        list[str] | None,
        Field(description="Restrict search to specific domains or sites (default None)"),
    ] = None,
    exclude_domains: Annotated[
        list[str] | None,
        Field(description="Exclude specific domains from search results (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh results from API (default False)"),
    ] = False,
    deadline: Annotated[
        float | None,
        Field(description="Overall timeout deadline in seconds; completed results returned if reached"),
    ] = None,
) -> dict[str, Any]:
    """Search the web with multiple queries concurrently and return ranked results per query.

    Batch multi-query search across 1-5 queries in parallel. Use when investigating multiple
    search facets, query formulations, or sub-topics at the same time. Preserves input order
    and deduplicates duplicate queries. Untrusted content is sanitized.

    When NOT to use: Do not use for a single search query (use deepseek_search), for reading
    known URLs (use deepseek_fetch_many), for structured schema extraction (use
    deepseek_extract_many), or for autonomous topic synthesis (use deepseek_research).

    Returns: {"ok": True, "results": [{"query": str, "results": [...]}], "cost_usd": float}.
    """
    if isinstance(content, bool):
        content = "full" if content else "snippets"
    if content not in ("snippets", "full"):
        return _fail("bad_request", "content must be 'snippets' or 'full'.")
    if not isinstance(queries, list) or not queries:
        return _fail("bad_request", "queries must be a non-empty list of strings.")
    if len(queries) > MAX_QUERIES:
        return _fail("bad_request", f"queries must contain between 1 and {MAX_QUERIES} entries.")
    if not all(isinstance(q, str) for q in queries):
        return _fail("bad_request", "all items in queries must be strings.")
    if count < 1 or count > MAX_COUNT:
        return _fail("bad_count", f"count must be between 1 and {MAX_COUNT}.")
    if max_total_chars < 1:
        return _fail("bad_request", "max_total_chars must be at least 1.")

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
        return await deepseek_search(
            q,
            count=count,
            content=content,
            lang=lang,
            max_total_chars=per_query_chars,
            rewrite=rewrite,
            exclude_terms=exclude_terms,
            exact_phrases=exact_phrases,
            after=after,
            before=before,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            refresh=refresh,
        )

    tasks = [asyncio.create_task(_run_one(q)) for q in unique_queries]
    timeout = deadline if deadline is not None else SEARCH_TIMEOUT
    try:
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    deadline_error = _fail("deadline_exceeded", f"Search query timed out after {timeout:g} s.")
    results: list[dict[str, Any]] = []
    requests_used = 0
    cached_count = 0
    failed_count = 0
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate query/queries")

    for q, task in zip(unique_queries, tasks, strict=True):
        if task.cancelled():
            res = deadline_error
        elif task.exception() is not None:
            res = _fail("network_error", redact(str(task.exception())))
        else:
            res = task.result()

        if isinstance(res, dict) and res.get("ok"):
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
        "content": content,
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
    rejection = _reject_url(url)
    if rejection is not None:
        return rejection
    egress_err = check_fetch_egress(url, get_settings())
    if egress_err is not None:
        return _fail("egress_blocked", egress_err)
    return None


@_timed
async def deepseek_fetch(
    url: Annotated[
        str,
        Field(description="HTTP or HTTPS URL to fetch (must be public, no private IP/intranet)"),
    ],
    *,
    clean: Annotated[
        bool,
        Field(description="Remove Markdown backslash escapes from the page text (default True)"),
    ] = True,
    max_chars: Annotated[
        int,
        Field(description="Max characters of page text to return (default 20000)"),
    ] = 20000,
    offset: Annotated[
        int | None,
        Field(description="Character offset to start reading from (mutually exclusive with selectors)"),
    ] = None,
    limit: Annotated[
        int | None,
        Field(description="Max characters to return from offset (mutually exclusive with selectors)"),
    ] = None,
    find_text: Annotated[
        str | None,
        Field(description="Locate text snippet and return surrounding context (mutually exclusive)"),
    ] = None,
    query: Annotated[
        str | None,
        Field(description="Search page content for query terms and extract best matching snippets"),
    ] = None,
    section: Annotated[
        int | None,
        Field(description="1-based section number to extract from document (mutually exclusive)"),
    ] = None,
    max_snippets: Annotated[
        int,
        Field(description=f"Max snippets to return when using query selector (1-{MAX_SNIPPETS}, default 3)"),
    ] = 3,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh page from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners, navigation and site UI boilerplate (default True)"),
    ] = True,
    fallback_to_search: Annotated[
        bool,
        Field(description="On 404, fall back to searching exact URL in index (default True)"),
    ] = True,
) -> dict[str, Any]:
    """Fetch one known URL and return its title and Markdown content.

    Direct retrieval tool for reading a single known web page. Supports section slicing,
    keyword snippet extraction, and character offsets. Automatically attempts search index
    fallback on 404. Untrusted content is sanitized (images neutralized, hidden text stripped).

    When NOT to use: Do not use for discovering pages by topic (use deepseek_search),
    for fetching multiple URLs in parallel (use deepseek_fetch_many), for structured entity
    or schema extraction (use deepseek_extract), or for topic synthesis (use deepseek_research).

    Returns: {"ok": True, "url": str, "title": str, "content": str, "cost_usd": float, "notes": [...]}.
    """
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
    selectors = [
        section is not None,
        query is not None,
        offset is not None or limit is not None,
        find_text is not None,
    ]
    if sum(bool(s) for s in selectors) > 1:
        return _fail(
            "bad_request", "Selectors section, query, offset/limit, and find_text are mutually exclusive."
        )
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


def _is_canonical_match(target_url: str, candidate_url: str) -> bool:
    """Check if candidate_url matches target_url canonically.

    Must share the same normalized host (no parent/subdomain match).
    For PDFs: same host and same filename (ignoring optional .pdf extension).
    For non-PDFs: same host and same normalized path.
    """
    try:
        t_parts = urllib.parse.urlsplit(target_url)
        c_parts = urllib.parse.urlsplit(candidate_url)
    except ValueError:
        return False

    t_host = (t_parts.hostname or "").lower()
    c_host = (c_parts.hostname or "").lower()
    if t_host.startswith("www."):
        t_host = t_host[4:]
    if c_host.startswith("www."):
        c_host = c_host[4:]
    if not t_host or t_host != c_host:
        return False

    is_pdf = t_parts.path.lower().endswith(".pdf") or _PDF_ID_PATH.search(t_parts.path.lower()) is not None

    t_path = t_parts.path.rstrip("/")
    c_path = c_parts.path.rstrip("/")

    if is_pdf:
        t_filename = t_path.split("/")[-1].removesuffix(".pdf") if t_path else ""
        c_filename = c_path.split("/")[-1].removesuffix(".pdf") if c_path else ""
        return bool(t_filename and t_filename == c_filename)

    return t_path == c_path


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
    try:
        search_query, _ = fit_query(query_terms, filter_ops=f"site:{host_for_site}")
    except ValueError:
        return None

    search_res = await _search(
        search_query,
        count=2,
        ori_question=None,
        content="full",
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
        if not _is_canonical_match(url, res_url):
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
    selectors = [
        section is not None,
        query is not None,
        offset is not None or limit is not None,
        find_text is not None,
    ]
    if sum(bool(s) for s in selectors) > 1:
        return _fail(
            "bad_request", "Selectors section, query, offset/limit, and find_text are mutually exclusive."
        )
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
            else:
                return _with_search_advice(record, url)
        else:
            return _with_search_advice(record, url)

    data = record["data"]
    raw_content = data.get("content") or ""
    raw_len = len(raw_content)
    content = _cap_raw_content(raw_content, max_chars)
    content_capped = raw_len > len(content)
    capped_limit = len(content)
    if clean:
        content = _unescape_markdown(content)
    only_boilerplate = False
    if strip_boilerplate:
        stripped = remove_boilerplate(content, url)
        only_boilerplate = bool(content.strip()) and not stripped
        content = stripped
    content, sanitize_notes = sanitize_untrusted_text(content)

    total_sections: int | None = None
    section_has_more: bool | None = None
    section_index: int | None = None
    snippets_count: int | None = None
    no_match_note: str | None = None

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
        had_content = bool(content.strip())
        content, snippets_count = extract_snippets(
            content, query, max_snippets=max_snippets, max_chars=max_chars
        )
        if had_content and snippets_count == 0:
            no_match_note = (
                f"No excerpt contains the query terms: '{query}'. The page was fetched but nothing matched."
            )

    total_chars = raw_len if content_capped else len(content)
    notes = [no_match_note] if no_match_note else _fetch_notes(content, only_boilerplate)
    notes.extend(sanitize_notes)
    if content_capped:
        notes.append(
            f"Content was capped at {capped_limit} characters before processing "
            f"(original document length: {raw_len} chars)."
        )

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
        if offset is not None and offset >= capped_limit:
            if content_capped:
                notes.append(
                    f"offset {offset} is beyond the capped content limit ({capped_limit} chars); "
                    f"original length is {raw_len} chars."
                )
            else:
                notes.append(f"offset {offset} is beyond the total document length ({capped_limit} chars).")
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
    source = "search_index" if (fallback_used or record.get("fallback_from_search")) else "fetch"
    envelope: dict[str, Any] = {
        "ok": True,
        "url": url,
        "source": source,
        "title": (
            strip_hidden_unicode(_collapse_repeated_title(data.get("title")))[0]
            if isinstance(_collapse_repeated_title(data.get("title")), str)
            else _collapse_repeated_title(data.get("title"))
        ),
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
    if content_capped:
        envelope["content_capped"] = True
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
    urls: Annotated[
        list[str],
        Field(description="List of HTTP/HTTPS URLs to fetch in parallel (max 10 URLs)"),
    ],
    *,
    max_chars_per_url: Annotated[
        int,
        Field(description="Max characters of page text returned per URL (default 4000)"),
    ] = 4000,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters across all fetched URLs (default 20000)"),
    ] = 20000,
    clean: Annotated[
        bool,
        Field(description="Remove Markdown backslash escapes from the page text (default True)"),
    ] = True,
    query: Annotated[
        str | None,
        Field(description="Optional query to extract relevant snippets from each page (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners and site UI boilerplate from content (default True)"),
    ] = True,
    deadline: Annotated[
        float | None,
        Field(description="Overall timeout deadline in seconds; partial results returned on expiry"),
    ] = None,
    fallback_to_search: Annotated[
        bool,
        Field(description="On 404, fall back to searching exact URL in index (default True)"),
    ] = True,
) -> dict[str, Any]:
    """Fetch up to 10 known URLs concurrently and return each page's title and Markdown content.

    Batch page retrieval across 1-10 known URLs in parallel. Use when target URLs are already
    known and full text or targeted snippets across pages are required. Skips blanks,
    deduplicates URLs, and preserves order. Untrusted content is sanitized.

    When NOT to use: Do not use for query-based web discovery (use deepseek_search_many),
    for fetching a single URL with section/offset selectors (use deepseek_fetch), for structured
    entity extraction (use deepseek_extract_many), or for research synthesis (use deepseek_research).

    Returns: {"ok": True, "results": [{"url": str, "content": str, ...}], "cost_usd": float}.
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


def _format_public_sources(
    sources: list[dict[str, Any]],
    *,
    include_snippet: bool = False,
) -> list[dict[str, Any]]:
    """Format research sources into a lean public list omitting full page content.

    Each item carries the identifiers extracted from the URL/content (``doi``, ``pmid``,
    ``pmcid``) and ``access``: ``full_text`` for a fetched page, ``abstract`` for search content
    of at least 200 characters and ``snippet_only`` otherwise.
    On success, sources carry no page content; on fallback paths, a 400-character sanitized
    snippet is included.
    """
    formatted: list[dict[str, Any]] = []
    for s in sources:
        title = s.get("title")
        if isinstance(title, str):
            title, _ = sanitize_untrusted_text(title)
        item: dict[str, Any] = {
            "id": s.get("id"),
            "title": title,
            "url": s.get("url"),
            "date": s.get("date"),
            "doi": s.get("doi"),
            "pmid": s.get("pmid"),
            "pmcid": s.get("pmcid"),
            "access": s.get("access"),
        }
        if include_snippet:
            raw_text = s.get("snippet") or s.get("content") or ""
            snip = raw_text[:400].strip() if isinstance(raw_text, str) else ""
            if snip:
                snip, _ = sanitize_untrusted_text(snip)
            item["snippet"] = snip
        formatted.append(item)
    return formatted


# Search results whose content is shorter than this are treated as gaps and fetched directly.
RESEARCH_GAP_CHARS = 200
# Default content budgets for deepseek_research: per source and across all sources.
RESEARCH_PAGE_CHARS = 6000
RESEARCH_TOTAL_CHARS = 30000


def _research_source(index: int, item: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Turn one shaped search result into a research source (``origin`` = ``search``)."""
    url = clean_source_url(item["url"]) if isinstance(item.get("url"), str) else item.get("url")
    content = item.get("content") or ""
    return {
        "id": index,
        "title": clean_source_title(item.get("title"), url or ""),
        "url": url,
        "date": item.get("date"),
        "date_source": item.get("date_source"),
        "site": item.get("site"),
        "authority": item.get("authority"),
        "rerank_score": item.get("rerank_score"),
        "content": content,
        "chars": 0,
        "origin": "search",
        "error": None,
        "cached": cached,
        **source_identifiers(url or "", content),
        "access": source_access("search", content),
    }


def _gap_check(source: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Classify a source: ``(url, None)`` to fetch, ``(None, error)`` if unfetchable, else no gap."""
    content = source.get("content") or ""
    if len(content) >= RESEARCH_GAP_CHARS:
        return None, None
    url = source["url"].strip() if isinstance(source.get("url"), str) else ""
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


def _select_gaps(
    sources: list[dict[str, Any]], fill: _GapFill, max_pages: int | None = None
) -> dict[str, dict[str, Any]]:
    """Pick the sources to fetch, keyed by canonical URL; mark rejected ones in place.

    At most ``max_pages`` distinct URLs are selected; rejections count as failures.
    """
    limit = max_pages if max_pages is not None else get_settings().research_max_pages
    gaps: dict[str, dict[str, Any]] = {}
    for source in sources:
        url, rejection = _gap_check(source)
        if rejection is not None:
            source["error"] = rejection
            fill.failed += 1
        key = None if url is None else (_canonical_url(url) or url)
        if key is not None and key not in gaps and len(gaps) < limit:
            gaps[key] = source
    return gaps


async def _fill_gaps(
    sources: list[dict[str, Any]],
    per_page_chars: int,
    total_chars: int,
    refresh: bool,
    strip_boilerplate: bool = True,
    deadline: float | None = None,
    max_pages: int | None = None,
    internal_batch: bool = False,
) -> _GapFill:
    """Fetch sources with missing or short content and merge the fetched pages in place."""
    fill = _GapFill()
    gaps = _select_gaps(sources, fill, max_pages=max_pages)
    if not gaps:
        return fill
    if internal_batch:
        items: list[_BatchItem] = [(source["url"].strip(), None) for source in gaps.values()]
        tasks = await _run_batch(
            items,
            clean=True,
            per_url=per_page_chars,
            refresh=refresh,
            strip_boilerplate=strip_boilerplate,
            deadline=deadline if deadline is not None else get_settings().research_gap_deadline,
            fallback_to_search=True,
        )
        batch = _assemble_batch(
            items,
            tasks,
            duplicates=0,
            max_total_chars=total_chars,
            deadline=deadline if deadline is not None else get_settings().research_gap_deadline,
        )
    else:
        batch = await deepseek_fetch_many(
            [source["url"].strip() for source in gaps.values()],
            max_chars_per_url=per_page_chars,
            max_total_chars=total_chars,
            refresh=refresh,
            strip_boilerplate=strip_boilerplate,
            deadline=deadline if deadline is not None else get_settings().research_gap_deadline,
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
            source.update(source_identifiers(source["url"] or "", source["content"]))
            source["access"] = source_access("fetch", source["content"])
            if source.get("date") is None and (found := _content_date(None, item["content"])):
                source["date"], source["date_source"] = found, "content"
            fill.fetched += 1
    return fill


@_timed
async def deepseek_research(
    topic: Annotated[
        str,
        Field(description="Complex research topic or question to investigate (non-empty, max 500 chars)"),
    ],
    *,
    depth: Annotated[
        Literal["low", "medium", "high", "max"],
        Field(description="Depth tier controlling queries, sources and reasoning (default 'medium')"),
    ] = "medium",
    lang: Annotated[
        str | None,
        Field(description="Two-letter ISO language code to restrict search sources to (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and perform fresh research pipeline (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove boilerplate from fetched source pages before synthesis (default True)"),
    ] = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Execute multi-angle research on a topic and generate a cited briefing report.

    Topic synthesis tool that decomposes an overarching subject into targeted subqueries,
    retrieves sources, and compiles a comprehensive cited briefing report whose numbered
    citations [1], [2] match the ids in the returned sources array. Depth tiers ("low", "medium",
    "high", "max") scale subquery breadth and source depth. Untrusted source content is sanitized.

    When NOT to use: Do not use for simple factual lookups or single queries (use deepseek_search),
    for multiple independent queries (use deepseek_search_many), for reading known URLs (use
    deepseek_fetch), or for structured entity or schema extraction (use deepseek_extract).

    Returns: {"ok": True, "topic": str, "depth": str, "report": str (no references list),
    "sources": [{id, title, url, date, doi, pmid, pmcid, access}] (ids match the [n] citations),
    "cost_usd": float}.
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.")
    if depth not in DEPTH_CONFIG:
        return _fail("bad_request", f"depth must be one of: {', '.join(DEPTH_CONFIG.keys())}")

    settings = get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(_clock())

    existing = store.find_active_match(clean_topic, depth, lang)
    if existing is None and not refresh:
        finished = store.find_finished_match(clean_topic, depth, lang, _clock())
        if finished is not None and finished.result is not None:
            return _reuse_finished_result(finished.result)
    if existing is not None and existing.task is not None and not existing.task.done():
        job = existing
        if ctx is not None:
            job.ctx = ctx
    else:
        if store.count_active() >= settings.research_max_jobs:
            return _fail(
                "rate_limited_local",
                "Maximum concurrent research jobs reached. Try again later.",
            )
        job_id = str(uuid.uuid4())
        job = ResearchJob(
            job_id=job_id,
            topic=clean_topic,
            depth=depth,
            lang=lang,
            created_at=_clock(),
            ctx=ctx,
        )
        job.task = asyncio.create_task(
            _run_research_job(
                job=job,
                clean_topic=clean_topic,
                depth=depth,
                lang=lang,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
            )
        )
        store.add(job)

    try:
        inline_wait = settings.research_inline_wait
        assert job.task is not None
        try:
            done, _ = await asyncio.wait([job.task], timeout=inline_wait)
        except asyncio.CancelledError:
            raise

        if job.task in done:
            if job.exception is not None:
                return _fail("internal_error", redact(f"Research job failed: {job.exception}"))
            res = dict(job.result) if job.result else _fail("internal_error", "Empty research result")
            if res.get("ok"):
                res["status"] = "done"
                res["delivery"] = "inline"
            return res

        elapsed_ms = int((_clock() - job.created_at) * 1000)
        return {
            "ok": True,
            "status": "running",
            "job_id": job.job_id,
            "depth": depth,
            "elapsed_ms": elapsed_ms,
            "progress": job.progress,
            "next_action": "Call deepseek_check_research with this job_id now. Do not start a new research.",
        }
    finally:
        job.ctx = None


def _reuse_finished_result(stored: dict[str, Any]) -> dict[str, Any]:
    """Copy a stored finished research result, marked as served from the job store at zero cost."""
    res = copy.deepcopy(stored)
    res["status"] = "done"
    res["delivery"] = "inline"
    res["cached"] = True
    res["cost_usd"] = 0
    if isinstance(res.get("cost_breakdown"), dict):
        res["cost_breakdown"] = {k: 0 for k in res["cost_breakdown"]}
    if isinstance(res.get("_meta"), dict):
        res["_meta"] = {**res["_meta"], "cost_usd": 0}
    notes = list(res.get("notes") or [])
    notes.append("Reused finished research result from this session (no new cost).")
    res["notes"] = notes
    return res


async def deepseek_check_research(
    job_id: Annotated[
        str,
        Field(description="Unique UUID identifier of the research job to poll (from deepseek_research)"),
    ],
    *,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Poll a long-running research job until completion or timeout.

    Long-polls an active background research task up to the inline wait limit (45s).
    Returns the completed synthesis report when done, updated progress while running,
    or a job_not_found error if expired or missing.

    When NOT to use: Do not use to start a new research investigation (use deepseek_research).
    Do not restart research when status is running; continue polling with this tool.

    Returns: {"ok": True, "status": "done"|"running", "job_id": str, ...}.
    """
    clean_id = job_id.strip() if isinstance(job_id, str) else ""
    if not clean_id:
        return _fail("bad_request", "job_id must not be empty.")

    settings = get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(_clock())

    job = store.get(clean_id)
    if job is None:
        return _fail(
            "job_not_found",
            "Research job not found or expired. Restart the research with deepseek_research.",
        )
    if ctx is not None:
        job.ctx = ctx

    try:
        if job.task is not None and job.task.done():
            if job.exception is not None:
                return _fail("internal_error", redact(f"Research job failed: {job.exception}"))
            res = dict(job.result) if job.result else _fail("internal_error", "Empty research result")
            if res.get("ok"):
                res["status"] = "done"
                res["delivery"] = "polled"
                res["elapsed_ms"] = max(0, int((_clock() - job.created_at) * 1000))
            return res

        inline_wait = settings.research_inline_wait
        assert job.task is not None
        try:
            done, _ = await asyncio.wait([job.task], timeout=inline_wait)
        except asyncio.CancelledError:
            raise

        if job.task in done:
            if job.exception is not None:
                return _fail("internal_error", redact(f"Research job failed: {job.exception}"))
            res = dict(job.result) if job.result else _fail("internal_error", "Empty research result")
            if res.get("ok"):
                res["status"] = "done"
                res["delivery"] = "polled"
                res["elapsed_ms"] = max(0, int((_clock() - job.created_at) * 1000))
            return res

        elapsed_ms = int((_clock() - job.created_at) * 1000)
        return {
            "ok": True,
            "status": "running",
            "job_id": job.job_id,
            "depth": job.depth,
            "elapsed_ms": elapsed_ms,
            "progress": job.progress,
            "next_action": "Call deepseek_check_research with this job_id now. Do not start a new research.",
        }
    finally:
        job.ctx = None


async def _report_job_progress(
    job: ResearchJob,
    phase: str,
    step: int,
    total_steps: int,
    message: str | None = None,
) -> None:
    """Update job progress state and emit FastMCP progress notification if context is attached."""
    event = {"phase": phase, "step": step, "of": total_steps}
    job.progress = event
    job.progress_history.append(event)
    if job.ctx is not None and hasattr(job.ctx, "report_progress"):
        try:
            await job.ctx.report_progress(float(step), float(total_steps), message=message or phase)
        except Exception as exc:
            logger.debug("Failed to report progress to context for job %s: %s", job.job_id, exc)


async def _run_research_job(job: ResearchJob, **kwargs: Any) -> dict[str, Any]:
    try:
        res = await _execute_research_pipeline(job=job, **kwargs)
        job.result = res
        job.completed_at = _clock()
        return res
    except asyncio.CancelledError:
        job.completed_at = _clock()
        raise
    except Exception as exc:
        job.exception = exc
        job.completed_at = _clock()
        job.result = _fail("internal_error", redact(str(exc)))
        return job.result


async def _condense_report(
    report: str,
    *,
    depth: str,
    allowed_urls: set[str],
    max_id: int,
    deadline_at: float,
) -> tuple[str, float, bool, list[str]]:
    """Shorten an over-length report once; return (report, cost, call_made, notes).

    The condensed text goes through the same output filter and citation normalization as the original
    and is accepted only when it is non-empty, shorter and cites no id the original did not cite.
    """
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    words = report_body_words(report)
    if words <= cfg.max_words:
        return report, 0.0, False, []
    remaining = deadline_at - _clock()
    if remaining < 8.0:
        return report, 0.0, False, ["Report exceeds the word ceiling; condensing skipped (deadline)."]
    res = await _chat_request(
        messages=build_condense_messages(report, depth),
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0.0,
        max_tokens=min(cfg.max_tokens, cfg.target_words * 4),
        timeout=max(5.0, remaining - 2.0),
        deadline=max(5.0, remaining - 2.0),
        attempts=1,
    )
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    kept = f"Report had {words} words (ceiling {cfg.max_words}); condensing was not usable, original kept."
    if not res.get("ok") or not (res.get("content") or "").strip():
        return report, cost, True, [kept]
    candidate, _, _ = filter_synthesis_output(res["content"], allowed_urls)
    candidate = normalize_citations(candidate, max_id=max_id)
    if (
        not candidate.strip()
        or report_body_words(candidate) >= words
        or not citation_ids(res["content"]) <= citation_ids(report)
        or not citation_ids(candidate) <= citation_ids(report)
    ):
        return report, cost, True, [kept]
    return (
        candidate,
        cost,
        True,
        [
            f"Report condensed from {words} to {report_body_words(candidate)} words "
            f"(target {cfg.target_words})."
        ],
    )


async def _execute_research_pipeline(
    *,
    job: ResearchJob,
    clean_topic: str,
    depth: str,
    lang: str | None,
    refresh: bool,
    strip_boilerplate: bool,
) -> dict[str, Any]:
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    max_queries = cfg.subqueries
    max_pages_per_query = cfg.count
    deadline_total = cfg.deadline_s
    t0 = _clock()
    deadline_at = t0 + deadline_total

    synthesis_reserve = {
        "low": 20.0,
        "medium": 30.0,
        "high": 45.0,
        "max": 60.0,
    }.get(depth, 30.0)

    dedupe = True
    total_steps = 5 if depth == "max" else 4

    # Step 1: Decompose topic into subqueries
    await _report_job_progress(job, "decomposition", 1, total_steps, "Decomposing topic into subqueries")
    rem1 = deadline_at - _clock()
    dec_timeout = min(20.0, max(5.0, rem1 - synthesis_reserve))
    dec_messages = build_decomposition_messages(clean_topic, max_queries=max_queries)
    dec_res = await _chat_request(
        messages=dec_messages,
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        timeout=dec_timeout,
        deadline=dec_timeout,
        attempts=2,
    )
    dec_cost = dec_res.get("cost_usd", 0.0) if dec_res.get("ok") else 0.0
    if dec_res.get("ok") and dec_res.get("content"):
        raw_dec = dec_res["content"]
        subqueries = parse_subqueries(raw_dec, clean_topic, max_queries=max_queries)
    else:
        fallback_subq, _ = fit_query(clean_topic)
        subqueries = [fallback_subq]

    # Step 2: Concurrently execute search across subqueries
    await _report_job_progress(job, "retrieval", 2, total_steps, "Searching across subqueries")
    rem2 = deadline_at - _clock()
    search_timeout = min(30.0, max(5.0, rem2 - synthesis_reserve))
    search_coros = [
        asyncio.wait_for(
            _search(
                subq,
                max_pages_per_query,
                None,
                "full",
                lang,
                cfg.per_page_char_cap,
                cfg.per_page_char_cap * max_pages_per_query,
                dedupe,
                False,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
            ),
            timeout=search_timeout,
        )
        for subq in subqueries
    ]
    search_results = await asyncio.gather(*search_coros)

    successful_searches = [r for r in search_results if r.get("ok")]
    if not successful_searches:
        return search_results[0]

    subquery_result_lists = [s.get("results", []) for s in successful_searches]
    cached_count = sum(s.get("cached_count", 0) for s in successful_searches)
    requests_used = sum(s.get("requests_used", 0) for s in successful_searches)
    search_cost = sum(s.get("cost_usd", 0.0) for s in successful_searches)
    r2_notes: list[str] = []

    # Round 2: Gap analysis for 'max' depth
    if depth == "max":
        await _report_job_progress(
            job, "round_2_retrieval", 3, total_steps, "Analyzing gaps and searching additional angles"
        )
        rem_r2 = deadline_at - _clock()
        if rem_r2 - synthesis_reserve < 10.0:
            r2_notes.append("Round 2 skipped due to deadline budget.")
        else:
            r1_items: list[dict[str, Any]] = []
            r1_domains: set[str] = set()
            for r in successful_searches:
                for it in r.get("results", []):
                    r1_items.append(it)
                    u = it.get("url") or ""
                    d = _url_etld1(u)
                    if d:
                        r1_domains.add(d)

            r2_timeout = min(25.0, max(5.0, (rem_r2 - synthesis_reserve) / 2))
            try:
                r2_messages = build_r2_gap_messages(clean_topic, r1_items, subqueries)
                r2_res = await _chat_request(
                    messages=r2_messages,
                    thinking={"type": "disabled"},
                    model=REWRITE_MODEL,
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    timeout=r2_timeout,
                    deadline=r2_timeout,
                    attempts=1,
                )
                if r2_res.get("ok") and r2_res.get("content"):
                    dec_cost += r2_res.get("cost_usd", 0.0)
                    r2_parsed = parse_subqueries(r2_res["content"], clean_topic, max_queries=3)
                    valid_r2 = validate_r2_queries(r2_parsed, clean_topic, subqueries, r1_domains, lang=lang)
                    if valid_r2:
                        rem_r2_search = deadline_at - _clock() - synthesis_reserve
                        r2_search_timeout = min(20.0, max(5.0, rem_r2_search))
                        r2_search_coros = [
                            asyncio.wait_for(
                                _search(
                                    r2q,
                                    max_pages_per_query,
                                    None,
                                    "full",
                                    lang,
                                    cfg.per_page_char_cap,
                                    cfg.per_page_char_cap * max_pages_per_query,
                                    dedupe,
                                    False,
                                    refresh=refresh,
                                    strip_boilerplate=strip_boilerplate,
                                ),
                                timeout=r2_search_timeout,
                            )
                            for r2q in valid_r2
                        ]
                        r2_search_results = await asyncio.gather(*r2_search_coros, return_exceptions=True)
                        successful_r2 = [r for r in r2_search_results if isinstance(r, dict) and r.get("ok")]
                        if successful_r2:
                            for r2q, res_dict in zip(valid_r2, r2_search_results, strict=True):
                                if isinstance(res_dict, dict) and res_dict.get("ok"):
                                    subqueries.append(r2q)
                                    subquery_result_lists.append(res_dict.get("results", []))
                            cached_count += sum(s.get("cached_count", 0) for s in successful_r2)
                            requests_used += sum(s.get("requests_used", 0) for s in successful_r2)
                            search_cost += sum(s.get("cost_usd", 0.0) for s in successful_r2)
                        else:
                            r2_notes.append(
                                "Round 2 search produced no results; proceeding with Round 1 sources."
                            )
                    else:
                        r2_notes.append(
                            "Round 2 gap query validation failed; proceeding with Round 1 sources."
                        )
                else:
                    r2_notes.append("Round 2 gap analysis failed; proceeding with Round 1 sources.")
            except Exception as exc:
                logger.debug("Round 2 gap analysis exception: %s", exc)
                r2_notes.append("Round 2 gap analysis failed; proceeding with Round 1 sources.")

    # Step 3: Round-robin interleave across subqueries, dedupe, domain cap, top K
    selected_items, rank_notes = rank_and_diversify_sources(
        subquery_result_lists, k_sources=cfg.k_sources, denylist=get_settings().source_denylist
    )
    sources = [
        _research_source(idx, item, item.get("cached", False)) for idx, item in enumerate(selected_items, 1)
    ]

    # Step 4: Gap-fill stub pages
    gap_step = 4 if depth == "max" else 3
    await _report_job_progress(job, "gap_fill", gap_step, total_steps, "Fetching gap pages for top sources")
    rem4 = deadline_at - _clock()
    gap_timeout = min(30.0, max(5.0, rem4 - synthesis_reserve))
    fill = await _fill_gaps(
        sources,
        cfg.per_page_char_cap,
        cfg.total_source_chars,
        refresh,
        strip_boilerplate,
        deadline=gap_timeout,
        max_pages=cfg.k_sources,
        internal_batch=True,
    )
    requests_used += fill.requests_used
    cached_count += fill.cached_count

    notes: list[str] = r2_notes + list(rank_notes)
    if fill.fetched:
        notes.append(f"fetched {fill.fetched} page(s) with missing or short search content")
    if fill.failed:
        notes.append(f"{fill.failed} fetch(es) failed; see the source's error")

    # Step 5: Select synthesis context and guard token limits
    synth_step = 5 if depth == "max" else 4
    await _report_job_progress(
        job, "synthesis", synth_step, total_steps, "Synthesizing cited research report"
    )
    synthesis_context = select_synthesis_context(sources, depth=depth)
    synthesis_context, guard_notes = apply_token_guard(synthesis_context)
    notes.extend(guard_notes)
    fetch_cost = fill.requests_used * COST_PER_REQUEST
    if not synthesis_context:
        total_llm_cost = round(dec_cost, 6)
        total_cost = round(search_cost + fetch_cost + total_llm_cost, 6)
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": _format_public_sources(sources, include_snippet=True),
            "partial": True,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, cached_count),
            "cached_count": cached_count,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(search_cost, 6),
                "fetch": round(fetch_cost, 6),
                "llm": total_llm_cost,
            },
            "untrusted_content": True,
            "derived_from_untrusted": True,
            "injection_suspected": 0,
            "notes": notes + ["No readable source content available for synthesis."],
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }

    # S7: Cost ceiling guard
    total_source_chars = sum(len(s.get("content") or "") for s in synthesis_context)
    est_input_tokens = total_source_chars // 2
    pricing = get_settings().llm_pricing.peak
    est_synth_cost = (est_input_tokens / 1_000_000) * pricing.cache_miss_per_1m + (
        cfg.max_tokens / 1_000_000
    ) * pricing.output_per_1m
    already_spent = search_cost + fetch_cost + dec_cost
    est_total_call_cost = already_spent + est_synth_cost
    ceiling = get_settings().research_cost_ceiling_usd
    if est_total_call_cost > ceiling:
        total_llm_cost = round(dec_cost, 6)
        total_cost = round(search_cost + fetch_cost + total_llm_cost, 6)
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": _format_public_sources(sources, include_snippet=True),
            "partial": True,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, cached_count),
            "cached_count": cached_count,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(search_cost, 6),
                "fetch": round(fetch_cost, 6),
                "llm": total_llm_cost,
            },
            "untrusted_content": True,
            "derived_from_untrusted": True,
            "injection_suspected": 0,
            "notes": notes
            + [
                f"Estimated research cost (${est_total_call_cost:.4f}) exceeds ceiling "
                f"of ${ceiling:.2f}; returning gathered sources without synthesis."
            ],
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }

    # Deadline check before synthesis
    rem_synth = deadline_at - _clock()
    if rem_synth < 5.0:
        total_llm_cost = round(dec_cost, 6)
        total_cost = round(search_cost + fetch_cost + total_llm_cost, 6)
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": _format_public_sources(sources, include_snippet=True),
            "partial": True,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, cached_count),
            "cached_count": cached_count,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(search_cost, 6),
                "fetch": round(fetch_cost, 6),
                "llm": total_llm_cost,
            },
            "untrusted_content": True,
            "derived_from_untrusted": True,
            "injection_suspected": 0,
            "notes": notes + ["Synthesis deadline reached; returning gathered sources."],
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }

    # Step 6: Synthesize cited briefing report with nonce delimiter
    synth_nonce = uuid.uuid4().hex[:12]
    synthesis_messages = build_synthesis_messages(
        clean_topic, synthesis_context, depth=depth, nonce=synth_nonce
    )
    synth_timeout = max(10.0, rem_synth - 2.0)
    synth_kwargs: dict[str, Any] = {
        "messages": synthesis_messages,
        "model": REWRITE_MODEL,
        "max_tokens": cfg.max_tokens,
        "timeout": synth_timeout,
        "deadline": synth_timeout,
        "attempts": 1,
    }
    if cfg.thinking:
        synth_kwargs["thinking"] = {"type": "enabled"}
        if cfg.reasoning_effort:
            synth_kwargs["reasoning_effort"] = cfg.reasoning_effort
    else:
        synth_kwargs["thinking"] = {"type": "disabled"}
        synth_kwargs["temperature"] = 0.0

    synth_res = await _chat_request(**synth_kwargs)

    # Retry policy: If finish_reason == "length" or empty content, retry once with thinking OFF
    needs_retry = (
        not synth_res.get("ok")
        or synth_res.get("finish_reason") == "length"
        or not (synth_res.get("content") or "").strip()
    )
    rem_retry = deadline_at - _clock()
    discarded_attempt_cost = 0.0
    if needs_retry and rem_retry >= 8.0:
        notes.append("Synthesis initial attempt truncated or empty; retrying with thinking disabled.")
        retry_kwargs = {
            "messages": synthesis_messages,
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "model": REWRITE_MODEL,
            "max_tokens": min(cfg.max_tokens, 16_384),
            "timeout": max(5.0, rem_retry - 2.0),
            "deadline": max(5.0, rem_retry - 2.0),
            "attempts": 1,
        }
        retry_res = await _chat_request(**retry_kwargs)
        # Every attempt is billed, so the attempt that is not used still counts toward the cost.
        retry_used = (
            retry_res.get("ok")
            and (retry_res.get("content") or "").strip()
            and retry_res.get("finish_reason") != "length"
        )
        if retry_used:
            discarded_attempt_cost = float(synth_res.get("cost_usd", 0.0) or 0.0)
            synth_res = retry_res
        else:
            discarded_attempt_cost = float(retry_res.get("cost_usd", 0.0) or 0.0)

    synth_cost = synth_res.get("cost_usd", 0.0) if synth_res.get("ok") else 0.0
    total_llm_cost = round(dec_cost + synth_cost + discarded_attempt_cost, 6)
    total_cost = round(search_cost + fetch_cost + total_llm_cost, 6)

    if not synth_res.get("ok") or not (synth_res.get("content") or "").strip():
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "subqueries": subqueries,
            "report": None,
            "synthesized": False,
            "sources": _format_public_sources(sources, include_snippet=True),
            "partial": True,
            "requests_used": requests_used + 1,
            "cached": _all_cached(requests_used + 1, cached_count),
            "cached_count": cached_count,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(search_cost, 6),
                "fetch": round(fetch_cost, 6),
                "llm": total_llm_cost,
            },
            "untrusted_content": True,
            "derived_from_untrusted": True,
            "injection_suspected": 0,
            "notes": notes
            + [
                f"Synthesis model call failed: {synth_res.get('error', {}).get('message', 'unknown')}; "
                "returning gathered sources."
            ],
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }

    report_text = synth_res.get("content", "")
    allowed_urls = {
        s["url"].strip() for s in synthesis_context if isinstance(s.get("url"), str) and s["url"].strip()
    }
    filtered_report, injection_suspected, filter_notes = filter_synthesis_output(report_text, allowed_urls)
    clean_report = normalize_citations(filtered_report, max_id=len(synthesis_context))
    clean_report, condense_cost, condense_called, condense_notes = await _condense_report(
        clean_report,
        depth=depth,
        allowed_urls=allowed_urls,
        max_id=len(synthesis_context),
        deadline_at=deadline_at,
    )
    clean_report, s_notes = sanitize_untrusted_text(clean_report)
    notes.extend(filter_notes)
    notes.extend(condense_notes)
    notes.extend(s_notes)
    if condense_called:
        requests_used += 1
    total_llm_cost = round(total_llm_cost + condense_cost, 6)
    total_cost = round(total_cost + condense_cost, 6)

    return {
        "ok": True,
        "status": "done",
        "topic": clean_topic,
        "depth": depth,
        "subqueries": subqueries,
        "report": clean_report,
        "synthesized": True,
        "sources": _format_public_sources(synthesis_context, include_snippet=False),
        "partial": bool(fill.failed > 0),
        "requests_used": requests_used + 1,
        "cached": _all_cached(requests_used + 1, cached_count),
        "cached_count": cached_count,
        "cost_usd": total_cost,
        "cost_breakdown": {
            "search": round(search_cost, 6),
            "fetch": round(fetch_cost, 6),
            "llm": total_llm_cost,
        },
        "untrusted_content": True,
        "derived_from_untrusted": True,
        "injection_suspected": injection_suspected,
        "notes": notes,
        "_meta": _envelope_meta(total_cost, untrusted_content=True),
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


def _calculate_extract_budget(
    fields: list[str] | None,
    schema: dict[str, Any] | None,
    content_len: int,
) -> tuple[int, float]:
    """Calculate adaptive max_tokens and timeout for extraction.

    Tokens: 4096 base, scales up to 16384 with large field sets or complex schemas.
    Timeout: 20s to 90s based on page content length.
    """
    field_count = len(fields) if fields else 0
    if schema:
        props = schema.get("properties")
        if isinstance(props, dict):
            field_count = max(field_count, len(props))
        else:
            field_count = max(field_count, 10)

    tokens = min(16384, max(4096, 4096 + max(0, field_count - 5) * 256))
    timeout = min(90.0, max(20.0, 20.0 + (content_len / 5000.0) * 1.0))
    return tokens, timeout


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

    max_tokens, timeout = _calculate_extract_budget(fields, schema, len(content))
    chat_res = await _chat_request(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=timeout,
        attempts=2,
    )
    if not chat_res.get("ok"):
        return {
            "ok": False,
            "url": clean_url,
            "error": chat_res.get("error")
            or {"code": "extraction_failed", "message": "Extraction model request failed."},
            "llm_cost_usd": 0.0,
        }, 1

    finish_reason = chat_res.get("finish_reason")
    extraction_truncated = finish_reason == "length"

    raw_text = chat_res.get("content")
    llm_cost = chat_res.get("cost_usd", 0.0)
    if not isinstance(raw_text, str) or not raw_text.strip():
        err_code = "extraction_truncated" if extraction_truncated else "extraction_failed"
        err_msg = (
            "Extraction response was truncated before valid JSON could be completed."
            if extraction_truncated
            else "Invalid response payload from extraction model."
        )
        return {
            "ok": False,
            "url": clean_url,
            "error": {
                "code": err_code,
                "message": err_msg,
            },
            "llm_cost_usd": llm_cost,
        }, 1

    parsed, err = sanitize_json(raw_text)
    if err or parsed is None:
        err_code = "extraction_truncated" if extraction_truncated else "extraction_failed"
        err_msg = (
            "Extraction response was truncated before valid JSON could be completed."
            if extraction_truncated
            else (err or "Failed to parse extraction response as JSON.")
        )
        return {
            "ok": False,
            "url": clean_url,
            "error": {
                "code": err_code,
                "message": err_msg,
            },
            "llm_cost_usd": llm_cost,
        }, 1

    notes: list[str] = []
    if extraction_truncated:
        notes.append("Extraction output was truncated by token limit.")
    if fetch_res.get("cached"):
        notes.append("Page fetched from local cache.")

    if isinstance(parsed, dict) and "data" in parsed:
        raw_data = parsed["data"]
        raw_evidence = parsed.get("evidence")
        if isinstance(raw_data, dict):
            verified_data, verified_evidence, ev_notes = verify_evidence(
                raw_data, raw_evidence if isinstance(raw_evidence, dict) else {}, content
            )
            notes.extend(ev_notes)
        else:
            verified_data = raw_data
            verified_evidence = {}
    elif isinstance(parsed, dict):
        verified_data = parsed
        verified_evidence = {}
    else:
        verified_data = {}
        verified_evidence = {}

    if fields is not None and isinstance(verified_data, dict):
        verified_data = validate_fields(verified_data, fields)
    elif schema is not None and isinstance(verified_data, dict):
        schema_errors = validate_schema(verified_data, schema)
        if schema_errors:
            notes.extend(schema_errors)

    if isinstance(verified_data, dict):
        verified_data, s_notes = sanitize_structure(verified_data)
        notes.extend(s_notes)
    if verified_evidence:
        verified_evidence, ev_s_notes = sanitize_structure(verified_evidence)
        notes.extend(ev_s_notes)

    return {
        "ok": True,
        "url": clean_url,
        "title": fetch_res.get("title"),
        "data": verified_data,
        "evidence": verified_evidence,
        "extraction_truncated": extraction_truncated,
        "cached": fetch_res.get("cached", False),
        "notes": notes,
        "llm_cost_usd": llm_cost,
    }, 1


@_timed
async def deepseek_extract(
    url: Annotated[
        str,
        Field(description="HTTP/HTTPS URL of the page from which to extract structured data"),
    ],
    *,
    fields: Annotated[
        list[str] | None,
        Field(description="List of field names to extract (mutually exclusive with schema)"),
    ] = None,
    schema: Annotated[
        dict[str, Any] | None,
        Field(description="JSON schema describing desired output structure (mutually exclusive with fields)"),
    ] = None,
    instruction: Annotated[
        str | None,
        Field(description="Optional natural-language guidance for extraction (default None)"),
    ] = None,
    clean: Annotated[
        bool,
        Field(
            description="Remove Markdown backslash escapes from page text before extraction (default True)"
        ),
    ] = True,
    max_chars: Annotated[
        int,
        Field(description="Max characters of page text to feed to the extractor LLM (default 40000)"),
    ] = 40000,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh page from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners and site UI boilerplate before extraction (default True)"),
    ] = True,
) -> dict[str, Any]:
    """Extract structured JSON from a web page conforming to target fields or a schema.

    Semantic entity extraction from a single known web page. Returns extracted values under
    'data' along with verifiable verbatim quotes under 'evidence'. Fields failing evidence
    verification are nullified with notes. Untrusted content and outputs are sanitized.

    When NOT to use: Do not use for reading full page prose or browsing (use deepseek_fetch),
    for searching pages by topic (use deepseek_search), for extracting across multiple URLs (use
    deepseek_extract_many), or for comprehensive topic research (use deepseek_research).

    Returns: {"ok": True, "data": {...}, "evidence": {...}, "extraction_truncated": bool, "cost_usd": float}.
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

    fetch_cost = fetch_res.get("cost_usd", 0.0)
    llm_cost = item.get("llm_cost_usd", 0.0)
    total_cost = round(fetch_cost + llm_cost, 6)
    return {
        "ok": True,
        "url": clean_url,
        "title": item.get("title"),
        "data": item.get("data"),
        "evidence": item.get("evidence", {}),
        "extraction_truncated": item.get("extraction_truncated", False),
        "cached": fetch_res.get("cached", False),
        "cost_usd": total_cost,
        "cost_breakdown": {
            "search": 0.0,
            "fetch": round(fetch_cost, 6),
            "llm": round(llm_cost, 6),
        },
        "untrusted_content": True,
        "notes": item.get("notes", []),
        "_meta": _envelope_meta(total_cost, untrusted_content=True),
    }


@_timed
async def deepseek_extract_many(
    urls: Annotated[
        list[str],
        Field(description="List of HTTP/HTTPS URLs to extract structured data from concurrently"),
    ],
    *,
    fields: Annotated[
        list[str] | None,
        Field(description="List of field names to extract (mutually exclusive with schema)"),
    ] = None,
    schema: Annotated[
        dict[str, Any] | None,
        Field(description="JSON schema describing output structure (mutually exclusive with fields)"),
    ] = None,
    instruction: Annotated[
        str | None,
        Field(description="Optional natural-language extraction guidance (default None)"),
    ] = None,
    clean: Annotated[
        bool,
        Field(
            description="Remove Markdown backslash escapes from page text before extraction (default True)"
        ),
    ] = True,
    max_chars_per_url: Annotated[
        int,
        Field(description="Max characters per page to feed to extraction LLM (default 20000)"),
    ] = 20000,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters across all pages for extraction (default 60000)"),
    ] = 60000,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners and boilerplate before extraction (default True)"),
    ] = True,
    deadline: Annotated[
        float | None,
        Field(description="Overall timeout deadline in seconds; partial results returned on expiry"),
    ] = None,
) -> dict[str, Any]:
    """Extract structured JSON from multiple web pages concurrently conforming to target fields or a schema.

    Batch entity extraction across 1-10 web pages in parallel. Returns structured values under
    'data' and verifiable quote strings under 'evidence' per URL. Fields failing evidence
    verification are nullified with notes. Untrusted content is sanitized.

    When NOT to use: Do not use for extracting from a single URL (use deepseek_extract),
    for reading unformatted full text across pages (use deepseek_fetch_many), for discovering
    pages via search queries (use deepseek_search_many), or for topic reports (use deepseek_research).

    Returns: {"ok": True, "results": [{"url": str, "data": {...}, "evidence": {...}}], "cost_usd": float}.
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

    extract_sem = asyncio.Semaphore(4)

    async def _bounded_extract_page(u: str, e: dict[str, Any]) -> tuple[dict[str, Any], int]:
        async with extract_sem:
            return await _extract_page(u, e, system_prompt, fields, schema)

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
            extract_coros.append(_bounded_extract_page(url, env))
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
                    "error": env.get("error", _fail("request_failed", "Failed to fetch page.")["error"]),
                }
            )

    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")
    if any(item.get("extraction_truncated") for item in results):
        notes.append("One or more extractions were truncated by token limit.")

    failed_count = sum(not item.get("ok", False) for item in results)
    cached_count = sum(item.get("cached", False) for item in results)
    total_fetch_cost = sum(env.get("cost_usd", 0.0) for env in fetch_envelopes.values() if env.get("ok"))
    total_llm_cost = sum(item.get("llm_cost_usd", 0.0) for item in results if item.get("ok"))
    total_cost = round(total_fetch_cost + total_llm_cost, 6)

    return {
        "ok": True,
        "count": len(results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": total_cost,
        "cost_breakdown": {
            "search": 0.0,
            "fetch": round(total_fetch_cost, 6),
            "llm": round(total_llm_cost, 6),
        },
        "untrusted_content": True,
        "results": results,
        "notes": notes,
        "_meta": _envelope_meta(total_cost, untrusted_content=True),
    }


@_timed
async def deepseek_check_balance() -> dict[str, Any]:
    """Check account balance (CNY/USD) and session telemetry.

    Administrative tool for inspecting DeepSeek upstream account balance, currency breakdown,
    and session usage metrics (requests handled, cache hit rate, estimated spend). Strictly
    read-only and non-destructive.

    When NOT to use: Do not use for web searching, page retrieval, or extraction (use
    deepseek_search, deepseek_fetch, or deepseek_extract).

    Returns: {"ok": True, "is_available": bool, "balances": [...], "session": {...}}.
    """
    result = await _request("GET", "/user/balance", timeout=BALANCE_TIMEOUT)
    if not result.get("ok"):
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

    return {
        "ok": True,
        "is_available": data.get("is_available"),
        "balances": data.get("balance_infos") or data.get("balances") or data,
        "session": session_data,
    }


TOOLS: tuple[Callable[..., Awaitable[dict[str, Any]]], ...] = (
    deepseek_search,
    deepseek_search_many,
    deepseek_fetch,
    deepseek_fetch_many,
    deepseek_research,
    deepseek_check_research,
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
    "deepseek_check_research": ToolAnnotations(
        readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
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
