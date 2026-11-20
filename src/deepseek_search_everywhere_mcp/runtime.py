"""Process-wide runtime state and the shared request gateway.

Holds the lazily built caches, single-flight maps, rate limiter, session metrics and daily budget
tracker (all dropped by ``reset_state()``), the rate-limited upstream and chat-completions calls,
the monotonic clock behind every envelope's ``elapsed_ms`` and the ``_timed`` tool decorator.
Callers in other modules look patchable names up here at call time (``runtime._request``,
``runtime.get_settings``, ``runtime._clock``, ...), so one patch covers every caller.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .budget import DailyBudgetTracker, Reservation
from .cache import SingleFlight, TTLCache
from .deep_research import (
    reset_state as reset_deep_research_state,
)
from .errors import (
    _fail,  # re-exported for callers and tests
)
from .ratelimit import CHAT, RateLimiter, RateLimitExceeded
from .settings import (
    active_chat_pricing,
    get_pricing_period,
    get_settings,
)
from .transport import _api_key as _api_key  # patch point: callers use runtime._api_key
from .transport import _chat_model as _chat_model  # patch point: chat host may rename models
from .transport import request as _request  # patch point: callers use runtime._request

CHAT_TIMEOUT = 8.0


REWRITE_MODEL = "deepseek-flash"


# Every module logs under the server's logger name, as before the split into modules.
logger = logging.getLogger(f"{__package__}.server")


# Monotonic clock behind every envelope's ``elapsed_ms``; tests may replace it.
_clock: Callable[[], float] = time.monotonic


@dataclass
class SessionMetrics:
    """In-memory request and cache-hit counts for the current session."""

    start_time: float
    requests_upstream: int = 0
    cache_hits: int = 0


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
        "model": _chat_model(model),
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

    pricing = active_chat_pricing()
    period = "flat" if pricing.flat is not None else get_pricing_period()
    if pricing.flat is not None:
        rates = pricing.flat
    else:
        rates = pricing.peak if period == "peak" else pricing.off_peak

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


def _timed[**P](
    tool: Callable[P, Awaitable[dict[str, Any]]],
) -> Callable[P, Awaitable[dict[str, Any]]]:
    """Add ``elapsed_ms``, the call's wall time on the monotonic clock, to every envelope ``tool`` returns.

    Success and error envelopes alike get it, on a copy; a cache hit reports its own (small) time.
    ``functools.wraps`` keeps the signature and docstring that the MCP SDK turns into the tool schema.
    """

    @functools.wraps(tool)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        started = _clock()
        envelope = await tool(*args, **kwargs)
        # A copy: the envelope may be shared (single-flight followers, the upstream's error dict).
        return {**envelope, "elapsed_ms": max(0, round((_clock() - started) * 1000))}

    return wrapper
