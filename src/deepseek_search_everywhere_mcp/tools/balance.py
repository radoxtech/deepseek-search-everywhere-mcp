"""The deepseek_check_balance tool: account balance and this session's usage."""

from __future__ import annotations

from typing import Any

from .. import runtime
from ..envelope import (
    COST_PER_REQUEST,
)
from ..runtime import (
    _get_cache_stats,
    _get_session_metrics,
    _timed,
)

BALANCE_TIMEOUT = 20.0


@_timed
async def deepseek_check_balance() -> dict[str, Any]:
    """Check the DeepSeek account balance and this session's usage before spending more.

    Free ($0.00) and read-only: no upstream search or fetch charge. Returns the balance per currency
    and session metrics: upstream requests, cache hits, hit rate and estimated spend. The call is
    capped at a 20 s timeout.

    When NOT to use: for web data (deepseek_search, deepseek_fetch or deepseek_extract).

    Errors: account_error (missing_api_key, auth_failed, insufficient_balance) → stop and ask the user.

    Returns: {"ok": True, "is_available": bool, "balances": [...], "session": {...}}.
    """
    result = await runtime._request("GET", "/user/balance", timeout=BALANCE_TIMEOUT)
    if not result.get("ok"):
        return result
    data = result["data"]
    metrics = _get_session_metrics()
    uptime = round(runtime._clock() - metrics.start_time, 1)
    total_handled = metrics.requests_upstream + metrics.cache_hits
    hit_rate = round(metrics.cache_hits / total_handled, 3) if total_handled > 0 else 0.0
    cost_usd = round(metrics.requests_upstream * COST_PER_REQUEST, 5)
    savings_usd = round(metrics.cache_hits * COST_PER_REQUEST, 5)

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
