"""Integration tests for session metrics and cost auditing in deepseek_check_balance."""

from __future__ import annotations

import asyncio

from deepseek_search_everywhere_mcp import server


def test_balance_returns_session_telemetry(fake_upstream):
    fake_upstream(
        responses={"/user/balance": {"ok": True, "data": {"is_available": True, "balance_infos": []}}}
    )
    result = asyncio.run(server.deepseek_check_balance())
    assert result["ok"] is True
    session = result["session"]
    assert "uptime_seconds" in session
    assert "requests_upstream" in session
    assert "cache_hits" in session
    assert "total_queries_handled" in session
    assert "cache_hit_rate" in session
    assert "estimated_upstream_cost_usd" in session
    assert "estimated_cache_savings_usd" in session


def test_session_tracks_upstream_and_cache(fake_upstream):
    server.reset_state()
    fake_upstream(
        responses={
            "/user/balance": {"ok": True, "data": {"is_available": True, "balance_infos": []}},
            "/v1/tools/search": {"ok": True, "data": {"results": [{"title": "T", "url": "https://a.test"}]}},
        }
    )

    # 1st call -> upstream request
    r1 = asyncio.run(server.deepseek_search("query1"))
    assert r1["cached"] is False

    # 2nd identical call -> cache hit
    r2 = asyncio.run(server.deepseek_search("query1"))
    assert r2["cached"] is True

    balance = asyncio.run(server.deepseek_check_balance())
    session = balance["session"]
    assert session["requests_upstream"] >= 1
    assert session["cache_hits"] >= 1
    assert session["total_queries_handled"] >= 2
    assert session["estimated_upstream_cost_usd"] > 0
    assert session["estimated_cache_savings_usd"] > 0


def test_reset_state_clears_session_metrics():
    server.reset_state()
    metrics = server._get_session_metrics()
    assert metrics.requests_upstream == 0
    assert metrics.cache_hits == 0


def test_balance_reports_cache_stats(fake_upstream):
    server.reset_state()
    fake_upstream(
        responses={
            "/user/balance": {"ok": True, "data": {"is_available": True, "balance_infos": []}},
            "/v1/tools/search": {"ok": True, "data": {"results": [{"title": "T", "url": "https://a.test"}]}},
        }
    )

    # 1. Warm search cache
    r1 = asyncio.run(server.deepseek_search("query1"))
    assert r1["cached"] is False
    r2 = asyncio.run(server.deepseek_search("query1"))
    assert r2["cached"] is True

    # 2. Check balance: verify cache stats reported
    b1 = asyncio.run(server.deepseek_check_balance())
    assert b1["ok"] is True
    cache_stats = b1["session"]["cache"]
    assert cache_stats["search"]["entries"] >= 1
