"""Unit tests for the LLM gateway (_chat_request) and related concurrency guards."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import ratelimit, server
from deepseek_search_everywhere_mcp.budget import DailyBudgetTracker
from deepseek_search_everywhere_mcp.settings import get_pricing_period


async def test_chat_request_requires_thinking_argument() -> None:
    # Omitting 'thinking' raises TypeError
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'thinking'"):
        await server._chat_request([{"role": "user", "content": "hi"}])  # type: ignore[call-arg]

    # Non-dict 'thinking' raises ValueError
    with pytest.raises(ValueError, match="'thinking' must be a dictionary"):
        await server._chat_request([{"role": "user", "content": "hi"}], thinking=None)  # type: ignore[arg-type]


async def test_chat_request_payload_always_contains_thinking(fake_upstream: Any) -> None:
    upstream = fake_upstream(
        responses={
            "/v1/chat/completions": {
                "ok": True,
                "data": {
                    "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            }
        }
    )

    res = await server._chat_request(
        [{"role": "user", "content": "ping"}],
        thinking={"type": "disabled"},
    )

    assert res["ok"] is True
    assert res["content"] == "pong"
    assert res["finish_reason"] == "stop"

    call = upstream.calls[0]
    assert call["path"] == "/v1/chat/completions"
    assert call["payload"]["thinking"] == {"type": "disabled"}
    assert call["payload"]["model"] == "deepseek-flash"


async def test_chat_request_acquires_chat_rate_limiter(
    monkeypatch: pytest.MonkeyPatch, fake_upstream: Any
) -> None:
    fake_upstream(
        responses={
            "/v1/chat/completions": {
                "ok": True,
                "data": {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            }
        }
    )

    acquired: list[str] = []
    real_acquire = ratelimit.TokenBucket.acquire

    async def mock_acquire(self: ratelimit.TokenBucket) -> ratelimit.Ticket:
        acquired.append(self.name)
        return await real_acquire(self)

    monkeypatch.setattr(ratelimit.TokenBucket, "acquire", mock_acquire)

    res = await server._chat_request(
        [{"role": "user", "content": "test"}],
        thinking={"type": "enabled"},
    )
    assert res["ok"] is True
    assert ratelimit.CHAT in acquired


async def test_chat_request_releases_reservation_on_upstream_failure(
    tmp_path: Any, fake_upstream: Any
) -> None:
    state_file = tmp_path / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)
    server._budget_tracker = tracker

    fake_upstream(
        responses={
            "/v1/chat/completions": {
                "ok": False,
                "error": {"code": "upstream_error", "message": "server error"},
            }
        }
    )

    res = await server._chat_request(
        [{"role": "user", "content": "fail test"}],
        thinking={"type": "disabled"},
    )

    assert res["ok"] is False
    status = await tracker.get_status()
    # Reservation was released on failure; reserved_usd should be 0.0
    assert status["reserved_usd"] == 0.0
    assert status["spent_usd"] == 0.0


async def test_chat_request_releases_reservation_on_cancellation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)
    server._budget_tracker = tracker

    async def hanging_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(10.0)
        return {"ok": True, "data": {}}

    monkeypatch.setattr(server, "_request", hanging_request)

    task = asyncio.create_task(
        server._chat_request([{"role": "user", "content": "cancel test"}], thinking={"type": "disabled"})
    )
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    status = await tracker.get_status()
    assert status["reserved_usd"] == 0.0


async def test_extract_many_bounds_llm_concurrency_to_at_most_four(
    monkeypatch: pytest.MonkeyPatch, fake_upstream: Any
) -> None:
    urls = [f"https://example.com/{i}" for i in range(10)]
    fake_upstream(
        responses={
            server.FETCH_PATH: {
                "ok": True,
                "data": {"title": "Title", "content": "Item price is $10"},
            }
        }
    )

    current_concurrent = 0
    max_seen_concurrent = 0

    async def mock_extract_page(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
        nonlocal current_concurrent, max_seen_concurrent
        current_concurrent += 1
        max_seen_concurrent = max(max_seen_concurrent, current_concurrent)
        await asyncio.sleep(0.02)
        current_concurrent -= 1
        return {"ok": True, "url": args[0], "data": {"price": 10}}, 1

    monkeypatch.setattr(server, "_extract_page", mock_extract_page)

    res = await server.deepseek_extract_many(urls, fields=["price"])
    assert res["ok"] is True
    assert res["count"] == 10
    assert max_seen_concurrent <= 4


def test_get_pricing_period() -> None:
    # Wednesday 02:30 UTC -> peak (1-4)
    assert get_pricing_period(datetime(2026, 11, 11, 2, 30, tzinfo=UTC)) == "peak"
    # Wednesday 07:15 UTC -> peak (6-10)
    assert get_pricing_period(datetime(2026, 11, 11, 7, 15, tzinfo=UTC)) == "peak"
    # Wednesday 04:30 UTC -> off_peak
    assert get_pricing_period(datetime(2026, 11, 11, 4, 30, tzinfo=UTC)) == "off_peak"
    # Wednesday 11:00 UTC -> off_peak
    assert get_pricing_period(datetime(2026, 11, 11, 11, 0, tzinfo=UTC)) == "off_peak"
    # Saturday 02:30 UTC -> off_peak (weekend)
    assert get_pricing_period(datetime(2026, 11, 14, 2, 30, tzinfo=UTC)) == "off_peak"
    # Sunday 07:30 UTC -> off_peak (weekend)
    assert get_pricing_period(datetime(2026, 11, 15, 7, 30, tzinfo=UTC)) == "off_peak"


async def test_chat_request_prices_from_usage_tokens(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_upstream(
        responses={
            "/v1/chat/completions": {
                "ok": True,
                "data": {
                    "choices": [{"message": {"content": "result"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1000,
                        "prompt_cache_hit_tokens": 400,
                        "prompt_cache_miss_tokens": 600,
                        "completion_tokens": 200,
                    },
                },
            }
        }
    )

    # Test peak period
    monkeypatch.setattr(server, "get_pricing_period", lambda: "peak")
    res_peak = await server._chat_request(
        [{"role": "user", "content": "hi"}],
        thinking={"type": "disabled"},
    )
    assert res_peak["ok"] is True
    assert res_peak["pricing_period"] == "peak"
    # Peak rates: cache_hit=0.006/1M, cache_miss=0.30/1M, output=1.20/1M
    # (400 / 1e6) * 0.006 + (600 / 1e6) * 0.30 + (200 / 1e6) * 1.20
    # = 0.0000024 + 0.00018 + 0.00024 = 0.0004224 -> round 0.000422
    assert res_peak["cost_usd"] == 0.000422

    # Test off_peak period
    monkeypatch.setattr(server, "get_pricing_period", lambda: "off_peak")
    res_off = await server._chat_request(
        [{"role": "user", "content": "hi"}],
        thinking={"type": "disabled"},
    )
    assert res_off["ok"] is True
    assert res_off["pricing_period"] == "off_peak"
    # Off-peak rates: cache_hit=0.003/1M, cache_miss=0.15/1M, output=0.60/1M
    # (400 / 1e6) * 0.003 + (600 / 1e6) * 0.15 + (200 / 1e6) * 0.60
    # = 0.0000012 + 0.00009 + 0.00012 = 0.0002112 -> round 0.000211
    assert res_off["cost_usd"] == 0.000211


async def test_chat_request_missing_usage_falls_back_to_estimate(fake_upstream: Any) -> None:
    fake_upstream(
        responses={
            "/v1/chat/completions": {
                "ok": True,
                "data": {
                    "choices": [{"message": {"content": "no usage"}, "finish_reason": "stop"}],
                },
            }
        }
    )

    res = await server._chat_request(
        [{"role": "user", "content": "short message"}],
        thinking={"type": "disabled"},
    )
    assert res["ok"] is True
    assert res["cost_usd"] > 0
    assert any("Usage missing from upstream response; cost estimated." in note for note in res["notes"])
