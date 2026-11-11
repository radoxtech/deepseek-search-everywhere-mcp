"""Unit tests for the LLM gateway (_chat_request) and related concurrency guards."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import ratelimit, server
from deepseek_search_everywhere_mcp.budget import DailyBudgetTracker


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
