"""Transport tests: streamed size caps, Retry-After parsing and error-body limits (offline)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from deepseek_search_everywhere_mcp import server, transport

MIB = 1024 * 1024
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(transport.asyncio, "sleep", fake_sleep)
    return recorded


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[httpx.Request]]:
    """Route the transport's ``httpx.AsyncClient`` to an in-process handler."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    real_client = httpx.AsyncClient

    def install(handler: Handler) -> list[httpx.Request]:
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        def client(**kwargs: object) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(record), **kwargs)

        monkeypatch.setattr(transport.httpx, "AsyncClient", client)
        return seen

    return install


def _chunks(count: int, produced: list[int], size: int = MIB) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for _ in range(count):
            produced.append(size)
            yield b"a" * size

    return gen()


async def test_server_request_alias_is_transport_request() -> None:
    assert server._request is transport.request


async def test_oversized_search_response_is_rejected(upstream, sleeps) -> None:
    produced: list[int] = []
    upstream(lambda request: httpx.Response(200, content=_chunks(16, produced)))

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5)

    assert result["ok"] is False
    assert result["error"]["code"] == "response_too_large"
    assert len(produced) <= 9  # stopped right after crossing 8 MiB
    assert sleeps == []


async def test_fetch_allows_bodies_above_the_search_cap(upstream) -> None:
    body = json.dumps({"content": "a" * (9 * MIB)}).encode()
    upstream(lambda request: httpx.Response(200, content=body))

    result = await transport.request("POST", "/v1/tools/fetch", payload={}, timeout=5)

    assert result["ok"] is True
    assert len(result["data"]["content"]) == 9 * MIB


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(timedelta(seconds=5), 5.0), (timedelta(hours=1), 10.0), (timedelta(seconds=-30), 0.0)],
)
async def test_retry_after_date_is_honored(upstream, sleeps, monkeypatch, offset, expected) -> None:
    retry_at = datetime(2026, 10, 21, 7, 28, tzinfo=UTC)
    monkeypatch.setattr(transport, "_utcnow", lambda: retry_at - offset)
    responses = iter([
        httpx.Response(503, text="busy", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        httpx.Response(200, json={"results": []}),
    ])
    seen = upstream(lambda request: next(responses))

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5)

    assert result == {"ok": True, "data": {"results": []}}
    assert len(seen) == 2
    assert sleeps == [expected]


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, 2.0), ("", 2.0), ("3", 3.0), ("1.5", 1.5), ("120", 10.0), ("-4", 0.0),
     ("soon", 2.0), ("nan", 2.0), ("inf", 2.0), ("Mon, 99 Foo 2026 25:61:00 GMT", 2.0)],
)
def test_retry_after_values_are_clamped(header, expected) -> None:
    assert transport._retry_after(header) == expected


async def test_large_error_body_is_capped(upstream, sleeps) -> None:
    produced: list[int] = []
    upstream(lambda request: httpx.Response(400, content=_chunks(16, produced, 64 * 1024)))

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5)

    assert result["error"]["status"] == 400
    assert len(result["error"]["message"]) <= 64 * 1024
    assert len(produced) <= 2
    assert sleeps == []


async def test_megabyte_json_error_message_stays_bounded(upstream, sleeps) -> None:
    body = json.dumps({"error": {"message": "x" * MIB, "code": "bad"}})
    upstream(lambda request: httpx.Response(500, text=body))

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5)

    assert len(result["error"]["message"]) <= 64 * 1024
    assert sleeps == [2.0]


async def test_client_errors_other_than_429_are_not_retried(upstream, sleeps) -> None:
    seen = upstream(lambda request: httpx.Response(404, json={"error": {"message": "nope"}}))

    result = await transport.request("POST", "/v1/tools/fetch", payload={}, timeout=5)

    assert result["error"]["status"] == 404
    assert len(seen) == 1
    assert sleeps == []


async def test_non_json_success_is_invalid_json(upstream) -> None:
    upstream(lambda request: httpx.Response(200, text="<html>"))

    result = await transport.request("GET", "/user/balance", timeout=5)

    assert result["error"]["code"] == "invalid_json"


async def test_timeouts_are_retried_then_reported(upstream, sleeps) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    seen = upstream(handler)

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5)

    assert result["error"]["code"] == "timeout"
    assert len(seen) == 2
    assert sleeps == [2.0]


async def test_network_errors_keep_their_code(upstream, sleeps) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    upstream(handler)

    result = await transport.request("POST", "/v1/tools/search", payload={}, timeout=5, retries=1)

    assert result["error"]["code"] == "network_error"
    assert sleeps == []
