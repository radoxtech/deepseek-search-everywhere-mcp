"""``elapsed_ms`` on every tool envelope, measured on an injectable monotonic clock."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

PAGE = {"ok": True, "data": {"title": "T", "content": "page text " * 50}}
RESULT = {"title": "T", "url": "https://example.com/x", "content": "c " * 200}
SEARCH = {"ok": True, "data": {"results": [RESULT]}}
BALANCE = {"ok": True, "data": {"is_available": True, "balance_infos": []}}
ERROR = {"ok": False, "error": {"code": "http_error", "message": "boom", "status": 500}}


class SlowUpstream:
    """Answers per path and advances the fake clock by ``step`` seconds on every call."""

    def __init__(self, clock: Any, step: float, answer: dict[str, Any] | None = None) -> None:
        self.clock, self.step, self.answer, self.calls = clock, step, answer, 0

    async def __call__(self, method: str, path: str, **_: Any) -> dict[str, Any]:
        self.calls += 1
        self.clock.advance(self.step)
        if self.answer is not None:
            return self.answer
        if path == server.SEARCH_PATH:
            return SEARCH
        if path == server.FETCH_PATH:
            return PAGE
        return BALANCE


@pytest.fixture
def clocked(monkeypatch, fake_clock, fake_upstream):
    """Install the fake clock and return a factory for a slow upstream on it."""
    monkeypatch.setattr(server, "_clock", fake_clock.monotonic)

    def install(step: float, answer: dict[str, Any] | None = None) -> SlowUpstream:
        return fake_upstream(SlowUpstream(fake_clock, step, answer))

    return install


CALLS = {
    "web_search": lambda: server.deepseek_web_search("q", count=1),
    "search_and_read": lambda: server.deepseek_search_and_read("q", count=1),
    "fetch": lambda: server.deepseek_fetch("https://example.com/x"),
    "fetch_many": lambda: server.deepseek_fetch_many(["https://example.com/x"]),
    "research": lambda: server.deepseek_research("q", max_pages=1),
    "balance": lambda: server.deepseek_balance(),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_success_envelope_reports_elapsed_ms(clocked, name):
    clocked(0.25)
    envelope = asyncio.run(CALLS[name]())
    assert envelope["ok"] is True
    assert envelope["elapsed_ms"] == 250


@pytest.mark.parametrize("name", sorted(CALLS))
def test_error_envelope_reports_elapsed_ms(clocked, name):
    clocked(0.04, ERROR)
    envelope = asyncio.run(CALLS[name]())
    if name == "fetch_many":  # the batch itself succeeds; the failure is per item
        assert envelope["partial"] is True
    else:
        assert envelope["error"]["code"] == "http_error"
    assert envelope["elapsed_ms"] == 40


def test_local_rejection_reports_elapsed_ms():
    envelope = asyncio.run(server.deepseek_web_search("   "))
    assert envelope["error"]["code"] == "empty_query"
    assert isinstance(envelope["elapsed_ms"], int) and envelope["elapsed_ms"] >= 0


def test_cache_hit_reports_its_own_time(clocked):
    upstream = clocked(0.5)
    first = asyncio.run(server.deepseek_fetch("https://example.com/x"))
    second = asyncio.run(server.deepseek_fetch("https://example.com/x"))
    assert upstream.calls == 1
    assert (first["elapsed_ms"], second["elapsed_ms"]) == (500, 0)
    assert second["cached"] is True


def test_shared_error_envelope_is_not_mutated(clocked):
    error = {"ok": False, "error": {"code": "http_error", "message": "boom"}}
    clocked(0.01, error)
    envelope = asyncio.run(server.deepseek_fetch("https://example.com/x"))
    assert "elapsed_ms" in envelope
    assert "elapsed_ms" not in error
