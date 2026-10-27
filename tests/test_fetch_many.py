"""Batch fetching: validation, dedupe, ordering, error isolation, budget and deadline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

A = "https://example.com/a"
B = "https://example.org/b"
C = "https://example.net/c"


class Upstream:
    """Fake ``server._request`` serving per-URL pages, failures and delays."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}
        self.delays: dict[str, float] = {}
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = kwargs["payload"]["url"]
        self.calls.append(url)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delays.get(url, 0))
        finally:
            self.active -= 1
        if url in self.pages:
            return self.pages[url]
        return {"ok": True, "data": {"title": f"Title {url}", "content": f"Body of {url} " + "x" * 200}}


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    fake = Upstream()
    monkeypatch.setattr(server, "_request", fake)
    return fake


async def test_partial_failure_keeps_order_and_ok(upstream: Upstream) -> None:
    upstream.delays[C] = 0.05
    upstream.pages[B] = {"ok": False, "error": {"code": "http_404", "message": "failed", "status": 404}}
    result = await server.deepseek_fetch_many([C, "ftp://bad", B, A])
    assert result["ok"] is True
    assert [item["url"] for item in result["results"]] == [C, "ftp://bad", B, A]
    assert [item["ok"] for item in result["results"]] == [True, False, False, True]
    assert result["results"][1]["error"]["code"] == "bad_url"
    assert result["results"][2]["error"]["code"] == "http_404"
    assert result["results"][0]["title"] == f"Title {C}"
    assert result["count"] == 4
    assert result["partial"] is True
    assert result["failed_count"] == 2
    assert result["requests_used"] == 3
    assert sorted(upstream.calls) == sorted([A, B, C])


async def test_ssrf_url_is_rejected_without_upstream_call(upstream: Upstream) -> None:
    result = await server.deepseek_fetch_many(["http://127.0.0.1/admin", A])
    assert result["results"][0]["error"]["code"] == "ssrf_blocked"
    assert result["results"][1]["ok"] is True
    assert upstream.calls == [A]


async def test_duplicates_and_empty_entries_are_fetched_once(upstream: Upstream) -> None:
    result = await server.deepseek_fetch_many(
        [A, "", "  ", "https://www.example.com/a/?utm_source=x#top", A, B]
    )
    assert upstream.calls.count(A) == 1
    assert result["count"] == 2
    assert [item["url"] for item in result["results"]] == [A, B]
    assert "deduped 2 duplicate URL(s)" in result["notes"]


async def test_cache_is_shared_with_deepseek_fetch(upstream: Upstream) -> None:
    await server.deepseek_fetch(A)
    result = await server.deepseek_fetch_many([A, B])
    assert upstream.calls == [A, B]
    assert [item["cached"] for item in result["results"]] == [True, False]
    assert result["cached_count"] == 1
    assert result["requests_used"] == 1
    refreshed = await server.deepseek_fetch_many([A], refresh=True)
    assert refreshed["cached_count"] == 0 and upstream.calls.count(A) == 2


@pytest.mark.parametrize("urls", [[], [f"https://example.com/{i}" for i in range(11)], ["", " "]])
async def test_bad_batch_size_is_rejected_without_upstream_calls(upstream: Upstream, urls: list[str]) -> None:
    result = await server.deepseek_fetch_many(urls)
    assert result["ok"] is False
    assert result["error"]["code"] == "bad_request"
    assert upstream.calls == []


async def test_budget_caps_total_chars(upstream: Upstream) -> None:
    urls = [A, B, C]
    result = await server.deepseek_fetch_many(urls, max_chars_per_url=4000, max_total_chars=100)
    total = sum(item["chars"] for item in result["results"])
    assert total <= 100
    assert all(item["chars"] == len(item["content"]) for item in result["results"])
    assert server.TRUNCATED_NOTE in result["notes"]


async def test_budget_untouched_when_pages_fit(upstream: Upstream) -> None:
    result = await server.deepseek_fetch_many([A, B])
    assert result["partial"] is False
    assert result["failed_count"] == 0
    assert server.TRUNCATED_NOTE not in result["notes"]
    assert sum(item["chars"] for item in result["results"]) <= 20000


async def test_concurrency_is_bounded(upstream: Upstream) -> None:
    urls = [f"https://example.com/{i}" for i in range(10)]
    for url in urls:
        upstream.delays[url] = 0.02
    result = await server.deepseek_fetch_many(urls)
    assert result["count"] == 10 and all(item["ok"] for item in result["results"])
    assert upstream.peak == 5


async def test_deadline_returns_partial_results(upstream: Upstream) -> None:
    set_settings(Settings(batch_deadline=0.3, batch_concurrency=1))
    upstream.delays.update({A: 0.2, B: 0.2, C: 0.2})
    started = time.monotonic()
    result = await server.deepseek_fetch_many([A, B, C])
    assert time.monotonic() - started < 1
    assert result["ok"] is True
    assert [item["url"] for item in result["results"]] == [A, B, C]
    assert result["results"][0]["ok"] is True
    codes = [item["error"]["code"] for item in result["results"] if not item["ok"]]
    assert codes and set(codes) == {"deadline_exceeded"}
    assert result["requests_used"] == 1


async def test_unexpected_exception_is_isolated(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    original = server._fetch

    async def flaky(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if url == B:
            raise RuntimeError("boom")
        return await original(url, *args, **kwargs)

    monkeypatch.setattr(server, "_fetch", flaky)
    result = await server.deepseek_fetch_many([A, B])
    assert result["ok"] is True
    assert result["results"][0]["ok"] is True
    assert result["results"][1]["error"]["code"] == "internal_error"


async def test_tool_is_registered() -> None:
    tools = {tool.name for tool in await server.mcp.list_tools()}
    assert "deepseek_fetch_many" in tools
