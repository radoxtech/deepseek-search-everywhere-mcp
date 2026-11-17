"""Batch fetching: validation, dedupe, ordering, error isolation, budget and deadline."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import fetch_core, server
from deepseek_search_everywhere_mcp.errors import _fail
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
def upstream(fake_upstream: Any) -> Upstream:
    return fake_upstream(Upstream())


async def test_partial_failure_keeps_order_and_ok(upstream: Upstream) -> None:
    upstream.delays[C] = 0.05
    upstream.pages[B] = _fail("not_found", "failed", 404)
    result = await server.deepseek_fetch([C, "ftp://bad", B, A])
    assert result["ok"] is True
    assert [item["url"] for item in result["results"]] == [C, "ftp://bad", B, A]
    assert [item["ok"] for item in result["results"]] == [True, False, False, True]
    assert result["results"][1]["error"]["reason"] == "bad_url"
    assert result["results"][2]["error"]["reason"] == "not_found"
    assert result["results"][0]["title"] == f"Title {C}"
    assert result["count"] == 4
    assert result["partial"] is True
    assert result["failed_count"] == 2
    assert result["requests_used"] == 3
    assert sorted(upstream.calls) == sorted([A, B, C])


async def test_ssrf_url_is_rejected_without_upstream_call(upstream: Upstream) -> None:
    result = await server.deepseek_fetch(["http://127.0.0.1/admin", A])
    assert result["results"][0]["error"]["reason"] == "ssrf_blocked"
    assert result["results"][1]["ok"] is True
    assert upstream.calls == [A]


async def test_duplicates_and_empty_entries_are_fetched_once(upstream: Upstream) -> None:
    result = await server.deepseek_fetch([A, "", "  ", "https://www.example.com/a/?utm_source=x#top", A, B])
    assert upstream.calls.count(A) == 1
    assert result["count"] == 2
    assert [item["url"] for item in result["results"]] == [A, B]
    assert "deduped 2 duplicate URL(s)" in result["notes"]


async def test_cache_is_shared_with_deepseek_fetch(upstream: Upstream) -> None:
    await server.deepseek_fetch(A)
    result = await server.deepseek_fetch([A, B])
    assert upstream.calls == [A, B]
    assert [item["cached"] for item in result["results"]] == [True, False]
    assert result["cached_count"] == 1
    assert result["requests_used"] == 1
    refreshed = await server.deepseek_fetch([A], refresh=True)
    assert refreshed["cached_count"] == 0 and upstream.calls.count(A) == 2


@pytest.mark.parametrize("urls", [[], [f"https://example.com/{i}" for i in range(11)], ["", " "]])
async def test_bad_batch_size_is_rejected_without_upstream_calls(upstream: Upstream, urls: list[str]) -> None:
    result = await server.deepseek_fetch(urls)
    assert result["ok"] is False
    assert result["error"]["reason"] == "bad_request"
    assert upstream.calls == []


async def test_budget_caps_total_chars(upstream: Upstream) -> None:
    urls = [A, B, C]
    result = await server.deepseek_fetch(urls, max_chars=4000, max_total_chars=100)
    total = sum(item["chars"] for item in result["results"])
    assert total <= 100
    assert all(item["chars"] == len(item["content"]) for item in result["results"])
    assert server.TRUNCATED_NOTE in result["notes"]


async def test_budget_untouched_when_pages_fit(upstream: Upstream) -> None:
    result = await server.deepseek_fetch([A, B])
    assert result["partial"] is False
    assert result["failed_count"] == 0
    assert server.TRUNCATED_NOTE not in result["notes"]
    assert sum(item["chars"] for item in result["results"]) <= 20000


async def test_concurrency_is_bounded(upstream: Upstream) -> None:
    urls = [f"https://example.com/{i}" for i in range(10)]
    for url in urls:
        upstream.delays[url] = 0.02
    result = await server.deepseek_fetch(urls)
    assert result["count"] == 10 and all(item["ok"] for item in result["results"])
    assert upstream.peak == 5


async def test_deadline_returns_partial_results(upstream: Upstream) -> None:
    set_settings(Settings(batch_deadline=0.3, batch_concurrency=1))
    upstream.delays.update({A: 0.2, B: 0.2, C: 0.2})
    started = time.monotonic()
    result = await server.deepseek_fetch([A, B, C])
    assert time.monotonic() - started < 1
    assert result["ok"] is True
    assert [item["url"] for item in result["results"]] == [A, B, C]
    assert result["results"][0]["ok"] is True
    codes = [item["error"]["reason"] for item in result["results"] if not item["ok"]]
    assert codes and set(codes) == {"deadline_exceeded"}
    assert result["requests_used"] == 1


async def test_unexpected_exception_is_isolated(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    original = server._fetch

    async def flaky(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if url == B:
            raise RuntimeError("boom")
        return await original(url, *args, **kwargs)

    monkeypatch.setattr(fetch_core, "_fetch", flaky)
    result = await server.deepseek_fetch([A, B])
    assert result["ok"] is True
    assert result["results"][0]["ok"] is True
    assert result["results"][1]["error"]["reason"] == "internal_error"


async def test_unexpected_exception_is_logged_redacted(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom Bearer sk-abcdefghijklmnop")

    monkeypatch.setattr(fetch_core, "_fetch", broken)
    with caplog.at_level(logging.DEBUG, logger=server.__name__):
        result = await server.deepseek_fetch([A])
    assert result["results"][0]["error"]["reason"] == "internal_error"
    messages = [record.getMessage() for record in caplog.records if record.name == server.__name__]
    assert any("RuntimeError" in message for message in messages)
    assert all("sk-abcdefghijklmnop" not in message for message in messages)


async def test_tool_is_registered() -> None:
    tools = {tool.name for tool in await server.mcp.list_tools()}
    assert "deepseek_fetch" in tools and "deepseek_fetch_many" not in tools


@pytest.mark.parametrize(("name", "value"), [("section", 1), ("offset", 0), ("find_text", "Body")])
async def test_single_url_selectors_are_rejected_for_a_list(
    upstream: Upstream, name: str, value: Any
) -> None:
    result = await server.deepseek_fetch([A, B], **{name: value})
    assert result["ok"] is False
    assert (result["error"]["code"], result["error"]["param"]) == ("invalid_input", name)
    assert upstream.calls == []


async def test_max_chars_defaults_differ_for_one_url_and_a_list(upstream: Upstream) -> None:
    upstream.pages[A] = {"ok": True, "data": {"title": "Long", "content": "y" * 30000}}
    single = await server.deepseek_fetch(A)
    listed = await server.deepseek_fetch([A])
    # Each limit applies to page text; a truncation marker of a few characters may follow it.
    assert 20000 <= single["chars"] < 20100 and single["truncated"] is True
    assert 4000 <= len(listed["results"][0]["content"]) < 4100
