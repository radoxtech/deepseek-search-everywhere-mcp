"""Research composition: gap filling, cost accounting, budget and partial failures."""

from __future__ import annotations

from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

LONG = "Long search content. " * 30


def _result(index: int, content: str) -> dict[str, Any]:
    return {
        "title": f"Result {index}",
        "url": f"https://site{index}.example.com/page",
        "authority": 2,
        "rerank_score": 1.0 - index / 10,
        "site_name": f"Site {index}",
        "content": content,
    }


class Upstream:
    """Fake ``server._request`` serving one search response and per-URL fetches."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.searches: list[dict[str, Any]] = []
        self.fetches: list[str] = []
        self.chats = 0
        self.failures: dict[str, dict[str, Any]] = {}

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.searches.append(payload)
            return {"ok": True, "data": {"results": self.results}}
        if path == server.FETCH_PATH:
            url = payload["url"]
            self.fetches.append(url)
            if url in self.failures:
                return {"ok": False, "error": self.failures[url]}
            body = f"Fetched body {url} " + "y" * 500
            return {"ok": True, "data": {"title": f"Fetched {url}", "content": body}}
        self.chats += 1
        return {"ok": False, "error": {"code": "unexpected", "message": path}}


def _mixed() -> list[dict[str, Any]]:
    return [_result(1, LONG), _result(2, ""), _result(3, LONG), _result(4, "too short"), _result(5, LONG)]


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    fake = Upstream(_mixed())
    monkeypatch.setattr(server, "_request", fake)
    return fake


async def test_research_fetches_only_the_gaps(upstream: Upstream) -> None:
    result = await server.deepseek_research("what is a research question")
    assert result["ok"] is True
    assert result["partial"] is False
    assert sorted(upstream.fetches) == ["https://site2.example.com/page", "https://site4.example.com/page"]
    assert result["requests_used"] == 3
    assert result["cached_count"] == 0
    assert all(source["cached"] is False for source in result["sources"])
    assert [source["id"] for source in result["sources"]] == [1, 2, 3, 4, 5]
    origins = [source["origin"] for source in result["sources"]]
    assert origins == ["search", "fetch", "search", "fetch", "search"]
    assert result["sources"][1]["content"].startswith("Fetched body https://site2.example.com/page")
    assert all(source["error"] is None for source in result["sources"])
    assert upstream.searches[0]["need_content"] is True
    assert upstream.searches[0]["count"] == 5
    assert set(result["sources"][0]) == {"id", "title", "url", "authority", "rerank_score", "content",
                                         "chars", "origin", "error", "cached"}


async def test_research_partial_fetch_failure(upstream: Upstream) -> None:
    upstream.failures["https://site4.example.com/page"] = {"code": "http_404", "message": "not found",
                                                           "status": 404}
    result = await server.deepseek_research("partial failure question")
    assert result["ok"] is True
    assert result["partial"] is True
    failed = result["sources"][3]
    assert failed["error"]["code"] == "http_404"
    assert failed["origin"] == "search"
    assert failed["content"] == "too short"
    assert result["sources"][1]["origin"] == "fetch"
    assert result["requests_used"] == 3


async def test_research_ssrf_url_is_not_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    results = _mixed()
    results[1]["url"] = "http://127.0.0.1/admin"
    fake = Upstream(results)
    monkeypatch.setattr(server, "_request", fake)
    result = await server.deepseek_research("internal url question")
    assert fake.fetches == ["https://site4.example.com/page"]
    assert result["sources"][1]["origin"] == "search"
    assert result["requests_used"] == 2


async def test_research_respects_budget(upstream: Upstream) -> None:
    result = await server.deepseek_research("budget question", per_page_chars=300, total_chars=1000)
    chars = [source["chars"] for source in result["sources"]]
    assert all(value <= 300 for value in chars)
    assert sum(chars) <= 1000
    assert all(source["chars"] == len(source["content"]) for source in result["sources"])
    assert "truncated_to_budget" in result["notes"]


async def test_research_default_budget(upstream: Upstream) -> None:
    upstream.results = [_result(index, "z" * 9000) for index in range(1, 6)]
    result = await server.deepseek_research("default budget question")
    assert sum(source["chars"] for source in result["sources"]) <= 30000
    assert "truncated_to_budget" in result["notes"]
    assert upstream.fetches == []


async def test_research_never_rewrites(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    calls = 0

    async def fake_rewrite(query: str) -> str:
        nonlocal calls
        calls += 1
        return "rewritten"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result = await server.deepseek_research("no rewrite question")
    assert calls == 0
    assert upstream.chats == 0
    assert upstream.searches[0]["query"] == "no rewrite question"
    assert result["ok"] is True


async def test_research_repeat_is_served_from_cache(upstream: Upstream) -> None:
    await server.deepseek_research("cached question")
    again = await server.deepseek_research("cached question")
    assert again["requests_used"] == 0
    assert again["cached_count"] == 3
    assert all(source["cached"] is True for source in again["sources"])
    assert len(upstream.searches) == 1
    assert len(upstream.fetches) == 2


async def test_research_search_error_is_returned(upstream: Upstream) -> None:
    result = await server.deepseek_research("")
    assert result["ok"] is False
    assert result["error"]["code"] == "empty_query"
    assert upstream.searches == []


async def test_research_rejects_bad_budget(upstream: Upstream) -> None:
    result = await server.deepseek_research("question", total_chars=0)
    assert result["error"]["code"] == "bad_request"
