"""End-to-end tests against the real DeepSeek API, graded by an LLM judge.

Run with ``pytest -m live`` and ``DEEPSEEK_API_KEY`` set; without a key every test is skipped.
Budgets are kept small (3-5 results, short content) so a full run costs well under $0.01.
Thresholds tolerate normal web variability while still catching off-topic or broken output.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from live_judge import assert_quality, judge_answerable, judge_items

from deepseek_search_everywhere_mcp import server

pytestmark = pytest.mark.live

RequestFn = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
def upstream_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the path of every upstream request the server makes (requests still go out)."""
    calls: list[str] = []
    real: RequestFn = server._request

    async def spy(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(path)
        return await real(method, path, **kwargs)

    monkeypatch.setattr(server, "_request", spy)
    return calls


def _assert_ok(envelope: dict[str, Any]) -> None:
    assert envelope.get("ok") is True, f"tool failed: {envelope.get('error')}"


def _assert_results(envelope: dict[str, Any], minimum: int) -> list[dict[str, Any]]:
    _assert_ok(envelope)
    results = envelope["results"]
    assert len(results) >= minimum, f"expected >= {minimum} results, got {len(results)}"
    for item in results:
        assert isinstance(item.get("url"), str) and item["url"].startswith("http"), item
        assert isinstance(item.get("title"), str), item
    # Upstream occasionally returns an empty title; most results must still carry one.
    titled = sum(bool(item["title"].strip()) for item in results)
    assert titled * 2 > len(results), f"only {titled} of {len(results)} results have a title"
    return results


async def test_web_search_returns_relevant_results() -> None:
    query = "Python asyncio TaskGroup exception handling"
    envelope = await server.deepseek_web_search(query, count=5)
    results = _assert_results(envelope, 3)
    assert_quality(await judge_items("web_search", query, results))


async def test_search_and_read_returns_relevant_content() -> None:
    query = "PostgreSQL VACUUM FULL versus regular VACUUM"
    envelope = await server.deepseek_search_and_read(
        query, count=3, max_chars_per_result=1500, max_total_chars=4500
    )
    results = _assert_results(envelope, 2)
    with_content = [item for item in results if item.get("content")]
    assert len(with_content) >= 2, "expected page content for at least two results"
    assert all(item["content_chars"] > 0 for item in with_content)
    assert_quality(await judge_items("search_and_read", query, with_content))


@pytest.mark.parametrize(
    ("query", "lang"),
    [
        ("cómo funciona la fotosíntesis en las plantas", "es"),
        ("富士山 登山 シーズン", "ja"),
    ],
    ids=["es", "ja"],
)
async def test_non_english_query_with_lang(query: str, lang: str) -> None:
    envelope = await server.deepseek_web_search(query, count=4, lang=lang)
    results = _assert_results(envelope, 3)
    assert f"lang:{lang}" in envelope["query_used"]
    assert_quality(await judge_items(f"web_search[{lang}]", query, results))


async def test_rewrite_keeps_results_relevant(upstream_calls: list[str]) -> None:
    query = "Wie installiere ich Docker unter Ubuntu"
    envelope = await server.deepseek_web_search(query, count=4, rewrite=True)
    results = _assert_results(envelope, 3)
    assert "/v1/chat/completions" in upstream_calls, "rewrite did not call the model"
    assert envelope.get("query_original") == query, f"query was not rewritten; notes: {envelope['notes']}"
    assert envelope["query_used"] != query
    assert_quality(await judge_items("web_search[rewrite]", query, results))


async def test_fetch_stable_page_matches_its_content() -> None:
    url = "https://docs.python.org/3/library/asyncio.html"
    envelope = await server.deepseek_fetch(url, max_chars=3000)
    _assert_ok(envelope)
    assert envelope["url"] == url
    assert envelope["content_chars"] > 200, "fetched page is (nearly) empty"
    assert "asyncio" in envelope["content"].lower()
    item = {"title": envelope["title"], "url": url, "content": envelope["content"]}
    record = await judge_items("fetch", "Python asyncio standard library documentation", [item])
    assert_quality(record, min_mean=4.0, min_share=1.0)


async def test_fetch_many_keeps_order_and_dedupes(upstream_calls: list[str]) -> None:
    urls = [
        "https://docs.python.org/3/library/asyncio.html",
        "https://www.python.org/",
        "https://docs.python.org/3/library/asyncio.html#section",
    ]
    envelope = await server.deepseek_fetch_many(urls, max_chars_per_url=2000, max_total_chars=4000)
    _assert_ok(envelope)
    assert envelope["count"] == 2
    assert [item["url"] for item in envelope["results"]] == urls[:2]
    assert "deduped 1 duplicate URL(s)" in envelope["notes"]
    assert upstream_calls.count(server.FETCH_PATH) == 2
    for item in envelope["results"]:
        assert item["ok"] is True, f"{item['url']}: {item.get('error')}"
        assert item["chars"] > 200, f"{item['url']}: content is (nearly) empty"
    first, second = envelope["results"]
    assert_quality(
        await judge_items("fetch_many[asyncio]", "Python asyncio documentation", [first]),
        min_mean=4.0,
        min_share=1.0,
    )
    assert_quality(
        await judge_items(
            "fetch_many[python.org]", "Official homepage of the Python programming language", [second]
        ),
        min_mean=4.0,
        min_share=1.0,
    )


async def test_research_sources_can_answer_the_question() -> None:
    question = "What is the difference between TCP and UDP?"
    envelope = await server.deepseek_research(question, max_pages=3, per_page_chars=1500, total_chars=4500)
    _assert_ok(envelope)
    sources = [source for source in envelope["sources"] if source["content"]]
    assert len(sources) >= 2, f"expected >= 2 sources with content; notes: {envelope['notes']}"
    assert envelope["requests_used"] >= 1
    record = await judge_answerable("research", question, sources)
    assert_quality(record, min_mean=3.0, min_share=1.0)


async def test_repeated_search_is_served_from_cache(upstream_calls: list[str]) -> None:
    query = "HTTP 418 I'm a teapot status code"
    first = await server.deepseek_web_search(query, count=3)
    _assert_results(first, 1)
    assert first["cached"] is False
    assert upstream_calls == [server.SEARCH_PATH]

    second = await server.deepseek_web_search(query, count=3)
    _assert_ok(second)
    assert second["cached"] is True
    assert upstream_calls == [server.SEARCH_PATH], "the repeat made another upstream call"
    assert [r["url"] for r in second["results"]] == [r["url"] for r in first["results"]]


async def test_fetch_of_loopback_is_blocked_without_network(upstream_calls: list[str]) -> None:
    envelope = await server.deepseek_fetch("http://127.0.0.1/")
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "ssrf_blocked"
    assert upstream_calls == []


async def test_balance_is_available() -> None:
    envelope = await server.deepseek_balance()
    _assert_ok(envelope)
    assert envelope["is_available"] is True
    assert envelope["balances"], "no balance entries"
