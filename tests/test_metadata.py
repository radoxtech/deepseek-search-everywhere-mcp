"""Unit tests for tool response metadata, cost tracking, and untrusted_content security markers."""

from __future__ import annotations

import asyncio
from typing import Any

from deepseek_search_everywhere_mcp import server


def _search_response(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"ok": True, "data": {"results": results or []}}


def _fetch_response(title: str = "Example", content: str = "Content") -> dict[str, Any]:
    return {"ok": True, "data": {"title": title, "content": content}}


def test_search_meta_and_cost(fake_upstream):
    fake_upstream(
        responses={"/v1/tools/search": _search_response([{"title": "T", "url": "https://example.com"}])}
    )

    # Cache miss: cost is COST_PER_REQUEST
    res1 = asyncio.run(server.deepseek_search("quantum computing"))
    assert res1["ok"] is True
    assert res1["cached"] is False
    assert res1["cost_usd"] == 0.00075
    assert res1["untrusted_content"] is True
    assert res1["_meta"]["cost_usd"] == 0.00075
    assert res1["_meta"]["untrusted_content"] is True

    # Cache hit: cost is 0.0
    res2 = asyncio.run(server.deepseek_search("quantum computing"))
    assert res2["ok"] is True
    assert res2["cached"] is True
    assert res2["cost_usd"] == 0.0
    assert res2["untrusted_content"] is True
    assert res2["_meta"]["cost_usd"] == 0.0
    assert res2["_meta"]["untrusted_content"] is True


def test_fetch_meta_and_cost(fake_upstream):
    fake_upstream(responses={"/v1/tools/fetch": _fetch_response()})

    # Cache miss
    f1 = asyncio.run(server.deepseek_fetch("https://example.com/page"))
    assert f1["ok"] is True
    assert f1["cached"] is False
    assert f1["cost_usd"] == 0.00075
    assert f1["untrusted_content"] is True
    assert f1["_meta"]["cost_usd"] == 0.00075
    assert f1["_meta"]["untrusted_content"] is True

    # Cache hit
    f2 = asyncio.run(server.deepseek_fetch("https://example.com/page"))
    assert f2["ok"] is True
    assert f2["cached"] is True
    assert f2["cost_usd"] == 0.0
    assert f2["untrusted_content"] is True
    assert f2["_meta"]["cost_usd"] == 0.0
    assert f2["_meta"]["untrusted_content"] is True


def test_fetch_many_meta_and_cost(fake_upstream):
    fake_upstream(responses={"/v1/tools/fetch": _fetch_response()})

    urls = ["https://example.com/1", "https://example.com/2"]
    res = asyncio.run(server.deepseek_fetch_many(urls))
    assert res["ok"] is True
    assert res["untrusted_content"] is True
    # 2 upstream requests = 2 * 0.00075 = 0.0015
    assert res["cost_usd"] == 0.0015
    assert res["_meta"]["cost_usd"] == 0.0015
    assert res["_meta"]["untrusted_content"] is True
    assert all(item["untrusted_content"] is True for item in res["results"])


def test_research_meta_and_cost(fake_upstream):
    fake_upstream(
        responses={
            "/v1/tools/search": _search_response(
                [{"title": "T1", "url": "https://example.com/1", "content": "Full content" * 50}]
            ),
        }
    )

    res = asyncio.run(server.deepseek_research("quantum computing"))
    assert res["ok"] is True
    assert res["untrusted_content"] is True
    assert res["cost_usd"] == 0.00075
    assert res["_meta"]["cost_usd"] == 0.00075
    assert res["_meta"]["untrusted_content"] is True
