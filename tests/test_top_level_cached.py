"""Top-level ``cached`` on ``deepseek_fetch_many`` and ``deepseek_research``."""

from __future__ import annotations

import asyncio

from deepseek_search_everywhere_mcp import server

A = "https://example.com/a"
B = "https://example.com/b"
PAGE = {"ok": True, "data": {"title": "T", "content": "page text " * 50}}


def _search_answer(*urls: str) -> dict:
    return {"ok": True, "data": {"results": [{"title": u, "url": u, "content": ""} for u in urls]}}


def test_fetch_many_cached_only_when_every_page_is_a_hit(fake_upstream):
    upstream = fake_upstream(default=PAGE)
    first = asyncio.run(server.deepseek_fetch_many([A]))
    mixed = asyncio.run(server.deepseek_fetch_many([A, B]))
    warm = asyncio.run(server.deepseek_fetch_many([A, B]))
    assert (first["cached"], first["cached_count"]) == (False, 0)
    assert (mixed["cached"], mixed["cached_count"], mixed["requests_used"]) == (False, 1, 1)
    assert (warm["cached"], warm["cached_count"], warm["requests_used"]) == (True, 2, 0)
    assert len(upstream.calls) == 2


def test_fetch_many_with_only_rejected_urls_is_not_cached():
    envelope = asyncio.run(server.deepseek_fetch_many(["http://127.0.0.1/x"]))
    assert envelope["requests_used"] == 0
    assert envelope["cached"] is False


def test_research_cached_only_without_upstream_requests(fake_upstream):
    fake_upstream(responses={server.SEARCH_PATH: _search_answer(A), server.FETCH_PATH: PAGE})
    cold = asyncio.run(server.deepseek_research("q", max_pages=1))
    warm = asyncio.run(server.deepseek_research("q", max_pages=1))
    assert (cold["cached"], cold["requests_used"]) == (False, 2)
    assert (warm["cached"], warm["requests_used"], warm["cached_count"]) == (True, 0, 2)


def test_research_with_cached_search_but_fresh_fetch_is_not_cached(fake_upstream):
    fake_upstream(responses={server.SEARCH_PATH: _search_answer(A), server.FETCH_PATH: PAGE})
    asyncio.run(server.deepseek_research("q", max_pages=1))
    server._get_fetch_cache().clear()
    envelope = asyncio.run(server.deepseek_research("q", max_pages=1))
    assert (envelope["cached"], envelope["cached_count"], envelope["requests_used"]) == (False, 1, 1)
