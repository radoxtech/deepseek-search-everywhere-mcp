"""Top-level ``cached`` on ``deepseek_fetch`` with a URL list and ``deepseek_research``."""

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
    first = asyncio.run(server.deepseek_fetch([A]))
    mixed = asyncio.run(server.deepseek_fetch([A, B]))
    warm = asyncio.run(server.deepseek_fetch([A, B]))
    assert (first["cached"], first["cached_count"]) == (False, 0)
    assert (mixed["cached"], mixed["cached_count"], mixed["requests_used"]) == (False, 1, 1)
    assert (warm["cached"], warm["cached_count"], warm["requests_used"]) == (True, 2, 0)
    assert len(upstream.calls) == 2


def test_fetch_many_with_only_rejected_urls_is_not_cached():
    envelope = asyncio.run(server.deepseek_fetch(["http://127.0.0.1/x"]))
    assert envelope["requests_used"] == 0
    assert envelope["cached"] is False


def test_search_many_cached_only_when_every_query_is_a_hit(fake_upstream):
    fake_upstream(responses={server.SEARCH_PATH: _search_answer(A)})
    first = asyncio.run(server.deepseek_search(["q1"]))
    mixed = asyncio.run(server.deepseek_search(["q1", "q2"]))
    warm = asyncio.run(server.deepseek_search(["q1", "q2"]))
    assert (first["cached"], first["cached_count"]) == (False, 0)
    assert (mixed["cached"], mixed["cached_count"], mixed["requests_used"]) == (False, 1, 1)
    assert (warm["cached"], warm["cached_count"], warm["requests_used"]) == (True, 2, 0)
