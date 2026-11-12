"""``chars`` is the returned content length in every tool; the old ``content_chars`` is gone."""

from __future__ import annotations

import asyncio

from deepseek_search_everywhere_mcp import server

URL = "https://example.com/article"
PAGE = "Plain page text with an escaped \\(paren\\). " * 100


def test_search_results_report_chars_only(fake_upstream):
    item = {"title": "T", "url": URL, "site_name": "Example", "content": PAGE}
    fake_upstream(default={"ok": True, "data": {"results": [item]}})
    envelope = asyncio.run(server.deepseek_search("q", count=1, content="full", max_total_chars=300))
    result = envelope["results"][0]
    assert result["chars"] == len(result["content"])
    assert "content_chars" not in result


def test_fetch_reports_chars_only(fake_upstream):
    fake_upstream(default={"ok": True, "data": {"title": "T", "content": PAGE}})
    envelope = asyncio.run(server.deepseek_fetch(URL, max_chars=500))
    assert envelope["chars"] == len(envelope["content"])
    assert "content_chars" not in envelope


def test_every_tool_uses_chars(fake_upstream):
    item = {"title": "T", "url": URL, "site_name": "Example", "content": PAGE}
    fake_upstream(
        responses={
            server.SEARCH_PATH: {"ok": True, "data": {"results": [item]}},
            server.FETCH_PATH: {"ok": True, "data": {"title": "T", "content": PAGE}},
        }
    )
    single = asyncio.run(server.deepseek_fetch(URL))
    many = asyncio.run(server.deepseek_fetch_many([URL]))
    search = asyncio.run(server.deepseek_search("q", count=1, content="full"))
    search_many = asyncio.run(server.deepseek_search_many(["q"], count=1, content="full"))
    for entry in (single, many["results"][0], search["results"][0], search_many["results"][0]["results"][0]):
        assert entry["chars"] == len(entry["content"])
        assert "content_chars" not in entry
