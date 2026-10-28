"""``chars`` is the returned content length in every tool; the old ``content_chars`` is gone."""

from __future__ import annotations

import asyncio

from deepseek_search_everywhere_mcp import server

URL = "https://example.com/article"
PAGE = "Plain page text with an escaped \\(paren\\). " * 100


def test_search_results_report_chars_only(fake_upstream):
    item = {"title": "T", "url": URL, "site_name": "Example", "content": PAGE}
    fake_upstream(default={"ok": True, "data": {"results": [item]}})
    envelope = asyncio.run(server.deepseek_search_and_read("q", count=1, max_chars_per_result=300))
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
    many = asyncio.run(server.deepseek_fetch_many([URL]))
    research = asyncio.run(server.deepseek_research("q", max_pages=1))
    search = asyncio.run(server.deepseek_search_and_read("q", count=1))
    for entry in (many["results"][0], research["sources"][0], search["results"][0]):
        assert entry["chars"] == len(entry["content"])
        assert "content_chars" not in entry
