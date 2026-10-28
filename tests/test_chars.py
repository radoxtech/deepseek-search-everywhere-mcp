"""``chars`` is the returned content length in every tool; ``content_chars`` is a deprecated alias."""

from __future__ import annotations

import asyncio

from deepseek_search_everywhere_mcp import server

URL = "https://example.com/article"
PAGE = "Plain page text with an escaped \\(paren\\). " * 100


def test_search_results_report_chars_and_deprecated_content_chars(fake_upstream):
    item = {"title": "T", "url": URL, "site_name": "Example", "content": PAGE}
    fake_upstream(default={"ok": True, "data": {"results": [item]}})
    envelope = asyncio.run(server.deepseek_search_and_read("q", count=1, max_chars_per_result=300))
    result = envelope["results"][0]
    assert result["chars"] == len(result["content"])
    assert result["content_chars"] == len(server._unescape_markdown(PAGE))
    assert result["content_chars"] > result["chars"]


def test_fetch_reports_chars_with_content_chars_alias(fake_upstream):
    fake_upstream(default={"ok": True, "data": {"title": "T", "content": PAGE}})
    envelope = asyncio.run(server.deepseek_fetch(URL, max_chars=500))
    assert envelope["chars"] == len(envelope["content"])
    assert envelope["content_chars"] == envelope["chars"]


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
