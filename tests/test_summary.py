"""Summary caps for search results without page content (``summary_chars``)."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server


def _realistic_results(n: int = 3) -> list[dict[str, Any]]:
    """Upstream results sized like real ones: ~500-character snippets and ~4,000-character summaries."""
    return [
        {
            "title": f"Result number {i} about asynchronous programming in Python",
            "url": f"https://docs.example.org/guide/section-{i}/asyncio-tasks-and-coroutines",
            "date": "2026-01-15T00:00:00Z",
            "authority": 3,
            "rerank_score": 0.8123456789012345,
            "site_name": f"Example Docs {i}",
            "snippet": ("Snippet text describing the page in a couple of sentences. " * 10)[:500],
            "summary": ("A long upstream summary sentence that keeps going and going. " * 70)[:4000],
            "content": "Full page content. " * 400,
        }
        for i in range(1, n + 1)
    ]


def _search(fake_upstream, **kwargs: Any) -> dict[str, Any]:
    fake_upstream(default={"ok": True, "data": {"results": _realistic_results()}})
    return asyncio.run(server.deepseek_web_search("asyncio tasks", count=3, **kwargs))


def _size(envelope: dict[str, Any]) -> int:
    return len(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))


def test_default_summary_is_short_without_content(fake_upstream):
    envelope = _search(fake_upstream)
    assert envelope["ok"] is True
    assert all(len(item["summary"]) <= server.SEARCH_SUMMARY_CHARS + 20 for item in envelope["results"])
    assert _size(envelope) <= 4096


def test_summary_chars_zero_omits_summary(fake_upstream):
    envelope = _search(fake_upstream, summary_chars=0)
    assert all("summary" not in item for item in envelope["results"])
    assert server.TRUNCATED_NOTE not in envelope["notes"]


def test_summary_chars_raises_the_cap(fake_upstream):
    envelope = _search(fake_upstream, summary_chars=1000)
    assert all(item["summary"].startswith("A long upstream") for item in envelope["results"])
    assert all(1000 <= len(item["summary"]) < 1100 for item in envelope["results"])


def test_summary_is_bounded_by_max_chars_per_result(fake_upstream):
    envelope = _search(fake_upstream, summary_chars=3000, max_chars_per_result=200)
    assert all(len(item["summary"]) < 250 for item in envelope["results"])
    assert server.TRUNCATED_NOTE in envelope["notes"]


def test_negative_summary_chars_is_rejected(fake_upstream):
    envelope = _search(fake_upstream, summary_chars=-1)
    assert envelope["error"]["code"] == "bad_request"


def test_content_is_unaffected_by_summary_chars(fake_upstream):
    envelope = _search(fake_upstream, need_content=True, summary_chars=0)
    assert all(len(item["content"]) > server.SEARCH_SUMMARY_CHARS for item in envelope["results"])
    assert all("summary" not in item for item in envelope["results"])


def test_fallback_summary_with_content_keeps_max_chars_per_result(fake_upstream):
    results = [dict(item, content="") for item in _realistic_results(1)]
    fake_upstream(default={"ok": True, "data": {"results": results}})
    envelope = asyncio.run(
        server.deepseek_search_and_read("asyncio tasks", count=1, max_chars_per_result=2000)
    )
    assert 2000 <= len(envelope["results"][0]["summary"]) < 2100

    capped = asyncio.run(
        server.deepseek_search_and_read("asyncio tasks", count=1, summary_chars=0, refresh=True)
    )
    assert "summary" not in capped["results"][0]


@pytest.mark.parametrize("name", ["deepseek_web_search", "deepseek_search_and_read", "_search"])
def test_summary_chars_is_keyword_only(name):
    parameter = inspect.signature(getattr(server, name)).parameters["summary_chars"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
