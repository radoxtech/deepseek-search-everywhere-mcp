"""Summary caps for search results without page content (``summary_chars``)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

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
    return asyncio.run(server.deepseek_search("asyncio tasks", count=3, **kwargs))


def _size(envelope: dict[str, Any]) -> int:
    return len(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))


def test_default_summary_is_short_without_content(fake_upstream):
    envelope = _search(fake_upstream)
    assert envelope["ok"] is True
    assert all(len(item["summary"]) <= server.SEARCH_SUMMARY_CHARS + 20 for item in envelope["results"])
    assert _size(envelope) <= 4096


def test_fallback_summary_with_content_derived_budget(fake_upstream):
    results = [dict(item, content="") for item in _realistic_results(1)]
    fake_upstream(default={"ok": True, "data": {"results": results}})
    envelope = asyncio.run(
        server.deepseek_search("asyncio tasks", count=1, content="full", max_total_chars=2000)
    )
    assert 2000 <= len(envelope["results"][0]["summary"]) < 2100


def test_per_result_budget_derived_from_max_total_chars(fake_upstream):
    results = [dict(item, content="A" * 10000) for item in _realistic_results(2)]
    fake_upstream(default={"ok": True, "data": {"results": results}})
    envelope = asyncio.run(
        server.deepseek_search("asyncio tasks", count=2, content="full", max_total_chars=2000)
    )
    assert envelope["ok"] is True
    # Derived per_result is max(500, 2000 // 2) = 1000
    for res in envelope["results"]:
        assert len(res["content"]) <= 1000 + len("\n\n…[truncated]")
