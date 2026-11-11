"""Tests for smart search fallback on 404 for PDF and Reddit in deepseek_fetch."""

from __future__ import annotations

import json
from typing import Any

from deepseek_search_everywhere_mcp import server


class FallbackUpstream:
    def __init__(
        self,
        fetch_status: int = 404,
        search_results: list[dict[str, Any]] | None = None,
        search_ok: bool = True,
        chat_response: str = "",
    ) -> None:
        self.fetch_status = fetch_status
        self.search_results = search_results if search_results is not None else []
        self.search_ok = search_ok
        self.chat_response = chat_response
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        self.calls.append({"method": method, "path": path, "payload": payload})
        if path == server.FETCH_PATH:
            if self.fetch_status == 200:
                return {
                    "ok": True,
                    "data": {
                        "url": payload.get("url"),
                        "title": "Standard Title",
                        "content": "Standard Content",
                    },
                }
            return {
                "ok": False,
                "error": {
                    "code": f"http_{self.fetch_status}",
                    "message": "failed to fetch URL",
                    "status": self.fetch_status,
                },
            }
        if path == server.SEARCH_PATH:
            if not self.search_ok:
                return {"ok": False, "error": {"code": "upstream_500", "message": "Search error"}}
            return {
                "ok": True,
                "data": {
                    "results": self.search_results,
                },
            }
        if path == "/v1/chat/completions":
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self.chat_response,
                            }
                        }
                    ]
                },
            }
        return {"ok": False, "error": {"code": "not_found", "message": f"Unexpected path {path}"}}


async def test_fetch_pdf_fallback_success(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    search_item = {
        "url": "https://arxiv.org/abs/1706.03762",
        "title": "Attention Is All You Need",
        "content": "The dominant sequence transduction models are based on attention.",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res = await server.deepseek_fetch(pdf_url)
    assert res["ok"] is True
    assert res["title"] == "Attention Is All You Need"
    assert "The dominant sequence" in res["content"]
    assert any("Content retrieved via DeepSeek search index" in note for note in res["notes"])
    # 1 fetch attempt (404) + 1 search attempt = 2 requests
    assert res["cost_usd"] == round(2 * server.COST_PER_REQUEST, 5)

    # Verify calls made
    assert len(upstream.calls) == 2
    assert upstream.calls[0]["path"] == server.FETCH_PATH
    assert upstream.calls[1]["path"] == server.SEARCH_PATH
    assert "arxiv.org/pdf/1706.03762" in upstream.calls[1]["payload"]["query"]


async def test_fetch_reddit_fallback_success(fake_upstream: Any) -> None:
    reddit_url = "https://www.reddit.com/r/MachineLearning/comments/123/discussion"
    search_item = {
        "url": "https://reddit.com/r/MachineLearning/comments/123/discussion",
        "title": "Discussion on LLMs",
        "content": "Interesting thread about reasoning models.",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res = await server.deepseek_fetch(reddit_url)
    assert res["ok"] is True
    assert res["title"] == "Discussion on LLMs"
    assert "Interesting thread" in res["content"]
    assert any("Content retrieved via DeepSeek search index" in note for note in res["notes"])


async def test_fetch_pdf_fallback_caching(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    search_item = {
        "url": "https://arxiv.org/pdf/1706.03762",
        "title": "Cached Paper",
        "content": "Paper content to cache.",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res1 = await server.deepseek_fetch(pdf_url)
    assert res1["ok"] is True
    assert res1["cached"] is False
    assert len(upstream.calls) == 2

    # Second call should be served from fetch cache at 0 cost
    res2 = await server.deepseek_fetch(pdf_url)
    assert res2["ok"] is True
    assert res2["cached"] is True
    assert res2["cost_usd"] == 0.0
    assert res2["content"] == "Paper content to cache."
    assert any("Content retrieved via DeepSeek search index" in note for note in res2["notes"])
    # No new upstream calls
    assert len(upstream.calls) == 2


async def test_fetch_fallback_disabled(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    search_item = {
        "url": "https://arxiv.org/pdf/1706.03762",
        "title": "Paper",
        "content": "Content",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res = await server.deepseek_fetch(pdf_url, fallback_to_search=False)
    assert res["ok"] is False
    assert res["error"]["code"] == "http_404"
    assert "deepseek_search" in res["error"]["message"]
    # Only 1 fetch call, no search call
    assert len(upstream.calls) == 1
    assert upstream.calls[0]["path"] == server.FETCH_PATH


async def test_fetch_fallback_domain_mismatch_fails_to_404(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    # Result domain is spam-site.com, not arxiv.org
    search_item = {
        "url": "https://spam-site.com/arxiv-pdf",
        "title": "Spam Paper",
        "content": "Untrusted spam content.",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res = await server.deepseek_fetch(pdf_url)
    assert res["ok"] is False
    assert res["error"]["code"] == "http_404"
    assert "deepseek_search" in res["error"]["message"]


async def test_fetch_fallback_empty_content_fails_to_404(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    search_item = {
        "url": "https://arxiv.org/pdf/1706.03762",
        "title": "Paper",
        "content": "",
    }
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item])
    fake_upstream(upstream)

    res = await server.deepseek_fetch(pdf_url)
    assert res["ok"] is False
    assert res["error"]["code"] == "http_404"
    assert "deepseek_search" in res["error"]["message"]


async def test_extract_pdf_uses_fallback(fake_upstream: Any) -> None:
    pdf_url = "https://arxiv.org/pdf/1706.03762"
    search_item = {
        "url": "https://arxiv.org/pdf/1706.03762",
        "title": "Attention Is All You Need",
        "content": "Transformers by Vaswani et al.",
    }
    chat_response = json.dumps({"title": "Attention Is All You Need", "year": 2017})
    upstream = FallbackUpstream(fetch_status=404, search_results=[search_item], chat_response=chat_response)
    fake_upstream(upstream)

    res = await server.deepseek_extract(pdf_url, fields=["title", "year"])
    assert res["ok"] is True
    assert res["data"] == {"title": "Attention Is All You Need", "year": 2017}
    # 1 fetch (404) + 1 search + 1 chat completion = 3 requests
    assert res["cost_usd"] == round(3 * server.COST_PER_REQUEST, 5)
