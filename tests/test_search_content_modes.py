"""Tests for search content modes ('snippets' vs 'full') and gap-filling."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server


def test_search_content_invalid_mode() -> None:
    res = asyncio.run(server.deepseek_search("python", content="invalid"))  # type: ignore[arg-type]
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"


def test_search_content_snippets(fake_upstream: Any) -> None:
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Snippets Result",
                        "url": "https://example.com/item",
                        "snippet": "Short snippet",
                        "content": "Full content that should not be included in snippets mode",
                    }
                ]
            },
        }
    )
    res = asyncio.run(server.deepseek_search("python", content="snippets", refresh=True))
    assert res["ok"] is True
    assert res["content"] == "snippets"
    first = res["results"][0]
    assert "snippet" in first
    assert "content" not in first


def test_search_content_full_gap_fills_stub(fake_upstream: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Upstream returns a result with stub content (< 200 chars)
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Stub Result",
                        "url": "https://example.com/stub",
                        "snippet": "Short snippet",
                        "content": "Short stub.",  # < 200 chars triggers gap-fill
                    }
                ]
            },
        }
    )

    async def fake_fetch_many(urls: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "requests_used": 1,
            "cached_count": 0,
            "notes": [],
            "results": [
                {
                    "ok": True,
                    "url": urls[0],
                    "title": "Fetched Full Title",
                    "content": "Full fetched page content from isolated fetch " * 20,
                    "cached": False,
                }
            ],
        }

    monkeypatch.setattr(server, "deepseek_fetch_many", fake_fetch_many)

    res = asyncio.run(server.deepseek_search("python", content="full", refresh=True))
    assert res["ok"] is True
    assert res["content"] == "full"
    # Search used 1 request, gap-fill used 1 fetch -> requests_used == 2
    assert res["requests_used"] == 2
    assert res["cost_usd"] == round(2 * server.COST_PER_REQUEST, 6)
    first = res["results"][0]
    assert "Full fetched page content" in first["content"]
    assert any("fetched 1 page(s)" in n for n in res["notes"])
