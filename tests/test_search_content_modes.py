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


def test_snippets_mode_omits_summary_when_snippet_exists(fake_upstream: Any) -> None:
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Lean Result",
                        "url": "https://example.com/lean",
                        "snippet": "Short snippet",
                        "summary": "A longer upstream summary that must not be returned",
                    }
                ]
            },
        }
    )
    res = asyncio.run(server.deepseek_search("python", content="snippets", refresh=True))
    first = res["results"][0]
    assert first["snippet"] == "Short snippet"
    assert "summary" not in first
    assert "need_content" not in res


def test_snippets_mode_keeps_summary_as_fallback_without_snippet(fake_upstream: Any) -> None:
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Fallback Result",
                        "url": "https://example.com/fallback",
                        "snippet": "",
                        "summary": "Upstream summary used because there is no snippet",
                    }
                ]
            },
        }
    )
    res = asyncio.run(server.deepseek_search("python", content="snippets", refresh=True))
    first = res["results"][0]
    assert first["summary"] == "Upstream summary used because there is no snippet"
    assert "snippet" not in first


def test_snippets_mode_has_no_truncation_note(fake_upstream: Any) -> None:
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Long Summary",
                        "url": "https://example.com/long",
                        "snippet": "",
                        "summary": "Long upstream summary sentence. " * 200,
                    }
                ]
            },
        }
    )
    res = asyncio.run(server.deepseek_search("python", content="snippets", refresh=True))
    assert len(res["results"][0]["summary"]) <= server.SEARCH_SUMMARY_CHARS + 20
    assert server.TRUNCATED_NOTE not in res["notes"]


def test_search_many_snippets_are_lean(fake_upstream: Any) -> None:
    fake_upstream(
        default={
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "Many Result",
                        "url": "https://example.com/many",
                        "snippet": "Snippet for many",
                        "summary": "Summary that must be dropped",
                    }
                ]
            },
        }
    )
    res = asyncio.run(server.deepseek_search_many(["python one", "python two"], content="snippets"))
    assert res["ok"] is True
    for block in res["results"]:
        for item in block["results"]:
            assert item["snippet"] == "Snippet for many"
            assert "summary" not in item
            assert "need_content" not in item
