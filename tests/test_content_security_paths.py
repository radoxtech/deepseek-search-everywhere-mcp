"""Parametrized integration tests verifying content sanitization on all paths."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from deepseek_search_everywhere_mcp.server import (
    deepseek_deep_research,
    deepseek_extract,
    deepseek_extract_many,
    deepseek_fetch,
    deepseek_fetch_many,
    deepseek_research,
    deepseek_search,
    deepseek_search_many,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "injection"


@pytest.fixture
def injection_payload() -> str:
    """Combined injection payload including tag characters, zero width, images and javascript links."""
    tag_sample = (FIXTURES_DIR / "tag_chars_hidden.md").read_text(encoding="utf-8")
    zw_sample = (FIXTURES_DIR / "zero_width_split.md").read_text(encoding="utf-8")
    img_sample = (FIXTURES_DIR / "markdown_image_exfil.md").read_text(encoding="utf-8")
    link_sample = (FIXTURES_DIR / "unsafe_links.md").read_text(encoding="utf-8")
    return f"{tag_sample}\n\n{zw_sample}\n\n{img_sample}\n\n{link_sample}"


def _assert_content_is_sanitized(text: str) -> None:
    # 1. No Unicode Tag characters (U+E0000 - U+E007F)
    assert not any(0xE0000 <= ord(c) <= 0xE007F for c in text)
    # 2. No zero-width spaces
    assert "\u200b" not in text
    assert "\u200c" not in text
    assert "\u2060" not in text
    assert "\ufeff" not in text
    # 3. No active unescaped markdown images
    assert "![Company Logo]" not in text
    # 4. No javascript: or data: link destinations
    assert "javascript:" not in text
    assert "data:text/html" not in text


@pytest.mark.asyncio
async def test_search_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "code": 0,
            "data": {
                "results": [
                    {
                        "title": "Untrusted\u200b Result 1",
                        "url": "https://example.com/untrusted1",
                        "snippet": f"Snippet: {injection_payload}",
                        "content": injection_payload,
                    },
                    {
                        "title": "Untrusted\u200b Result 2",
                        "url": "https://example.com/untrusted2",
                        "snippet": f"Snippet: {injection_payload}",
                        "summary": f"Summary: {injection_payload}",
                    },
                ]
            },
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._request", fake_request)
    res = await deepseek_search("test query", need_content=True, refresh=True)
    assert res["ok"] is True
    first = res["results"][0]
    _assert_content_is_sanitized(first["content"])
    _assert_content_is_sanitized(first["snippet"])
    assert "\u200b" not in first["title"]
    assert "[image: Company Logo]" in first["content"]

    second = res["results"][1]
    _assert_content_is_sanitized(second["summary"])


@pytest.mark.asyncio
async def test_search_many_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "code": 0,
            "data": {
                "results": [
                    {
                        "title": "Untrusted Result",
                        "url": "https://example.com/untrusted",
                        "snippet": "clean",
                        "content": injection_payload,
                    }
                ]
            },
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._request", fake_request)
    res = await deepseek_search_many(["test query 1"], need_content=True, refresh=True)
    assert res["ok"] is True
    first = res["results"][0]["results"][0]
    _assert_content_is_sanitized(first["content"])


@pytest.mark.asyncio
async def test_fetch_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "code": 0,
            "data": {
                "title": "Test\u200b Page",
                "content": injection_payload,
            },
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._request", fake_request)
    res = await deepseek_fetch("https://example.com/page", refresh=True)
    assert res["ok"] is True
    _assert_content_is_sanitized(res["content"])
    assert "\u200b" not in res["title"]
    # Check that sanitization notes were recorded
    assert any("Sanitized" in n or "Neutralized" in n for n in res["notes"])


@pytest.mark.asyncio
async def test_fetch_many_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "code": 0,
            "data": {
                "title": "Batch Page",
                "content": injection_payload,
            },
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._request", fake_request)
    res = await deepseek_fetch_many(["https://example.com/page1"], refresh=True)
    assert res["ok"] is True
    first = res["results"][0]
    _assert_content_is_sanitized(first["content"])


@pytest.mark.asyncio
async def test_extract_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "url": "https://example.com/extract",
            "title": "Extract Page",
            "content": "Short text for extraction",
            "cached": False,
            "cost_usd": 0.0002,
        }

    async def fake_chat_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "content": (
                '{"notes": "Hello\\u200b world", "tracker": "![Image](https://evil.com/leak.png)", '
                '"action": "[Login](javascript:steal())"}'
            ),
            "cost_usd": 0.0001,
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._fetch", fake_fetch)
    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._chat_request", fake_chat_request)

    res = await deepseek_extract("https://example.com/extract", fields=["notes", "tracker", "action"])
    assert res["ok"] is True
    data = res["data"]
    assert "\u200b" not in data["notes"]
    assert "[image: Image]" in data["tracker"]
    assert "javascript:" not in data["action"]


@pytest.mark.asyncio
async def test_extract_many_path_sanitization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "url": args[0] if args else "https://example.com/1",
            "title": "Page",
            "content": "Short text for extraction",
            "cached": False,
            "cost_usd": 0.0002,
        }

    async def fake_chat_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "content": '{"field": "Safe\\u200b text with ![Img](https://evil.com/x.png)"}',
            "cost_usd": 0.0001,
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._fetch", fake_fetch)
    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._chat_request", fake_chat_request)

    res = await deepseek_extract_many(["https://example.com/1"], fields=["field"])
    assert res["ok"] is True
    data = res["results"][0]["data"]
    assert "\u200b" not in data["field"]
    assert "[image: Img]" in data["field"]


@pytest.mark.asyncio
async def test_research_path_sanitization(monkeypatch: pytest.MonkeyPatch, injection_payload: str) -> None:
    async def fake_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "code": 0,
            "data": {
                "results": [
                    {
                        "title": "Untrusted\u200b Source",
                        "url": "https://example.com/source1",
                        "content": injection_payload,
                    }
                ]
            },
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._request", fake_request)
    res = await deepseek_research("ai security", refresh=True)
    assert res["ok"] is True
    first_source = res["sources"][0]
    _assert_content_is_sanitized(first_source["content"])
    assert "\u200b" not in first_source["title"]


@pytest.mark.asyncio
async def test_deep_research_path_sanitization(
    monkeypatch: pytest.MonkeyPatch, injection_payload: str
) -> None:
    async def fake_chat_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        messages = kwargs.get("messages") or (args[0] if args else [])
        sys_content = messages[0]["content"] if messages else ""
        if "decomposition" in sys_content or "sub-queries" in sys_content or "search queries" in sys_content:
            return {"ok": True, "content": '["query 1", "query 2"]', "cost_usd": 0.0001}
        # Synthesis call
        return {
            "ok": True,
            "content": f"Research report summary [1].\n\n{injection_payload}",
            "cost_usd": 0.0005,
        }

    async def fake_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "results": [
                {
                    "title": "Untrusted Source",
                    "url": "https://example.com/source1",
                    "content": "Source body content with enough information to cite [1].",
                }
            ],
            "cost_usd": 0.0002,
        }

    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._chat_request", fake_chat_request)
    monkeypatch.setattr("deepseek_search_everywhere_mcp.server._search", fake_search)

    res = await deepseek_deep_research("ai security", refresh=True)
    assert res["ok"] is True
    assert res["synthesized"] is True
    _assert_content_is_sanitized(res["report"])
