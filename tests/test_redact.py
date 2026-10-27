"""Redaction tests: credential patterns, exact secrets, false positives and result content (offline)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from deepseek_search_everywhere_mcp import server, transport
from deepseek_search_everywhere_mcp.errors import _map_http_error
from deepseek_search_everywhere_mcp.redact import redact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bearer abc123", "Bearer [REDACTED]"),
        ("bearer eyJhbGciOiJIUzI1NiJ9.payload", "bearer [REDACTED]"),
        ("key sk-abcdefghijkl", "key [REDACTED]"),
        ("https://user:pass@host/x", "https://[REDACTED]@host/x"),
        ("api_key=abc123 end", "api_key=[REDACTED] end"),
        ("https://x.test/?token=abc&q=1", "https://x.test/?token=[REDACTED]&q=1"),
        ("access_token=xyz&x=1", "access_token=[REDACTED]&x=1"),
        ("secret=hunter2", "secret=[REDACTED]"),
    ],
)
def test_redact_patterns(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_redact_exact_secret_anywhere() -> None:
    assert redact("the value exact-SECRET-value leaked", ("exact-SECRET-value",)) == (
        "the value [REDACTED] leaked"
    )


def test_redact_ignores_empty_secret() -> None:
    assert redact("plain text", ("",)) == "plain text"


@pytest.mark.parametrize(
    "text",
    [
        "tokenization and apiary",
        "bearer responsibility is ours",
        "risk-assessment-tool and task-manager-app",
        "https://host:8080/a@b",
        "https://example.com/path",
        "the secretary said nothing",
    ],
)
def test_redact_no_false_positives(text: str) -> None:
    assert redact(text) == text


def test_redact_is_idempotent() -> None:
    once = redact("Bearer abc123 and sk-abcdefghijkl")
    assert redact(once) == once


def test_http_error_message_is_redacted() -> None:
    body = '{"error": {"code": "invalid_key", "message": "bad key sk-abcdefghijkl"}}'
    envelope = _map_http_error(401, body)
    assert envelope["error"]["message"].startswith("bad key [REDACTED]")
    assert "sk-abcdefghijkl" not in envelope["error"]["message"]


async def _no_sleep(delay: float) -> None:
    return None


def test_transport_network_error_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-livekeyvalue123")

    async def boom(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("connect failed for sk-livekeyvalue123")

    monkeypatch.setattr(transport, "_attempt", boom)
    monkeypatch.setattr(transport, "_sleep", _no_sleep)

    result = asyncio.run(transport.request("GET", "/user/balance", timeout=1.0, retries=1))
    assert result["ok"] is False
    assert result["error"]["code"] == "network_error"
    assert "sk-livekeyvalue123" not in result["error"]["message"]


def test_notes_are_redacted_but_result_content_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    content = "Docs say to use sk-abcdefghijkl for testing."

    async def fake_rewrite(query: str) -> str:
        return "find sk-abcdefghijkl"

    async def fake_request(method: str, path: str, **kwargs: object) -> dict:
        return {
            "ok": True,
            "data": {"results": [{"title": "T", "url": "https://a.test", "content": content}]},
        }

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    monkeypatch.setattr(server, "_request", fake_request)

    envelope = asyncio.run(server._search("original question", 5, None, True, None, 4000, 30000, True, True))
    assert any("query rewritten" in note for note in envelope["notes"])
    assert all("sk-abcdefghijkl" not in note for note in envelope["notes"])
    assert envelope["results"][0]["content"] == content
