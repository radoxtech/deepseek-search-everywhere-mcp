"""Backward-compatibility contract: positional signatures, tool names, error codes and notes."""

import asyncio
import inspect
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

TOOL_NAMES = {"deepseek_web_search", "deepseek_search_and_read", "deepseek_fetch", "deepseek_balance"}


def _install_fake_request(monkeypatch: pytest.MonkeyPatch, results: list[dict[str, Any]]) -> list[dict]:
    calls: list[dict] = []

    async def fake_request(method, path, *, payload=None, timeout=None, retries=2, **_):
        calls.append({"path": path, "payload": payload})
        return {"ok": True, "data": {"results": results}}

    monkeypatch.setattr(server, "_request", fake_request)
    monkeypatch.setattr(server, "_api_key", lambda: "test-key")
    return calls


def test_search_accepts_nine_positional_arguments(monkeypatch):
    calls = _install_fake_request(monkeypatch, [])
    result = asyncio.run(server._search("q", 5, None, False, None, 1000, 4000, True, False))
    assert result["ok"] is True
    assert len(calls) == 1


def test_tool_names_are_stable():
    tools = asyncio.run(server.mcp.list_tools())
    assert TOOL_NAMES <= {tool.name for tool in tools}


@pytest.mark.parametrize(
    "name", ["_search", "deepseek_web_search", "deepseek_search_and_read", "deepseek_fetch"]
)
def test_new_parameters_are_keyword_only(name):
    signature = inspect.signature(getattr(server, name))
    assert signature.parameters["refresh"].kind is inspect.Parameter.KEYWORD_ONLY


def test_search_parameter_order_is_unchanged():
    names = list(inspect.signature(server._search).parameters)
    assert names[:9] == [
        "query",
        "count",
        "ori_question",
        "need_content",
        "lang",
        "max_chars_per_result",
        "max_total_chars",
        "dedupe",
        "rewrite",
    ]


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"query": "  "}, "empty_query"),
        ({"query": "q", "count": 0}, "bad_count"),
        ({"query": "q", "lang": "english"}, "bad_lang"),
    ],
)
def test_search_error_codes(kwargs, code):
    assert asyncio.run(server.deepseek_web_search(**kwargs))["error"]["code"] == code


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("https://x.test/" + "a" * 9000, "url_too_long"),
        ("ftp://x.test/file", "bad_url"),
    ],
)
def test_fetch_error_codes(url, code):
    assert asyncio.run(server.deepseek_fetch(url))["error"]["code"] == code


def test_missing_api_key_error_code():
    result = asyncio.run(server.deepseek_web_search("q"))
    assert result["error"]["code"] == "missing_api_key"


def test_dedupe_note(monkeypatch):
    item = {"title": "T", "url": "https://a.test/x", "site_name": "a.test"}
    _install_fake_request(monkeypatch, [item, dict(item, url="https://a.test/x?utm_source=z")])
    result = asyncio.run(server._search("q", 5, None, False, None, 1000, 4000))
    assert "deduped 1 duplicate result(s)" in result["notes"]


def test_rewrite_note(monkeypatch):
    async def fake_rewrite(query):
        return "better query"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    _install_fake_request(monkeypatch, [])
    result = asyncio.run(server._search("q", 5, None, False, None, 1000, 4000, True, True))
    assert "query rewritten (flash): 'better query'" in result["notes"]
