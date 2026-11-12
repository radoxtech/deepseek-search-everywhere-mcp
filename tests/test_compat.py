"""Backward-compatibility contract: positional signatures, tool names, error codes and notes."""

import asyncio
import inspect
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

TOOL_NAMES = {
    "deepseek_search",
    "deepseek_search_many",
    "deepseek_fetch",
    "deepseek_fetch_many",
    "deepseek_research",
    "deepseek_extract",
    "deepseek_extract_many",
    "deepseek_check_balance",
}


def _install_fake_request(fake_upstream: Any, results: list[dict[str, Any]]) -> list[dict]:
    return fake_upstream(default={"ok": True, "data": {"results": results}}).calls


def test_search_accepts_nine_positional_arguments(fake_upstream):
    calls = _install_fake_request(fake_upstream, [])
    result = asyncio.run(server._search("q", 5, None, False, None, 1000, 4000, True, False))
    assert result["ok"] is True
    assert len(calls) == 1


def test_tool_names_are_stable():
    tools = asyncio.run(server.mcp.list_tools())
    assert {tool.name for tool in tools} == TOOL_NAMES


@pytest.mark.parametrize("tool_name", sorted(TOOL_NAMES))
def test_all_tool_parameters_after_primary_are_keyword_only(tool_name: str) -> None:
    tool_func = getattr(server, tool_name)
    params = list(inspect.signature(tool_func).parameters.values())
    if tool_name == "deepseek_check_balance":
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)
    else:
        assert params[0].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
        assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params[1:])


def test_ori_question_dropped_from_public_search() -> None:
    assert "ori_question" not in inspect.signature(server.deepseek_search).parameters


@pytest.mark.parametrize(
    "name",
    [
        "_search",
        "deepseek_search",
        "deepseek_search_many",
        "deepseek_fetch",
        "deepseek_research",
        "deepseek_extract",
        "deepseek_extract_many",
    ],
)
def test_new_parameters_are_keyword_only(name):
    signature = inspect.signature(getattr(server, name))
    assert signature.parameters["refresh"].kind is inspect.Parameter.KEYWORD_ONLY


def test_check_balance_has_no_parameters() -> None:
    signature = inspect.signature(server.deepseek_check_balance)
    assert len(signature.parameters) == 0


def test_search_parameter_order_is_unchanged():
    names = list(inspect.signature(server._search).parameters)
    assert names[:9] == [
        "query",
        "count",
        "ori_question",
        "content",
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
    assert asyncio.run(server.deepseek_search(**kwargs))["error"]["code"] == code


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
    result = asyncio.run(server.deepseek_search("q"))
    assert result["error"]["code"] == "missing_api_key"


def test_dedupe_note(fake_upstream):
    item = {"title": "T", "url": "https://a.test/x", "site_name": "a.test"}
    _install_fake_request(fake_upstream, [item, dict(item, url="https://a.test/x?utm_source=z")])
    result = asyncio.run(server._search("q", 5, None, "snippets", None, 1000, 4000))
    assert "deduped 1 duplicate result(s)" in result["notes"]


def test_rewrite_note(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return "better query"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    _install_fake_request(fake_upstream, [])
    result = asyncio.run(server._search("q", 5, None, "snippets", None, 1000, 4000, True, True))
    assert "query rewritten (flash): 'better query'" in result["notes"]


def test_truncation_note_is_shared(fake_upstream):
    item = {"title": "T", "url": "https://a.test/x", "site_name": "a.test", "content": "y" * 500}
    _install_fake_request(fake_upstream, [item])
    result = asyncio.run(server._search("q", 5, None, "full", None, 100, 4000))
    expected = "Some page content was truncated to respect the character budget."
    assert server.TRUNCATED_NOTE == expected
    assert expected in result["notes"]


def test_search_tool_descriptions_are_distinct_and_budgeted() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}

    first_lines = [tool.description.splitlines()[0] for tool in tools.values()]
    assert len(set(first_lines)) == len(tools)

    search_desc = tools["deepseek_search"].description
    assert "primary search and retrieval tool" in search_desc.lower()

    total_chars = sum(len(tool.description) for tool in tools.values())
    assert total_chars < 7500


def test_tool_descriptions_sibling_references_are_valid() -> None:
    import re

    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    for name, tool in tools.items():
        refs = set(re.findall(r"deepseek_[a-z0-9_]+", tool.description))
        sibling_refs = refs - {name}
        assert sibling_refs, f"Tool '{name}' must reference sibling tools for disambiguation."
        invalid = sibling_refs - TOOL_NAMES
        assert not invalid, f"Tool '{name}' references invalid sibling tools: {invalid}"


def test_tool_annotations_are_truthful() -> None:
    annotations = server.TOOL_ANNOTATIONS
    assert set(annotations.keys()) == TOOL_NAMES
    for name, ann in annotations.items():
        assert ann.readOnlyHint is True
        assert ann.destructiveHint is False
        assert ann.idempotentHint is True
        if name == "deepseek_check_balance":
            assert ann.openWorldHint is False
        else:
            assert ann.openWorldHint is True
