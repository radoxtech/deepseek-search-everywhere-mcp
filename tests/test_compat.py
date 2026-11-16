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
    "deepseek_check_research",
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
    "dropped_param",
    ["localization", "summary_chars", "max_chars_per_result", "strip_boilerplate", "dedupe"],
)
def test_dropped_parameters_not_in_search_tools(dropped_param: str) -> None:
    assert dropped_param not in inspect.signature(server.deepseek_search).parameters
    assert dropped_param not in inspect.signature(server.deepseek_search_many).parameters


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
        if name in ("deepseek_check_balance", "deepseek_check_research"):
            assert ann.openWorldHint is False
        else:
            assert ann.openWorldHint is True


def test_every_tool_input_schema_property_has_description() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    for tool in tools:
        props = tool.inputSchema.get("properties", {})
        for prop_name, prop in props.items():
            desc = prop.get("description")
            assert desc, f"Tool '{tool.name}' property '{prop_name}' is missing description."
            assert len(desc) <= 120, (
                f"Tool '{tool.name}' property '{prop_name}' description exceeds 120 chars: {len(desc)}"
            )


def test_server_instructions_are_present_and_guide_tool_choice() -> None:
    assert hasattr(server, "SERVER_INSTRUCTIONS")
    assert server.mcp.instructions is not None
    assert server.mcp.instructions == server.SERVER_INSTRUCTIONS
    assert "Default to deepseek_research" in server.SERVER_INSTRUCTIONS
    assert 'sources="academic"' in server.SERVER_INSTRUCTIONS
    assert 'sources="technical"' in server.SERVER_INSTRUCTIONS
    assert "deepseek_check_research" in server.SERVER_INSTRUCTIONS


def test_tool_descriptions_contain_cost_or_size_facts() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    for name, tool in tools.items():
        desc = tool.description
        assert any(token in desc for token in ("$0.", "free", "Free", "$0.00")), (
            f"Tool {name} description must mention cost facts: {desc}"
        )
    assert "128" in tools["deepseek_search"].description
    desc_search_many = tools["deepseek_search_many"].description
    assert "20,000" in desc_search_many or "20000" in desc_search_many
    desc_fetch = tools["deepseek_fetch"].description
    assert "20,000" in desc_fetch or "20000" in desc_fetch
    desc_fetch_many = tools["deepseek_fetch_many"].description
    assert "20,000" in desc_fetch_many and "4,000" in desc_fetch_many
    desc_research = tools["deepseek_research"].description
    assert "6-24 sources" in desc_research or "sources" in desc_research


def test_every_tool_has_a_title() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    titles = [tool.annotations.title for tool in tools]
    assert all(titles) and len(set(titles)) == len(titles)


def test_research_description_explains_polling_and_client_synthesis() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    research = tools["deepseek_research"].description
    assert "deepseek_check_research" in research and '"running"' in research
    assert 'synthesis="client"' in research
    check = tools["deepseek_check_research"].description
    assert "page=" in check and "report=" in check and "awaiting_report" in check
    assert 'synthesis="client"' in server.SERVER_INSTRUCTIONS


@pytest.mark.parametrize(
    ("tool", "param"),
    [
        ("deepseek_search", "max_total_chars"),
        ("deepseek_search_many", "max_total_chars"),
        ("deepseek_fetch", "max_chars"),
        ("deepseek_fetch_many", "max_chars_per_url"),
        ("deepseek_fetch_many", "max_total_chars"),
        ("deepseek_extract", "max_chars"),
        ("deepseek_extract_many", "max_chars_per_url"),
        ("deepseek_extract_many", "max_total_chars"),
    ],
)
def test_size_defaults_in_descriptions_match_the_schema(tool: str, param: str) -> None:
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    default = tools[tool].inputSchema["properties"][param]["default"]
    assert f"{default:,}" in tools[tool].description, f"{tool}.{param} default {default} not in description"
