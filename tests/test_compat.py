"""Backward-compatibility contract: positional signatures, tool names, error codes and notes."""

import asyncio
import inspect
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import search_core, server

TOOL_NAMES = {
    "deepseek_search",
    "deepseek_deep_agent_research",
    "deepseek_fetch",
    "deepseek_quick_research",
    "deepseek_check_research",
    "deepseek_extract",
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


def test_refresh_is_a_keyword_only_parameter_of_internal_search():
    names = ["_search", "_fetch"]
    for name in names:
        refresh = inspect.signature(getattr(server, name)).parameters.get("refresh")
        assert refresh is not None and refresh.kind is inspect.Parameter.KEYWORD_ONLY, name


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
    assert asyncio.run(server.deepseek_search(**kwargs))["error"]["reason"] == code


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("https://x.test/" + "a" * 9000, "url_too_long"),
        ("ftp://x.test/file", "bad_url"),
    ],
)
def test_fetch_error_codes(url, code):
    assert asyncio.run(server.deepseek_fetch(url))["error"]["reason"] == code


def test_missing_api_key_error_code():
    result = asyncio.run(server.deepseek_search("q"))
    assert result["error"]["reason"] == "missing_api_key"


def test_dedupe_note(fake_upstream):
    item = {"title": "T", "url": "https://a.test/x", "site_name": "a.test"}
    _install_fake_request(fake_upstream, [item, dict(item, url="https://a.test/x?utm_source=z")])
    result = asyncio.run(server._search("q", 5, None, "snippets", None, 1000, 4000))
    assert "deduped 1 duplicate result(s)" in result["notes"]


def test_rewrite_note(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return "better query"

    monkeypatch.setattr(search_core, "_rewrite_query", fake_rewrite)
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
    # The cap bounds the tools/list context cost; it is not a TDQS target.
    assert total_chars < 8500


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
        assert ann.read_only_hint is True
        assert ann.destructive_hint is False
        assert ann.idempotent_hint is True
        if name == "deepseek_check_balance":
            assert ann.open_world_hint is False
        else:
            assert ann.open_world_hint is True


def test_every_tool_input_schema_property_has_description() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    for tool in tools:
        props = tool.input_schema.get("properties", {})
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
    assert "deepseek_quick_research" in server.SERVER_INSTRUCTIONS
    assert "deepseek_deep_agent_research" in server.SERVER_INSTRUCTIONS
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
    desc_search = tools["deepseek_search"].description
    assert "20,000" in desc_search or "20000" in desc_search
    desc_fetch = tools["deepseek_fetch"].description
    assert "20,000" in desc_fetch or "20000" in desc_fetch
    assert "4,000" in desc_fetch
    desc_research = tools["deepseek_quick_research"].description
    assert "6-24 sources" in desc_research or "sources" in desc_research


def test_every_tool_has_a_title() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    titles = [tool.title for tool in tools]
    annotation_titles = [tool.annotations.title for tool in tools]
    assert all(titles), "Every tool must expose a non-empty top-level title."
    assert all(annotation_titles), "Every tool must expose a non-empty annotations.title."
    assert len(set(titles)) == len(titles), "Top-level titles must be distinct across tools."
    assert len(set(annotation_titles)) == len(annotation_titles), (
        "Annotation titles must be distinct across tools."
    )
    for tool in tools:
        assert tool.title != tool.name, f"Tool '{tool.name}' title must differ from its name."
        assert tool.annotations.title != tool.name, (
            f"Tool '{tool.name}' annotation title must differ from its name."
        )
        assert tool.title == tool.annotations.title, (
            f"Tool '{tool.name}' top-level title must match its annotation title."
        )


def test_research_description_explains_polling() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    research = tools["deepseek_quick_research"].description
    assert "deepseek_check_research" in research and '"running"' in research
    check = tools["deepseek_check_research"].description
    assert "job_id" in check and "poll" in check.lower()


def test_error_codes_named_in_server_instructions_exist() -> None:
    import re

    from deepseek_search_everywhere_mcp.errors import CLOSED_ERROR_CODES, CODE_ACTIONS

    named = set(re.findall(r"\b[a-z]+(?:_[a-z]+)+\b", server.SERVER_INSTRUCTIONS))
    codes = {word for word in named if not word.startswith("deepseek_")}
    codes -= {"job_id", "cost_usd", "error_code", "synthesis_instructions"}
    codes -= set(CODE_ACTIONS.values())
    assert codes <= CLOSED_ERROR_CODES, codes - CLOSED_ERROR_CODES


@pytest.mark.parametrize(
    ("tool", "single", "per_url"), [("deepseek_fetch", 20000, 4000), ("deepseek_extract", 40000, 20000)]
)
def test_dual_max_chars_defaults_are_in_the_description(tool: str, single: int, per_url: int) -> None:
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    description = tools[tool].description
    assert f"{single:,}" in description and f"{per_url:,}" in description


def test_every_tool_description_fits_the_length_limit() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    for tool in tools:
        # The cap bounds the tools/list context cost; it is not a TDQS target.
        assert len(tool.description) <= 1400, (tool.name, len(tool.description))
