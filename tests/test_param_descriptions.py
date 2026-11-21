"""Parameter descriptions must state the limits and defaults that the code enforces."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from deepseek_search_everywhere_mcp import server, snippets
from deepseek_search_everywhere_mcp.settings import get_settings


def _schema_description(tool_name: str, param: str) -> str:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    return tools[tool_name].input_schema["properties"][param]["description"]


def _stated(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    assert match is not None, f"no stated limit matching {pattern!r} in {text!r}"
    return int(match.group(1))


def test_search_count_description_matches_max_count() -> None:
    for tool in ("deepseek_search",):
        desc = _schema_description(tool, "count")
        assert _stated(r"\(1-(\d+), default \d+\)", desc) == server.MAX_COUNT


def test_search_count_limit_is_enforced_at_the_stated_maximum() -> None:
    res = asyncio.run(server.deepseek_search("python", count=server.MAX_COUNT + 1))
    assert res["ok"] is False and res["error"]["reason"] == "bad_count"


def test_search_query_list_description_matches_validation() -> None:
    desc = _schema_description("deepseek_search", "query")
    stated = _stated(r"a list of up to (\d+)", desc)
    assert stated == server.MAX_QUERIES
    too_many = [f"query number {i}" for i in range(stated + 1)]
    res = asyncio.run(server.deepseek_search(too_many))
    assert res["count"] == stated
    assert res["skipped_queries"] == too_many[stated:]


def test_fetch_snippets_clamp() -> None:
    doc = "".join(f"## Section {i}\n\nquantum notes for section {i}.\n\n" for i in range(12))
    _, count = snippets.extract_snippets(doc, "quantum", max_snippets=50)
    assert count == snippets.MAX_SNIPPETS


def test_fetch_url_list_limit_matches_settings() -> None:
    desc = _schema_description("deepseek_fetch", "url")
    assert _stated(r"up to (\d+) URLs", desc) == get_settings().batch_max_urls


def test_removed_shaping_flags_are_not_tool_parameters() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    pruned = {
        "clean",
        "strip_boilerplate",
        "fallback_to_search",
        "limit",
        "max_total_chars",
        "max_snippets",
        "section",
        "offset",
        "find_text",
        "include_domains",
        "exclude_domains",
        "exclude_terms",
        "exact_phrases",
        "after",
        "before",
        "rewrite",
        "deadline",
        "unverified",
    }
    for tool in tools:
        params = set(tool.input_schema.get("properties", {}))
        assert not params & pruned, (tool.name, params & pruned)


def test_every_described_default_matches_the_signature() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    checks: list[tuple[str, str, Any]] = [
        ("deepseek_search", "count", 5),
        ("deepseek_search", "content", "snippets"),
        ("deepseek_quick_research", "depth", "medium"),
        ("deepseek_quick_research", "sources", "any"),
        ("deepseek_deep_agent_research", "depth", "medium"),
        ("deepseek_deep_agent_research", "sources", "any"),
    ]
    for tool_name, param, default in checks:
        prop = tools[tool_name].input_schema["properties"][param]
        assert prop.get("default") == default, (tool_name, param)
    # max_chars defaults differ for one URL and a list, so the schema default is None.
    prop = tools["deepseek_fetch"].input_schema["properties"]["max_chars"]
    assert prop.get("default") is None
    assert "default 20000" in prop["description"] and "4000 per URL" in prop["description"]
