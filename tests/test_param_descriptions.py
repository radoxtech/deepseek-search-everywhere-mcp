"""Parameter descriptions must state the limits and defaults that the code enforces."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from deepseek_search_everywhere_mcp import server, snippets
from deepseek_search_everywhere_mcp.settings import get_settings


def _schema_description(tool_name: str, param: str) -> str:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    return tools[tool_name].inputSchema["properties"][param]["description"]


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


def test_fetch_max_snippets_description_matches_clamp() -> None:
    desc = _schema_description("deepseek_fetch", "max_snippets")
    assert _stated(r"\(1-(\d+), default 3\)", desc) == snippets.MAX_SNIPPETS
    doc = "".join(f"## Section {i}\n\nquantum notes for section {i}.\n\n" for i in range(12))
    _, count = snippets.extract_snippets(doc, "quantum", max_snippets=50)
    assert count == snippets.MAX_SNIPPETS


def test_fetch_url_list_limit_matches_settings() -> None:
    desc = _schema_description("deepseek_fetch", "url")
    assert _stated(r"up to (\d+) URLs", desc) == get_settings().batch_max_urls


def test_removed_shaping_flags_are_not_tool_parameters() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    for tool in tools:
        params = set(tool.inputSchema.get("properties", {}))
        assert not params & {"clean", "strip_boilerplate", "fallback_to_search", "limit"}, tool.name


def test_every_described_default_matches_the_signature() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    checks: list[tuple[str, str, Any]] = [
        ("deepseek_search", "max_total_chars", 20000),
        ("deepseek_fetch", "max_snippets", 3),
        ("deepseek_fetch", "max_total_chars", 20000),
        ("deepseek_extract", "max_total_chars", 60000),
    ]
    for tool_name, param, default in checks:
        prop = tools[tool_name].inputSchema["properties"][param]
        assert prop.get("default") == default, (tool_name, param)
        assert f"default {default}" in prop["description"], (tool_name, param, prop["description"])
    # max_chars defaults differ for one URL and a list, so the schema default is None.
    for tool_name, single, per_url in (("deepseek_fetch", 20000, 4000), ("deepseek_extract", 40000, 20000)):
        prop = tools[tool_name].inputSchema["properties"]["max_chars"]
        assert prop.get("default") is None, tool_name
        assert f"default {single}" in prop["description"] and f"{per_url} per URL" in prop["description"]
