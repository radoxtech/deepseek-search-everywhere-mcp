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
    for tool in ("deepseek_search", "deepseek_search_many"):
        desc = _schema_description(tool, "count")
        assert _stated(r"\(1-(\d+), default \d+\)", desc) == server.MAX_COUNT


def test_search_count_limit_is_enforced_at_the_stated_maximum() -> None:
    res = asyncio.run(server.deepseek_search("python", count=server.MAX_COUNT + 1))
    assert res["ok"] is False and res["error"]["code"] == "bad_count"


def test_search_many_queries_description_matches_validation() -> None:
    desc = _schema_description("deepseek_search_many", "queries")
    stated = _stated(r"\(1-(\d+) queries", desc)
    assert stated == server.MAX_QUERIES
    too_many = [f"query number {i}" for i in range(stated + 1)]
    res = asyncio.run(server.deepseek_search_many(too_many))
    assert res["ok"] is False and res["error"]["code"] == "bad_request"
    assert str(stated) in res["error"]["message"]


def test_fetch_max_snippets_description_matches_clamp() -> None:
    desc = _schema_description("deepseek_fetch", "max_snippets")
    assert _stated(r"\(1-(\d+), default 3\)", desc) == snippets.MAX_SNIPPETS
    doc = "".join(f"## Section {i}\n\nquantum notes for section {i}.\n\n" for i in range(12))
    _, count = snippets.extract_snippets(doc, "quantum", max_snippets=50)
    assert count == snippets.MAX_SNIPPETS


def test_fetch_many_url_limit_matches_settings() -> None:
    desc = _schema_description("deepseek_fetch_many", "urls")
    assert _stated(r"max (\d+) URLs", desc) == get_settings().batch_max_urls


def test_clean_descriptions_describe_markdown_unescaping() -> None:
    tools_params: list[tuple[str, str]] = [
        ("deepseek_fetch", "clean"),
        ("deepseek_fetch_many", "clean"),
        ("deepseek_extract", "clean"),
        ("deepseek_extract_many", "clean"),
    ]
    for tool, param in tools_params:
        desc = _schema_description(tool, param)
        assert "structured markdown" not in desc, (tool, desc)
        assert "backslash" in desc.lower(), (tool, desc)


def test_every_described_default_matches_the_signature() -> None:
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    checks: list[tuple[str, str, Any]] = [
        ("deepseek_search", "max_total_chars", 30000),
        ("deepseek_search_many", "max_total_chars", 30000),
        ("deepseek_fetch", "max_chars", 20000),
        ("deepseek_fetch", "max_snippets", 3),
        ("deepseek_fetch_many", "max_chars_per_url", 4000),
        ("deepseek_fetch_many", "max_total_chars", 20000),
        ("deepseek_extract", "max_chars", 40000),
        ("deepseek_extract_many", "max_chars_per_url", 20000),
        ("deepseek_extract_many", "max_total_chars", 60000),
    ]
    for tool_name, param, default in checks:
        prop = tools[tool_name].inputSchema["properties"][param]
        assert prop.get("default") == default, (tool_name, param)
        assert f"default {default}" in prop["description"], (tool_name, param, prop["description"])
