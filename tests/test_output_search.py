"""Published output schema and key coverage for ``deepseek_search``."""

from __future__ import annotations

import asyncio
from typing import Any

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.output_models.search import SearchOutput
from deepseek_search_everywhere_mcp.server import create_server


def _search_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"ok": True, "data": {"results": results}}


def _published_output_schema() -> dict[str, Any]:
    tools = {tool.name: tool for tool in asyncio.run(create_server().list_tools())}
    schema = tools["deepseek_search"].output_schema
    assert schema is not None, "deepseek_search publishes no outputSchema"
    return schema


def _allowed_result_keys() -> set[str]:
    """Every key a returned dict may carry: model field names plus their published aliases."""
    return set(SearchOutput.model_fields) | {
        field.alias for field in SearchOutput.model_fields.values() if field.alias
    }


def test_published_output_schema_is_a_closed_object_with_the_search_fields() -> None:
    schema = _published_output_schema()
    properties = schema.get("properties")
    assert isinstance(properties, dict) and properties
    assert "additionalProperties" not in schema
    assert {"_meta", "results", "elapsed_ms", "skipped_queries"} <= set(properties)
    assert schema.get("required") == ["ok"]


def _assert_covered(result: dict[str, Any]) -> None:
    """Every returned key is modelled (so it survives structuredContent) and validates."""
    extra = set(result) - _allowed_result_keys()
    assert not extra, f"unmodelled result keys would be dropped: {extra}"
    SearchOutput.model_validate(result)


def test_single_query_result_round_trips_through_the_model(fake_upstream: Any) -> None:
    fake_upstream(default=_search_results([{"title": "A", "url": "https://a.test", "snippet": "s"}]))
    result = asyncio.run(server.deepseek_search("python", refresh=True))
    assert result["ok"] is True
    _assert_covered(result)


def test_six_query_list_result_round_trips_through_the_model(fake_upstream: Any) -> None:
    fake_upstream(default=_search_results([{"title": "A", "url": "https://a.test", "snippet": "s"}]))
    result = asyncio.run(server.deepseek_search(["q1", "q2", "q3", "q4", "q5", "q6"], refresh=True))
    assert result["ok"] is True
    assert result["skipped_queries"] == ["q6"]
    _assert_covered(result)


def test_error_result_round_trips_through_the_model(fake_upstream: Any) -> None:
    fake_upstream(default=_search_results([]))
    result = asyncio.run(server.deepseek_search("python", content="invalid"))  # type: ignore[arg-type]
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_input"
    _assert_covered(result)
