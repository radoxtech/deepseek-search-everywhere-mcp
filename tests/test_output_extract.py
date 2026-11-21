"""Output-schema tests for ``deepseek_extract``: every emitted shape fits ``ExtractOutput``.

The tool publishes one annotation for three shapes (single URL, URL list, error), so every shape has to
validate against the same model: ``ExtractOutput.model_validate(result)`` succeeds and no emitted key
falls outside ``ExtractOutput.model_fields`` (the ``_meta`` alias is normalised to ``meta``).
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.errors import _fail
from deepseek_search_everywhere_mcp.output_models.extract import ExtractOutput

MODEL_FIELDS = set(ExtractOutput.model_fields)


def _assert_extract_output(result: dict[str, Any]) -> None:
    """Assert ``result`` is a subset of the model fields and validates against ``ExtractOutput``.

    ``_meta`` is the published alias of the inherited ``meta`` attribute, so it is mapped back to
    ``meta`` before the subset check; ``model_validate`` resolves the alias itself.
    """
    keys = {"meta" if key == "_meta" else key for key in result}
    assert keys <= MODEL_FIELDS
    assert ExtractOutput.model_validate(result)


class ExtractUpstream:
    """Fake upstream answering both the fetch and the chat-completions path."""

    def __init__(self, page_content: str = "", chat_response: str = "") -> None:
        self.page_content = page_content
        self.chat_response = chat_response

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.FETCH_PATH:
            return {
                "ok": True,
                "data": {
                    "url": payload.get("url"),
                    "title": "Test Page Title",
                    "content": self.page_content,
                },
            }
        if path == "/v1/chat/completions":
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": self.chat_response},
                        }
                    ]
                },
            }
        return _fail("not_found", f"Unexpected path {path}")


def test_extract_output_schema_has_properties_and_no_open_top_level_object() -> None:
    schema = ExtractOutput.model_json_schema()
    properties = schema["properties"]
    assert "ok" in properties
    assert "_meta" in properties
    assert "meta" not in properties
    assert "additionalProperties" not in schema
    assert {"url", "data", "evidence", "results", "cost_breakdown", "elapsed_ms"} <= set(properties)


def test_extract_output_every_field_has_a_description() -> None:
    for name, field in ExtractOutput.model_fields.items():
        assert field.description, f"ExtractOutput.{name} has no description"


def test_extract_output_single_success(fake_upstream: Any) -> None:
    model_output = '```json\n{"product": "Pro Widget", "price": 49.99}\n```'
    fake_upstream(ExtractUpstream(page_content="Our Pro Widget costs $49.99.", chat_response=model_output))

    result = asyncio.run(server.deepseek_extract("https://example.com/product", fields=["product", "price"]))

    assert result["ok"] is True
    assert result["data"] == {"product": "Pro Widget", "price": 49.99}
    assert result["url"] == "https://example.com/product"
    _assert_extract_output(result)


def test_extract_output_list_with_one_failure(fake_upstream: Any) -> None:
    class MixedUpstream:
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs.get("payload") or {}
            if path == server.FETCH_PATH:
                url = payload.get("url") or ""
                if "empty" in url:
                    return {"ok": True, "data": {"url": url, "title": "Empty", "content": ""}}
                return {"ok": True, "data": {"url": url, "title": "Valid", "content": "Product info."}}
            if path == "/v1/chat/completions":
                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": '{"name": "Widget"}'}}]
                    },
                }
            return _fail("not_found", "unexpected")

    fake_upstream(MixedUpstream())
    urls = ["https://example.com/valid", "https://example.com/empty"]
    result = asyncio.run(server.deepseek_extract(urls, fields=["name"]))

    assert result["ok"] is True
    assert result["partial"] is True
    assert result["failed_count"] == 1
    assert result["results"][0]["ok"] is True
    assert result["results"][1]["ok"] is False
    _assert_extract_output(result)


def test_extract_output_top_level_error(fake_upstream: Any) -> None:
    fake_upstream(ExtractUpstream())
    result = asyncio.run(server.deepseek_extract("https://example.com"))

    assert result["ok"] is False
    assert result["error"]["reason"] == "bad_request"
    _assert_extract_output(result)


def test_extract_output_bare_error_item(fake_upstream: Any) -> None:
    """A single-URL extraction failure returns the bare ``{ok: false, error}`` shape."""
    fake_upstream(ExtractUpstream(page_content="Hello world", chat_response="Not json"))
    result = asyncio.run(server.deepseek_extract("https://example.com", fields=["item"]))

    assert result["ok"] is False
    assert result["error"]["reason"] == "extraction_failed"
    assert set(result) - {"elapsed_ms"} == {"ok", "error"}
    _assert_extract_output(result)
