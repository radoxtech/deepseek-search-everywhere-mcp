"""Integration tests for deepseek_extract tool."""

from __future__ import annotations

import json
from typing import Any

from deepseek_search_everywhere_mcp import server


class ExtractUpstream:
    def __init__(self, page_content: str = "", chat_response: str = "") -> None:
        self.page_content = page_content
        self.chat_response = chat_response
        self.fetch_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.FETCH_PATH:
            self.fetch_calls.append(payload)
            return {
                "ok": True,
                "data": {
                    "url": payload.get("url"),
                    "title": "Test Page Title",
                    "content": self.page_content,
                },
            }
        if path == "/v1/chat/completions":
            self.chat_calls.append(payload)
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self.chat_response,
                            }
                        }
                    ]
                },
            }
        return {"ok": False, "error": {"code": "not_found", "message": f"Unexpected path {path}"}}


async def test_extract_validates_parameters(fake_upstream: Any) -> None:
    fake_upstream(ExtractUpstream())

    # Neither fields nor schema
    res = await server.deepseek_extract("https://example.com")
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"
    assert "Either 'fields' or 'schema'" in res["error"]["message"]

    # Both fields and schema
    res = await server.deepseek_extract("https://example.com", fields=["a"], schema={"type": "object"})
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"
    assert "not both" in res["error"]["message"]

    # Invalid fields
    res = await server.deepseek_extract("https://example.com", fields=[])
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"

    # Invalid schema
    res = await server.deepseek_extract("https://example.com", schema="not a dict")  # type: ignore[arg-type]
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"

    # Bad URL
    res = await server.deepseek_extract("ftp://example.com", fields=["item"])
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_url"


async def test_extract_empty_page_fails_fast(fake_upstream: Any) -> None:
    upstream = ExtractUpstream(page_content="")
    fake_upstream(upstream)

    res = await server.deepseek_extract("https://example.com/empty", fields=["price"])
    assert res["ok"] is False
    assert res["error"]["code"] == "empty_page"
    assert len(upstream.fetch_calls) == 1
    assert len(upstream.chat_calls) == 0  # No LLM calls wasted


async def test_extract_fields_success_with_fences(fake_upstream: Any) -> None:
    model_output = '```json\n{"product": "Pro Widget", "price": 49.99}\n```'
    upstream = ExtractUpstream(page_content="Our Pro Widget costs $49.99.", chat_response=model_output)
    fake_upstream(upstream)

    res = await server.deepseek_extract(
        "https://example.com/product",
        fields=["product", "price", "stock_count"],
        instruction="Extract product details",
    )
    assert res["ok"] is True
    assert res["title"] == "Test Page Title"
    assert res["data"] == {
        "product": "Pro Widget",
        "price": 49.99,
        "stock_count": None,  # Missing field auto-populated
    }
    assert len(upstream.fetch_calls) == 1
    assert len(upstream.chat_calls) == 1

    chat_payload = upstream.chat_calls[0]
    assert chat_payload["model"] == "deepseek-flash"
    assert chat_payload["thinking"] == {"type": "disabled"}
    assert "Additional guidance: Extract product details" in chat_payload["messages"][0]["content"]


async def test_extract_schema_validation_warnings(fake_upstream: Any) -> None:
    model_output = json.dumps({"name": "Gadget", "rating": "four"})
    upstream = ExtractUpstream(page_content="Gadget with rating four.", chat_response=model_output)
    fake_upstream(upstream)

    schema = {
        "required": ["name", "price"],
        "properties": {
            "name": {"type": "string"},
            "rating": {"type": "number"},
        },
    }

    res = await server.deepseek_extract("https://example.com/item", schema=schema)
    assert res["ok"] is True
    assert res["data"] == {"name": "Gadget", "rating": "four"}
    # Notes should contain warnings for missing required 'price' and wrong type for 'rating'
    assert any("Missing required property: 'price'" in note for note in res["notes"])
    assert any("expected number, got str" in note for note in res["notes"])


async def test_extract_chat_failure(fake_upstream: Any) -> None:
    upstream = ExtractUpstream(page_content="Hello world", chat_response="Not json")
    fake_upstream(upstream)

    res = await server.deepseek_extract("https://example.com", fields=["item"])
    assert res["ok"] is False
    assert res["error"]["code"] == "extraction_failed"
