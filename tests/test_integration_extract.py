"""Integration tests for deepseek_extract tool."""

from __future__ import annotations

import json
from typing import Any

from deepseek_search_everywhere_mcp import server


class ExtractUpstream:
    def __init__(
        self,
        page_content: str = "",
        chat_response: str = "",
        finish_reason: str = "stop",
    ) -> None:
        self.page_content = page_content
        self.chat_response = chat_response
        self.finish_reason = finish_reason
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
                            "finish_reason": self.finish_reason,
                            "message": {
                                "role": "assistant",
                                "content": self.chat_response,
                            },
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


async def test_extract_many_parameter_validation(fake_upstream: Any) -> None:
    fake_upstream(ExtractUpstream())

    # Empty urls
    res = await server.deepseek_extract_many([], fields=["item"])
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"

    # Too many urls (> 10)
    res = await server.deepseek_extract_many([f"https://example.com/{i}" for i in range(11)], fields=["item"])
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"

    # Neither fields nor schema
    res = await server.deepseek_extract_many(["https://example.com/1"])
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"

    # Both fields and schema
    res = await server.deepseek_extract_many(
        ["https://example.com/1"], fields=["item"], schema={"type": "object"}
    )
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"


async def test_extract_many_success_with_fields(fake_upstream: Any) -> None:
    model_output = json.dumps({"product": "Test Item", "price": 10.0})
    upstream = ExtractUpstream(page_content="Test Item is $10.", chat_response=model_output)
    fake_upstream(upstream)

    urls = [
        "https://example.com/p1",
        "https://example.com/p2",
        "https://example.com/p1",  # Duplicate URL
    ]
    res = await server.deepseek_extract_many(
        urls,
        fields=["product", "price"],
        instruction="Extract product details",
    )
    assert res["ok"] is True
    assert res["count"] == 2
    assert res["partial"] is False
    assert res["failed_count"] == 0
    assert any("deduped 1 duplicate URL(s)" in note for note in res["notes"])

    for item in res["results"]:
        assert item["ok"] is True
        assert item["data"] == {"product": "Test Item", "price": 10.0}
        assert item["title"] == "Test Page Title"

    assert len(upstream.fetch_calls) == 2
    assert len(upstream.chat_calls) == 2
    assert res["requests_used"] == 4
    assert "cost_breakdown" in res
    assert res["cost_breakdown"]["fetch"] == round(2 * server.COST_PER_REQUEST, 6)
    assert res["cost_breakdown"]["llm"] > 0.0
    assert res["cost_usd"] == round(res["cost_breakdown"]["fetch"] + res["cost_breakdown"]["llm"], 6)


async def test_extract_many_partial_failure(fake_upstream: Any) -> None:
    class MixedUpstream:
        def __init__(self) -> None:
            self.fetch_calls: list[dict[str, Any]] = []
            self.chat_calls: list[dict[str, Any]] = []

        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs.get("payload") or {}
            if path == server.FETCH_PATH:
                self.fetch_calls.append(payload)
                url = payload.get("url") or ""
                if "empty" in url:
                    return {"ok": True, "data": {"url": url, "title": "Empty", "content": ""}}
                return {"ok": True, "data": {"url": url, "title": "Valid", "content": "Product info."}}
            if path == "/v1/chat/completions":
                self.chat_calls.append(payload)
                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": '{"name": "Widget"}'}}]
                    },
                }
            return {"ok": False, "error": {"code": "not_found", "message": "unexpected"}}

    upstream = MixedUpstream()
    fake_upstream(upstream)

    urls = [
        "https://example.com/valid",
        "http://127.0.0.1/admin",  # SSRF blocked locally
        "https://example.com/empty",  # Empty page
    ]
    res = await server.deepseek_extract_many(urls, fields=["name"])
    assert res["ok"] is True
    assert res["count"] == 3
    assert res["partial"] is True
    assert res["failed_count"] == 2

    valid_item = res["results"][0]
    assert valid_item["ok"] is True
    assert valid_item["data"] == {"name": "Widget"}

    blocked_item = res["results"][1]
    assert blocked_item["ok"] is False
    assert blocked_item["error"]["code"] == "ssrf_blocked"

    empty_item = res["results"][2]
    assert empty_item["ok"] is False
    assert empty_item["error"]["code"] == "empty_page"

    # Only 2 fetch calls reached upstream (blocked item was rejected locally)
    assert len(upstream.fetch_calls) == 2
    # Only 1 chat call reached upstream (empty page fast-failed before LLM)
    assert len(upstream.chat_calls) == 1
    # 2 upstream fetch calls + 1 upstream chat call = 3
    assert res["requests_used"] == 3


async def test_extract_evidence_verified(fake_upstream: Any) -> None:
    page = "The Enterprise Widget model X9 costs $199 and includes a 3-year warranty."
    model_output = json.dumps(
        {
            "data": {"model": "model X9", "price": "$199", "warranty": "3-year warranty"},
            "evidence": {
                "model": "model X9",
                "price": "costs $199",
                "warranty": "3-year warranty",
            },
        }
    )
    upstream = ExtractUpstream(page_content=page, chat_response=model_output)
    fake_upstream(upstream)

    res = await server.deepseek_extract(
        "https://example.com/item",
        fields=["model", "price", "warranty"],
    )
    assert res["ok"] is True
    assert res["data"] == {
        "model": "model X9",
        "price": "$199",
        "warranty": "3-year warranty",
    }
    assert res["evidence"] == {
        "model": "model X9",
        "price": "costs $199",
        "warranty": "3-year warranty",
    }
    assert res["extraction_truncated"] is False
    assert not any("nullified" in note for note in res["notes"])

    # Verify JSON mode was passed to LLM gateway
    assert len(upstream.chat_calls) == 1
    call = upstream.chat_calls[0]
    assert call.get("response_format") == {"type": "json_object"}


async def test_extract_evidence_unverified_nullified(fake_upstream: Any) -> None:
    page = "Only basic gadgets are listed on this inventory page."
    model_output = json.dumps(
        {
            "data": {"gadget": "basic gadgets", "price": "$499"},
            "evidence": {
                "gadget": "basic gadgets",
                "price": "unheard of cheap price of $499",  # Not in page
            },
        }
    )
    upstream = ExtractUpstream(page_content=page, chat_response=model_output)
    fake_upstream(upstream)

    res = await server.deepseek_extract(
        "https://example.com/gadgets",
        fields=["gadget", "price"],
    )
    assert res["ok"] is True
    assert res["data"]["gadget"] == "basic gadgets"
    assert res["data"]["price"] is None
    assert any(
        "Field 'price' nullified: evidence quote not found in page content." in n for n in res["notes"]
    )


async def test_extract_evidence_missing_nullified(fake_upstream: Any) -> None:
    page = "The server blade is listed at $1200 with rack mount included."
    model_output = json.dumps(
        {
            "data": {"item": "server blade", "price": "$1200"},
            "evidence": {
                "item": "server blade",
                # "price" evidence omitted
            },
        }
    )
    upstream = ExtractUpstream(page_content=page, chat_response=model_output)
    fake_upstream(upstream)

    res = await server.deepseek_extract(
        "https://example.com/blade",
        fields=["item", "price"],
    )
    assert res["ok"] is True
    assert res["data"]["item"] == "server blade"
    assert res["data"]["price"] is None
    assert any("Field 'price' nullified: missing evidence quote." in n for n in res["notes"])


async def test_extract_truncated_invalid_json(fake_upstream: Any) -> None:
    page = "Page with a large amount of content."
    broken_json = '{"data": {"item": "Incomplete text and no closing'
    upstream = ExtractUpstream(
        page_content=page,
        chat_response=broken_json,
        finish_reason="length",
    )
    fake_upstream(upstream)

    res = await server.deepseek_extract("https://example.com/large", fields=["item"])
    assert res["ok"] is False
    assert res["error"]["code"] == "extraction_truncated"
    assert "truncated before valid JSON" in res["error"]["message"]


async def test_extract_truncated_valid_json(fake_upstream: Any) -> None:
    page = "Page with items."
    valid_json = json.dumps(
        {
            "data": {"item": "items"},
            "evidence": {"item": "items"},
        }
    )
    upstream = ExtractUpstream(
        page_content=page,
        chat_response=valid_json,
        finish_reason="length",
    )
    fake_upstream(upstream)

    res = await server.deepseek_extract("https://example.com/items", fields=["item"])
    assert res["ok"] is True
    assert res["data"] == {"item": "items"}
    assert res["extraction_truncated"] is True
    assert any("Extraction output was truncated by token limit." in n for n in res["notes"])


async def test_extract_schema_mode_evidence_verification(fake_upstream: Any) -> None:
    page = "Author: John Doe. Published in 2024. Available in Hardcover."
    schema = {
        "type": "object",
        "properties": {
            "author": {"type": "string"},
            "format": {"type": "string"},
            "price": {"type": "string"},
        },
    }
    model_output = json.dumps(
        {
            "data": {"author": "John Doe", "format": "Hardcover", "price": "$29"},
            "evidence": {
                "author": "Author: John Doe",
                "format": "Available in Hardcover",
                "price": "cost is $29",  # Hallucinated quote
            },
        }
    )
    upstream = ExtractUpstream(page_content=page, chat_response=model_output)
    fake_upstream(upstream)

    res = await server.deepseek_extract("https://example.com/book", schema=schema)
    assert res["ok"] is True
    assert res["data"]["author"] == "John Doe"
    assert res["data"]["format"] == "Hardcover"
    assert res["data"]["price"] is None
    assert any(
        "Field 'price' nullified: evidence quote not found in page content." in n for n in res["notes"]
    )
