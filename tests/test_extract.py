"""Unit tests for structured data extraction helpers."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.extract import (
    build_extract_prompt,
    sanitize_json,
    validate_fields,
    validate_schema,
)


def test_build_extract_prompt_with_fields():
    prompt = build_extract_prompt(["price", "title"], None, "Focus on monthly plans")
    assert '"price", "title"' in prompt
    assert "Additional guidance: Focus on monthly plans" in prompt


def test_build_extract_prompt_with_schema():
    schema = {"properties": {"price": {"type": "number"}}, "required": ["price"]}
    prompt = build_extract_prompt(None, schema, None)
    assert '"price"' in prompt
    assert '"required"' in prompt


def test_sanitize_json_clean():
    parsed, err = sanitize_json('{"price": 99.99, "currency": "USD"}')
    assert err is None
    assert parsed == {"price": 99.99, "currency": "USD"}


def test_sanitize_json_with_code_fences_and_preamble():
    raw = 'Here is the extracted data:\n```json\n{"status": "active"}\n```\nHope this helps!'
    parsed, err = sanitize_json(raw)
    assert err is None
    assert parsed == {"status": "active"}


def test_sanitize_json_invalid():
    parsed, err = sanitize_json("No json here at all")
    assert parsed is None
    assert "No JSON object found" in str(err)


def test_validate_fields_populates_missing():
    data = {"price": 10}
    validated = validate_fields(data, ["price", "discount"])
    assert validated == {"price": 10, "discount": None}


def test_validate_schema_checks_required_and_types():
    schema = {
        "required": ["name", "age", "active"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
        },
    }
    # Missing 'active', wrong type for 'name', boolean for 'age'
    errors = validate_schema({"name": 123, "age": True}, schema)
    assert len(errors) == 3
    assert any("Missing required property: 'active'" in e for e in errors)
    assert any("expected string" in e for e in errors)
    assert any("expected integer, got boolean" in e for e in errors)
