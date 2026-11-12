"""Unit tests for structured data extraction helpers."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.extract import (
    build_extract_prompt,
    normalize_for_evidence,
    sanitize_json,
    validate_fields,
    validate_schema,
    verify_evidence,
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


def test_normalize_for_evidence_full_pipeline():
    raw = "  “DeepSeek—V3” released on 2024–12–26 with `100%` **speed**! It’s «great»   "
    normalized = normalize_for_evidence(raw)
    assert normalized == '"deepseek-v3" released on 2024-12-26 with 100% speed! it\'s "great"'


def test_verify_evidence_scalar_found_with_formatting():
    page = "Quarterly profit surged by 1,250,000 USD, reaching 42.5% YoY growth."
    data = {"profit": 1250000, "growth": 42.5}
    evidence = {
        "profit": "surged by 1,250,000 USD",
        "growth": "reaching 42.5% YoY growth",
    }
    v_data, v_ev, notes = verify_evidence(data, evidence, page)
    assert v_data == data
    assert v_ev == evidence
    assert len(notes) == 0


def test_verify_evidence_scalar_not_in_quote_nullified():
    page = "Quarterly profit reached 500 USD."
    data = {"profit": 999}
    evidence = {"profit": "profit reached 500 USD"}
    v_data, v_ev, notes = verify_evidence(data, evidence, page)
    assert v_data["profit"] is None
    assert "profit" not in v_ev
    assert any("extracted value not found in evidence quote" in n for n in notes)


def test_verify_evidence_quote_not_in_page_nullified():
    page = "Only standard features are present."
    data = {"feature": "advanced"}
    evidence = {"feature": "advanced enterprise mode"}
    v_data, v_ev, notes = verify_evidence(data, evidence, page)
    assert v_data["feature"] is None
    assert "feature" not in v_ev
    assert any("evidence quote not found in page content" in n for n in notes)
