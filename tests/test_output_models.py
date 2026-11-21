"""Tests for the shared output models: schema shape, required fields and the ``_meta`` alias."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepseek_search_everywhere_mcp.output_models import Envelope, ErrorInfo, Meta


def test_envelope_schema_has_properties_and_no_open_object() -> None:
    schema = Envelope.model_json_schema()
    properties = schema["properties"]
    assert set(properties) == {"ok", "_meta", "error"}
    assert schema.get("additionalProperties") is not True


def test_envelope_schema_uses_meta_alias_not_attribute_name() -> None:
    properties = Envelope.model_json_schema()["properties"]
    assert "_meta" in properties
    assert "meta" not in properties


def test_meta_requires_both_cost_fields() -> None:
    assert Meta.model_json_schema()["required"] == ["cost_usd", "untrusted_content"]
    with pytest.raises(ValidationError):
        Meta()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Meta(cost_usd=0.001)  # type: ignore[call-arg]


def test_error_info_minimal_has_no_action_or_reason() -> None:
    error = ErrorInfo(code="rate_limited_local", message="Too many requests; retry shortly.")
    assert error.action is None
    assert error.reason is None
    assert error.model_dump() == {
        "code": "rate_limited_local",
        "message": "Too many requests; retry shortly.",
        "action": None,
        "reason": None,
        "param": None,
        "status": None,
        "upstream_code": None,
        "upstream_message": None,
    }


def test_envelope_accepts_and_drops_extra_keys() -> None:
    envelope = Envelope(ok=True, extra_key=1)  # type: ignore[call-arg]
    assert envelope.ok is True
    assert "extra_key" not in envelope.model_dump()


def test_envelope_publishes_meta_alias() -> None:
    envelope = Envelope(ok=True, meta=Meta(cost_usd=0.001, untrusted_content=True))
    dumped = envelope.model_dump(by_alias=True)
    assert dumped["_meta"] == {"cost_usd": 0.001, "untrusted_content": True}
    assert "meta" not in dumped


@pytest.mark.parametrize("model", [Envelope, ErrorInfo, Meta])
def test_every_field_has_a_description(model: type[Envelope] | type[ErrorInfo] | type[Meta]) -> None:
    for name, field in model.model_fields.items():
        assert field.description, f"{model.__name__}.{name} has no description"
