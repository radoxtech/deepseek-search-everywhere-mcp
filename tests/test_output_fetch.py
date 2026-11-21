"""Tests for ``FetchOutput``: the schema shape and real envelopes from single, list and error calls."""

from __future__ import annotations

from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.errors import _fail
from deepseek_search_everywhere_mcp.output_models.fetch import FetchOutput

A = "https://example.com/a"
B = "https://example.org/b"
C = "https://example.net/c"


class Upstream:
    """Fake ``server._request`` serving pages and scripted failures per URL."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = kwargs["payload"]["url"]
        if url in self.pages:
            return self.pages[url]
        return {"ok": True, "data": {"title": f"Title {url}", "content": f"Body of {url}"}}


@pytest.fixture
def upstream(fake_upstream: Any) -> Upstream:
    return fake_upstream(Upstream())


# ``model_fields`` is keyed by Python attribute name, while envelopes publish the ``_meta`` alias,
# so the published key set adds every field alias to the field names.
_MODELED_KEYS = set(FetchOutput.model_fields) | {
    field.alias for field in FetchOutput.model_fields.values() if field.alias
}


def _assert_modeled(result: dict[str, Any]) -> None:
    assert set(result) <= _MODELED_KEYS
    assert "_meta" in FetchOutput.model_json_schema()["properties"]
    FetchOutput.model_validate(result)


def test_schema_has_described_properties_and_meta_alias() -> None:
    schema = FetchOutput.model_json_schema()
    properties = schema["properties"]
    assert properties
    assert "_meta" in properties
    assert "meta" not in properties
    assert schema.get("additionalProperties") is not True
    assert schema["required"] == ["ok"]
    for name, prop in properties.items():
        assert prop.get("description"), f"FetchOutput.{name} has no description"


async def test_single_url_envelope_is_modeled(upstream: Upstream) -> None:
    result = await server.deepseek_fetch(A)
    assert result["ok"] is True
    assert result["source"] == "fetch"
    _assert_modeled(result)


async def test_list_with_one_failure_is_modeled(upstream: Upstream) -> None:
    upstream.pages[B] = _fail("not_found", "failed", 404)
    result = await server.deepseek_fetch([A, B, C])
    assert result["partial"] is True
    assert result["failed_count"] == 1
    assert all(isinstance(item, dict) for item in result["results"])
    _assert_modeled(result)


async def test_error_envelope_is_modeled(upstream: Upstream) -> None:
    result = await server.deepseek_fetch("ftp://bad")
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_input"
    _assert_modeled(result)
