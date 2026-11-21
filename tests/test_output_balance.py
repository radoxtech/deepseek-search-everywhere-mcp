"""Schema tests for the ``deepseek_check_balance`` output model (TDQS T12)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.output_models.balance import BalanceOutput


def _schema() -> dict[str, Any]:
    return BalanceOutput.model_json_schema()


def _published_schema() -> dict[str, Any]:
    tools = asyncio.run(server.mcp.list_tools())
    tool = next(tool for tool in tools if tool.name == "deepseek_check_balance")
    assert tool.output_schema is not None
    return tool.output_schema


def test_schema_declares_properties() -> None:
    schema = _schema()
    assert set(schema["properties"]) >= {
        "ok",
        "_meta",
        "error",
        "is_available",
        "balances",
        "session",
        "elapsed_ms",
    }
    assert schema["required"] == ["ok"]


@pytest.mark.parametrize("schema", [_schema(), _published_schema()], ids=["model", "published"])
def test_top_level_schema_is_not_an_open_object(schema: dict[str, Any]) -> None:
    assert schema.get("additionalProperties") is not True
    assert schema["properties"]
    for prop in schema["properties"].values():
        assert prop.get("description"), f"property without description: {prop}"


@pytest.mark.parametrize("schema", [_schema(), _published_schema()], ids=["model", "published"])
def test_schema_has_no_cost_usd(schema: dict[str, Any]) -> None:
    assert "cost_usd" not in schema["properties"]


@pytest.mark.parametrize(
    "balances",
    [
        None,
        [{"currency": "USD", "total_balance": "1.00"}],
        {"currency": "USD", "total_balance": "1.00"},
    ],
    ids=["none", "list", "mapping"],
)
def test_both_balances_shapes_validate(balances: Any) -> None:
    model = BalanceOutput.model_validate({"ok": True, "is_available": True, "balances": balances})
    dumped = model.model_dump(mode="json", by_alias=True)
    assert dumped["balances"] == balances


def test_is_available_stays_a_top_level_boolean() -> None:
    prop = _schema()["properties"]["is_available"]
    types = prop.get("type") or [branch.get("type") for branch in prop.get("anyOf", [])]
    assert "boolean" in types
    assert "properties" not in prop

    for flag in (True, False):
        dumped = BalanceOutput.model_validate({"ok": True, "is_available": flag}).model_dump(
            mode="json", by_alias=True
        )
        assert dumped["is_available"] is flag
        assert dumped["session"] is None


def test_error_envelope_validates() -> None:
    dumped = BalanceOutput.model_validate(
        {"ok": False, "error": {"code": "auth_failed", "message": "denied", "action": "ask_user"}}
    ).model_dump(mode="json", by_alias=True)
    assert dumped["ok"] is False
    assert dumped["error"]["code"] == "auth_failed"
