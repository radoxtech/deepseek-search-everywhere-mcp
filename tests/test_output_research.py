"""Output models of the two research tools: every status a caller can receive must be representable.

The runtime envelope is a plain dict with status-conditional keys, so each model is checked for two
properties: ``model_validate`` accepts the dict (``_meta`` is the model's ``meta`` alias) and every
returned top-level key is a modelled field (no key would be dropped from ``structuredContent``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import runtime, server
from deepseek_search_everywhere_mcp.output_models.research import (
    CheckResearchOutput,
    QuickResearchOutput,
)
from deepseek_search_everywhere_mcp.settings import Settings

SOURCES = [
    {
        "title": "Quantum Error Correction Guide",
        "url": "https://physics.org/qec",
        "content": "Surface codes provide fault tolerance in quantum computing. " * 5,
    },
    {
        "title": "Qubit Fidelity Benchmarks",
        "url": "https://physics.org/benchmarks",
        "content": "Two-qubit gate fidelities exceed 99.9% in recent trials. " * 5,
    },
]


class Upstream:
    """Fake DeepSeek: decomposition answers subqueries; the next chat call returns a report."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.chats: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        import asyncio

        if self.delay:
            await asyncio.sleep(self.delay)
        if path == server.SEARCH_PATH:
            return {"ok": True, "data": {"results": SOURCES}}
        if path == server.FETCH_PATH:
            return {"ok": True, "data": {"title": "Fetched", "content": "Fetched page content " * 20}}
        if path == "/v1/chat/completions":
            self.chats.append(kwargs.get("payload") or {})
            content = (
                json.dumps({"queries": ["subquery 1", "subquery 2"]})
                if len(self.chats) == 1
                else "Surface codes provide fault tolerance [1]. Gate fidelities exceed 99.9% [2]."
            )
            return {"ok": True, "data": {"choices": [{"message": {"role": "assistant", "content": content}}]}}
        return {"ok": False, "error": {"code": "not_found", "message": path}}


def assert_representable(model: type[QuickResearchOutput] | type[CheckResearchOutput], result: dict) -> None:
    """Every returned key is modelled and the whole dict validates against ``model``."""
    returned = {"meta" if key == "_meta" else key for key in result}
    missing = returned - set(model.model_fields)
    assert not missing, f"{model.__name__} does not model: {sorted(missing)}"
    validated = model.model_validate(result)
    assert validated.ok is result["ok"]


async def test_quick_done_envelope_is_modelled(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    result = await server.deepseek_research("quantum error correction", depth="low")
    assert result["status"] == "done" and result["delivery"] == "inline"
    assert "elapsed_ms" in result  # @_timed adds it
    assert_representable(QuickResearchOutput, result)


async def test_quick_running_envelope_is_modelled(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.01))
    fake_upstream(Upstream(delay=0.2))
    result = await server.deepseek_research("slow quantum topic", depth="low")
    assert result["status"] == "running" and result["job_id"]
    assert_representable(QuickResearchOutput, result)


async def test_quick_needs_input_envelope_is_modelled(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    result = await server.deepseek_research("quantum error correction", depth="low", synthesis="client")
    assert result["status"] == "needs_input" and result["step"] == "report"
    assert_representable(QuickResearchOutput, result)


async def test_quick_error_envelope_is_modelled(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    result = await server.deepseek_research("   ")
    assert result["ok"] is False and result["error"]["reason"] == "empty_query"
    assert_representable(QuickResearchOutput, result)


async def test_check_done_polled_envelope_is_modelled(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    await server.deepseek_research("quantum error correction", depth="low")
    job = server.get_job_store().find_finished_match("quantum error correction", "low", None, server._clock())
    assert job is not None
    result = await server.deepseek_check_research(job.job_id)
    assert result["status"] == "done" and result["delivery"] == "polled"
    assert_representable(CheckResearchOutput, result)


async def test_check_running_envelope_is_modelled(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.01))
    fake_upstream(Upstream(delay=0.2))
    running = await server.deepseek_research("slow quantum topic", depth="low")
    assert running["status"] == "running"
    result = await server.deepseek_check_research(running["job_id"])
    assert result["status"] == "running"
    assert_representable(CheckResearchOutput, result)


async def test_check_needs_input_envelope_is_modelled(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    awaiting = await server.deepseek_research("quantum error correction", depth="low", synthesis="client")
    result = await server.deepseek_check_research(awaiting["job_id"])
    assert result["status"] == "needs_input" and result["context_page"] == 1
    assert_representable(CheckResearchOutput, result)


async def test_check_error_envelope_is_modelled() -> None:
    result = await server.deepseek_check_research("00000000-0000-4000-8000-000000000000")
    assert result["ok"] is False and result["error"]["reason"] == "job_not_found"
    assert_representable(CheckResearchOutput, result)


async def test_verified_client_report_has_no_elapsed_ms_and_no_job_id(fake_upstream: Any) -> None:
    fake_upstream(Upstream())
    awaiting = await server.deepseek_research("quantum error correction", depth="low", synthesis="client")
    result = await server.deepseek_check_research(
        awaiting["job_id"], answer="Surface codes provide fault tolerance [1]."
    )
    assert result["status"] == "done" and result["delivery"] == "client"
    assert "elapsed_ms" not in result and "job_id" not in result
    assert_representable(CheckResearchOutput, result)


@pytest.mark.parametrize("model", [QuickResearchOutput, CheckResearchOutput])
def test_bare_error_envelope_validates(
    model: type[QuickResearchOutput] | type[CheckResearchOutput],
) -> None:
    result = {"ok": False, "error": {"code": "internal_error", "message": "boom"}}
    validated = model.model_validate(result)
    assert validated.ok is False and validated.error is not None
    assert validated.error.code == "internal_error"
    assert_representable(model, result)


@pytest.mark.parametrize("model", [QuickResearchOutput, CheckResearchOutput])
def test_research_schemas_are_closed_and_described(
    model: type[QuickResearchOutput] | type[CheckResearchOutput],
) -> None:
    schema = model.model_json_schema()
    assert "properties" in schema
    assert "additionalProperties" not in schema
    assert schema["properties"]["_meta"]
    assert "meta" not in schema["properties"]
    for name, field in model.model_fields.items():
        assert field.description, f"{model.__name__}.{name} has no description"
