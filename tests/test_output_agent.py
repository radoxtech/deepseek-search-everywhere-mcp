"""Tests for ``AgentResearchOutput``: synthesized, unsynthesized, running and budget-guard envelopes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

from test_agent_research import PlannedUpstream, ReportPlanUpstream

from deepseek_search_everywhere_mcp import agent_research, runtime, server
from deepseek_search_everywhere_mcp.deep_research import DECOMPOSE_SYSTEM_PROMPT, ResearchJob
from deepseek_search_everywhere_mcp.output_models.agent import AgentResearchOutput
from deepseek_search_everywhere_mcp.settings import set_settings


def _job(**overrides: Any) -> ResearchJob:
    fields: dict[str, Any] = {
        "job_id": "agent-job",
        "topic": "quantum error correction",
        "depth": "medium",
        "lang": None,
        "created_at": runtime._clock(),
        "kind": "agent",
        "max_rounds": 3,
    }
    fields.update(overrides)
    return ResearchJob(**fields)


async def _run(job: ResearchJob) -> dict[str, Any]:
    return await agent_research.run_agent(
        job,
        clean_topic=job.topic,
        depth=job.depth,
        sources_pref="any",
        lang=None,
        refresh=False,
        strip_boilerplate=True,
    )


# ``model_fields`` is keyed by Python attribute name, while envelopes publish the ``_meta`` alias,
# so the published key set adds every field alias to the field names.
_MODELED_KEYS = set(AgentResearchOutput.model_fields) | {
    field.alias for field in AgentResearchOutput.model_fields.values() if field.alias
}


def _assert_modeled(result: dict[str, Any]) -> AgentResearchOutput:
    assert set(result) <= _MODELED_KEYS
    return AgentResearchOutput.model_validate(result)


class FailedSynthesisUpstream(PlannedUpstream):
    """The plan call succeeds, every later chat call fails, so no report can be written."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/chat/completions":
            payload = kwargs.get("payload") or {}
            self.chats.append(payload)
            messages = payload.get("messages", [])
            system = messages[0].get("content", "") if messages else ""
            if system == DECOMPOSE_SYSTEM_PROMPT:
                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(self.PLAN)}}]
                    },
                }
            return {"ok": False, "error": {"code": "upstream_error", "message": "model unavailable"}}
        return await super().__call__(method, path, **kwargs)


class SlowUpstream:
    """A request that never returns before the inline wait expires, keeping the job running."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {"ok": False, "error": {"code": "temporary_failure", "message": "slow"}}


async def test_done_synthesized_matches_model(fake_upstream: Any) -> None:
    fake_upstream(ReportPlanUpstream())

    result = await server.deepseek_deep_agent_research("quantum error correction")

    assert result["ok"] is True
    assert result["status"] == "done"
    assert result["delivery"] == "inline"
    assert result["synthesized"] is True
    for key in ("report", "sources", "plan", "rounds", "facts_per_round", "estimate_usd"):
        assert key in result
    _assert_modeled(result)


async def test_done_unsynthesized_matches_model(fake_upstream: Any) -> None:
    fake_upstream(FailedSynthesisUpstream())

    result = await server.deepseek_deep_agent_research("quantum error correction without a synthesis")

    assert result["ok"] is True
    assert result["status"] == "done"
    assert result["delivery"] == "inline"
    assert result["synthesized"] is False
    assert result["partial"] is True
    assert result["report"] is None
    # plan and rounds are only published once a report is synthesized.
    assert "plan" not in result and "rounds" not in result
    _assert_modeled(result)


async def test_running_job_matches_model(fake_upstream: Any) -> None:
    fake_upstream(SlowUpstream())
    set_settings(replace(runtime.get_settings(), research_inline_wait=0.01))

    result = await server.deepseek_deep_agent_research("quantum error correction running")

    assert result["ok"] is True
    assert result["status"] == "running"
    assert result["job_id"]
    assert "progress" in result and "next_action" in result
    _assert_modeled(result)


async def test_budget_guard_error_matches_model(fake_upstream: Any) -> None:
    fake_upstream(PlannedUpstream())

    result = await _run(_job(max_cost_usd=0.001))

    assert result["ok"] is False
    assert result["error"]["code"] == "budget_exceeded"
    assert set(result["error"]) == {"code", "message"}
    _assert_modeled(result)
    minimal = {"ok": False, "error": {"code": "budget_exceeded", "message": "over budget"}}
    assert _assert_modeled(minimal).error is not None
