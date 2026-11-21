from __future__ import annotations

import json
from typing import Any

from deepseek_search_everywhere_mcp import agent_research, runtime
from deepseek_search_everywhere_mcp.deep_research import ResearchJob


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


async def test_agent_stops_when_a_round_adds_nothing(fake_upstream: Any) -> None:
    from test_research import DeepResearchUpstream

    fake = DeepResearchUpstream()
    fake_upstream(fake)

    res = await _run(_job())

    assert res["ok"] is True and res["synthesized"] is True
    assert res["stopped_reason"] in {"saturated", "no_gaps"}
    assert len(res["rounds"]) == 1
    assert "plan" in res and res["estimate_usd"] > 0
    queries = [payload.get("query") for payload in fake.searches]
    assert len(queries) == len(set(queries))


class PlannedUpstream:
    """A plan with one aspect that its own pages satisfy, so round 1 gains and round 2 repeats."""

    PLAN = {
        "aspects": [
            {
                "kind": "numbers",
                "aspect": "gate fidelity",
                "looking_for": "percent",
                "query": "two qubit gate fidelity percent",
            },
            {
                "kind": "numbers",
                "aspect": "error budget",
                "looking_for": "logical error rate per cycle",
                "query": "surface code logical error rate per cycle",
            },
        ]
    }
    PAGES = [
        {
            "title": "Fidelity",
            "url": "https://physics.org/fidelity",
            "content": "Two-qubit gate fidelity reached 99.9 percent in 2025 trials. " * 4,
        }
    ]

    def __init__(self) -> None:
        self.chats: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        from deepseek_search_everywhere_mcp import server

        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.searches.append(payload)
            return {"ok": True, "data": {"results": self.PAGES}}
        if path == server.FETCH_PATH:
            return {"ok": True, "data": {"title": "Fetched", "content": "Nothing new here. " * 20}}
        if path == "/v1/chat/completions":
            self.chats.append(payload)
            if len(self.chats) == 1:
                import json

                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(self.PLAN)}}]
                    },
                }
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Gate fidelity reached 99.9 percent in 2025 [1].",
                            }
                        }
                    ]
                },
            }
        return {"ok": False, "error": {"code": "not_found", "message": path}}


async def test_agent_saturates_on_the_second_round(fake_upstream: Any) -> None:
    fake = PlannedUpstream()
    fake_upstream(fake)

    res = await _run(_job())

    assert res["ok"] is True
    assert res["rounds"][0]["new_numeric_facts"] >= 1
    assert res["stopped_reason"] in {"saturated", "no_gaps"}
    assert len(res["rounds"]) == 2
    assert res["facts_per_round"][-1] == 0
    queries = [payload.get("query") for payload in fake.searches]
    assert len(queries) == len(set(queries))
    assert len(queries) == 3
    assert "logical error rate" in queries[-1]


async def test_agent_records_max_rounds_when_it_cannot_saturate(fake_upstream: Any) -> None:
    from test_research import DeepResearchUpstream

    fake_upstream(DeepResearchUpstream())

    res = await _run(_job(max_rounds=1))

    assert res["ok"] is True
    assert res["stopped_reason"] == "max_rounds"
    assert len(res["rounds"]) == 1


async def test_agent_refuses_a_budget_below_its_estimate(fake_upstream: Any) -> None:
    from test_research import DeepResearchUpstream

    fake = DeepResearchUpstream()
    fake_upstream(fake)

    res = await _run(_job(max_cost_usd=0.001))

    assert res["ok"] is False
    assert res["error"]["code"] == "budget_exceeded"
    assert fake.searches == []
    assert fake.chats == []


class FailingSearchUpstream(PlannedUpstream):
    """Every search reports an upstream failure."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        from deepseek_search_everywhere_mcp import server

        if path == server.SEARCH_PATH:
            return {"ok": False, "error": {"code": "upstream_error", "message": "boom"}}
        return await super().__call__(method, path, **kwargs)


class EmptyPageUpstream(PlannedUpstream):
    """Searches succeed but every page has no readable content, and fetching adds none."""

    PAGES = [{"title": "Empty", "url": "https://example.org/empty", "content": ""}]

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        from deepseek_search_everywhere_mcp import server

        if path == server.FETCH_PATH:
            return {"ok": True, "data": {"title": "Empty", "content": ""}}
        return await super().__call__(method, path, **kwargs)


async def test_agent_reports_when_every_search_fails(fake_upstream: Any) -> None:
    fake_upstream(FailingSearchUpstream())

    res = await _run(_job())

    assert res["ok"] is False
    assert res["error"]["code"] == "temporary_failure"
    assert res["stopped_reason"] == "retrieval_failed"


async def test_agent_reports_when_every_page_is_empty(fake_upstream: Any) -> None:
    fake_upstream(EmptyPageUpstream())

    res = await _run(_job())

    assert res["ok"] is False
    assert res["error"]["code"] == "page_unavailable"
    assert res["stopped_reason"] == "empty_pages"


class ReportPlanUpstream(PlannedUpstream):
    """An upstream returning a valid report-plan JSON on plan call, then section text."""

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        from deepseek_search_everywhere_mcp import server
        from deepseek_search_everywhere_mcp.agent_research import (
            REPORT_PLAN_PROMPT,
            UNIVERSAL_GAP_PROMPT,
        )
        from deepseek_search_everywhere_mcp.deep_research import DECOMPOSE_SYSTEM_PROMPT

        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.searches.append(payload)
            return {"ok": True, "data": {"results": self.PAGES}}
        if path == "/v1/chat/completions":
            self.chats.append(payload)
            msgs = payload.get("messages", [])
            sys_msg = msgs[0].get("content", "") if msgs else ""
            if sys_msg == DECOMPOSE_SYSTEM_PROMPT:
                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(self.PLAN)}}]
                    },
                }
            if sys_msg == UNIVERSAL_GAP_PROMPT:
                gap_data = {"queries": ["surface code logical error rate per cycle"]}
                choice = {"message": {"role": "assistant", "content": json.dumps(gap_data)}}
                return {"ok": True, "data": {"choices": [choice]}}
            if sys_msg == REPORT_PLAN_PROMPT:
                plan_json = {
                    "subquestions": ["How high is gate fidelity?", "What is the logical error rate?"],
                    "sections": [
                        {"title": "Gate Fidelity Achievements", "answers": [0], "facts": [0]},
                    ],
                    "unsupported": [1],
                }
                return {
                    "ok": True,
                    "data": {
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(plan_json)}}]
                    },
                }
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Two-qubit gate fidelity reached 99.9% in 2025 trials [1].",
                            }
                        }
                    ]
                },
            }
        return await super().__call__(method, path, **kwargs)


async def test_agent_synthesize_report_plan(fake_upstream: Any) -> None:
    fake = ReportPlanUpstream()
    fake_upstream(fake)

    res = await _run(_job())

    assert res["ok"] is True
    assert res["synthesized"] is True
    report = res["report"]
    assert "## Gate Fidelity Achievements" in report
    assert "Two-qubit gate fidelity reached 99.9%" in report
    assert "## What the sources do not settle" in report
    assert "What is the logical error rate?" in report
    assert "> **How this was researched:**" in report
