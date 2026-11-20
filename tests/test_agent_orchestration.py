"""Offline tests for agent_loop.py and agent_roles.py."""

from __future__ import annotations

from typing import Any

from ai.types.usage import Usage

from deepseek_search_everywhere_mcp.agent_loop import ResearchAgent
from deepseek_search_everywhere_mcp.agent_roles import (
    EvidenceAudit,
    PlanCriticism,
    PlannedSection,
    PlanningRoles,
    ReportPlan,
    ScreenedQueries,
    TopicAnalysis,
)


class MockContext:
    def __init__(self, running_count: int = 20) -> None:
        self.count = running_count
        self.messages: list[Any] = []

    def keep_running(self) -> bool:
        return self.count > 0

    def add(self, message: Any) -> None:
        self.messages.append(message)
        self.count -= 1


def test_research_agent_defaults() -> None:
    agent = ResearchAgent(max_steps=5)
    assert agent.LOOP_BUFFER == 0
    assert agent.max_steps == 5
    assert agent.stopped_reason == "running"
    assert isinstance(agent.total_usage, Usage)


async def test_planning_roles_methods() -> None:
    class MockStructured:
        def __init__(self, answer: Any) -> None:
            self.answer = answer

        async def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, int]]:
            return self.answer, {"total_tokens": 10}

    roles = PlanningRoles(sdk_client=None, model="deepseek-flash")

    # 1. Screen queries
    roles.sdk_client = "mock"
    import deepseek_search_everywhere_mcp.agent_roles as ar

    original_gen = ar.generate_structured

    try:
        ar.generate_structured = MockStructured(ScreenedQueries(queries=["q1 refined", "q2 refined"]))
        screened = await roles.screen_queries("topic", ["q1", "q2"], n=2)
        assert screened == ["q1 refined", "q2 refined"]

        # 2. Critique plan
        ar.generate_structured = MockStructured(
            PlanCriticism(aspects_missing=["aspect 3"], refined_queries=["q3 focus"])
        )
        critic = await roles.critique_plan("topic", [{"aspect": "a1", "query": "q1"}])
        assert critic.aspects_missing == ["aspect 3"]
        assert critic.refined_queries == ["q3 focus"]

        # 3. Audit evidence
        ar.generate_structured = MockStructured(EvidenceAudit(uncovered_aspects=["a2"]))
        audit = await roles.audit_evidence([{"aspect": "a1"}], [{"id": "s1", "says": "fact"}])
        assert audit.uncovered_aspects == ["a2"]

        # 4. Analyze topic (verify no askable or default!)
        ar.generate_structured = MockStructured(
            TopicAnalysis(
                seed_covered=["covered branch"],
                seed_missing=["missing perspective"],
                must_find=["must find fact 1", "must find fact 2"],
            )
        )
        analysis = await roles.analyze_topic("topic", ["recon snippet 1"])
        assert "seed_missing" in analysis.model_dump()
        assert "must_find" in analysis.model_dump()
        assert "askable" not in analysis.model_dump()
        assert "default" not in analysis.model_dump()
        assert analysis.must_find[0] == "must find fact 1"

        # 5. Plan report
        ar.generate_structured = MockStructured(
            ReportPlan(
                subquestions=["subq1", "subq2"],
                sections=[PlannedSection(title="Sec 1", answers=[0], facts=[0])],
                unsupported=[1],
            )
        )
        plan = await roles.plan_report("topic", ["fact1"], [{"id": "s1"}])
        assert plan.subquestions == ["subq1", "subq2"]
        assert plan.sections[0].title == "Sec 1"
        assert plan.unsupported == [1]

    finally:
        ar.generate_structured = original_gen
