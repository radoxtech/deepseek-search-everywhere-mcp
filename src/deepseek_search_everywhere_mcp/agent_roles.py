"""Planning and check roles for agent orchestration (ADR 0009).

Implements the three check roles: screener, critic, and auditor, plus topic analyzer
and report planner. Prompts enforce reasoning_effort="high" and strict JSON.
Crucially: no 'default' or 'askable' concepts — the agent is 100% autonomous.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .structured import generate_structured

QUERY_SCREENER_PROMPT = (
    "You review search queries before they are sent. For each query say whether it will return generic "
    "pages, repeated pages or the wrong kind of source, and if so give a better query of 4 to 12 words. "
    "Return exactly the same number of queries you were given, one for each input query, in the same "
    "order, so the plan keeps its coverage. Return only JSON matching the schema."
)

PLAN_CRITIC_PROMPT = (
    "You critique a research plan before execution. Evaluate whether aspects and queries cover "
    "core trade-offs and empirical data needed to answer the question, or if they lean on generic overviews. "
    "Identify any critical angles completely missing and propose up to 3 refined, narrower search queries "
    "specifically targeting hard facts, dates, or quantitative metrics. Return only JSON matching the schema."
)

EVIDENCE_AUDITOR_PROMPT = (
    "You audit whether gathered research sources contain sufficient evidence for each aspect. "
    "For each aspect, decide if the sources provide direct empirical facts or figures. "
    "List aspects that remain uncovered and specify the concrete missing facts. "
    "Return only JSON matching the schema."
)

ANALYZER_PROMPT = (
    "You analyze a research topic against reconnaissance snippets showing what the web already holds.\n"
    "Identify what the preliminary reconnaissance covers (seed_covered), what major branches or "
    "perspectives are completely missing from the sample (seed_missing), and what concrete facts, numbers "
    "or specifications must be established from the web (must_find).\n"
    "Order must_find by importance, with the most decisive fact first. Do NOT generate questions for user. "
    "Return only JSON matching the schema."
)

REPORT_PLAN_PROMPT = (
    "You design the section structure of a research report. Given the question, facts to establish, "
    "and what the gathered sources actually support:\n"
    "Split the question into sub-questions. Provide one section per supported sub-question in logical order. "
    "Do not add a section for generic background or definitions unless requested. "
    "Write section titles in the language of the original question, integrating facts across multilingual "
    "sources. List any sub-questions that sources do not settle in unsupported. "
    "Return only JSON matching the schema."
)


class ScreenedQueries(BaseModel):
    queries: list[str] = Field(description="Screened and refined queries matching the input count.")


class PlanCriticism(BaseModel):
    aspects_missing: list[str] = Field(default_factory=list, description="Missing angles or branches.")
    refined_queries: list[str] = Field(
        default_factory=list, description="Narrower evidence-targeted queries."
    )
    notes: list[str] = Field(default_factory=list, description="Critique notes.")


class AspectAudit(BaseModel):
    aspect: str
    has_evidence: bool
    missing_facts: list[str] = Field(default_factory=list)


class EvidenceAudit(BaseModel):
    aspects: list[AspectAudit] = Field(default_factory=list)
    uncovered_aspects: list[str] = Field(default_factory=list)


class TopicAnalysis(BaseModel):
    seed_covered: list[str] = Field(default_factory=list, description="Covered by reconnaissance.")
    seed_missing: list[str] = Field(default_factory=list, description="Missing perspectives or branches.")
    must_find: list[str] = Field(default_factory=list, description="Facts to establish from the web.")


class PlannedSection(BaseModel):
    title: str
    answers: list[int] = Field(default_factory=list, description="Indices of subquestions answered.")
    facts: list[int] = Field(default_factory=list, description="Indices of facts supporting this section.")


class ReportPlan(BaseModel):
    subquestions: list[str] = Field(default_factory=list)
    sections: list[PlannedSection] = Field(default_factory=list)
    unsupported: list[int] = Field(default_factory=list)


class PlanningRoles:
    """Check and planning roles executed with reasoning_effort='high'."""

    def __init__(self, sdk_client: Any, model: str) -> None:
        self.sdk_client = sdk_client
        self.model = model

    async def screen_queries(self, topic: str, queries: list[str], n: int) -> list[str]:
        """Review and sharpen queries, maintaining exact count n."""
        messages = [
            {"role": "system", "content": QUERY_SCREENER_PROMPT},
            {
                "role": "user",
                "content": f"Research question: {topic}\nQueries ({len(queries)}): {json.dumps(queries)}",
            },
        ]
        try:
            parsed, _ = await generate_structured(
                self.sdk_client,
                model=self.model,
                messages=messages,
                schema=ScreenedQueries,
                reasoning_effort="high",
            )
            answered = [str(q).strip()[:128] for q in parsed.queries if str(q).strip()]
            final = list(dict.fromkeys(answered))
            if len(final) >= n:
                return final[:n]
            # Pad with original if screener under-returned
            for q in queries:
                if q not in final:
                    final.append(q)
                if len(final) >= n:
                    break
            return final[:n]
        except Exception:
            return queries[:n]

    async def critique_plan(self, topic: str, aspects: list[dict[str, str]]) -> PlanCriticism:
        """Critique the decomposition plan before retrieval."""
        messages = [
            {"role": "system", "content": PLAN_CRITIC_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\nPlanned aspects: {json.dumps(aspects)}",
            },
        ]
        try:
            parsed, _ = await generate_structured(
                self.sdk_client,
                model=self.model,
                messages=messages,
                schema=PlanCriticism,
                reasoning_effort="high",
            )
            return parsed
        except Exception:
            return PlanCriticism()

    async def audit_evidence(
        self, aspects: list[dict[str, str]], sources_digest: list[dict[str, str]]
    ) -> EvidenceAudit:
        """Audit whether gathered sources support the planned aspects."""
        messages = [
            {"role": "system", "content": EVIDENCE_AUDITOR_PROMPT},
            {
                "role": "user",
                "content": f"Aspects: {json.dumps(aspects)}\nSources: {json.dumps(sources_digest)}",
            },
        ]
        try:
            parsed, _ = await generate_structured(
                self.sdk_client,
                model=self.model,
                messages=messages,
                schema=EvidenceAudit,
                reasoning_effort="high",
            )
            return parsed
        except Exception:
            return EvidenceAudit()

    async def analyze_topic(self, topic: str, reconnaissance: list[str]) -> TopicAnalysis:
        """Analyze topic against reconnaissance sample: identify seed_missing and must_find."""
        messages = [
            {"role": "system", "content": ANALYZER_PROMPT},
            {
                "role": "user",
                "content": f"Topic: {topic}\nReconnaissance snippets:\n" + "\n".join(reconnaissance),
            },
        ]
        try:
            parsed, _ = await generate_structured(
                self.sdk_client,
                model=self.model,
                messages=messages,
                schema=TopicAnalysis,
                reasoning_effort="high",
            )
            return parsed
        except Exception:
            return TopicAnalysis(must_find=[topic])

    async def plan_report(self, topic: str, facts: list[str], sources: list[dict[str, Any]]) -> ReportPlan:
        """Structure report into sub-questions and supported sections."""
        src_repr = json.dumps(sources)[:6000]
        messages = [
            {"role": "system", "content": REPORT_PLAN_PROMPT},
            {
                "role": "user",
                "content": f"Question: {topic}\nFacts: {json.dumps(facts)}\nSources: {src_repr}",
            },
        ]
        try:
            parsed, _ = await generate_structured(
                self.sdk_client,
                model=self.model,
                messages=messages,
                schema=ReportPlan,
                reasoning_effort="high",
            )
            return parsed
        except Exception:
            return ReportPlan(
                subquestions=[topic],
                sections=[PlannedSection(title=topic, answers=[0], facts=[0])],
                unsupported=[],
            )
