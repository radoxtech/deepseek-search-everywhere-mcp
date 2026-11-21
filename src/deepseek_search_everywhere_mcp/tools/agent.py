"""The deepseek_deep_agent_research tool: it decides when more searching stops paying."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Literal

from mcp.server.mcpserver.context import Context
from pydantic import Field

from .. import agent_research, runtime
from ..deep_research import ResearchJob, get_job_store
from ..errors import _fail
from ..output_models.agent import AgentResearchOutput
from ..research_jobs import _finished_job_envelope, _reuse_finished_result, _running_job_envelope
from ..runtime import _timed

MAX_TOPIC = 4000
DEPTH_ROUNDS: dict[str, int] = {"medium": 2, "high": 3, "max": 5}


@_timed
async def deepseek_deep_agent_research(
    topic: Annotated[
        str,
        Field(description="Complex research topic or question to investigate (non-empty, max 4000 chars)"),
    ],
    *,
    depth: Annotated[
        Literal["medium", "high", "max"],
        Field(
            description=(
                "Depth tier: 'medium' (default, 2 rounds, ~60s), 'high' (up to 3 rounds, ~90s), "
                "or 'max' (up to 5 rounds, ~135s)"
            )
        ),
    ] = "medium",
    sources: Annotated[
        Literal["any", "academic", "technical"],
        Field(
            description=(
                "Source preference: 'any' (default), 'academic' (papers) or 'technical' (docs, specs, repos)"
            )
        ),
    ] = "any",
    lang: Annotated[
        str | None,
        Field(
            description=(
                "Language code for sources (e.g. 'en', 'de', 'pl'). Prefer 'en' for tech/science docs; "
                "set code for local laws/markets."
            )
        ),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and perform fresh research (default False)"),
    ] = False,
    ctx: Context | None = None,
) -> AgentResearchOutput:
    """Research a topic with an autonomous agent that plans, verifies and stops when rounds stop paying.

    Flagship multi-round agent (~55-140s, ~$0.011-$0.035). Plans an aspect matrix, audits factual
    gains per round, queries gaps with entity anchoring, and autostops when evidence saturates. topic
    is at most 4,000 characters.

    When to use:
    - Comprehensive investigations, contested disputes and multi-faceted state-of-the-art reports.
    - In-depth research where opposing claims and subtle details must be uncovered across rounds.
    - When you need explicit tracking of open questions ("What the sources do not settle").
    (For rapid single-pass briefings prefer deepseek_quick_research; for single facts use deepseek_search).

    Depth sets rounds and deadline: medium 2 rounds within 180 s, high 3 within 300 s, max 5 within
    420 s. A call waits 45 s inline, then returns status "running" with a job_id; poll it with
    deepseek_check_research. At most 3 jobs run at once, and a finished result is reused for 30
    minutes at no cost.

    Returns: {"ok": True, "status": "done"|"running", "job_id"?, "report": str, "sources": [...],
    "plan": [...], "rounds": [...], "facts_per_round": [...], "estimate_usd": float,
    "stopped_reason": str, "cost_usd": float}. Errors: job_not_found (start again), limit_reached
    (max_cost_reached), temporary_failure (retry).
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.", param="topic")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.", param="topic")
    # Graceful backward compatibility: map legacy 'low' to 'medium' (minimum 2 rounds)
    if depth == "low":
        depth = "medium"
    if depth not in DEPTH_ROUNDS:
        return _fail("bad_request", "depth must be one of: medium, high, max", param="depth")
    if sources not in {"any", "academic", "technical"}:
        return _fail(
            "bad_request",
            f"Invalid sources preference '{sources}'. Expected one of: any, academic, technical.",
            param="sources",
        )

    max_rounds = DEPTH_ROUNDS.get(depth, 3)
    settings = runtime.get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(runtime._clock())
    key = {"sources": sources, "unverified": "flag", "synthesis": "server", "kind": "agent"}

    existing = store.find_active_match(clean_topic, depth, lang, **key)
    if existing is None and not refresh:
        finished = store.find_finished_match(clean_topic, depth, lang, runtime._clock(), **key)
        if finished is not None and finished.result is not None:
            return _reuse_finished_result(finished.result)
    if existing is not None and existing.task is not None and not existing.task.done():
        job = existing
        if ctx is not None:
            job.ctx = ctx
    else:
        if store.count_active() >= settings.research_max_jobs:
            return _fail(
                "rate_limited_local",
                f"At most {settings.research_max_jobs} research jobs run at once and that many are "
                "running. Collect a running job with deepseek_check_research or wait for one to finish.",
            )
        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            topic=clean_topic,
            depth=depth,
            lang=lang,
            sources=sources,
            kind="agent",
            max_rounds=max_rounds,
            max_cost_usd=None,
            created_at=runtime._clock(),
            ctx=ctx,
        )
        job.task = asyncio.create_task(
            agent_research.run_agent_job(
                job,
                clean_topic=clean_topic,
                depth=depth,
                sources_pref=sources,
                lang=lang,
                refresh=refresh,
                strip_boilerplate=True,
            )
        )
        store.add(job)

    assert job.task is not None
    try:
        done, _ = await asyncio.wait([job.task], timeout=settings.research_inline_wait)
        if job.task in done:
            return _finished_job_envelope(job, "inline")
        return _running_job_envelope(job)
    finally:
        job.ctx = None


# Alias for backwards compatibility
deepseek_agent_deep_research = deepseek_deep_agent_research
