"""The deepseek_deep_agent_research tool: it decides when more searching stops paying."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, Literal

from mcp.server.mcpserver.context import Context
from pydantic import Field

from .. import agent_research, runtime
from ..deep_research import DEPTH_CONFIG, ResearchJob, get_job_store
from ..errors import _fail
from ..research_jobs import _finished_job_envelope, _reuse_finished_result, _running_job_envelope
from ..runtime import _timed

MAX_TOPIC = 500
DEPTH_ROUNDS: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "max": 3}


@_timed
async def deepseek_deep_agent_research(
    topic: Annotated[
        str,
        Field(description="Complex research topic or question to investigate (non-empty, max 500 chars)"),
    ],
    *,
    depth: Annotated[
        Literal["low", "medium", "high", "max"],
        Field(description="Depth tier controlling queries, sources and reasoning (default 'medium')"),
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
) -> dict[str, Any]:
    """Research a topic with an autonomous agent that plans, verifies and stops when rounds stop paying.

    The plan is written first, then each round is measured: a round that adds no fact carrying a figure,
    closes no gap and brings no new source ends the loop. The depth tier controls how much the writer
    may say and how many sources it reads. About $0.015-0.025 at depth 'medium'. Returns the plan,
    the per-round gains and why the loop stopped.

    When NOT to use: a single question where one pass is enough (deepseek_quick_research), one fact
    (deepseek_search), known URLs (deepseek_fetch), or fields as JSON (deepseek_extract).

    Returns: {"ok": True, "status": "done"|"running", "job_id"?, "report": str, "sources": [...],
    "plan": [...], "rounds": [...], "facts_per_round": [...], "estimate_usd": float,
    "stopped_reason": str, "cost_usd": float}. Poll a running job with deepseek_check_research.
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.", param="topic")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.", param="topic")
    if depth not in DEPTH_CONFIG:
        return _fail("bad_request", f"depth must be one of: {', '.join(DEPTH_CONFIG)}", param="depth")
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
