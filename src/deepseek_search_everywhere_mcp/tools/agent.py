"""The agent research tool: it decides when more searching stops paying."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, Literal

from pydantic import Field

from .. import agent_research, runtime
from ..deep_research import DEPTH_CONFIG, ResearchJob, get_job_store
from ..errors import _fail
from ..research_jobs import _finished_job_envelope, _reuse_finished_result, _running_job_envelope

MAX_TOPIC = 500


async def deepseek_agent_deep_research(
    topic: Annotated[
        str,
        Field(description="Complex research topic or question to investigate (non-empty, max 500 chars)"),
    ],
    *,
    depth: Annotated[
        Literal["low", "medium", "high", "max"],
        Field(description="Writer size and source count (default 'medium'); rounds come from max_rounds"),
    ] = "medium",
    max_rounds: Annotated[
        int,
        Field(description="Upper bound on search rounds; the loop stops earlier when rounds stop paying"),
    ] = 3,
    max_cost_usd: Annotated[
        float | None,
        Field(description="Refuse the job when its cost estimate exceeds this, in US dollars (default none)"),
    ] = None,
    sources: Annotated[
        Literal["any", "academic", "technical"],
        Field(description="Source preference: 'any' (default), 'academic' (papers), 'technical' (docs)"),
    ] = "any",
    lang: Annotated[
        str | None,
        Field(description="Language code such as 'en' or 'pt-BR' to restrict search sources to"),
    ] = None,
    refresh: Annotated[
        bool, Field(description="Bypass the cache and run a fresh pipeline (default False)")
    ] = False,
) -> dict[str, Any]:
    """Research a topic with an agent that searches again only when a round still adds evidence.

    The plan is written first, then each round is measured: a round that adds no fact carrying a figure,
    closes no gap and brings no new source ends the loop. The depth tier still controls how much the
    writer may say and how many sources it reads. About $0.018 for three rounds at depth "medium" against
    $0.010 for one deepseek_research pass, and it returns the plan, the per-round gains, the estimate and
    why the loop stopped.

    When NOT to use: a single question where one pass is enough (deepseek_research), one fact
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
    if not 1 <= max_rounds <= agent_research.MAX_ROUNDS:
        return _fail(
            "bad_request",
            f"max_rounds must be between 1 and {agent_research.MAX_ROUNDS}.",
            param="max_rounds",
        )
    if max_cost_usd is not None and max_cost_usd <= 0:
        return _fail("bad_request", "max_cost_usd must be positive.", param="max_cost_usd")

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
    else:
        if store.count_active() >= settings.research_max_jobs:
            return _fail(
                "rate_limited_local",
                f"At most {settings.research_max_jobs} research jobs run at once and that many are "
                "running. Collect a running job with deepseek_check_research or wait for one to finish.",
            )
        estimate = agent_research.estimate_cost(depth, max_rounds, sources_pref=sources)
        if max_cost_usd is not None and estimate > max_cost_usd:
            return _fail(
                "budget_exceeded",
                f"Estimated ${estimate:.4f} for {max_rounds} round(s) at depth {depth!r} exceeds the "
                f"${max_cost_usd:.4f} budget. Raise max_cost_usd or lower max_rounds or depth.",
                param="max_cost_usd",
            )
        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            topic=clean_topic,
            depth=depth,
            lang=lang,
            sources=sources,
            kind="agent",
            max_rounds=max_rounds,
            max_cost_usd=max_cost_usd,
            created_at=runtime._clock(),
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
    done, _ = await asyncio.wait([job.task], timeout=settings.research_inline_wait)
    if job.task in done:
        return _finished_job_envelope(job, "inline")
    return _running_job_envelope(job)
