"""The deepseek_quick_research and deepseek_check_research tools (job start, join, reuse and collection)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, Literal

from mcp.server.mcpserver.context import Context
from pydantic import Field

from .. import runtime
from ..deep_research import (
    ResearchJob,
    get_job_store,
)
from ..errors import (
    _fail,  # re-exported for callers and tests
)
from ..research_jobs import (
    _client_page,
    _finished_job_envelope,
    _reuse_finished_result,
    _run_research_job,
    _running_job_envelope,
    _verify_client_report,
    _with_page_note,
)
from ..runtime import (
    _timed,
)

MAX_TOPIC = 4000


@_timed
async def deepseek_quick_research(
    topic: Annotated[
        str,
        Field(description="Complex research topic or question to investigate (non-empty, max 4000 chars)"),
    ],
    *,
    depth: Annotated[
        Literal["low", "medium", "high"],
        Field(
            description=(
                "Depth tier: 'low' (~10s fact-check), 'medium' (default, ~20s briefing), "
                "or 'high' (~35s comprehensive single-pass)"
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
    synthesis: Annotated[
        Literal["server", "client"],
        Field(
            description=(
                "Report author: 'server' (default) or 'client' (subagent orchestrators like "
                "Claude Code+Opus: you write, server checks)."
            )
        ),
    ] = "server",
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and perform fresh research pipeline (default False)"),
    ] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Research a topic across the web and get a cited report in one quick single pass.

    Fast single-pass pipeline (~25-35s, $0.003-$0.01). Searches several angles, reads 6-24 sources,
    and writes a cited report with [n] citations mapped to source ids. If status is "running", poll
    deepseek_check_research with job_id. With synthesis="client", returns status "needs_input".

    When to use:
    - Technical documentation, specifications, architecture comparisons, and state-of-the-art overviews.
    - Any question where high factual density, speed (~30s), and instant synthesis matter most.
    (For multi-round investigations of contested disputes, choose deepseek_deep_agent_research).

    Returns: {"ok": True, "status": "done"|"running"|"needs_input", "job_id"?, "report": str,
    "sources": [{id, title, url, ...}], "unverified_numbers": [...], "cost_usd": float}.
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.", param="topic")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.", param="topic")
    # Graceful backward compatibility: map legacy 'max' to 'high' (comprehensive single pass)
    if depth == "max":
        depth = "high"
    if depth not in ("low", "medium", "high"):
        return _fail("bad_request", "depth must be one of: low, medium, high", param="depth")
    if sources not in {"any", "academic", "technical"}:
        return _fail(
            "bad_request",
            f"Invalid sources preference '{sources}'. Expected one of: any, academic, technical.",
            param="sources",
        )
    if synthesis not in {"server", "client"}:
        return _fail(
            "bad_request", f"synthesis must be 'server' or 'client', got {synthesis!r}.", param="synthesis"
        )

    settings = runtime.get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(runtime._clock())

    key = {"sources": sources, "unverified": "flag", "synthesis": synthesis}
    existing = store.find_active_match(clean_topic, depth, lang, **key)
    if existing is None and not refresh:
        finished = store.find_finished_match(clean_topic, depth, lang, runtime._clock(), **key)
        if finished is not None and finished.result is not None:
            if finished.has_client_sources and finished.client_sources_result is not None:
                finished.completed_at = runtime._clock()
                return _reuse_finished_result(finished.client_sources_result)
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
                "running. Collect a running job with deepseek_check_research or wait for one to "
                "finish (usually 30-180 s), then retry with the same arguments.",
            )
        job_id = str(uuid.uuid4())
        job = ResearchJob(
            job_id=job_id,
            topic=clean_topic,
            depth=depth,
            lang=lang,
            sources=sources,
            unverified="flag",
            synthesis=synthesis,
            created_at=runtime._clock(),
            ctx=ctx,
        )
        job.task = asyncio.create_task(
            _run_research_job(
                job=job,
                clean_topic=clean_topic,
                depth=depth,
                sources_pref=sources,
                lang=lang,
                refresh=refresh,
                strip_boilerplate=True,
            )
        )
        store.add(job)

    try:
        inline_wait = settings.research_inline_wait
        assert job.task is not None
        try:
            done, _ = await asyncio.wait([job.task], timeout=inline_wait)
        except asyncio.CancelledError:
            raise

        if job.task in done:
            return _finished_job_envelope(job, "inline")

        return _running_job_envelope(job)
    finally:
        job.ctx = None


# Alias for backwards compatibility
deepseek_research = deepseek_quick_research


async def deepseek_check_research(
    job_id: Annotated[
        str,
        Field(
            description="Unique UUID identifier of the research job to poll (from deepseek_quick_research "
            "or deepseek_deep_agent_research)"
        ),
    ],
    *,
    page: Annotated[
        int | None,
        Field(description="Page of sources to read for a synthesis='client' job (default None)"),
    ] = None,
    answer: Annotated[
        str | dict[str, Any] | None,
        Field(description="Input report text for a job with status needs_input (default None)"),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Continue a research job: poll it, read its sources, or answer when it needs input.

    Free ($0.00); waits up to 45 s per call. Returns the finished report (status 'done'),
    progress while the job runs ('running': call again with the same job_id) or job_not_found
    once it expired. Status 'needs_input' names a step: for step 'report' (synthesis='client'),
    page= reads the next page of sources and answer= submits your report with [id] citations for
    verification. Never restart a running job; poll it with this tool.

    When NOT to use: to start research (deepseek_quick_research or deepseek_deep_agent_research).

    Returns: {"ok": True, "status": "done"|"running"|"needs_input", "step"?: str, "job_id": str, ...}.
    """
    clean_id = job_id.strip() if isinstance(job_id, str) else ""
    if not clean_id:
        return _fail("bad_request", "job_id must not be empty.", param="job_id")

    settings = runtime.get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(runtime._clock())

    job = store.get(clean_id)
    if job is None:
        return _fail(
            "job_not_found",
            "Research job not found or expired. Restart the research with deepseek_quick_research.",
        )
    if answer is not None:
        if page is not None:
            return _fail("bad_request", "Pass either page= or answer=, not both.", param="page")
        if not isinstance(answer, str):
            return _fail("bad_request", "For step 'report', answer must be the report text.", param="answer")
        return await _verify_client_report(job, answer)
    if job.has_client_sources and job.task is not None and job.task.done():
        job.completed_at = runtime._clock()
        if page is not None:
            return _client_page(job, page)
        return _client_page(job, 1) if job.awaiting_report else _finished_job_envelope(job, "polled")
    if ctx is not None:
        job.ctx = ctx

    try:
        if job.task is not None and job.task.done():
            return _with_page_note(_finished_job_envelope(job, "polled"), page)

        inline_wait = settings.research_inline_wait
        assert job.task is not None
        try:
            done, _ = await asyncio.wait([job.task], timeout=inline_wait)
        except asyncio.CancelledError:
            raise

        if job.task in done:
            if job.has_client_sources:
                return _client_page(job, page or 1)
            return _with_page_note(_finished_job_envelope(job, "polled"), page)

        return _running_job_envelope(job)
    finally:
        job.ctx = None
