"""The deepseek_research and deepseek_check_research tools (job start, join, reuse and collection)."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context
from pydantic import Field

from .. import runtime
from ..deep_research import (
    DEPTH_CONFIG,
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

MAX_TOPIC = 500


@_timed
async def deepseek_research(
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
            description="Language code such as 'en' or 'pt-BR' to restrict search sources to (default None)"
        ),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and perform fresh research pipeline (default False)"),
    ] = False,
    unverified: Annotated[
        Literal["flag", "drop"],
        Field(
            description=(
                "Sentences with numbers not found in their sources: 'flag' marks them (default), "
                "'drop' removes them"
            )
        ),
    ] = "flag",
    synthesis: Annotated[
        Literal["server", "client"],
        Field(
            description=(
                "Who writes the report: 'server' (default) or 'client' (you get ranked sources, "
                "write it, server verifies it)"
            )
        ),
    ] = "server",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Research a topic across the web and get a cited report in one call.

    The server searches several angles, reads 6-24 sources and returns a compact report whose [n]
    citations match the ids in the sources array, instead of filling your context with raw pages. About
    $0.003 (depth "low") to $0.03 ("max"). depth: "low" quick overview, "medium" (default) most
    questions, "high" decisions with several viewpoints, "max" broad or contested topics.
    sources="academic" favours peer-reviewed papers, "technical" official docs, specs and repos.
    Numbers missing from their cited source are marked [unverified] (unverified="drop" removes
    those sentences). If status is "running", call deepseek_check_research with the job_id; do not
    start the same research again. With synthesis="client" you get the ranked sources in pages
    and write the report yourself; the server then verifies its citations and numbers. At most 3
    jobs run at once.

    When NOT to use: one fact or a few independent queries (deepseek_search), known URLs
    (deepseek_fetch), or fields as JSON (deepseek_extract).

    Returns: {"ok": True, "status": "done"|"running"|"needs_input", "job_id"?, "report": str,
    "sources": [{id, title, url, date, doi, pmid, pmcid, access, source_type, authors}],
    "unverified_numbers": [...], "cost_usd": float}.
    """
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return _fail("empty_query", "topic must not be empty.", param="topic")
    if len(clean_topic) > MAX_TOPIC:
        return _fail("topic_too_long", f"topic must be at most {MAX_TOPIC} characters.", param="topic")
    if depth not in DEPTH_CONFIG:
        return _fail("bad_request", f"depth must be one of: {', '.join(DEPTH_CONFIG.keys())}", param="depth")
    if sources not in {"any", "academic", "technical"}:
        return _fail(
            "bad_request",
            f"Invalid sources preference '{sources}'. Expected one of: any, academic, technical.",
            param="sources",
        )
    if unverified not in {"flag", "drop"}:
        return _fail(
            "bad_request", f"unverified must be 'flag' or 'drop', got {unverified!r}.", param="unverified"
        )
    if synthesis not in {"server", "client"}:
        return _fail(
            "bad_request", f"synthesis must be 'server' or 'client', got {synthesis!r}.", param="synthesis"
        )

    settings = runtime.get_settings()
    store = get_job_store(ttl=settings.research_job_ttl, max_active=settings.research_max_jobs)
    store.prune_expired(runtime._clock())

    key = {"sources": sources, "unverified": unverified, "synthesis": synthesis}
    existing = store.find_active_match(clean_topic, depth, lang, **key)
    if existing is None and not refresh:
        finished = store.find_finished_match(clean_topic, depth, lang, runtime._clock(), **key)
        if finished is not None and finished.result is not None:
            if finished.has_client_sources and finished.client_sources_result is not None:
                # Another caller gets the sources to write its own report, never someone else's report.
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
            unverified=unverified,
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


async def deepseek_check_research(
    job_id: Annotated[
        str,
        Field(description="Unique UUID identifier of the research job to poll (from deepseek_research)"),
    ],
    *,
    page: Annotated[
        int | None,
        Field(description="Page of sources to read for a synthesis='client' job (default None)"),
    ] = None,
    answer: Annotated[
        str | dict[str, Any] | None,
        Field(description="Input for a job with status needs_input; its step says what (default None)"),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Continue a research job: poll it, read its sources, or answer when it needs input.

    Free ($0.00); waits up to 45 s per call. Returns the finished report (status "done"),
    progress while the job runs ("running": call again with the same job_id) or job_not_found
    once it expired. Status "needs_input" names a step: for step "report" (synthesis="client"),
    page= reads the next page of sources and answer= submits your report with [id] citations for
    verification. Sending the same answer again costs nothing.

    When NOT to use: to start research (deepseek_research). Never restart a running job; poll it.

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
            "Research job not found or expired. Restart the research with deepseek_research.",
        )
    if answer is not None:
        if page is not None:
            return _fail("bad_request", "Pass either page= or answer=, not both.", param="page")
        if not isinstance(answer, str):
            return _fail("bad_request", "For step 'report', answer must be the report text.", param="answer")
        return await _verify_client_report(job, answer)
    if job.has_client_sources and job.task is not None and job.task.done():
        job.completed_at = runtime._clock()  # a client still reading or writing keeps the job
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
