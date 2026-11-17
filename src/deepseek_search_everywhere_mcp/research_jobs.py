"""Research job orchestration behind deepseek_research and deepseek_check_research.

``_ResearchRun`` carries one job through its pipeline phases (decompose, retrieve, round two,
rank and fill, synthesize); ``_finalize_report`` is shared by server and client synthesis. Also
holds the job envelopes (running, finished, awaiting a client report) and client report checks.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from . import runtime, search_core
from .deep_research import (
    DEPTH_CONFIG,
    ResearchJob,
    apply_token_guard,
    best_source_title,
    build_condense_messages,
    build_decomposition_messages,
    build_r2_gap_messages,
    build_synthesis_messages,
    build_synthesis_system_prompt,
    citation_ids,
    classify_source_type,
    clean_source_url,
    filter_synthesis_output,
    ground_report_numbers,
    normalize_citations,
    paginate_context,
    parse_subqueries,
    rank_and_diversify_sources,
    report_body_words,
    select_synthesis_context,
    source_access,
    source_authors,
    source_identifiers,
    validate_r2_queries,
)
from .envelope import (
    COST_PER_REQUEST,
    _all_cached,
    _envelope_meta,
)
from .errors import (
    _fail,  # re-exported for callers and tests
)
from .filters import (
    fit_query,
)
from .gap_fill import (
    _fill_gaps,
)
from .redact import redact
from .runtime import (
    REWRITE_MODEL,
    logger,
)
from .sanitize import (
    sanitize_untrusted_text,
)
from .settings import (
    ENV_DEBUG_DUMP_DIR,
)
from .urls import (
    _url_etld1,
)


def _format_public_sources(
    sources: list[dict[str, Any]],
    *,
    include_snippet: bool = False,
) -> list[dict[str, Any]]:
    """Format research sources into a lean public list omitting full page content.

    Each item carries the identifiers extracted from the URL/content (``doi``, ``pmid``,
    ``pmcid``) and ``access``: ``full_text`` for a fetched page or search content of at least 2000
    characters, ``abstract`` for search content of at least 200 characters and ``snippet_only``
    otherwise.
    On success, sources carry no page content; on fallback paths, a 400-character sanitized
    snippet is included.
    """
    formatted: list[dict[str, Any]] = []
    for s in sources:
        title = s.get("title")
        if isinstance(title, str):
            title, _ = sanitize_untrusted_text(title)
        item: dict[str, Any] = {
            "id": s.get("id"),
            "title": title,
            "url": s.get("url"),
            "date": s.get("date"),
            "doi": s.get("doi"),
            "pmid": s.get("pmid"),
            "pmcid": s.get("pmcid"),
            "access": s.get("access"),
            "source_type": s.get("source_type")
            or classify_source_type(s.get("url") or "", s.get("title") or "", s.get("content") or ""),
            "authors": s.get("authors"),
        }
        if include_snippet:
            raw_text = s.get("snippet") or s.get("content") or ""
            snip = raw_text[:400].strip() if isinstance(raw_text, str) else ""
            if snip:
                snip, _ = sanitize_untrusted_text(snip)
            item["snippet"] = snip
        formatted.append(item)
    return formatted


def _research_source(index: int, item: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Turn one shaped search result into a research source (``origin`` = ``search``)."""
    url = clean_source_url(item["url"]) if isinstance(item.get("url"), str) else item.get("url")
    content = item.get("content") or ""
    title = best_source_title(item.get("title"), url or "", content)
    return {
        "id": index,
        "title": title,
        "url": url,
        "date": item.get("date"),
        "date_source": item.get("date_source"),
        "site": item.get("site"),
        "authority": item.get("authority"),
        "rerank_score": item.get("rerank_score"),
        "content": content,
        "chars": 0,
        "origin": "search",
        "error": None,
        "cached": cached,
        **source_identifiers(url or "", content),
        "access": source_access("search", content),
        "authors": source_authors(url or "", content),
        "source_type": item.get("source_type") or classify_source_type(url or "", title, content),
    }


def _job_elapsed_ms(job: ResearchJob) -> int:
    """Milliseconds since ``job`` was created."""
    return max(0, int((runtime._clock() - job.created_at) * 1000))


def _finished_job_envelope(job: ResearchJob, delivery: Literal["inline", "polled"]) -> dict[str, Any]:
    """Return a finished job's result marked ``status: "done"``, or an error if the job failed.

    A polled result reports the job's own elapsed time instead of the time of the polling call.
    """
    if job.exception is not None:
        return _fail("internal_error", redact(f"Research job failed: {job.exception}"))
    res = dict(job.result) if job.result else _fail("internal_error", "Empty research result")
    if job.awaiting_report:
        if job.client_charged:
            return _awaiting_report_envelope(job, 1)  # the gathering cost was already reported
        job.client_charged = True
    if res.get("ok"):
        if res.get("status") != "awaiting_report":
            res["status"] = "done"
        res["delivery"] = delivery
        if delivery == "polled":
            res["elapsed_ms"] = _job_elapsed_ms(job)
    return res


def _running_job_envelope(job: ResearchJob) -> dict[str, Any]:
    """Return the ``status: "running"`` envelope telling the agent to poll the job."""
    return {
        "ok": True,
        "status": "running",
        "job_id": job.job_id,
        "depth": job.depth,
        "elapsed_ms": _job_elapsed_ms(job),
        "progress": _progress_view(job),
        "next_action": "Call deepseek_check_research with this job_id now. Do not start a new research.",
    }


def _reuse_finished_result(stored: dict[str, Any]) -> dict[str, Any]:
    """Copy a stored finished research result, marked as served from the job store at zero cost."""
    res = copy.deepcopy(stored)
    if res.get("status") != "awaiting_report":
        res["status"] = "done"
    res["delivery"] = "inline"
    res["cached"] = True
    res["cost_usd"] = 0
    if isinstance(res.get("cost_breakdown"), dict):
        res["cost_breakdown"] = {k: 0 for k in res["cost_breakdown"]}
    if isinstance(res.get("_meta"), dict):
        res["_meta"] = {**res["_meta"], "cost_usd": 0}
    notes = list(res.get("notes") or [])
    notes.append("Reused finished research result from this session (no new cost).")
    res["notes"] = notes
    return res


def _with_page_note(envelope: dict[str, Any], page: int | None) -> dict[str, Any]:
    """Note that ``page`` was ignored on a job that has no client-synthesis pages."""
    if page is not None and envelope.get("ok"):
        envelope["notes"] = [
            *envelope.get("notes", []),
            "page applies only to synthesis='client' jobs; ignored.",
        ]
    return envelope


def _progress_view(job: ResearchJob) -> dict[str, Any]:
    """Return the public progress of a running job.

    Besides ``phase``, ``step`` and ``of`` (and ``detail`` for a sub-step such as a synthesis retry
    or condensing), it carries ``phase_elapsed_ms``, the time spent in the current phase or sub-step,
    and ``deadline_ms``, the hard time cap of the job's depth, so a caller polling a long synthesis can
    tell that the job is alive and how long it can still take.
    """
    view = dict(job.progress)
    if job.phase_started_at is not None:
        view["phase_elapsed_ms"] = max(0, int((runtime._clock() - job.phase_started_at) * 1000))
    cfg = DEPTH_CONFIG.get(job.depth)
    if cfg is not None:
        view["deadline_ms"] = int(cfg.deadline_s * 1000)
    return view


async def _report_job_progress(
    job: ResearchJob,
    phase: str,
    step: int,
    total_steps: int,
    message: str | None = None,
    *,
    detail: str | None = None,
) -> None:
    """Update job progress state and emit FastMCP progress notification if context is attached."""
    event: dict[str, Any] = {"phase": phase, "step": step, "of": total_steps}
    if detail:
        event["detail"] = detail
    job.progress = event
    job.phase_started_at = runtime._clock()
    job.progress_history.append(event)
    if job.ctx is not None and hasattr(job.ctx, "report_progress"):
        try:
            await job.ctx.report_progress(float(step), float(total_steps), message=message or phase)
        except Exception as exc:
            logger.debug("Failed to report progress to context for job %s: %s", job.job_id, exc)


async def _run_research_job(job: ResearchJob, **kwargs: Any) -> dict[str, Any]:
    try:
        res = await _execute_research_pipeline(job=job, **kwargs)
        job.result = res
        job.completed_at = runtime._clock()
        return res
    except asyncio.CancelledError:
        job.completed_at = runtime._clock()
        raise
    except Exception as exc:
        job.exception = exc
        job.completed_at = runtime._clock()
        job.result = _fail("internal_error", redact(str(exc)))
        return job.result


async def _condense_report(
    report: str,
    *,
    depth: str,
    allowed_urls: set[str],
    max_id: int,
    deadline_at: float,
) -> tuple[str, float, bool, list[str]]:
    """Shorten an over-length report once; return (report, cost, call_made, notes).

    The condensed text goes through the same output filter and citation normalization as the original
    and is accepted only when it is non-empty, shorter and cites no id the original did not cite.
    """
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    words = report_body_words(report)
    if words <= cfg.max_words:
        return report, 0.0, False, []
    remaining = deadline_at - runtime._clock()
    if remaining < 8.0:
        return report, 0.0, False, ["Report exceeds the word ceiling; condensing skipped (deadline)."]
    res = await runtime._chat_request(
        messages=build_condense_messages(report, depth),
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0.0,
        max_tokens=min(cfg.max_tokens, cfg.target_words * 4),
        timeout=max(5.0, remaining - 2.0),
        deadline=max(5.0, remaining - 2.0),
        attempts=1,
    )
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    kept = f"Report had {words} words (ceiling {cfg.max_words}); condensing was not usable, original kept."
    if not res.get("ok") or not (res.get("content") or "").strip():
        return report, cost, True, [kept]
    candidate, _, _ = filter_synthesis_output(res["content"], allowed_urls)
    candidate = normalize_citations(candidate, max_id=max_id)
    if (
        not candidate.strip()
        or report_body_words(candidate) >= words
        or not citation_ids(res["content"]) <= citation_ids(report)
        or not citation_ids(candidate) <= citation_ids(report)
    ):
        return report, cost, True, [kept]
    return (
        candidate,
        cost,
        True,
        [
            f"Report condensed from {words} to {report_body_words(candidate)} words "
            f"(target {cfg.target_words})."
        ],
    )


def _maybe_dump_research_job(
    *,
    job_id: str,
    topic: str,
    depth: str,
    synthesis_context: list[dict[str, Any]],
    raw_model_report: str,
    final_report: str,
    unverified_numbers: list[dict[str, str]],
) -> None:
    """Optionally dump research job synthesis inputs and outputs for offline debugging/audits."""
    dump_dir = runtime.get_settings().debug_dump_dir or os.environ.get(ENV_DEBUG_DUMP_DIR, "").strip()
    if not dump_dir:
        return
    try:
        p = Path(dump_dir)
        p.mkdir(parents=True, exist_ok=True)
        dump_file = p / f"{job_id}.json"
        synthesis_context_dump = [
            {
                "id": s.get("id"),
                "url": s.get("url"),
                "title": s.get("title"),
                "content": s.get("content"),
                "doi": s.get("doi"),
                "pmid": s.get("pmid"),
                "pmcid": s.get("pmcid"),
                "access": s.get("access"),
                "source_type": s.get("source_type"),
            }
            for s in synthesis_context
        ]
        dump_payload = {
            "job_id": job_id,
            "topic": topic,
            "depth": depth,
            "sources": _format_public_sources(synthesis_context, include_snippet=False),
            "synthesis_context": synthesis_context_dump,
            "raw_model_report": raw_model_report,
            "raw_report": raw_model_report,
            "final_report": final_report,
            "report": final_report,
            "unverified_numbers": unverified_numbers,
        }
        dump_file.write_text(json.dumps(dump_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write research debug dump for job %s to %s: %s", job_id, dump_dir, exc)


# Seconds of the depth deadline kept free for the synthesis call while the earlier phases run.
_SYNTHESIS_RESERVE_S: Final[dict[str, float]] = {"low": 20.0, "medium": 30.0, "high": 45.0, "max": 60.0}


@dataclass
class _ResearchRun:
    """State shared by the phases of one research job: inputs, deadline, costs, counters and notes."""

    job: ResearchJob
    topic: str
    depth: str
    sources_pref: str
    lang: str | None
    refresh: bool
    strip_boilerplate: bool
    cfg: Any
    deadline_at: float
    synthesis_reserve: float
    total_steps: int
    subqueries: list[str] = field(default_factory=list)
    result_lists: list[list[dict[str, Any]]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    search_cost: float = 0.0
    fetch_cost: float = 0.0
    llm_cost: float = 0.0
    requests_used: int = 0
    cached_count: int = 0
    fetch_failed: int = 0

    def remaining(self) -> float:
        """Seconds left before the job deadline."""
        return self.deadline_at - runtime._clock()

    def phase_timeout(self, cap: float) -> float:
        """Timeout for a phase: the time left after the synthesis reserve, between 5 s and ``cap``."""
        return min(cap, max(5.0, self.remaining() - self.synthesis_reserve))

    def add_searches(self, results: list[dict[str, Any]]) -> None:
        """Count the requests, cache hits and cost of successful search envelopes."""
        self.cached_count += sum(r.get("cached_count", 0) for r in results)
        self.requests_used += sum(r.get("requests_used", 0) for r in results)
        self.search_cost += sum(r.get("cost_usd", 0.0) for r in results)

    def envelope(
        self,
        *,
        report: str | None,
        sources: list[dict[str, Any]],
        unverified_numbers: list[dict[str, str]],
        unverified_count: int,
        partial: bool,
        requests_used: int,
        llm_cost: float,
        injection_suspected: int,
        notes: list[str],
    ) -> dict[str, Any]:
        """Build a finished research envelope with the run's cost and cache accounting."""
        total_llm = round(llm_cost, 6)
        total_cost = round(self.search_cost + self.fetch_cost + total_llm, 6)
        return {
            "ok": True,
            "status": "done",
            "topic": self.topic,
            "depth": self.depth,
            "subqueries": self.subqueries,
            "report": report,
            "synthesized": report is not None,
            "sources": sources,
            "unverified_numbers": unverified_numbers,
            "unverified_number_count": unverified_count,
            "partial": partial,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, self.cached_count),
            "cached_count": self.cached_count,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(self.search_cost, 6),
                "fetch": round(self.fetch_cost, 6),
                "llm": total_llm,
            },
            "untrusted_content": True,
            "derived_from_untrusted": True,
            "injection_suspected": injection_suspected,
            "notes": notes,
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }

    def unsynthesized(
        self, note: str, *, requests_used: int | None = None, llm_cost: float | None = None
    ) -> dict[str, Any]:
        """Envelope returned when no report is written: the gathered sources with a snippet each."""
        return self.envelope(
            report=None,
            sources=_format_public_sources(self.sources, include_snippet=True),
            unverified_numbers=[],
            unverified_count=0,
            partial=True,
            requests_used=self.requests_used if requests_used is None else requests_used,
            llm_cost=self.llm_cost if llm_cost is None else llm_cost,
            injection_suspected=0,
            notes=self.notes + [note],
        )


def _research_search_calls(run: _ResearchRun, queries: list[str], timeout: float) -> list[Awaitable[Any]]:
    """Build one full-content search per query, each bounded by ``timeout``."""
    cfg = run.cfg
    return [
        asyncio.wait_for(
            search_core._search(
                query,
                cfg.count,
                None,
                "full",
                run.lang,
                cfg.per_page_char_cap,
                cfg.per_page_char_cap * cfg.count,
                True,
                False,
                refresh=run.refresh,
                strip_boilerplate=run.strip_boilerplate,
            ),
            timeout=timeout,
        )
        for query in queries
    ]


async def _research_decompose(run: _ResearchRun) -> None:
    """Phase 1: split the topic into subqueries with deepseek-flash (the topic itself on failure)."""
    await _report_job_progress(
        run.job, "decomposition", 1, run.total_steps, "Decomposing topic into subqueries"
    )
    timeout = run.phase_timeout(20.0)
    res = await runtime._chat_request(
        messages=build_decomposition_messages(
            run.topic, max_queries=run.cfg.subqueries, sources=run.sources_pref
        ),
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        timeout=timeout,
        deadline=timeout,
        attempts=2,
    )
    if res.get("ok"):
        run.llm_cost += res.get("cost_usd", 0.0)
    if res.get("ok") and res.get("content"):
        run.subqueries = parse_subqueries(res["content"], run.topic, max_queries=run.cfg.subqueries)
    else:
        fallback, _ = fit_query(run.topic)
        run.subqueries = [fallback]


async def _research_retrieve(run: _ResearchRun) -> dict[str, Any] | None:
    """Phase 2: search every subquery; return the first error envelope when all of them fail."""
    await _report_job_progress(run.job, "retrieval", 2, run.total_steps, "Searching across subqueries")
    results = await asyncio.gather(*_research_search_calls(run, run.subqueries, run.phase_timeout(30.0)))
    successful = [r for r in results if r.get("ok")]
    if not successful:
        return results[0]
    run.result_lists = [r.get("results", []) for r in successful]
    run.add_searches(successful)
    return None


async def _research_round_two(run: _ResearchRun) -> None:
    """Round 2 (depth max): ask for gap queries over the round-1 results and search them too."""
    await _report_job_progress(
        run.job, "round_2_retrieval", 3, run.total_steps, "Analyzing gaps and searching additional angles"
    )
    budget = run.remaining() - run.synthesis_reserve
    if budget < 10.0:
        run.notes.append("Round 2 skipped due to deadline budget.")
        return
    r1_items = [item for results in run.result_lists for item in results]
    r1_domains = {d for item in r1_items if (d := _url_etld1(item.get("url") or ""))}
    timeout = min(25.0, max(5.0, budget / 2))
    try:
        res = await runtime._chat_request(
            messages=build_r2_gap_messages(run.topic, r1_items, run.subqueries),
            thinking={"type": "disabled"},
            model=REWRITE_MODEL,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            timeout=timeout,
            deadline=timeout,
            attempts=1,
        )
        if not (res.get("ok") and res.get("content")):
            run.notes.append("Round 2 gap analysis failed; proceeding with Round 1 sources.")
            return
        run.llm_cost += res.get("cost_usd", 0.0)
        parsed = parse_subqueries(res["content"], run.topic, max_queries=3)
        queries = validate_r2_queries(parsed, run.topic, run.subqueries, r1_domains, lang=run.lang)
        if not queries:
            run.notes.append("Round 2 gap query validation failed; proceeding with Round 1 sources.")
            return
        search_timeout = min(20.0, max(5.0, run.remaining() - run.synthesis_reserve))
        results = await asyncio.gather(
            *_research_search_calls(run, queries, search_timeout), return_exceptions=True
        )
        successful = [r for r in results if isinstance(r, dict) and r.get("ok")]
        if not successful:
            run.notes.append("Round 2 search produced no results; proceeding with Round 1 sources.")
            return
        for query, res_dict in zip(queries, results, strict=True):
            if isinstance(res_dict, dict) and res_dict.get("ok"):
                run.subqueries.append(query)
                run.result_lists.append(res_dict.get("results", []))
        run.add_searches(successful)
    except Exception as exc:
        logger.debug("Round 2 gap analysis exception: %s", exc)
        run.notes.append("Round 2 gap analysis failed; proceeding with Round 1 sources.")


async def _research_rank_and_fill(run: _ResearchRun) -> None:
    """Phases 3-4: rank and diversify the results into sources, then fetch pages with short content."""
    selected, rank_notes = rank_and_diversify_sources(
        run.result_lists,
        k_sources=run.cfg.k_sources,
        denylist=runtime.get_settings().source_denylist,
        sources=run.sources_pref,
    )
    run.notes.extend(rank_notes)
    run.sources = [
        _research_source(idx, item, item.get("cached", False)) for idx, item in enumerate(selected, 1)
    ]

    gap_step = 4 if run.depth == "max" else 3
    await _report_job_progress(
        run.job, "gap_fill", gap_step, run.total_steps, "Fetching gap pages for top sources"
    )
    fill = await _fill_gaps(
        run.sources,
        run.cfg.per_page_char_cap,
        run.cfg.total_source_chars,
        run.refresh,
        run.strip_boilerplate,
        deadline=run.phase_timeout(30.0),
        max_pages=run.cfg.k_sources,
        share_total=False,
    )
    run.requests_used += fill.requests_used
    run.cached_count += fill.cached_count
    run.fetch_cost = fill.requests_used * COST_PER_REQUEST
    run.fetch_failed = fill.failed
    if fill.fetched:
        run.notes.append(f"fetched {fill.fetched} page(s) with missing or short search content")
    if fill.failed:
        run.notes.append(f"{fill.failed} fetch(es) failed; see the source's error")


def _research_cost_guard(run: _ResearchRun, context: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the sources-only envelope when synthesis would break the cost ceiling or the deadline."""
    total_source_chars = sum(len(s.get("content") or "") for s in context)
    pricing = runtime.get_settings().llm_pricing.peak
    est_synth_cost = ((total_source_chars // 2) / 1_000_000) * pricing.cache_miss_per_1m + (
        run.cfg.max_tokens / 1_000_000
    ) * pricing.output_per_1m
    est_total = run.search_cost + run.fetch_cost + run.llm_cost + est_synth_cost
    ceiling = runtime.get_settings().research_cost_ceiling_usd
    if est_total > ceiling:
        return run.unsynthesized(
            f"Estimated research cost (${est_total:.4f}) exceeds ceiling "
            f"of ${ceiling:.2f}; returning gathered sources without synthesis."
        )
    if run.remaining() < 5.0:
        return run.unsynthesized("Synthesis deadline reached; returning gathered sources.")
    return None


async def _research_synthesize(
    run: _ResearchRun, context: list[dict[str, Any]], synth_step: int
) -> tuple[dict[str, Any], float]:
    """Write the report with deepseek-flash, retrying once without thinking when the first answer is
    cut off or empty. Return the answer used and the cost of an attempt that was billed but not used."""
    cfg = run.cfg
    messages = build_synthesis_messages(run.topic, context, depth=run.depth, nonce=uuid.uuid4().hex[:12])
    timeout = max(10.0, run.remaining() - 2.0)
    kwargs: dict[str, Any] = {
        "messages": messages,
        "model": REWRITE_MODEL,
        "max_tokens": cfg.max_tokens,
        "timeout": timeout,
        "deadline": timeout,
        "attempts": 1,
    }
    if cfg.thinking:
        kwargs["thinking"] = {"type": "enabled"}
        if cfg.reasoning_effort:
            kwargs["reasoning_effort"] = cfg.reasoning_effort
    else:
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["temperature"] = 0.0
    res = await runtime._chat_request(**kwargs)

    needs_retry = (
        not res.get("ok") or res.get("finish_reason") == "length" or not (res.get("content") or "").strip()
    )
    rem_retry = run.remaining()
    if not needs_retry or rem_retry < 8.0:
        return res, 0.0
    run.notes.append("Synthesis initial attempt truncated or empty; retrying with thinking disabled.")
    await _report_job_progress(
        run.job,
        "synthesis",
        synth_step,
        run.total_steps,
        "Retrying synthesis with thinking disabled",
        detail="retry",
    )
    retry = await runtime._chat_request(
        messages=messages,
        thinking={"type": "disabled"},
        temperature=0.0,
        model=REWRITE_MODEL,
        max_tokens=min(cfg.max_tokens, 16_384),
        timeout=max(5.0, rem_retry - 2.0),
        deadline=max(5.0, rem_retry - 2.0),
        attempts=1,
    )
    # Every attempt is billed, so the attempt that is not used still counts toward the cost.
    if retry.get("ok") and (retry.get("content") or "").strip() and retry.get("finish_reason") != "length":
        return retry, float(res.get("cost_usd", 0.0) or 0.0)
    return res, float(retry.get("cost_usd", 0.0) or 0.0)


async def _finalize_report(
    run: _ResearchRun,
    context: list[dict[str, Any]],
    report_text: str,
    *,
    requests_used: int,
    llm_cost: float,
    synth_step: int,
    condense: bool = True,
) -> dict[str, Any]:
    """Filter, normalize, condense, sanitize and ground a written report, then build the envelope.

    Without ``condense`` (a report the client wrote) an over-length report is only noted, never
    rewritten.
    """
    allowed_urls = {s["url"].strip() for s in context if isinstance(s.get("url"), str) and s["url"].strip()}
    filtered, injection_suspected, filter_notes = filter_synthesis_output(report_text, allowed_urls)
    report = normalize_citations(filtered, max_id=len(context))
    words = report_body_words(report)
    condense_cost, condense_called, condense_notes = 0.0, False, []
    if condense:
        if words > run.cfg.max_words:
            await _report_job_progress(
                run.job,
                "synthesis",
                synth_step,
                run.total_steps,
                "Condensing an over-length report",
                detail="condense",
            )
        report, condense_cost, condense_called, condense_notes = await _condense_report(
            report,
            depth=run.depth,
            allowed_urls=allowed_urls,
            max_id=len(context),
            deadline_at=run.deadline_at,
        )
    elif words > run.cfg.max_words:
        condense_notes = [
            f"Report has {words} words, over the {run.cfg.max_words}-word ceiling for depth "
            f"{run.depth!r}; it was kept as written."
        ]
    report, sanitize_notes = sanitize_untrusted_text(report)
    report, unverified_numbers, unverified_count, unverified_notes = ground_report_numbers(
        report, context, drop=run.job.unverified == "drop"
    )
    run.notes.extend(filter_notes)
    run.notes.extend(condense_notes)
    run.notes.extend(sanitize_notes)
    run.notes.extend(unverified_notes)
    if condense_called:
        requests_used += 1

    _maybe_dump_research_job(
        job_id=run.job.job_id,
        topic=run.topic,
        depth=run.depth,
        synthesis_context=context,
        raw_model_report=report_text,
        final_report=report,
        unverified_numbers=unverified_numbers,
    )
    return run.envelope(
        report=report,
        sources=_format_public_sources(context, include_snippet=False),
        unverified_numbers=unverified_numbers,
        unverified_count=unverified_count,
        partial=run.fetch_failed > 0,
        requests_used=requests_used,
        llm_cost=round(llm_cost, 6) + condense_cost,
        injection_suspected=injection_suspected,
        notes=run.notes,
    )


# Longest report a client may submit for verification.
MAX_CLIENT_REPORT_CHARS = 60_000


def _awaiting_report_envelope(job: ResearchJob, page: int, *, charge: bool = False) -> dict[str, Any]:
    """Return one page of a client-synthesis job's sources (``status: "awaiting_report"``).

    Page 1 carries the synthesis instructions; only its first delivery (``charge``) reports the cost
    of gathering the sources, every other page or repeat costs nothing. ``next_action`` points to the
    next page, then to submitting the report.
    """
    run: _ResearchRun = job.client_run
    entries, pages = paginate_context(job.synthesis_context or [], page)
    if page < pages:
        next_action = (
            f"Read page {page + 1} of {pages} with deepseek_check_research(job_id, page={page + 1}), then "
            "write the report following synthesis_instructions and send it with "
            "deepseek_check_research(job_id, report=...)."
        )
    else:
        next_action = (
            "Write the report following synthesis_instructions (cite sources as [id]) and send it with "
            "deepseek_check_research(job_id, report=...); the server verifies citations and numbers."
        )
    first = page == 1
    requests_used = run.requests_used if charge else 0
    llm_cost = round(run.llm_cost, 6) if charge else 0.0
    search_cost = run.search_cost if charge else 0.0
    fetch_cost = run.fetch_cost if charge else 0.0
    total_cost = round(search_cost + fetch_cost + llm_cost, 6)
    envelope: dict[str, Any] = {
        "ok": True,
        "status": "awaiting_report",
        "job_id": job.job_id,
        "topic": run.topic,
        "depth": run.depth,
        "subqueries": run.subqueries,
    }
    if first:
        envelope["synthesis_instructions"] = build_synthesis_system_prompt(run.depth, client=True)
    envelope.update(
        {
            "context_page": page,
            "context_pages": pages,
            "context": entries,
            "next_action": next_action,
            "requests_used": requests_used,
            "cached": _all_cached(requests_used, run.cached_count) if charge else True,
            "cached_count": run.cached_count if charge else 0,
            "cost_usd": total_cost,
            "cost_breakdown": {
                "search": round(search_cost, 6),
                "fetch": round(fetch_cost, 6),
                "llm": llm_cost,
            },
            "untrusted_content": True,
            # Round-2 subqueries (depth max) were written by the model from untrusted search results.
            "derived_from_untrusted": True,
            "notes": list(run.notes),
            "_meta": _envelope_meta(total_cost, untrusted_content=True),
        }
    )
    return envelope


def _client_page(job: ResearchJob, page: int) -> dict[str, Any]:
    """Return ``page`` of a client-synthesis job's sources, or ``bad_request`` when it does not exist.

    The first delivery of page 1 reports the gathering cost; every later page or repeat is free.
    """
    _, pages = paginate_context(job.synthesis_context or [], page)
    if not 1 <= page <= pages:
        return _fail("bad_request", f"page must be between 1 and {pages} for this job.")
    if page == 1 and not job.client_charged and job.client_sources_result is not None:
        job.client_charged = True
        res = dict(job.client_sources_result)
        res["delivery"] = "polled"
        res["elapsed_ms"] = _job_elapsed_ms(job)
        return res
    return _awaiting_report_envelope(job, page)


async def _verify_client_report(job: ResearchJob, report: str) -> dict[str, Any]:
    """Verify a report written by the client for an awaiting job and store it as the job's result.

    The report goes through the same output filter, citation normalization, sanitizer and number
    grounding as a server report, but is never condensed. It may be sent again; the latest one wins.
    """
    if job.synthesis != "client":
        return _fail(
            "bad_request",
            "This job was synthesized by the server; report= applies only to synthesis='client' jobs.",
        )
    if job.exception is not None:
        return _finished_job_envelope(job, "polled")
    if job.task is None or not job.task.done():
        return _fail(
            "bad_request",
            "The sources are not ready yet. Poll with deepseek_check_research(job_id) until status is "
            "awaiting_report, then send the report.",
        )
    if job.synthesis_context is None or job.client_run is None:
        return _fail("bad_request", "This job has no sources to verify a report against.")
    text = report.strip() if isinstance(report, str) else ""
    if not text:
        return _fail("bad_request", "report must not be empty.")
    if len(text) > MAX_CLIENT_REPORT_CHARS:
        return _fail("bad_request", f"report must be at most {MAX_CLIENT_REPORT_CHARS} characters.")

    base: _ResearchRun = job.client_run
    run = dataclasses.replace(base, notes=list(base.notes))  # each submission starts from the same notes
    result = await _finalize_report(
        run,
        job.synthesis_context,
        text,
        requests_used=run.requests_used,
        llm_cost=run.llm_cost,
        synth_step=run.total_steps,
        condense=False,
    )
    # Verifying makes no upstream call; the gathering cost was reported with the sources.
    result.update(
        {
            "delivery": "client",
            "requests_used": 0,
            "cached": True,
            "cost_usd": 0,
            "cost_breakdown": {"search": 0, "fetch": 0, "llm": 0},
            "_meta": _envelope_meta(0, untrusted_content=True),
        }
    )
    job.result = result
    job.completed_at = runtime._clock()
    return dict(result)


async def _execute_research_pipeline(
    *,
    job: ResearchJob,
    clean_topic: str,
    depth: str,
    sources_pref: str = "any",
    lang: str | None,
    refresh: bool,
    strip_boilerplate: bool,
) -> dict[str, Any]:
    """Run one research job: decompose, search (plus round 2 at depth max), rank, fill gaps, select the
    synthesis context, write the report and verify it."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    client = job.synthesis == "client"
    run = _ResearchRun(
        job=job,
        topic=clean_topic,
        depth=depth,
        sources_pref=sources_pref,
        lang=lang,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
        cfg=cfg,
        deadline_at=runtime._clock() + cfg.deadline_s,
        synthesis_reserve=0.0 if client else _SYNTHESIS_RESERVE_S.get(depth, 30.0),
        total_steps=5 if depth == "max" else 4,
    )

    await _research_decompose(run)
    failure = await _research_retrieve(run)
    if failure is not None:
        return failure
    if depth == "max":
        await _research_round_two(run)
    await _research_rank_and_fill(run)

    synth_step = 5 if depth == "max" else 4
    if client:
        await _report_job_progress(
            job, "context", synth_step, run.total_steps, "Preparing sources for client synthesis"
        )
    else:
        await _report_job_progress(
            job, "synthesis", synth_step, run.total_steps, "Synthesizing cited research report"
        )
    context = select_synthesis_context(run.sources, depth=depth, sources_pref=sources_pref, notes=run.notes)
    context, guard_notes = apply_token_guard(context)
    run.notes.extend(guard_notes)
    if not context:
        return run.unsynthesized("No readable source content available for synthesis.")
    if client:
        job.synthesis_context = context
        job.client_run = run
        job.client_sources_result = _awaiting_report_envelope(job, 1, charge=True)
        return job.client_sources_result
    if (stopped := _research_cost_guard(run, context)) is not None:
        return stopped

    synth_res, discarded_cost = await _research_synthesize(run, context, synth_step)
    synth_cost = synth_res.get("cost_usd", 0.0) if synth_res.get("ok") else 0.0
    llm_cost = run.llm_cost + synth_cost + discarded_cost
    if not synth_res.get("ok") or not (synth_res.get("content") or "").strip():
        message = synth_res.get("error", {}).get("message", "unknown")
        return run.unsynthesized(
            f"Synthesis model call failed: {message}; returning gathered sources.",
            requests_used=run.requests_used + 1,
            llm_cost=llm_cost,
        )
    return await _finalize_report(
        run,
        context,
        synth_res.get("content", ""),
        requests_used=run.requests_used + 1,
        llm_cost=llm_cost,
        synth_step=synth_step,
    )
