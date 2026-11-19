"""Agentic research: plan, search, measure what each round added, and stop when new rounds stop paying.

The differences from ``deepseek_research`` are the ones the contract names: the tier decides how much the
writer may say, the rounds are a separate axis, and the loop stops on evidence rather than on a fixed
count. Everything downstream of retrieval (ranking, gap filling, context selection, synthesis,
verification) is the research pipeline, unchanged, so a report from the agent is verified the same way.

See ``plans/kontrakt-agent-deep-research.md`` for the decisions this implements.
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import evidence, runtime
from .deep_research import DEPTH_CONFIG, ResearchJob, apply_token_guard, select_synthesis_context
from .filters import fit_query
from .research_jobs import (
    _finalize_report,
    _report_job_progress,
    _research_cost_guard,
    _research_decompose,
    _research_rank_and_fill,
    _research_synthesize,
    _ResearchRun,
    _stage,
)

MAX_ROUNDS = 3


async def run_agent_job(job: ResearchJob, **kwargs: Any) -> dict[str, Any]:
    """Run the agent pipeline for one job and keep the result on the job, like the research runner."""
    from .errors import _fail
    from .redact import redact
    from .research_jobs import _maybe_dump_research_job

    try:
        res = await run_agent(job, **kwargs)
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
        if job.run is not None:
            _maybe_dump_research_job(run=job.run, error=redact(str(exc)))
        return job.result


SEARCH_COST_USD = 0.00075
ESTIMATED_SYNTHESIS_USD = {"low": 0.0035, "medium": 0.0081, "high": 0.0243, "max": 0.0337}
ESTIMATE_MARGIN = 1.2


def estimate_cost(depth: str, max_rounds: int, *, sources_pref: str = "any") -> float:
    """A rough pre-run cost: the searches the loop may make plus the measured synthesis cost.

    The synthesis figure comes from the measurements in `E09`, not from the public price table, because
    reasoning tokens dominate it and are not visible in the token counts. It is a warning, not a promise:
    the hard stop is ``max_cost_usd``.
    """
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    searches = cfg.subqueries + 3 * (max_rounds - 1)
    round_cost = searches * SEARCH_COST_USD
    return round(ESTIMATE_MARGIN * (round_cost + ESTIMATED_SYNTHESIS_USD.get(depth, 0.01)), 4)


def _result_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten search results into the shape the evidence matcher reads."""
    return [
        {"id": item.get("url") or f"page-{index}", "content": item.get("content") or ""}
        for index, item in enumerate(results, 1)
    ]


def _gap_queries(
    aspects: list[Any], blocks: list[dict[str, Any]], used: set[str], *, topic: str, max_queries: int = 3
) -> list[str]:
    """Reformulated queries for the aspects that still have no evidence.

    Reusing the aspect's own query would repeat a search that has already run, so a gap query is built
    from what the aspect wants to learn plus a word that asks for a different kind of page. It is dropped
    when the result would repeat a query already sent.
    """
    queries: list[str] = []
    for aspect, block in zip(aspects, blocks, strict=False):
        if block["sentences"]:
            continue
        wanted = (getattr(aspect, "looking_for", "") or "").strip()
        name = (getattr(aspect, "aspect", "") or "").strip()
        for variant in (f"{wanted} data {topic}", f"{name} figures {topic}", f"{name} report {topic}"):
            candidate, _ = fit_query(" ".join(variant.split()))
            if candidate and candidate not in used and candidate not in queries:
                queries.append(candidate)
                break
    return queries[:max_queries]


async def _search_round(run: _ResearchRun, queries: list[str]) -> list[dict[str, Any]]:
    """Run one search round and record it on the run; returns the new result items."""
    from .research_jobs import _research_search_calls

    timeout = run.phase_timeout(20.0)
    raw = await asyncio.gather(*_research_search_calls(run, queries, timeout), return_exceptions=True)
    collected: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        run.result_lists.append(item.get("results", []))
        collected.extend(item.get("results", []))
        run.add_searches([item])
    return collected


async def run_agent(
    job: ResearchJob,
    *,
    clean_topic: str,
    depth: str,
    sources_pref: str = "any",
    lang: str | None,
    refresh: bool,
    strip_boilerplate: bool,
) -> dict[str, Any]:
    """Run an agent job and return its envelope, with the plan, the rounds and the stop reason."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["medium"])
    max_rounds = max(1, int(getattr(job, "max_rounds", MAX_ROUNDS)))
    estimate = estimate_cost(depth, max_rounds, sources_pref=sources_pref)
    budget = job.max_cost_usd
    if budget is not None and estimate > budget:
        return job.result or {
            "ok": False,
            "error": {
                "code": "budget_exceeded",
                "message": f"estimated ${estimate:.4f} exceeds the ${budget:.4f} budget",
            },
            "estimate_usd": estimate,
        }

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
        synthesis_reserve=70.0 if depth == "medium" else 90.0,
        total_steps=2 + max_rounds + 1,
    )
    job.run = run

    await _research_decompose(run)
    plan = [{"aspect": a.aspect, "query": a.query} for a in run.aspects]
    used = {query for query in run.subqueries if query}
    await _report_job_progress(job, "retrieval", 2, run.total_steps, "Searching the plan's queries")
    with _stage(run, "round") as counters:
        results = await _search_round(run, run.subqueries)
        counters["queries"] = len(run.subqueries)
        counters["results"] = len(results)

    rounds: list[dict[str, Any]] = []
    stopped_reason = "max_rounds"
    blocks = evidence.evidence_matrix(run.aspects, _result_sources(results))
    previous = [dict(block) for block in evidence.evidence_matrix(run.aspects, [])]
    seen_sources = {item.get("url") for item in results}
    gain = evidence.round_gain(previous, blocks, new_sources=len(seen_sources))
    rounds.append(gain)

    for index in range(2, max_rounds + 1):
        if gain["saturated"]:
            stopped_reason = "saturated"
            break
        spent = run.search_cost + run.fetch_cost + run.llm_cost
        if budget is not None and spent + estimate_cost(depth, 1) > budget:
            stopped_reason = "budget"
            break
        queries = _gap_queries(run.aspects, blocks, used, topic=run.topic, max_queries=3)
        if not queries:
            stopped_reason = "no_gaps"
            break
        used.update(queries)
        await _report_job_progress(
            job, "retrieval", 1 + index, run.total_steps, f"Round {index}: searching the aspects still empty"
        )
        with _stage(run, f"round_{index}") as counters:
            fresh = await _search_round(run, queries)
            counters["queries"] = len(queries)
            counters["results"] = len(fresh)
            counters["new_sources"] = len({item.get("url") for item in fresh} - seen_sources)
        seen_sources |= {item.get("url") for item in fresh}
        previous, blocks = blocks, evidence.evidence_matrix(run.aspects, _result_sources(results + fresh))
        gain = evidence.round_gain(previous, blocks, new_sources=counters["new_sources"])
        rounds.append(gain)
        results.extend(fresh)

    await _research_rank_and_fill(run)
    synth_step = run.total_steps
    await _report_job_progress(job, "synthesis", synth_step, run.total_steps, "Synthesizing cited report")
    with _stage(run, "select") as counters:
        context = select_synthesis_context(
            run.sources, depth=depth, sources_pref=sources_pref, notes=run.notes, lang=lang, topic=run.topic
        )
        context, guard_notes = apply_token_guard(context)
        run.notes.extend(guard_notes)
        counters["context_chars"] = sum(len(c.get("content") or "") for c in context)
    if not context:
        return run.unsynthesized("No readable source content available for synthesis.")
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
    envelope = await _finalize_report(
        run,
        context,
        synth_res.get("content", ""),
        requests_used=run.requests_used + 1,
        llm_cost=llm_cost,
        synth_step=synth_step,
    )
    envelope["plan"] = plan
    envelope["rounds"] = rounds
    envelope["facts_per_round"] = [gain["new_facts"] for gain in rounds]
    envelope["estimate_usd"] = estimate
    envelope["stopped_reason"] = stopped_reason
    return envelope
