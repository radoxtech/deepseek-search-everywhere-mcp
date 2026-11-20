"""Agentic research: plan, search, measure what each round added, and stop when new rounds stop paying.

The differences from ``deepseek_research`` are the ones the contract names: the tier decides how much the
writer may say, the rounds are a separate axis, and the loop stops on evidence rather than on a fixed
count. Everything downstream of retrieval (ranking, gap filling, context selection, synthesis,
verification) is the research pipeline, unchanged, so a report from the agent is verified the same way.

See ``plans/kontrakt-agent-deep-research.md`` for the decisions this implements.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

from . import evidence, runtime
from .deep_research import (
    DEPTH_CONFIG,
    ResearchJob,
    apply_token_guard,
    sanitize_json,
    select_synthesis_context,
)
from .errors import _fail
from .filters import fit_query
from .research_jobs import (
    REWRITE_MODEL,
    _finalize_report,
    _report_job_progress,
    _research_cost_guard,
    _research_decompose,
    _research_rank_and_fill,
    _research_synthesize,
    _ResearchRun,
    _stage,
)
from .snippets import prune_source_content

MAX_ROUNDS = 5


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
    """Reformulated queries for the aspects that still have no evidence."""
    queries: list[str] = []
    for aspect, block in zip(aspects, blocks, strict=False):
        if block["sentences"]:
            continue
        wanted = (getattr(aspect, "looking_for", "") or "").strip()
        name = (getattr(aspect, "aspect", "") or "").strip()
        for variant in (f"{wanted} {topic}", f"{name} {wanted}", f"{name} {topic}"):
            candidate, _ = fit_query(" ".join(variant.split()))
            if candidate and candidate not in used and candidate not in queries:
                queries.append(candidate)
                break
    return queries[:max_queries]


UNIVERSAL_GAP_PROMPT = (
    "You are an expert research gap detector. Given the research topic, executed queries, and "
    "preliminary findings:\n"
    "1. Identify what critical factual dimensions (such as named primary studies/standards/statutes/models, "
    "specific quantitative metrics/rates/thresholds/costs, failure modes/degradation mechanisms, or "
    "comparative counter-evidence) are STILL MISSING from the findings.\n"
    "2. Generate up to 3 distinct specialist keyword search queries (3 to 8 keywords each) "
    "targeting these missing dimensions.\n"
    "CRITICAL: Do NOT repeat or slightly rephrase the executed queries; explore specific entity, "
    "case or material names, parameter/mechanism names, or measured outcome metrics.\n"
    "Output strictly valid JSON with 'queries': [str]."
)


async def _gap_queries_model(
    run: _ResearchRun,
    blocks: list[dict[str, Any]],
    used: set[str],
    results: list[dict[str, Any]],
    max_queries: int = 3,
) -> list[str]:
    """Find search queries for aspects lacking evidence using deepseek-flash, with fallback."""
    empty_aspects = [
        getattr(a, "aspect", "") for a, b in zip(run.aspects, blocks, strict=False) if not b.get("sentences")
    ]
    if not empty_aspects:
        empty_aspects = [
            getattr(a, "aspect", "")
            for a, b in zip(run.aspects, blocks, strict=False)
            if len(b.get("numbers") or []) < 2
        ]

    summary_lines = [
        f"- {s.get('title')}: {(s.get('content') or '')[:250]}" for s in results[:12] if s.get("title")
    ]
    summary_text = "\n".join(summary_lines)
    aspect_context = (
        f"Aspects still lacking clear evidence: {', '.join(empty_aspects)}\n"
        if empty_aspects
        else "Aspects partially covered; seek missing critical perspectives, metrics, or disputes.\n"
    )
    user_prompt = (
        f"Research topic: {run.topic}\n"
        f"{aspect_context}"
        f"Executed queries: {', '.join(used)}\n"
        f"Preliminary findings summary:\n{summary_text}\n\n"
        f"Output up to {max_queries} new search queries as JSON targeting the missing evidence."
    )
    timeout = min(15.0, max(5.0, run.remaining() - 20.0))
    try:
        res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": UNIVERSAL_GAP_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            thinking={"type": "disabled"},
            temperature=0.0,
            model=REWRITE_MODEL,
            max_tokens=1024,
            response_format={"type": "json_object"},
            timeout=timeout,
            deadline=run.deadline_at,
            attempts=1,
        )
        if res.get("ok"):
            run.llm_cost += float(res.get("cost_usd", 0.0) or 0.0)
            parsed, _ = sanitize_json(res.get("content") or "{}")
            queries = parsed.get("queries") if isinstance(parsed, dict) else []
            clean_queries: list[str] = []
            for q in queries:
                if isinstance(q, str) and q.strip():
                    fitted, _ = fit_query(q.strip())
                    if fitted and fitted not in used and fitted not in clean_queries:
                        clean_queries.append(fitted)
            if clean_queries:
                return clean_queries[:max_queries]
    except Exception:
        pass

    return _gap_queries(run.aspects, blocks, used, topic=run.topic, max_queries=max_queries)


REPORT_ASK = (
    "Write only this section, about {words} words. Start with the section's first sentence of prose; do "
    "not write a heading, the system adds it. Use only the sources given. Every number carries the [n] "
    "citation of the source that states it. Preserve verbatim numerical rates, percentages and dates rather "
    "than generalizing into descriptive words. Do not spend more than two sentences or one table on a "
    "single product code, formula, or worked numerical example; balance the section across all diverse "
    "substantive perspectives, limits, and disputes supported by the sources. If the sources say nothing "
    "usable for this section, write one sentence saying so."
)


def _plan_fact(aspect: Any) -> str:
    looking = (getattr(aspect, "looking_for", "") or "").strip()
    name = (getattr(aspect, "aspect", "") or "").strip()
    return looking or name or "the question as asked"


def _source_digest(content: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", content) if len(p.strip()) > 40]
    scored = sorted(parts, key=lambda p: -sum(ch.isdigit() for ch in p))[:3]
    return " ".join(scored)[:450] or content[:300]


def _strip_heading(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
        lines.pop(0)
    return "\n".join(lines).strip()


def _gaps_sentence(unanswered: list[str], context: list[dict[str, Any]]) -> str:
    if not unanswered:
        return (
            "The gathered sources cover every sub-question the report set out to answer "
            f"({len(context)} sources)."
        )
    return " ".join(f"No source found that answers: {u}." for u in unanswered[:6])


def _search_note(run: Any) -> str:
    queries = [q for q in (getattr(run, "subqueries", None) or []) if q]
    pages = len(getattr(run, "sources", None) or [])
    asked = getattr(run, "requests_used", 0)
    parts = []
    if queries:
        parts.append(f"searched {len(queries)} queries: " + "; ".join(f"“{q}”" for q in queries[:6]))
    if pages:
        parts.append(f"read {pages} sources")
    if asked:
        parts.append(f"{asked} upstream requests")
    if not parts:
        return ""
    body = ", ".join(parts)
    return f"> **How this was researched:** {body}. Claims cite the numbered sources below."


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    subs = [str(x).strip() for x in plan.get("subquestions") or [] if str(x).strip()]
    kept: list[dict[str, Any]] = []
    answered: set[int] = set()
    dropped = 0
    for sec in plan.get("sections") or []:
        if not isinstance(sec, dict) or not str(sec.get("title") or "").strip():
            continue
        idx = [i for i in (sec.get("answers") or []) if isinstance(i, int) and 0 <= i < len(subs)]
        if not idx:
            dropped += 1
            continue
        kept.append({**sec, "answers": idx})
        answered.update(idx)
    unanswered = [subs[i] for i in range(len(subs)) if i not in answered]
    return {"subquestions": subs, "sections": kept, "unanswered": unanswered, "dropped": dropped}


async def _agent_synthesize_report_plan(
    run: _ResearchRun,
    context: list[dict[str, Any]],
    synth_step: int,
) -> tuple[dict[str, Any], float]:
    """Synthesize the report from dynamic sub-questions (E20-E24) rather than fixed templates."""
    from .agent_roles import REPORT_PLAN_PROMPT
    from .deep_research import sanitize_json
    from .research_jobs import REWRITE_MODEL, build_synthesis_messages

    query_terms: set[str] = set()
    for text in (
        run.topic,
        *run.subqueries,
        *(getattr(a, "aspect", "") for a in run.aspects),
        *(getattr(a, "looking_for", "") for a in run.aspects),
    ):
        query_terms.update(w.lower() for w in re.findall(r"\w+", text) if len(w) >= 3)

    synth_context = [
        {
            **src,
            "content": prune_source_content(
                src.get("content") or "",
                query_terms=query_terms,
                max_chars=8000,
            ),
        }
        for src in context
    ]

    base = build_synthesis_messages(
        run.topic,
        synth_context,
        depth=run.depth,
        nonce=uuid.uuid4().hex[:12],
        aspects=run.aspects,
        lang=run.lang,
    )
    system = base[0]["content"]
    facts = [_plan_fact(a) for a in list(run.aspects)[:20]]
    shown = json.dumps(
        [
            {"id": s.get("id"), "title": s.get("title"), "says": _source_digest(s.get("content") or "")}
            for s in synth_context
        ]
    )[:40000]

    timeout = max(10.0, run.remaining() - 2.0)
    plan_messages = [
        {"role": "system", "content": REPORT_PLAN_PROMPT},
        {
            "role": "user",
            "content": f"Question: {run.topic}\nFacts: {json.dumps(facts)}\nSources: {shown}",
        },
    ]
    plan_res = await runtime._chat_request(
        messages=plan_messages,
        thinking={"type": "disabled"},
        temperature=0.0,
        model=REWRITE_MODEL,
        max_tokens=8192,
        response_format={"type": "json_object"},
        timeout=min(25.0, timeout),
        deadline=timeout,
        attempts=2,
    )
    cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "")
    plan = _validate_plan(parsed) if isinstance(parsed, dict) else {"sections": []}
    sections = plan["sections"]
    if not sections and run.remaining() > 15.0:
        retry_plan = await runtime._chat_request(
            messages=[
                *plan_messages,
                {"role": "assistant", "content": plan_res.get("content") or "{}"},
                {
                    "role": "user",
                    "content": (
                        "Previous response was not valid JSON or was missing 'sections'. "
                        "Output strictly valid JSON with 'subquestions', 'sections' (each with "
                        "'title', 'answers', 'facts'), and 'unsupported'."
                    ),
                },
            ],
            thinking={"type": "disabled"},
            temperature=0.0,
            model=REWRITE_MODEL,
            max_tokens=8192,
            response_format={"type": "json_object"},
            timeout=min(20.0, run.remaining() - 2.0),
            deadline=run.deadline_at,
            attempts=2,
        )
        if retry_plan.get("ok"):
            cost += float(retry_plan.get("cost_usd", 0.0) or 0.0)
            parsed_retry, _ = sanitize_json(retry_plan.get("content") or "")
            if isinstance(parsed_retry, dict):
                plan_retry = _validate_plan(parsed_retry)
                if plan_retry and plan_retry.get("sections"):
                    plan = plan_retry
                    sections = plan["sections"]

    if not sections:
        return await _research_synthesize(run, context, synth_step)

    per_section_words = max(120, int(run.cfg.target_words or 800) // len(sections))
    sec_max_tokens = min(run.cfg.max_tokens, max(1200, per_section_words * 4))

    async def _write_section(sec: dict[str, Any]) -> tuple[str, str, float]:
        sec_title = str(sec.get("title") or "").strip()
        ask = REPORT_ASK.format(words=per_section_words)
        section_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": base[1]["content"] + f"\n\n{ask}\nSection: {sec_title}"},
        ]
        res = await runtime._chat_request(
            messages=section_messages,
            thinking={"type": "disabled"} if not run.cfg.thinking else {"type": "enabled"},
            model=REWRITE_MODEL,
            max_tokens=sec_max_tokens,
            timeout=max(10.0, run.remaining() - 2.0),
            deadline=run.deadline_at,
            attempts=2,
        )
        sec_cost = float(res.get("cost_usd", 0.0) or 0.0)
        text = _strip_heading((res.get("content") or "").strip())
        if (not text or res.get("finish_reason") == "length") and run.remaining() > 8.0:
            retry_sec = await runtime._chat_request(
                messages=section_messages,
                thinking={"type": "disabled"},
                temperature=0.0,
                model=REWRITE_MODEL,
                max_tokens=sec_max_tokens,
                timeout=max(5.0, run.remaining() - 2.0),
                deadline=run.deadline_at,
                attempts=1,
            )
            if retry_sec.get("ok"):
                sec_cost += float(retry_sec.get("cost_usd", 0.0) or 0.0)
                sec_text = _strip_heading((retry_sec.get("content") or "").strip())
                if sec_text:
                    text = sec_text
        return sec_title, text, sec_cost

    section_tasks = [_write_section(sec) for sec in sections]
    section_results = await asyncio.gather(*section_tasks, return_exceptions=True)

    parts: list[tuple[str, str]] = []
    for item in section_results:
        if isinstance(item, tuple) and len(item) == 3:
            sec_title, text, sec_cost = item
            cost += sec_cost
            if text:
                parts.append((sec_title, text))

    if not parts:
        return await _research_synthesize(run, context, synth_step)

    report = "\n\n".join(f"## {name}\n\n{text}" for name, text in parts)
    report += "\n\n## What the sources do not settle\n\n" + _gaps_sentence(plan["unanswered"], context)
    note = _search_note(run)
    if note:
        report = note + "\n\n" + report
    run.notes.append(
        f"report-plan: {len(parts)} sections, {len(plan['unanswered'])} unanswered, "
        f"{plan['dropped']} sections dropped for answering nothing"
    )
    return {"ok": True, "content": report, "finish_reason": "stop"}, cost


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

    if not results:
        # Every search failed: a report built on nothing would look complete and be wrong.
        return {
            **_fail("upstream_error", "All searches failed or timed out; no report was written."),
            "estimate_usd": estimate,
            "stopped_reason": "retrieval_failed",
            "plan": plan,
            "rounds": [
                {
                    "new_facts": 0,
                    "new_numeric_facts": 0,
                    "gaps_filled": 0,
                    "new_sources": 0,
                    "saturated": True,
                }
            ],
            "facts_per_round": [0],
        }

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
        queries = await _gap_queries_model(run, blocks, used, results, max_queries=3)
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
        run.synthesis_context = context
        job.synthesis_context = context
        run.notes.extend(guard_notes)
        counters["context_chars"] = sum(len(c.get("content") or "") for c in context)
    if not context:
        return {
            **_fail("empty_page", "Every selected page had no readable content; no report was written."),
            "estimate_usd": estimate,
            "stopped_reason": "empty_pages",
            "plan": plan,
            "rounds": rounds,
            "facts_per_round": [gain["new_facts"] for gain in rounds],
        }
    if (stopped := _research_cost_guard(run, context)) is not None:
        return stopped

    synth_res, discarded_cost = await _agent_synthesize_report_plan(run, context, synth_step)
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
