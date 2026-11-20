"""Universal Pipeline Remedies Benchmark.

Empirical evaluation of 4 universal, domain-agnostic improvement proposals:
- Arm 1: Universal Metric Density Directive in Synthesis (verbatim numbers/percentages/dates)
- Arm 2: Universal Multi-Perspective Aspect Decomposition (Mandate, Metrics, Limits/Disputes, Latest Status)
- Arm 3: LLM Dynamic Gap Inspector in Agent Research (smart gap queries targeting missing empirical facts)

Evaluates on the 3 diagnostic benchmark domains (104 rubric atoms total):
1. medicine-glp1 (31 atoms)
2. politics-eu-cbam (33 atoms)
3. it-pg-incremental-backup (40 atoms)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import judge  # noqa: E402
import keypoints  # noqa: E402

from deepseek_search_everywhere_mcp import (  # noqa: E402
    deep_research as dr,
)
from deepseek_search_everywhere_mcp import (  # noqa: E402
    research_jobs,
    runtime,
)
from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    DEPTH_CONFIG,
    filter_synthesis_output,
    normalize_citations,
    sanitize_json,
)
from deepseek_search_everywhere_mcp.snippets import prune_source_content  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results" / "universal_remedies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    {
        "id": "medicine-glp1",
        "domain": "medicine",
        "topic": (
            "Semaglutide and tirzepatide for obesity: weight-loss efficacy in "
            "clinical trials and main safety concerns"
        ),
        "context_file": "results/step_comparison/medicine-glp1__quick_full.json",
    },
    {
        "id": "politics-eu-cbam",
        "domain": "politics",
        "topic": (
            "The EU Carbon Border Adjustment Mechanism (CBAM): how it works, "
            "implementation timeline and international political reactions"
        ),
        "context_file": "results/w-k20-high/politics-eu-cbam__k__high__r1.json",
    },
    {
        "id": "it-pg-incremental-backup",
        "domain": "it",
        "topic": (
            "How PostgreSQL 17 incremental backups work (pg_basebackup --incremental, "
            "WAL summarization, pg_combinebackup) and their limitations"
        ),
        "context_file": "results/step_comparison/it-pg-incremental-backup__quick_full.json",
    },
]

UNIVERSAL_DECOMPOSE_SYSTEM_PROMPT = (
    "You plan web research. Split the topic into research aspects: the distinct things a specialist "
    "would need to establish to answer it well. Give each aspect exactly one search query.\n\n"
    "Aspect diversity rule: Your chosen aspects MUST distribute across distinct factual dimensions:\n"
    "1. primary/mechanism: what the primary standard, law, trial, or documentation specifies or proves; "
    "query its subject and outcome, never its publication details.\n"
    "2. numbers: measured quantitative results, sizes, rates, prices, percentages, or dates with scope.\n"
    "3. limits: limitations, risks, criticism, adverse effects, legal challenges, or open disputes.\n"
    "4. current: the latest status, revisions, official proposals, amendments, or deadlines as of today.\n\n"
    "Query rules: 3 to 8 keywords, not a sentence; use the names a specialist would search for (proper "
    "names, parameter names, study names, document titles) rather than topic wording; use a number or "
    "identifier only when certain it exists; put a year only in a current query about the present state; "
    "no URLs; at most 100 characters; local language for domestic laws/entities, English for global tech.\n\n"
    "Return only a json object in this form:\n"
    '{"aspects": ['
    '{"kind": "primary", "aspect": "what specification requires", "looking_for": "requirements", '
    '"query": "HTTP/3 stream multiplexing"}, '
    '{"kind": "numbers", "aspect": "measured benchmarks", "looking_for": "latency numbers", '
    '"query": "HTTP/3 vs HTTP/2 latency benchmark ms"}, '
    '{"kind": "limits", "aspect": "failure modes", "looking_for": "replay attack limits", '
    '"query": "QUIC 0-RTT replay attack mitigation"}, '
    '{"kind": "current", "aspect": "current adoption", "looking_for": "traffic share status", '
    '"query": "HTTP/3 share web requests Cloudflare Radar"}'
    "]}"
)

UNIVERSAL_METRIC_DIRECTIVE = (
    "\n14. PRESERVE VERBATIM METRICS: Never generalize numbers, percentages, rates, dates, sample sizes, "
    "doses, or statutory thresholds into descriptive words (e.g., do not write 'frequent', 'rare', or "
    "'substantial' when the source provides '44%', '1 in 10,000', or '€100M'). Always report the exact "
    "numerical figures verbatim alongside their source citations [n]."
)

UNIVERSAL_GAP_PROMPT = (
    "You are a research gap detector. Given the research topic, planned aspects, executed queries, and "
    "preliminary findings summary: identify what critical factual dimensions (specific numbers, recent "
    "disputes, official updates, or technical edge cases) are STILL MISSING from the gathered evidence.\n"
    "Output up to 3 specialist keyword search queries targeting these missing dimensions.\n"
    "Output strictly valid JSON with 'missing_dimensions': [str] and 'queries': [str]."
)


def build_arm1_synthesis_system(depth: str = "high") -> str:
    base = dr.build_synthesis_system_prompt(depth, client=False)
    return base + UNIVERSAL_METRIC_DIRECTIVE


async def run_arm1_metric_directive(
    topic: str,
    context: list[dict[str, Any]],
    subqueries: list[str] | None = None,
    depth: str = "high",
) -> tuple[str, float, float]:
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()
    query_terms: set[str] = set()
    for text in [topic, *(subqueries or [])]:
        query_terms.update(w.lower() for w in re.findall(r"\w+", text) if len(w) >= 3)

    pruned_context = []
    for s in context:
        s_copy = dict(s)
        s_copy["content"] = prune_source_content(s.get("content") or "", query_terms, max_chars=8000)
        pruned_context.append(s_copy)

    messages = dr.build_synthesis_messages(topic, pruned_context, depth=depth)
    messages[0]["content"] = build_arm1_synthesis_system(depth)

    res = await runtime._chat_request(
        messages=messages,
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=cfg.max_tokens,
        timeout=150.0,
        deadline=150.0,
        attempts=1,
    )
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    report = (res.get("content") or "").strip()
    elapsed = round(time.monotonic() - t0, 2)
    return report, cost, elapsed


async def run_arm2_universal_decomp(
    topic: str,
    depth: str = "high",
) -> tuple[str, list[dict[str, Any]], float, float]:
    t0 = time.monotonic()
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])

    user_prompt = (
        f"Today's date: {dr._today()}\nResearch topic: {topic}\n"
        f"Return exactly {cfg.subqueries} aspects as json."
    )
    decomp_messages = [
        {"role": "system", "content": UNIVERSAL_DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    res_decomp = await runtime._chat_request(
        messages=decomp_messages,
        thinking={"type": "disabled"},
        model=runtime.REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=20.0,
        deadline=20.0,
        attempts=2,
    )
    cost = float(res_decomp.get("cost_usd", 0.0) or 0.0)
    subqueries, aspects = dr.parse_research_plan(
        res_decomp.get("content") or "", topic, max_queries=cfg.subqueries
    )
    if not subqueries:
        subqueries = [topic[:128]]

    run_dummy = research_jobs._ResearchRun(
        job=dr.ResearchJob(job_id=f"test-{int(t0)}", topic=topic, depth=depth, lang=None, created_at=t0),
        topic=topic,
        depth=depth,
        sources_pref="any",
        lang=None,
        refresh=True,
        strip_boilerplate=True,
        cfg=cfg,
        deadline_at=t0 + 120.0,
        synthesis_reserve=30.0,
        total_steps=4,
    )

    search_tasks = research_jobs._research_search_calls(run_dummy, subqueries, 30.0)
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    raw_sources: list[dict[str, Any]] = []
    for r in search_results:
        if isinstance(r, dict) and r.get("ok"):
            cost += float(r.get("cost_usd", 0.0) or 0.0)
            res_items = r.get("results") or []
            run_dummy.result_lists.append(res_items)
            raw_sources.extend(res_items)

    selected, _ = dr.rank_and_diversify_sources(
        run_dummy.result_lists,
        k_sources=cfg.k_sources,
        sources=run_dummy.sources_pref,
    )
    context = dr.select_synthesis_context(selected, depth=depth, topic=topic)

    report, synth_cost, _ = await run_arm1_metric_directive(
        topic, context, subqueries=subqueries, depth=depth
    )
    cost += synth_cost
    elapsed = round(time.monotonic() - t0, 2)
    return report, context, cost, elapsed


async def run_arm3_agent_dynamic_gaps(
    topic: str,
    depth: str = "high",
) -> tuple[str, list[dict[str, Any]], float, float]:
    t0 = time.monotonic()
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])

    user_prompt = (
        f"Today's date: {dr._today()}\nResearch topic: {topic}\n"
        f"Return exactly {cfg.subqueries} aspects as json."
    )
    decomp_messages = [
        {"role": "system", "content": UNIVERSAL_DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    res_decomp = await runtime._chat_request(
        messages=decomp_messages,
        thinking={"type": "disabled"},
        model=runtime.REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=20.0,
        deadline=20.0,
        attempts=2,
    )
    cost = float(res_decomp.get("cost_usd", 0.0) or 0.0)
    subqueries, aspects = dr.parse_research_plan(
        res_decomp.get("content") or "", topic, max_queries=cfg.subqueries
    )
    if not subqueries:
        subqueries = [topic[:128]]

    run_dummy = research_jobs._ResearchRun(
        job=dr.ResearchJob(job_id=f"test-arm3-{int(t0)}", topic=topic, depth=depth, lang=None, created_at=t0),
        topic=topic,
        depth=depth,
        sources_pref="any",
        lang=None,
        refresh=True,
        strip_boilerplate=True,
        cfg=cfg,
        deadline_at=t0 + 150.0,
        synthesis_reserve=30.0,
        total_steps=5,
    )

    search_tasks = research_jobs._research_search_calls(run_dummy, subqueries, 30.0)
    r1_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    all_sources: list[dict[str, Any]] = []
    used_queries = set(subqueries)
    for r in r1_results:
        if isinstance(r, dict) and r.get("ok"):
            cost += float(r.get("cost_usd", 0.0) or 0.0)
            res_items = r.get("results") or []
            run_dummy.result_lists.append(res_items)
            all_sources.extend(res_items)

    summary_snippets = "\n".join(
        f"- {s.get('title')}: {(s.get('content') or '')[:300]}" for s in all_sources[:15]
    )
    gap_prompt_content = (
        f"Research topic: {topic}\n"
        f"Executed queries: {', '.join(used_queries)}\n"
        f"Preliminary findings summary:\n{summary_snippets}\n"
    )
    gap_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": UNIVERSAL_GAP_PROMPT},
            {"role": "user", "content": gap_prompt_content},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=20.0,
        deadline=20.0,
        attempts=2,
    )
    cost += float(gap_res.get("cost_usd", 0.0) or 0.0)
    parsed_gaps, _ = sanitize_json(gap_res.get("content") or "{}")
    r2_queries = parsed_gaps.get("queries") if isinstance(parsed_gaps, dict) else []
    r2_queries = [q for q in r2_queries if isinstance(q, str) and q.strip() and q not in used_queries][:3]

    if r2_queries:
        r2_tasks = research_jobs._research_search_calls(run_dummy, r2_queries, 30.0)
        r2_results = await asyncio.gather(*r2_tasks, return_exceptions=True)
        for r in r2_results:
            if isinstance(r, dict) and r.get("ok"):
                cost += float(r.get("cost_usd", 0.0) or 0.0)
                res_items = r.get("results") or []
                run_dummy.result_lists.append(res_items)
                all_sources.extend(res_items)

    selected, _ = dr.rank_and_diversify_sources(
        run_dummy.result_lists,
        k_sources=cfg.k_sources,
        sources=run_dummy.sources_pref,
    )
    context = dr.select_synthesis_context(selected, depth=depth, topic=topic)
    all_subqueries = [*subqueries, *r2_queries]
    report, synth_cost, _ = await run_arm1_metric_directive(
        topic, context, subqueries=all_subqueries, depth=depth
    )
    cost += synth_cost
    elapsed = round(time.monotonic() - t0, 2)
    return report, context, cost, elapsed


async def judge_report(
    topic_id: str,
    topic: str,
    report: str,
    context: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "topic_id": topic_id,
        "topic": topic,
        "envelope": {"report": report},
        "dump": {"synthesis_context": context},
    }
    atoms_text = judge._rubric_text(topic_id, use_atoms=True)
    evidence_lines = judge._evidence(record)

    prompt = (
        f"# Topic\n{topic}\n\n"
        f"# Key points\n{atoms_text}\n\n"
        f"# Report\n{report}\n\n"
        f"# Evidence (cited sentences and passages of the cited sources)\n" + "\n".join(evidence_lines)
    )

    res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": keypoints.KEYPOINT_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        response_format={"type": "json_object"},
        max_tokens=4096,
        timeout=90.0,
        deadline=90.0,
        attempts=2,
    )

    verdict, _ = sanitize_json(res.get("content") or "{}")
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    points = verdict.get("points") or {}
    total = len(points)
    covered = sum(1 for p in points.values() if p.get("covered"))
    supported = sum(1 for p in points.values() if p.get("supported"))
    return {
        "verdict": verdict,
        "total_atoms": total,
        "covered_atoms": covered,
        "supported_atoms": supported,
        "cost_usd": cost,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Pipeline Remedies Benchmark")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["medicine-glp1", "politics-eu-cbam", "it-pg-incremental-backup"],
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["arm1_metric_directive", "arm2_universal_decomp", "arm3_agent_dynamic_gaps"],
    )
    args = parser.parse_args()

    selected_topics = [t for t in TOPICS if t["id"] in args.topics]
    print("\n=================================================================")
    print(f"Universal Pipeline Remedies Benchmark (Evaluating {len(selected_topics)} Topics)")
    print(f"Arms: {args.arms}")
    print("=================================================================\n")

    summary_table: list[dict[str, Any]] = []

    for t in selected_topics:
        tid = t["id"]
        topic_text = t["topic"]
        rubric = keypoints._rubric(tid)
        print(f"\n>>> Running Topic: {tid} ...")

        ctx_file = ROOT / "evals" / t["context_file"]
        ctx_data = json.loads(ctx_file.read_text(encoding="utf-8"))
        base_context = (
            ctx_data.get("dump", {}).get("synthesis_context")
            or ctx_data.get("run", {}).get("dump", {}).get("synthesis_context")
            or []
        )

        base_subqueries = (
            ctx_data.get("run", {}).get("envelope", {}).get("subqueries") or ctx_data.get("subqueries") or []
        )

        for arm in args.arms:
            print(f"  -> Executing {arm} on {tid} ...")
            if arm == "arm1_metric_directive":
                rep_text, arm_cost, wall = await run_arm1_metric_directive(
                    topic_text, base_context, subqueries=base_subqueries
                )
                context = base_context
            elif arm == "arm2_universal_decomp":
                rep_text, context, arm_cost, wall = await run_arm2_universal_decomp(topic_text)
            elif arm == "arm3_agent_dynamic_gaps":
                rep_text, context, arm_cost, wall = await run_arm3_agent_dynamic_gaps(topic_text)
            else:
                continue

            filtered, _, _ = filter_synthesis_output(
                rep_text, {s["url"] for s in context if isinstance(s.get("url"), str)}
            )
            norm_report = normalize_citations(filtered, max_id=len(context))

            j_res = await judge_report(tid, topic_text, norm_report, context, rubric)
            tot = j_res["total_atoms"]
            sup = j_res["supported_atoms"]
            cov = j_res["covered_atoms"]
            rate = round(sup / tot * 100, 1) if tot > 0 else 0.0
            print(f"     Result [{arm}]: {sup}/{tot} supported ({rate}%) in {wall}s | Cost: ${arm_cost:.4f}")

            res_record = {
                "topic_id": tid,
                "arm": arm,
                "wall_s": wall,
                "cost_usd": arm_cost,
                "total_atoms": tot,
                "supported_atoms": sup,
                "covered_atoms": cov,
                "report": norm_report,
                "judgement": j_res,
            }
            out_file = RESULTS_DIR / f"{tid}__{arm}.json"
            out_file.write_text(json.dumps(res_record, indent=2), encoding="utf-8")

            summary_table.append(
                {
                    "topic": tid,
                    "arm": arm,
                    "supported": sup,
                    "total": tot,
                    "rate": rate,
                    "wall_s": wall,
                    "cost_usd": arm_cost,
                }
            )

    print("\n=================================================================")
    print("BENCHMARK SUMMARY")
    print("=================================================================")
    for row in summary_table:
        topic_s = f"{row['topic']:<26}"
        arm_s = f"{row['arm']:<25}"
        score_s = f"{row['supported']}/{row['total']} ({row['rate']:>5.1f}%)"
        print(f"{topic_s} | {arm_s} | {score_s:<18} | {row['wall_s']:>5.1f}s | ${row['cost_usd']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
