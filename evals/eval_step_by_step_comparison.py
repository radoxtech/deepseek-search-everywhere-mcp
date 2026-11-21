"""Step-by-step benchmark and ablation study: Quick Research vs Agent Deep Research.

Measures:
1. Quick Research (Full, modern context limits)
2. Agent Research (Full, report-plan sectional synthesis, modern context limits)
3. Step 1 Ablation: No decomposition (1 raw query, no aspect planning)
4. Step 2 Ablation: No gap retrieval (R1 search only, no R2 gap search)
5. Step 3 Ablation: Legacy context ceiling (150k chars, 10 sources vs modern 800k/30 sources)
6. Step 4 Ablation: No thinking (thinking disabled)

Evaluates reports against 71 atomic rubric facts across IT and Medicine:
- it-pg-incremental-backup (40 atoms)
- medicine-glp1 (31 atoms)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
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
    agent_research,
    gap_fill,
    research_jobs,
    runtime,
)
from deepseek_search_everywhere_mcp import (  # noqa: E402
    deep_research as dr,
)
from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    DEPTH_CONFIG,
    ResearchAspect,
    ResearchJob,
    report_body_words,
)
from deepseek_search_everywhere_mcp.filters import fit_query  # noqa: E402

TOPICS = [
    {
        "id": "it-pg-incremental-backup",
        "domain": "it",
        "topic": (
            "How PostgreSQL 17 incremental backups work (pg_basebackup --incremental, "
            "WAL summarization, pg_combinebackup) and their limitations"
        ),
    },
    {
        "id": "medicine-glp1",
        "domain": "medicine",
        "topic": (
            "Semaglutide and tirzepatide for obesity: weight-loss efficacy in "
            "clinical trials and main safety concerns"
        ),
    },
]

VARIANTS = [
    "quick_full",
    "agent_full",
    "step1_no_decomp",
    "step2_no_r2_gaps",
    "step3_legacy_context",
    "step4_no_thinking",
]

RESULTS_DIR = ROOT / "evals" / "results" / "step_comparison"


async def execute_run(topic_id: str, topic_text: str, variant: str, depth: str = "high") -> dict[str, Any]:
    """Execute one research run under the given ablation variant."""
    cfg = DEPTH_CONFIG[depth]
    max_rounds = 3

    # Original functions for patching
    orig_decompose = research_jobs._research_decompose
    orig_select_context = research_jobs.select_synthesis_context
    orig_fill_gaps = research_jobs._fill_gaps

    is_agent = variant == "agent_full"

    if variant == "step2_no_r2_gaps":
        max_rounds = 1

    job = ResearchJob(
        job_id=f"step-{topic_id}-{variant}-{int(time.time())}",
        topic=topic_text,
        depth=depth,
        lang=None,
        sources="any",
        kind="agent" if is_agent else "quick",
        max_rounds=max_rounds,
        max_cost_usd=None,
        created_at=runtime._clock(),
    )

    async def patched_decompose(run: Any) -> None:
        if variant == "step1_no_decomp":
            await research_jobs._report_job_progress(
                run.job, "decomposition", 1, run.total_steps, "Single query (no decomposition)"
            )
            fallback, _ = fit_query(run.topic)
            run.subqueries = [fallback]
            run.aspects = [
                ResearchAspect(
                    aspect="overview",
                    kind="primary",
                    looking_for=run.topic,
                    query=fallback,
                )
            ]
        else:
            await orig_decompose(run)

    def patched_select_context(
        sources: list[dict[str, Any]], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        if variant == "step3_legacy_context":
            # Emulate legacy context ceiling: 10 sources, 150k chars
            legacy_cfg = dataclasses.replace(
                cfg, k_sources=10, total_source_chars=150_000, per_page_char_cap=20_000
            )
            orig_depth_cfg = dr.DEPTH_CONFIG[depth]
            try:
                dr.DEPTH_CONFIG[depth] = legacy_cfg
                kwargs["depth"] = depth
                return orig_select_context(sources, *args, **kwargs)
            finally:
                dr.DEPTH_CONFIG[depth] = orig_depth_cfg
        kwargs["depth"] = depth
        return orig_select_context(sources, *args, **kwargs)

    async def patched_fill_gaps(*args: Any, **kwargs: Any) -> Any:
        if variant == "step2_no_r2_gaps":
            return gap_fill._GapFill()
        return await orig_fill_gaps(*args, **kwargs)

    # Patching
    research_jobs._research_decompose = patched_decompose
    agent_research._research_decompose = patched_decompose
    research_jobs.select_synthesis_context = patched_select_context
    research_jobs._fill_gaps = patched_fill_gaps

    if variant == "step4_no_thinking":
        dr.DEPTH_CONFIG[depth] = dataclasses.replace(cfg, thinking=False, reasoning_effort=None)

    t0 = time.monotonic()
    try:
        if is_agent:
            envelope = await agent_research.run_agent_job(
                job,
                clean_topic=topic_text,
                depth=depth,
                sources_pref="any",
                lang=None,
                refresh=True,
                strip_boilerplate=True,
            )
        else:
            envelope = await research_jobs._run_research_job(
                job,
                clean_topic=topic_text,
                depth=depth,
                sources_pref="any",
                lang=None,
                refresh=True,
                strip_boilerplate=True,
            )
    finally:
        # Restore original functions and config
        research_jobs._research_decompose = orig_decompose
        agent_research._research_decompose = orig_decompose
        research_jobs.select_synthesis_context = orig_select_context
        research_jobs._fill_gaps = orig_fill_gaps
        dr.DEPTH_CONFIG[depth] = cfg

    elapsed_s = round(time.monotonic() - t0, 2)
    synthesis_context = [
        {"id": idx, "url": s.get("url"), "title": s.get("title"), "content": s.get("content")}
        for idx, s in enumerate(getattr(job.run, "sources", []) or [], 1)
    ]
    return {
        "topic_id": topic_id,
        "topic": topic_text,
        "variant": variant,
        "elapsed_s": elapsed_s,
        "envelope": envelope,
        "dump": {"synthesis_context": synthesis_context},
    }


async def judge_record(record: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    """Judge the record's report against rubric atoms using deepseek-flash."""
    topic = record["topic"]
    topic_id = record["topic_id"]
    atoms_text = judge._rubric_text(topic_id, use_atoms=True)
    evidence_lines = judge._evidence(record)
    report = record.get("envelope", {}).get("report") or "(no report)"

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

    verdict, _ = dr.sanitize_json(res.get("content") or "{}")
    if not isinstance(verdict, dict) or not verdict.get("points"):
        retry_res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": keypoints.KEYPOINT_INSTRUCTIONS},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": res.get("content") or "{}"},
                {
                    "role": "user",
                    "content": (
                        "Previous response did not contain a valid 'points' mapping. "
                        "Return strictly a JSON object with 'points': {<point_id>: "
                        "{'covered': bool, 'sentence': int|null, 'supported': bool|null}}."
                    ),
                },
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
        if retry_res.get("ok"):
            verdict_retry, _ = dr.sanitize_json(retry_res.get("content") or "{}")
            if isinstance(verdict_retry, dict) and verdict_retry.get("points"):
                verdict = verdict_retry

    if not isinstance(verdict, dict):
        verdict = {"points": {}, "unsupported": []}

    scores = keypoints.atom_scores(verdict, rubric)
    points = verdict.get("points") or {}
    all_atoms = judge.atoms(rubric)
    covered_count = sum(1 for a in all_atoms if (points.get(a["id"]) or {}).get("covered"))
    supported_count = sum(
        1
        for a in all_atoms
        if (points.get(a["id"]) or {}).get("covered") and (points.get(a["id"]) or {}).get("supported") is True
    )

    return {
        "verdict": verdict,
        "scores": scores,
        "total_atoms": len(all_atoms),
        "covered_atoms": covered_count,
        "supported_atoms": supported_count,
        "unsupported_sentences": len(verdict.get("unsupported") or []),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step benchmark: Quick vs Agent Research")
    parser.add_argument("--topic", help="Specific topic_id to run (default: both)")
    parser.add_argument("--variant", help="Specific variant to run (default: all)")
    parser.add_argument("--depth", default="high", help="Depth tier (default: high)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    topics = [t for t in TOPICS if not args.topic or t["id"] == args.topic]
    variants = [v for v in VARIANTS if not args.variant or v == args.variant]

    print(f"=== Running Step-by-Step Benchmark ({len(topics)} topics x {len(variants)} variants) ===")
    records = []

    for t in topics:
        tid = t["id"]
        rubric_path = ROOT / "evals" / "rubrics" / f"{tid}.json"
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))

        for var in variants:
            out_file = RESULTS_DIR / f"{tid}__{var}.json"
            print(f"\n---> Running {tid} [{var}] ...", flush=True)
            run_data = await execute_run(tid, t["topic"], var, depth=args.depth)

            print(f"     Judging against {len(judge.atoms(rubric))} rubric atoms ...", flush=True)
            judgement = await judge_record(run_data, rubric)

            report = run_data["envelope"].get("report", "")
            words = report_body_words(report)
            cost = run_data["envelope"].get("cost_usd", 0.0)
            wall = run_data["elapsed_s"]
            sources = len(run_data["envelope"].get("sources", []))

            combined = {
                "topic_id": tid,
                "variant": var,
                "wall_s": wall,
                "cost_usd": cost,
                "words": words,
                "sources_count": sources,
                "total_atoms": judgement["total_atoms"],
                "covered_atoms": judgement["covered_atoms"],
                "supported_atoms": judgement["supported_atoms"],
                "unsupported_sentences": judgement["unsupported_sentences"],
                "run": run_data,
                "judgement": judgement,
            }
            out_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(combined)

            supp = judgement["supported_atoms"]
            tot = judgement["total_atoms"]
            pct = (supp / tot) * 100 if tot else 0.0
            cost_per_atom = (cost / supp) if supp else 0.0
            print(
                f"     DONE: Supported={supp}/{tot} ({pct:.1f}%), Words={words}, Cost=${cost:.4f}, "
                f"Wall={wall}s, Cost/Atom=${cost_per_atom:.5f}"
            )

    # Print summary table across all runs
    hdr = (
        f"{'Variant':<22} | {'Topic':<12} | {'Supported Atoms':<15} | "
        f"{'Words':<6} | {'Cost':<8} | {'Wall':<7} | {'$/Atom':<9}"
    )
    print("\n" + "=" * 90)
    print(hdr)
    print("=" * 90)
    for r in records:
        supp = r["supported_atoms"]
        tot = r["total_atoms"]
        pct = (supp / tot) * 100 if tot else 0.0
        c_atom = (r["cost_usd"] / supp) if supp else 0.0
        print(
            f"{r['variant']:<22} | {r['topic_id'][:12]:<12} | {supp:>2}/{tot} ({pct:>4.1f}%) | "
            f"{r['words']:<6} | ${r['cost_usd']:<7.4f} | {r['wall_s']:<7.1f} | ${c_atom:<8.5f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
