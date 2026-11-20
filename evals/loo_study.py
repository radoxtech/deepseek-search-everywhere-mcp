"""Leave-One-Out (LOO) Ablation and Dependency Study for depth 'max'.

Tests the 5 core stages of deep agent research across 4 domains:
- W0: full_max (all stages active)
- W1: no_decomp (1 raw query, no aspect planning)
- W2: no_gaps (1 retrieval round, no gap queries)
- W3: no_fetch (search snippets only, zero Fetch API calls)
- W4: no_sectional (single-pass synthesis instead of question-based report plan)
- W5: no_thinking (thinking disabled, temperature 0.0)
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
    {
        "id": "law-eu-ai-act-gpai",
        "domain": "law",
        "topic": (
            "EU AI Act obligations for providers of general-purpose AI models: "
            "requirements, systemic-risk models, code of practice and deadlines"
        ),
    },
    {
        "id": "ecommerce-free-shipping",
        "domain": "ecommerce",
        "topic": (
            "Effect of free-shipping thresholds on conversion rate, average "
            "order value and returns in online retail"
        ),
    },
]

VARIANTS = [
    "full_max",
    "no_decomp",
    "no_gaps",
    "no_fetch",
    "no_sectional",
    "no_thinking",
]

RESULTS_DIR = ROOT / "evals" / "results" / "loo"


async def execute_run(topic_id: str, topic_text: str, variant: str) -> dict[str, Any]:
    """Execute one research run under the given ablation variant."""
    cfg = DEPTH_CONFIG["max"]
    max_rounds = 3

    if variant == "no_gaps":
        max_rounds = 1

    if variant == "no_thinking":
        cfg = dataclasses.replace(cfg, thinking=False, reasoning_effort=None)

    job = ResearchJob(
        job_id=f"loo-{topic_id}-{variant}-{int(time.time())}",
        topic=topic_text,
        depth="max",
        lang=None,
        sources="any",
        kind="agent",
        max_rounds=max_rounds,
        max_cost_usd=None,
        created_at=runtime._clock(),
    )

    # Prepare patches
    orig_decompose = research_jobs._research_decompose
    orig_fill_gaps = research_jobs._fill_gaps
    orig_agent_synth = agent_research._agent_synthesize_report_plan

    async def patched_decompose(run: Any) -> None:
        if variant == "no_decomp":
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

    async def patched_fill_gaps(*args: Any, **kwargs: Any) -> Any:
        if variant == "no_fetch":
            return gap_fill._GapFill()
        return await orig_fill_gaps(*args, **kwargs)

    async def patched_synth(
        run: Any, context: list[dict[str, Any]], synth_step: int
    ) -> tuple[dict[str, Any], float]:
        if variant == "no_sectional":
            return await research_jobs._research_synthesize(run, context, synth_step)
        if variant == "no_thinking":
            # Force thinking disabled inside report-plan
            run.cfg = dataclasses.replace(run.cfg, thinking=False, reasoning_effort=None)
        return await orig_agent_synth(run, context, synth_step)

    # Apply patches
    research_jobs._research_decompose = patched_decompose
    agent_research._research_decompose = patched_decompose
    research_jobs._fill_gaps = patched_fill_gaps
    agent_research._agent_synthesize_report_plan = patched_synth

    t0 = time.monotonic()
    try:
        envelope = await agent_research.run_agent_job(
            job,
            clean_topic=topic_text,
            depth="max",
            sources_pref="any",
            lang=None,
            refresh=True,
            strip_boilerplate=True,
        )
    finally:
        # Restore original functions
        research_jobs._research_decompose = orig_decompose
        agent_research._research_decompose = orig_decompose
        research_jobs._fill_gaps = orig_fill_gaps
        agent_research._agent_synthesize_report_plan = orig_agent_synth

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

    timeout = 90.0
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
        timeout=timeout,
        deadline=timeout,
        attempts=2,
    )

    verdict, _ = dr.sanitize_json(res.get("content") or "{}")
    if not isinstance(verdict, dict):
        verdict = {"points": {}, "unsupported": []}

    scores = keypoints.atom_scores(verdict, rubric)

    # Also compute raw counts
    points = verdict.get("points") or {}
    all_atoms = judge.atoms(rubric)
    covered_count = sum(1 for a in all_atoms if (points.get(a["id"]) or {}).get("covered"))
    supported_count = sum(
        1
        for a in all_atoms
        if (points.get(a["id"]) or {}).get("covered") and (points.get(a["id"]) or {}).get("supported") is True
    )
    core_atoms = [a for a in all_atoms if a.get("importance") == "core"]
    core_covered = sum(1 for a in core_atoms if (points.get(a["id"]) or {}).get("covered"))
    core_supported = sum(
        1
        for a in core_atoms
        if (points.get(a["id"]) or {}).get("covered") and (points.get(a["id"]) or {}).get("supported") is True
    )

    return {
        "verdict": verdict,
        "scores": scores,
        "total_atoms": len(all_atoms),
        "covered_atoms": covered_count,
        "supported_atoms": supported_count,
        "core_atoms": len(core_atoms),
        "core_covered": core_covered,
        "core_supported": core_supported,
        "unsupported_sentences": len(verdict.get("unsupported") or []),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-One-Out Ablation Study on 'max'")
    parser.add_argument("--topic", help="Specific topic_id to run (default: all 4)")
    parser.add_argument("--variant", help="Specific variant to run (default: all 6)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    topics = [t for t in TOPICS if not args.topic or t["id"] == args.topic]
    variants = [v for v in VARIANTS if not args.variant or v == args.variant]

    print(f"=== LOO Ablation Study: {len(topics)} topics x {len(variants)} variants ===")
    print(f"Chat model: {transport_chat_model()}")
    print(f"Results dir: {RESULTS_DIR}\n")

    summary_rows = []

    for t_idx, topic_meta in enumerate(topics, 1):
        topic_id = topic_meta["id"]
        topic_text = topic_meta["topic"]
        rubric_path = ROOT / "evals" / "rubrics" / f"{topic_id}.json"
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))

        print(f"\n[{t_idx}/{len(topics)}] TOPIC: {topic_id}")
        print(f"Goal: {topic_text[:80]}...")

        topic_results: dict[str, dict[str, Any]] = {}

        for v_idx, variant in enumerate(variants, 1):
            out_file = RESULTS_DIR / f"{topic_id}__{variant}.json"
            if out_file.exists() and not args.force:
                print(f"  ({v_idx}/{len(variants)}) {variant}: loading cached {out_file.name}")
                saved = json.loads(out_file.read_text(encoding="utf-8"))
                topic_results[variant] = saved
            else:
                print(f"  ({v_idx}/{len(variants)}) {variant}: running research...", end="", flush=True)
                record = await execute_run(topic_id, topic_text, variant)
                print(f" done ({record['elapsed_s']}s). Judging...", end="", flush=True)
                eval_res = await judge_record(record, rubric)
                print(f" done (covered: {eval_res['covered_atoms']}/{eval_res['total_atoms']})")

                combined = {
                    "topic_id": topic_id,
                    "variant": variant,
                    "elapsed_s": record["elapsed_s"],
                    "cost_usd": record["envelope"].get("cost_usd", 0.0),
                    "cost_breakdown": record["envelope"].get("cost_breakdown", {}),
                    "requests_used": record["envelope"].get("requests_used", 0),
                    "rounds": len(record["envelope"].get("rounds", [])),
                    "eval": eval_res,
                    "envelope": record["envelope"],
                    "dump": record["dump"],
                }
                out_file.write_text(json.dumps(combined, indent=2), encoding="utf-8")
                topic_results[variant] = combined

        # Compute LOO deltas against full_max for this topic
        full = topic_results.get("full_max", {})
        full_q = full.get("eval", {}).get("supported_atoms", 0)
        full_c = full.get("cost_usd", 0.0)

        for variant in variants:
            res = topic_results.get(variant, {})
            ev = res.get("eval", {})
            q = ev.get("supported_atoms", 0)
            cov_q = ev.get("covered_atoms", 0)
            c = res.get("cost_usd", 0.0)
            t = res.get("elapsed_s", 0.0)
            unsupported = ev.get("unsupported_sentences", 0)

            delta_q = full_q - q
            delta_c = round(full_c - c, 4)
            cost_per_q = round(delta_c / delta_q, 4) if delta_q > 0 else 0.0

            summary_rows.append(
                {
                    "topic": topic_id,
                    "variant": variant,
                    "time_s": t,
                    "cost_usd": c,
                    "covered_q": cov_q,
                    "supported_q": q,
                    "total_atoms": ev.get("total_atoms", 0),
                    "delta_q": delta_q,
                    "delta_cost": delta_c,
                    "cost_per_q": cost_per_q,
                    "unsupported_s": unsupported,
                }
            )

    # Print final summary table
    print("\n" + "=" * 105)
    print("FINAL SUMMARY: LEAVE-ONE-OUT (LOO) ABLATION & MARGINAL COST PER QUANTUM")
    print("=" * 105)
    header = (
        f"{'Topic':<24} | {'Variant':<14} | {'Time':>5} | {'Cost':>7} | "
        f"{'Atoms (Supp/Tot)':>16} | {'Δ Quanta':>8} | {'Δ Cost':>8} | {'$/Quantum':>9} | {'Unsupp.S':>8}"
    )
    print(header)
    print("-" * 105)
    for row in summary_rows:
        atoms_str = f"{row['supported_q']}/{row['total_atoms']}"
        if row["delta_q"] > 0:
            dq_str = f"-{row['delta_q']}"
        elif row["delta_q"] < 0:
            dq_str = f"+{abs(row['delta_q'])}"
        else:
            dq_str = "0"

        if row["delta_cost"] > 0:
            dc_str = f"-${row['delta_cost']:.4f}"
        elif row["delta_cost"] < 0:
            dc_str = f"+${abs(row['delta_cost']):.4f}"
        else:
            dc_str = "$0.0000"

        cpq_str = f"${row['cost_per_q']:.4f}" if row["delta_q"] > 0 else "N/A"
        print(
            f"{row['topic']:<24} | {row['variant']:<14} | {row['time_s']:>4.0f}s | "
            f"${row['cost_usd']:>6.4f} | {atoms_str:>16} | {dq_str:>8} | {dc_str:>8} | "
            f"{cpq_str:>9} | {row['unsupported_s']:>8}"
        )
    print("=" * 105)


def transport_chat_model() -> str:
    from deepseek_search_everywhere_mcp import transport

    return transport._chat_model("deepseek-flash")


if __name__ == "__main__":
    asyncio.run(main())
