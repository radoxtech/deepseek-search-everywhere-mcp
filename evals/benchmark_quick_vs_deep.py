"""Comprehensive Benchmark: Quick Research vs Agent Deep Research across Contested Topics.

Evaluates on 10 difficult, contested, adversarial, and multi-perspective topics
with rigorous atomic rubrics (388 atoms total).

Compares:
- Quick Research (deepseek_quick_research)
- Deep Agent Research (deepseek_deep_agent_research)

Metrics per run:
- Latency (seconds)
- Cost (USD)
- Stage breakdown & timings
- Discovered queries & sources
- Covered atoms (from ground truth rubric)
- Supported atoms (backed by cited evidence passages)
- Unverified numbers count
- Explicit handling of uncertainties & disputes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

load_dotenv()

import judge  # noqa: E402
import keypoints  # noqa: E402

from deepseek_search_everywhere_mcp import (  # noqa: E402
    agent_research,
    research_jobs,
    runtime,
)
from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    ResearchJob,
    sanitize_json,
)

RESULTS_DIR = ROOT / "evals" / "results" / "quick_vs_deep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = [
    {
        "id": "finance-basel-endgame",
        "domain": "Finance & Bank Capital",
        "topic": (
            "Basel III endgame in the United States: what the capital rules require, "
            "how the proposals changed, and their implementation status"
        ),
        "atoms": 60,
    },
    {
        "id": "law-eu-ai-act-gpai",
        "domain": "AI Law & Regulatory Policy",
        "topic": (
            "EU AI Act obligations for providers of general-purpose AI models: "
            "requirements, systemic-risk models, code of practice and deadlines"
        ),
        "atoms": 35,
    },
    {
        "id": "politics-eu-cbam",
        "domain": "International Trade & Climate",
        "topic": (
            "The EU Carbon Border Adjustment Mechanism (CBAM): how it works, "
            "implementation timeline and international political reactions"
        ),
        "atoms": 33,
    },
    {
        "id": "medicine-glp1",
        "domain": "Clinical Pharmacology & Lawsuits",
        "topic": (
            "Semaglutide and tirzepatide for obesity: weight-loss efficacy in "
            "clinical trials and main safety concerns"
        ),
        "atoms": 31,
    },
    {
        "id": "medicine-sglt2-vs-glp1",
        "domain": "Comparative Trials (Cardio/Renal)",
        "topic": (
            "SGLT2 inhibitors versus GLP-1 receptor agonists in type 2 diabetes: "
            "cardiovascular and renal outcome trial results and main safety findings"
        ),
        "atoms": 32,
    },
    {
        "id": "engineering-solid-state-batteries",
        "domain": "Materials Science & Automotive",
        "topic": (
            "Solid-state batteries for electric vehicles: how they work, the "
            "main technical obstacles, and realistic commercialization timelines"
        ),
        "atoms": 60,
    },
    {
        "id": "nutrition-mediterranean-vs-lowfat",
        "domain": "Epidemiology & Retractions",
        "topic": (
            "Mediterranean diet versus low-fat diet for cardiovascular events: "
            "PREDIMED, CORDIOPREV and Lyon Diet Heart trial results and the main "
            "criticisms of the evidence"
        ),
        "atoms": 31,
    },
    {
        "id": "nutrition-time-restricted-eating",
        "domain": "Metabolic Health & Controversy",
        "topic": (
            "Evidence on time-restricted eating (intermittent fasting) versus "
            "continuous calorie restriction for weight loss and metabolic health"
        ),
        "atoms": 31,
    },
    {
        "id": "it-passkeys",
        "domain": "Cybersecurity & Identity",
        "topic": (
            "Passkeys (WebAuthn/FIDO2): how they work, security benefits over "
            "passwords and remaining limitations for adoption"
        ),
        "atoms": 35,
    },
    {
        "id": "it-pg-incremental-backup",
        "domain": "Database Architecture",
        "topic": (
            "How PostgreSQL 17 incremental backups work (pg_basebackup --incremental, "
            "WAL summarization, pg_combinebackup) and their limitations"
        ),
        "atoms": 40,
    },
]


async def run_quick_research(
    topic: str, depth: str = "medium"
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float]:
    """Execute deepseek_quick_research end-to-end and measure performance."""
    t0 = time.monotonic()
    job_id = f"bench-quick-{uuid.uuid4().hex[:8]}"
    job = ResearchJob(
        job_id=job_id,
        topic=topic,
        depth=depth,
        lang=None,
        sources="any",
        unverified="flag",
        synthesis="server",
        created_at=runtime._clock(),
    )
    res = await research_jobs._run_research_job(
        job=job,
        clean_topic=topic,
        depth=depth,
        sources_pref="any",
        lang=None,
        refresh=True,
        strip_boilerplate=True,
    )
    wall = round(time.monotonic() - t0, 2)
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    context = getattr(job.run, "sources", []) if job.run else []
    return res, context, cost, wall


async def run_deep_agent_research(
    topic: str, depth: str = "medium"
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float]:
    """Execute deepseek_deep_agent_research end-to-end and measure performance."""
    t0 = time.monotonic()
    job_id = f"bench-agent-{uuid.uuid4().hex[:8]}"
    job = ResearchJob(
        job_id=job_id,
        topic=topic,
        depth=depth,
        lang=None,
        sources="any",
        kind="agent",
        max_rounds=5,
        max_cost_usd=None,
        created_at=runtime._clock(),
    )
    res = await agent_research.run_agent_job(
        job=job,
        clean_topic=topic,
        depth=depth,
        sources_pref="any",
        lang=None,
        refresh=True,
        strip_boilerplate=True,
    )
    wall = round(time.monotonic() - t0, 2)
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    context = getattr(job.run, "sources", []) if job.run else []
    return res, context, cost, wall


async def judge_report(
    topic_id: str,
    topic: str,
    report: str,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run binary rubric atom judge using official KEYPOINT_INSTRUCTIONS."""
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
    unsupported = verdict.get("unsupported") or []
    return {
        "verdict": verdict,
        "total_atoms": total,
        "covered_atoms": covered,
        "supported_atoms": supported,
        "unsupported_sentences": len(unsupported),
        "cost_usd": cost,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Research vs Deep Agent Research Benchmark")
    parser.add_argument("--topics", nargs="+", help="Specific topic IDs to benchmark")
    parser.add_argument("--depth", default="medium", choices=["low", "medium", "high", "max"])
    parser.add_argument("--dry-run", action="store_true", help="Print selected topics and exit")
    args = parser.parse_args()

    selected = TOPICS
    if args.topics:
        selected = [t for t in TOPICS if t["id"] in args.topics]

    print("\n===============================================================================")
    print(f"BENCHMARK: Quick Research vs Agent Deep Research ({len(selected)} Contested Topics)")
    print(f"Depth: {args.depth} | Total Ground-Truth Atoms: {sum(t['atoms'] for t in selected)}")
    print("===============================================================================\n")

    if args.dry_run:
        for t in selected:
            print(f"- {t['id']:<35} ({t['atoms']} atoms) | {t['domain']}")
        return

    summary: list[dict[str, Any]] = []

    for t in selected:
        tid = t["id"]
        topic_text = t["topic"]
        atoms = t["atoms"]
        print("\n==================================================================")
        print(f"TOPIC: {tid} ({atoms} atoms)")
        print(f"Domain: {t['domain']}")
        print(f"Query:  {topic_text}")
        print("==================================================================")

        # -----------------------------------------------------------------
        # 1. Quick Research
        # -----------------------------------------------------------------
        print(f"\n[1/2] Running deepseek_quick_research (depth={args.depth}) ...")
        q_res, q_ctx, q_cost, q_wall = await run_quick_research(topic_text, depth=args.depth)
        q_rep = q_res.get("report", "")
        q_sources = q_res.get("sources", [])
        q_words = len(q_rep.split())
        print(f"      Done in {q_wall}s | Cost: ${q_cost:.4f} | Words: {q_words} | Sources: {len(q_sources)}")

        print(f"      Judging Quick Research report against {atoms} rubric atoms ...")
        q_judge = await judge_report(tid, topic_text, q_rep, q_ctx)
        q_sup = q_judge["supported_atoms"]
        q_cov = q_judge["covered_atoms"]
        q_rate = round(q_sup / atoms * 100, 1) if atoms else 0.0
        print(f"      -> Quick Score: {q_sup}/{atoms} supported ({q_rate}%), {q_cov} covered")

        # -----------------------------------------------------------------
        # 2. Deep Agent Research
        # -----------------------------------------------------------------
        print(f"\n[2/2] Running deepseek_deep_agent_research (depth={args.depth}) ...")
        a_res, a_ctx, a_cost, a_wall = await run_deep_agent_research(topic_text, depth=args.depth)
        a_rep = a_res.get("report", "")
        a_sources = a_res.get("sources", [])
        a_words = len(a_rep.split())
        a_rounds = a_res.get("rounds", [])
        a_stopped = a_res.get("stopped_reason", "unknown")
        print(
            f"      Done in {a_wall}s | Cost: ${a_cost:.4f} | Words: {a_words} | "
            f"Sources: {len(a_sources)} | Rounds: {len(a_rounds)} | Stop: {a_stopped}"
        )

        print(f"      Judging Agent Deep Research report against {atoms} rubric atoms ...")
        a_judge = await judge_report(tid, topic_text, a_rep, a_ctx)
        a_sup = a_judge["supported_atoms"]
        a_cov = a_judge["covered_atoms"]
        a_rate = round(a_sup / atoms * 100, 1) if atoms else 0.0
        print(f"      -> Agent Score: {a_sup}/{atoms} supported ({a_rate}%), {a_cov} covered")

        delta_atoms = a_sup - q_sup
        delta_rate = round(a_rate - q_rate, 1)
        delta_cost = a_cost - q_cost
        print(
            f"\n>>> Comparison for {tid}: "
            f"Agent delta: {delta_atoms:+d} atoms ({delta_rate:+0.1f} pp) | "
            f"Cost delta: ${delta_cost:+.4f} | Time delta: {a_wall - q_wall:+.1f}s"
        )

        row = {
            "topic_id": tid,
            "domain": t["domain"],
            "atoms": atoms,
            "quick": {
                "wall_s": q_wall,
                "cost_usd": q_cost,
                "words": q_words,
                "sources_count": len(q_sources),
                "supported": q_sup,
                "covered": q_cov,
                "rate": q_rate,
                "unsupported_sentences": q_judge["unsupported_sentences"],
                "report": q_rep,
                "subqueries": q_res.get("subqueries", []),
            },
            "agent": {
                "wall_s": a_wall,
                "cost_usd": a_cost,
                "words": a_words,
                "sources_count": len(a_sources),
                "supported": a_sup,
                "covered": a_cov,
                "rate": a_rate,
                "unsupported_sentences": a_judge["unsupported_sentences"],
                "rounds": a_rounds,
                "facts_per_round": a_res.get("facts_per_round", []),
                "stopped_reason": a_stopped,
                "plan": a_res.get("plan", []),
                "report": a_rep,
            },
            "delta": {
                "supported": delta_atoms,
                "rate_pp": delta_rate,
                "cost_usd": delta_cost,
                "wall_s": round(a_wall - q_wall, 2),
            },
        }
        summary.append(row)

        topic_out = RESULTS_DIR / f"{tid}.json"
        topic_out.write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 90)
    header = (
        f"{'Topic':<28} | {'Atoms':<5} | {'Quick Sup':<10} | {'Agent Sup':<10} | "
        f"{'Delta':<8} | {'Quick $':<8} | {'Agent $':<8} | {'Quick s':<7} | {'Agent s':<7}"
    )
    print(header)
    print("-" * len(header))
    tot_atoms = sum(r["atoms"] for r in summary)
    tot_q_sup = sum(r["quick"]["supported"] for r in summary)
    tot_a_sup = sum(r["agent"]["supported"] for r in summary)
    tot_q_cost = sum(r["quick"]["cost_usd"] for r in summary)
    tot_a_cost = sum(r["agent"]["cost_usd"] for r in summary)
    tot_q_wall = sum(r["quick"]["wall_s"] for r in summary)
    tot_a_wall = sum(r["agent"]["wall_s"] for r in summary)

    for r in summary:
        tid = r["topic_id"][:28]
        atoms = r["atoms"]
        q_s = f"{r['quick']['supported']}/{atoms} ({r['quick']['rate']}%)"
        a_s = f"{r['agent']['supported']}/{atoms} ({r['agent']['rate']}%)"
        delta_s = f"{r['delta']['supported']:+d} ({r['delta']['rate_pp']:+.1f}%)"
        q_c = f"${r['quick']['cost_usd']:.4f}"
        a_c = f"${r['agent']['cost_usd']:.4f}"
        q_w = f"{r['quick']['wall_s']:.1f}s"
        a_w = f"{r['agent']['wall_s']:.1f}s"
        row_str = (
            f"{tid:<28} | {atoms:<5} | {q_s:<10} | {a_s:<10} | "
            f"{delta_s:<8} | {q_c:<8} | {a_c:<8} | {q_w:<7} | {a_w:<7}"
        )
        print(row_str)

    print("-" * len(header))
    q_avg_rate = round(tot_q_sup / tot_atoms * 100, 1) if tot_atoms else 0.0
    a_avg_rate = round(tot_a_sup / tot_atoms * 100, 1) if tot_atoms else 0.0
    tot_delta = tot_a_sup - tot_q_sup
    tot_delta_pp = round(a_avg_rate - q_avg_rate, 1)
    tot_s = (
        f"{'OVERALL TOTAL':<28} | {tot_atoms:<5} | "
        f"{tot_q_sup}/{tot_atoms} ({q_avg_rate}%) | "
        f"{tot_a_sup}/{tot_atoms} ({a_avg_rate}%) | "
        f"{tot_delta:+d} ({tot_delta_pp:+.1f}%) | "
        f"${tot_q_cost:.4f} | ${tot_a_cost:.4f} | {tot_q_wall:.1f}s | {tot_a_wall:.1f}s"
    )
    print(tot_s)
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
