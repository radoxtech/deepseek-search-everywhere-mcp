"""Evaluation of 4 Solutions to the Quick Research Synthesis Bottleneck.

Investigates remedies for the 87-98s single-pass thinking latency on 450k-800k context:
- arm0_baseline: Standard single-pass synthesis with full context and unrestricted thinking
- arm1_prompt_surgery: Prompt-level fix (no raw facts transcription in reasoning)
- arm2_focused_parallel: Map-Reduce (outline planer -> concurrent section writers)
- arm3_source_pruning: Pre-synthesis salience pruning (shrinks to top dense paragraphs)
- arm4_hybrid_plan_no_thinking: Stage 1 Reasoning Blueprint + Stage 2 Fast Writer

Evaluates atomic facts across IT (PG-17: 40 atoms) and Medicine (GLP-1: 31 atoms), total 71 atoms.
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
    runtime,
)
from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    DEPTH_CONFIG,
    filter_synthesis_output,
    normalize_citations,
    report_body_words,
    sanitize_json,
)

TOPICS = [
    {
        "id": "it-pg-incremental-backup",
        "domain": "it",
        "topic": (
            "How PostgreSQL 17 incremental backups work (pg_basebackup --incremental, "
            "WAL summarization, pg_combinebackup) and their limitations"
        ),
        "context_file": "results/step_comparison/it-pg-incremental-backup__quick_full.json",
        "query_terms": {
            "postgresql",
            "pg_basebackup",
            "incremental",
            "wal",
            "summarization",
            "pg_combinebackup",
            "walsummarizer",
            "backup_label",
            "lsn",
            "manifest",
        },
    },
    {
        "id": "medicine-glp1",
        "domain": "medicine",
        "topic": (
            "Semaglutide and tirzepatide for obesity: weight-loss efficacy in "
            "clinical trials and main safety concerns"
        ),
        "context_file": "results/step_comparison/medicine-glp1__quick_full.json",
        "query_terms": {
            "semaglutide",
            "tirzepatide",
            "step",
            "surpass",
            "surmount",
            "weight",
            "obesity",
            "glp-1",
            "gip",
            "pancreatitis",
            "thyroid",
            "nausea",
            "bmi",
        },
    },
    {
        "id": "law-eu-ai-act-gpai",
        "domain": "law",
        "topic": (
            "EU AI Act obligations for providers of general-purpose AI models: "
            "requirements, systemic-risk models, code of practice and deadlines"
        ),
        "context_file": "results/loo/law-eu-ai-act-gpai__full_max.json",
        "query_terms": {
            "gpai",
            "general-purpose",
            "ai",
            "act",
            "systemic",
            "flops",
            "10^25",
            "copyright",
            "transparency",
            "code",
            "practice",
            "fine",
            "obligation",
            "deadline",
        },
    },
    {
        "id": "it-passkeys",
        "domain": "it",
        "topic": (
            "Passkeys (WebAuthn/FIDO2): how they work, security benefits over passwords "
            "and remaining limitations for adoption"
        ),
        "context_file": "results/w-k20-high/it-passkeys__k__high__r1.json",
        "query_terms": {
            "passkey",
            "passkeys",
            "webauthn",
            "fido2",
            "fido",
            "credential",
            "private",
            "public",
            "key",
            "biometric",
            "phishing",
            "sync",
            "attestation",
            "ctap",
        },
    },
    {
        "id": "politics-eu-cbam",
        "domain": "politics",
        "topic": (
            "The EU Carbon Border Adjustment Mechanism (CBAM): how it works, "
            "implementation timeline and international political reactions"
        ),
        "context_file": "results/w-k20-high/politics-eu-cbam__k__high__r1.json",
        "query_terms": {
            "cbam",
            "carbon",
            "border",
            "adjustment",
            "ets",
            "allowance",
            "embedded",
            "emissions",
            "steel",
            "cement",
            "fertilizer",
            "aluminum",
            "hydrogen",
            "tariff",
        },
    },
]

ARMS = [
    "arm0_baseline",
    "arm1_prompt_surgery",
    "arm2_focused_parallel",
    "arm3_source_pruning",
    "arm4_hybrid_plan_no_thinking",
    "arm5_hybrid_ab",
]

RESULTS_DIR = ROOT / "evals" / "results" / "quick_remedies"


# ---------------------------------------------------------------------------
# Arm 1: Prompt Surgery
# ---------------------------------------------------------------------------
def build_prompt_surgery_system(depth: str) -> str:
    """Build synthesis prompt instructing direct synthesis without raw fact listing."""
    base = dr.build_synthesis_system_prompt(depth, client=False)
    # Replace the listing rule that causes 70s of reasoning transcription
    old_rule = (
        "2. COVERAGE BEFORE STYLE: before writing, list in your reasoning the distinct key facts in the "
        "sources that bear on the topic: numbers, dates, names, versions, mechanisms, conditions and "
        "limitations. Every fact that answers a listed research aspect must reach the report; add the "
        "others as space allows, most specific first. When space runs short, shorten the wording, not the "
        "facts."
    )
    new_rule = (
        "2. DIRECT SYNTHESIS WITHOUT FACT TRANSCRIPTION: Plan your outline and thematic structure directly "
        "in reasoning. Do not sequentially transcribe or list raw facts or sources in reasoning; synthesize "
        "claims, metrics, and citations directly into the report sections. Every fact answering a research "
        "aspect must reach the report. When space runs short, shorten the wording, not the facts."
    )
    if old_rule in base:
        return base.replace(old_rule, new_rule)
    return base


async def run_arm1_prompt_surgery(
    topic: str, context: list[dict[str, Any]], depth: str = "high"
) -> tuple[str, float, float]:
    """Execute Arm 1: Single-pass on full context with streamlined reasoning instructions."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()
    messages = dr.build_synthesis_messages(topic, context, depth=depth)
    messages[0]["content"] = build_prompt_surgery_system(depth)

    res = await runtime._chat_request(
        messages=messages,
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=cfg.max_tokens,
        timeout=180.0,
        deadline=180.0,
        attempts=1,
    )
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    report = (res.get("content") or "").strip()
    elapsed = round(time.monotonic() - t0, 2)
    return report, cost, elapsed


# ---------------------------------------------------------------------------
# Arm 2: Focused Parallel Synthesis (Map-Reduce)
# ---------------------------------------------------------------------------
PARALLEL_PLAN_PROMPT = """You are a Research Report Architect.
Given a research topic and available sources, create an analytical section plan (3 to 4 sections).
Assign to each section the exact list of source IDs [id] that contain evidence for that section.

Return strictly valid JSON:
{
  "sections": [
    {
      "title": "Section Title",
      "focus": "1-2 sentences stating the specific questions and metrics this section addresses",
      "source_ids": [1, 2, 5]
    }
  ],
  "unsettled": [
    "Key question or detail that gathered sources do NOT contain evidence for"
  ]
}
"""


async def run_arm2_focused_parallel(
    topic: str, context: list[dict[str, Any]], depth: str = "high"
) -> tuple[str, float, float]:
    """Execute Arm 2: Map-Reduce parallel synthesis with partitioned sources."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()

    # Step 1: Lightweight Outline Planer on compact digest
    digest_items = [
        {"id": s["id"], "title": s["title"], "snippet": (s.get("content") or "")[:400]} for s in context
    ]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": PARALLEL_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nSources:\n{json.dumps(digest_items)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1500,
        response_format={"type": "json_object"},
        timeout=30.0,
        deadline=30.0,
        attempts=2,
    )
    total_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        # Fallback to single-pass if planner fails
        return await run_arm1_prompt_surgery(topic, context, depth)

    sources_by_id = {s["id"]: s for s in context}
    per_sec_target = max(350, cfg.target_words // len(sections))

    async def _write_section(sec: dict[str, Any]) -> tuple[str, float]:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        sec_ids = [sid for sid in sec.get("source_ids", []) if sid in sources_by_id]
        if not sec_ids:
            sec_ids = [s["id"] for s in context[:4]]
        sec_sources = [sources_by_id[sid] for sid in sec_ids]

        # Compact context for this section only
        sec_messages = dr.build_synthesis_messages(topic, sec_sources, depth=depth)
        prompt_instruction = (
            f"\n\n---\n### SECTION WRITING ASSIGNMENT:\n"
            f"You are writing ONLY the section titled: '## {title}'\n"
            f"Focus: {focus}\n"
            f"Target length: ~{per_sec_target} words. Cite source IDs like [n] strictly.\n"
            f"Output markdown text starting with '## {title}'."
        )
        sec_messages[1]["content"] += prompt_instruction

        sres = await runtime._chat_request(
            messages=sec_messages,
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(3000, per_sec_target * 4),
            timeout=120.0,
            deadline=120.0,
            attempts=1,
        )
        scost = float(sres.get("cost_usd", 0.0) or 0.0)
        scontent = (sres.get("content") or "").strip()
        # Clean heading
        cleaned = re.sub(r"^#+\s*", "", scontent.split("\n")[0]).strip() if scontent else title
        body = "\n".join(scontent.split("\n")[1:]).strip() if scontent else ""
        return f"## {cleaned}\n\n{body}", scost

    # Step 2: Concurrent section generation
    sec_results = await asyncio.gather(*(_write_section(s) for s in sections), return_exceptions=True)

    sec_texts = []
    for r in sec_results:
        if isinstance(r, tuple):
            text, c = r
            total_cost += c
            sec_texts.append(text)

    # Step 3: Assembly
    if unsettled:
        unsettled_md = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        sec_texts.append(unsettled_md)

    full_report = "\n\n".join(sec_texts)
    elapsed = round(time.monotonic() - t0, 2)
    return full_report, total_cost, elapsed


# ---------------------------------------------------------------------------
# Arm 3: Source Pruning (Salience Extractive Pruning)
# ---------------------------------------------------------------------------
def prune_source_content(content: str, query_terms: set[str], max_chars: int = 6000) -> str:
    """Extract top-scoring paragraphs by query term density and metrics to shrink page noise."""
    if len(content) <= max_chars:
        return content
    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 40]
    if not paragraphs:
        paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 40]
    if not paragraphs:
        return content[:max_chars]

    scored = []
    for idx, p in enumerate(paragraphs):
        p_lower = p.lower()
        score = sum(3 for term in query_terms if term in p_lower)
        # Bonus for numerical metrics, percentages, versions
        digits = sum(1 for ch in p if ch.isdigit())
        score += min(6, digits // 2)
        if any(w in p_lower for w in ("limit", "error", "warning", "table", "benchmark", "result")):
            score += 3
        scored.append((score, idx, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked_indices = []
    accum_len = 0
    for _score, idx, p in scored:
        if accum_len + len(p) > max_chars and picked_indices:
            continue
        picked_indices.append(idx)
        accum_len += len(p)
        if accum_len >= max_chars:
            break

    picked_indices.sort()
    return "\n\n".join(paragraphs[i] for i in picked_indices)


async def run_arm3_source_pruning(
    topic: str,
    context: list[dict[str, Any]],
    query_terms: set[str],
    depth: str = "high",
) -> tuple[str, float, float]:
    """Execute Arm 3: Salience-pruned sources feeding single-pass synthesis with thinking."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()

    # Prune each source to ~6,000 dense characters
    pruned_context = []
    for s in context:
        s_copy = dict(s)
        s_copy["content"] = prune_source_content(s.get("content") or "", query_terms, max_chars=6000)
        pruned_context.append(s_copy)

    messages = dr.build_synthesis_messages(topic, pruned_context, depth=depth)
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


# ---------------------------------------------------------------------------
# Arm 4: Hybrid Plan + Fast Writer (No Thinking)
# ---------------------------------------------------------------------------
BLUEPRINT_PROMPT = """You are a Senior Research Analyst.
Given a topic and compact source digests, produce an analytical research blueprint for the report.
For each section:
1. Provide a descriptive title.
2. List the mandatory numerical data points, names, versions, and clinical/technical facts to include.
3. List which source IDs [id] prove those facts.

Return strictly valid JSON:
{
  "sections": [
    {
      "title": "Section Title",
      "must_include_facts": ["Specific fact or metric with [id] citation"]
    }
  ],
  "unsettled_aspects": ["Any question not answered by sources"]
}
"""


async def run_arm4_hybrid_plan_no_thinking(
    topic: str, context: list[dict[str, Any]], depth: str = "high"
) -> tuple[str, float, float]:
    """Execute Arm 4: Stage 1 Reasoning Blueprint on digest + Stage 2 Fast Writer without thinking."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()

    # Stage 1: Blueprint with lightweight reasoning on digest
    digest_items = [
        {"id": s["id"], "title": s["title"], "snippet": (s.get("content") or "")[:400]} for s in context
    ]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": BLUEPRINT_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nSources:\n{json.dumps(digest_items)}"},
        ],
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=2048,
        response_format={"type": "json_object"},
        timeout=45.0,
        deadline=45.0,
        attempts=1,
    )
    plan_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    blueprint_raw = plan_res.get("content") or "{}"

    # Stage 2: Fast Single-Pass Writer on full context (thinking disabled)
    messages = dr.build_synthesis_messages(topic, context, depth=depth)
    blueprint_injection = (
        f"\n\n---\n### REQUIRED FACTUAL BLUEPRINT & OUTLINE:\n"
        f"You MUST organize the report following this blueprint and cover every fact listed:\n"
        f"{blueprint_raw}\n\n"
        f"Write the complete, highly detailed research report now. Cite sources [n] strictly."
    )
    messages[1]["content"] += blueprint_injection

    writer_res = await runtime._chat_request(
        messages=messages,
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=cfg.max_tokens,
        timeout=150.0,
        deadline=150.0,
        attempts=1,
    )
    writer_cost = float(writer_res.get("cost_usd", 0.0) or 0.0)
    report = (writer_res.get("content") or "").strip()
    total_cost = plan_cost + writer_cost
    elapsed = round(time.monotonic() - t0, 2)
    return report, total_cost, elapsed


# ---------------------------------------------------------------------------
# Arm 5: Hybrid A+B (Salience Pruning + Focused Parallel + Prompt Surgery)
# ---------------------------------------------------------------------------
async def run_arm_hybrid_ab(
    topic: str,
    context: list[dict[str, Any]],
    query_terms: set[str],
    depth: str = "high",
) -> tuple[str, float, float]:
    """Execute Hybrid A+B: Pruned dense sources + Map-Reduce parallel synthesis + Prompt surgery."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["high"])
    t0 = time.monotonic()

    # Step 1: Pre-synthesis salience pruning (cuts boilerplate/noise to dense facts)
    pruned_context = []
    for s in context:
        s_copy = dict(s)
        s_copy["content"] = prune_source_content(s.get("content") or "", query_terms, max_chars=6000)
        pruned_context.append(s_copy)

    # Step 2: Lightweight outline planer on dense digests
    digest_items = [
        {"id": s["id"], "title": s["title"], "snippet": (s.get("content") or "")[:400]}
        for s in pruned_context
    ]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": PARALLEL_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n\nSources:\n{json.dumps(digest_items)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1500,
        response_format={"type": "json_object"},
        timeout=30.0,
        deadline=30.0,
        attempts=2,
    )
    total_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        return await run_arm3_source_pruning(topic, context, query_terms, depth)

    sources_by_id = {s["id"]: s for s in pruned_context}
    per_sec_target = max(350, cfg.target_words // len(sections))

    async def _write_hybrid_section(sec: dict[str, Any]) -> tuple[str, float]:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        sec_ids = [sid for sid in sec.get("source_ids", []) if sid in sources_by_id]
        if not sec_ids:
            sec_ids = [s["id"] for s in pruned_context[:4]]
        sec_sources = [sources_by_id[sid] for sid in sec_ids]

        # Compact context with prompt surgery system prompt
        sec_messages = dr.build_synthesis_messages(topic, sec_sources, depth=depth)
        sec_messages[0]["content"] = build_prompt_surgery_system(depth)
        prompt_instruction = (
            f"\n\n---\n### SECTION WRITING ASSIGNMENT:\n"
            f"You are writing ONLY the section titled: '## {title}'\n"
            f"Focus: {focus}\n"
            f"Target length: ~{per_sec_target} words. Cite source IDs like [n] strictly from your sources.\n"
            f"Synthesize directly without transcribing raw facts in reasoning. Start with '## {title}'."
        )
        sec_messages[1]["content"] += prompt_instruction

        sres = await runtime._chat_request(
            messages=sec_messages,
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(2500, per_sec_target * 3),
            timeout=90.0,
            deadline=90.0,
            attempts=1,
        )
        scost = float(sres.get("cost_usd", 0.0) or 0.0)
        scontent = (sres.get("content") or "").strip()
        cleaned = re.sub(r"^#+\s*", "", scontent.split("\n")[0]).strip() if scontent else title
        body = "\n".join(scontent.split("\n")[1:]).strip() if scontent else ""
        return f"## {cleaned}\n\n{body}", scost

    # Step 3: Concurrent execution via asyncio.gather
    sec_results = await asyncio.gather(*(_write_hybrid_section(s) for s in sections), return_exceptions=True)

    sec_texts = []
    for r in sec_results:
        if isinstance(r, tuple):
            text, c = r
            total_cost += c
            sec_texts.append(text)

    # Step 4: Assembly
    if unsettled:
        unsettled_md = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        sec_texts.append(unsettled_md)

    full_report = "\n\n".join(sec_texts)
    elapsed = round(time.monotonic() - t0, 2)
    return full_report, total_cost, elapsed


# ---------------------------------------------------------------------------
# Judge Execution
# ---------------------------------------------------------------------------
async def judge_report(
    topic_id: str,
    topic: str,
    report: str,
    context: list[dict[str, Any]],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    """Judge a synthesized report against rubric atoms."""
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
            verdict_retry, _ = sanitize_json(retry_res.get("content") or "{}")
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
    }


# ---------------------------------------------------------------------------
# Main Driver
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4 remedies for synthesis bottleneck")
    parser.add_argument("--topic", help="Specific topic_id to run (default: both)")
    parser.add_argument("--arm", help="Specific arm to run (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    topics = [t for t in TOPICS if not args.topic or t["id"] == args.topic]
    arms = [a for a in ARMS if not args.arm or a == args.arm]

    print(f"=== Evaluating {len(arms)} Remedies across {len(topics)} Topics ===")
    results_summary = []

    for t in topics:
        tid = t["id"]
        t_topic = t["topic"]
        qterms = t["query_terms"]

        rubric_path = ROOT / "evals" / "rubrics" / f"{tid}.json"
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))

        # Load identical gathered context
        ctx_file = ROOT / "evals" / t["context_file"]
        ctx_data = json.loads(ctx_file.read_text(encoding="utf-8"))
        context = (
            ctx_data.get("dump", {}).get("synthesis_context")
            or ctx_data.get("run", {}).get("dump", {}).get("synthesis_context")
            or []
        )
        total_chars = sum(len(s.get("content") or "") for s in context)

        print("\n========================================================")
        print(f"Topic: {tid} ({len(context)} sources, ~{total_chars} chars)")
        print("========================================================")

        for arm in arms:
            out_file = RESULTS_DIR / f"{tid}__{arm}.json"
            print(f"\n---> Running [{arm}] on {tid} ...", flush=True)

            if arm == "arm0_baseline":
                # Stored baseline from full run
                envelope = ctx_data.get("envelope") or ctx_data.get("run", {}).get("envelope") or {}
                rep_text = envelope.get("report") or ""
                cost = float(ctx_data.get("cost_usd") or 0.02)
                wall = float(ctx_data.get("wall_s") or ctx_data.get("elapsed_s") or 80.0)
            elif arm == "arm1_prompt_surgery":
                rep_text, cost, wall = await run_arm1_prompt_surgery(t_topic, context)
            elif arm == "arm2_focused_parallel":
                rep_text, cost, wall = await run_arm2_focused_parallel(t_topic, context)
            elif arm == "arm3_source_pruning":
                rep_text, cost, wall = await run_arm3_source_pruning(t_topic, context, qterms)
            elif arm == "arm4_hybrid_plan_no_thinking":
                rep_text, cost, wall = await run_arm4_hybrid_plan_no_thinking(t_topic, context)
            elif arm == "arm5_hybrid_ab":
                rep_text, cost, wall = await run_arm_hybrid_ab(t_topic, context, qterms)
            else:
                raise ValueError(f"Unknown arm {arm}")

            # Filter & normalize report
            allowed_urls = {s["url"] for s in context if isinstance(s.get("url"), str)}
            filtered, _, _ = filter_synthesis_output(rep_text, allowed_urls)
            norm_report = normalize_citations(filtered, max_id=len(context))
            words = report_body_words(norm_report)

            # Judge against rubric
            print(f"     Judging report ({words} words, wall={wall}s, cost=${cost:.4f}) ...", flush=True)
            j_res = await judge_report(tid, t_topic, norm_report, context, rubric)

            sup = j_res["supported_atoms"]
            tot = j_res["total_atoms"]
            pct = sup / tot * 100 if tot else 0.0
            cost_per_atom = cost / sup if sup else 0.0

            print(
                f"     RESULT: Supported={sup}/{tot} ({pct:.1f}%), Words={words}, "
                f"Cost=${cost:.4f}, Wall={wall}s, $/Atom=${cost_per_atom:.5f}"
            )

            record = {
                "topic_id": tid,
                "arm": arm,
                "wall_s": wall,
                "cost_usd": cost,
                "words": words,
                "total_atoms": tot,
                "supported_atoms": sup,
                "cost_per_atom": cost_per_atom,
                "report": norm_report,
                "judgement": j_res,
            }
            out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
            results_summary.append(record)

    # Print summary table
    header = (
        f"{'Arm':<28} | {'Topic':<14} | {'Supported':<14} | "
        f"{'Words':<7} | {'Cost ($)':<9} | {'Wall (s)':<8} | {'$/Atom'}"
    )
    print("\n\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results_summary:
        tid_short = r["topic_id"][:12]
        pct_val = r["supported_atoms"] / r["total_atoms"] * 100
        sup_str = f"{r['supported_atoms']}/{r['total_atoms']} ({pct_val:.1f}%)"
        print(
            f"{r['arm']:<28} | {tid_short:<14} | {sup_str:<14} | {r['words']:<7} | "
            f"${r['cost_usd']:<8.4f} | {r['wall_s']:<8.1f} | ${r['cost_per_atom']:.5f}"
        )
    print("=" * len(header))


if __name__ == "__main__":
    asyncio.run(main())
