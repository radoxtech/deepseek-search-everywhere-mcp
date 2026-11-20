"""Focused Parallel Synthesis Benchmark (E27).

Compares 3 synthesis architectures on identical gathered context across 4 domains (117 atoms):
- Arm A: synth_1pass (Single-pass synthesis, standard fallback)
- Arm B: synth_sequential (Sequential sectional synthesis)
- Arm C: synth_focused_parallel (Focused Parallel Synthesis: plan -> asyncio.gather -> assembly)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
    deep_research,
    runtime,
)
from deepseek_search_everywhere_mcp.deep_research import (  # noqa: E402
    DEPTH_CONFIG,
    citation_ids,
    filter_synthesis_output,
    ground_report_numbers,
    normalize_citations,
    report_body_words,
    sanitize_json,
)
from deepseek_search_everywhere_mcp.sanitize import sanitize_untrusted_text  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results" / "parallel_synthesis"

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

FOCUSED_PLAN_PROMPT = """You design the section structure of an in-depth research report.
Given the research topic, key facts to establish, and source digests:
1. Divide topic into 3-4 logical sections covering decisive aspects without overlap or fluff.
2. For each section, provide a clear title and a 1-sentence "focus" stating what it must answer.
3. List in "unsettled" any key aspects that the gathered sources do NOT contain evidence for.

Output strictly valid JSON with this exact schema:
{
  "sections": [
    {"title": "Section Title", "focus": "What this section answers"}
  ],
  "unsettled": [
    "Aspect or question not settled by sources"
  ]
}
"""

BLUEPRINT_PLAN_PROMPT = (
    "You are a Principal Research Architect. Given a research topic and source digests, produce an\n"
    "exhaustive Section Blueprint for an in-depth report.\n\n"
    "Instructions:\n"
    "1. Identify decisive factual entities in the sources: named trials/studies (e.g. STEP, SURMOUNT,\n"
    "   pg_basebackup), numbers, mechanisms, boxed warnings/contraindications, and limits.\n"
    "2. Structure the report into 3 to 5 deep, non-overlapping sections.\n"
    "3. For EACH section, provide:\n"
    "   - 'title': Specific markdown section heading\n"
    "   - 'focus': The primary analytical question this section must answer\n"
    "   - 'must_cover': List of 3-5 concrete named entities, trials, specific numbers, or critical warnings\n"
    "     found in the sources that MUST be included in this section.\n"
    "4. 'unsettled': List aspects where gathered sources lack data.\n\n"
    "Output strictly valid JSON with schema:\n"
    "{\n"
    '  "sections": [\n'
    "    {\n"
    '      "title": "Section Heading",\n'
    '      "focus": "Core analytical question",\n'
    '      "must_cover": ["entity / number / trial 1", "warning / metric 2"]\n'
    "    }\n"
    "  ],\n"
    '  "unsettled": ["Unanswered question 1"]\n'
    "}\n"
)

AUDIT_MATRIX_PROMPT = (
    "You are an Executive Research Editor.\n"
    "Given the research topic, the blueprint requirements, and the drafted report sections:\n"
    "1. Construct an Executive Comparison Matrix (markdown table) highlighting the core metrics, clinical\n"
    "   trial endpoints, numbers, or key technical parameters.\n"
    "2. Check if any critical item from the required blueprint was completely omitted from the drafted\n"
    "   sections. If so, write 1-2 dense paragraphs under heading '## Supplementary Key Findings' citing\n"
    "   sources. If all blueprint items were covered, do NOT add supplementary paragraphs.\n\n"
    "Output strictly valid markdown starting with:\n"
    "## Executive Summary & Key Findings Matrix\n"
    "[Markdown Table]\n\n"
    "(if needed)\n"
    "## Supplementary Key Findings\n"
    "[Paragraphs with citations]\n"
)


def _clean_section_heading(text: str, expected_title: str) -> str:
    """Ensure section text starts with ## expected_title."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("#"):
        # Strip leading heading if model wrote one
        rest = "\n".join(lines[1:]).strip()
        return f"## {expected_title}\n\n{rest}"
    return f"## {expected_title}\n\n{text.strip()}"


async def run_synth_1pass(
    topic: str,
    context: list[dict[str, Any]],
    depth: str = "max",
) -> tuple[str, float, float]:
    """Arm A: Single-pass synthesis."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    messages = deep_research.build_synthesis_messages(topic, context, depth=depth)
    t0 = time.monotonic()
    res = await runtime._chat_request(
        messages=messages,
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=cfg.max_tokens,
        timeout=180.0,
        deadline=180.0,
        attempts=1,
    )
    elapsed = time.monotonic() - t0
    cost = float(res.get("cost_usd", 0.0) or 0.0)
    report = res.get("content") or ""
    return report, cost, elapsed


async def run_synth_sequential(
    topic: str,
    context: list[dict[str, Any]],
    depth: str = "max",
) -> tuple[str, float, float]:
    """Arm B: Sequential Sectional Synthesis."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    base_msgs = deep_research.build_synthesis_messages(topic, context, depth=depth)
    system = base_msgs[0]["content"]
    ctx_content = base_msgs[1]["content"]

    t0 = time.monotonic()
    # Step 1: Plan
    plan_src = [{"id": s["id"], "title": s["title"]} for s in context]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": FOCUSED_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nSources: {json.dumps(plan_src)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=30.0,
        deadline=30.0,
        attempts=1,
    )
    total_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        # Fallback to 1pass
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, total_cost + c, time.monotonic() - t0

    per_sec_words = max(150, cfg.target_words // len(sections))
    parts: list[str] = []

    for sec in sections:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        ask = (
            f"\n\n---\n### Section Writing Task:\n"
            f"Section: {title}\nFocus: {focus}\nTarget length: ~{per_sec_words} words.\n"
            f"Write ONLY the text for this section starting directly with markdown heading '## {title}'. "
            f"Cite sources [n] strictly."
        )
        res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx_content}{ask}"},
            ],
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(1200, per_sec_words * 3),
            timeout=90.0,
            deadline=90.0,
            attempts=1,
        )
        total_cost += float(res.get("cost_usd", 0.0) or 0.0)
        content = (res.get("content") or "").strip()
        if content:
            parts.append(_clean_section_heading(content, title))

    if unsettled:
        unsettled_text = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        parts.append(unsettled_text)

    report = "\n\n".join(parts)
    elapsed = time.monotonic() - t0
    return report, total_cost, elapsed


async def run_synth_focused_parallel(
    topic: str,
    context: list[dict[str, Any]],
    depth: str = "max",
) -> tuple[str, float, float]:
    """Arm C: Focused Parallel Synthesis (asyncio.gather with token caps)."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    base_msgs = deep_research.build_synthesis_messages(topic, context, depth=depth)
    system = base_msgs[0]["content"]
    ctx_content = base_msgs[1]["content"]

    t0 = time.monotonic()
    # Step 1: Plan
    plan_src = [{"id": s["id"], "title": s["title"]} for s in context]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": FOCUSED_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nSources: {json.dumps(plan_src)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=30.0,
        deadline=30.0,
        attempts=1,
    )
    plan_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        # Fallback to 1pass
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + c, time.monotonic() - t0

    per_sec_words = max(150, cfg.target_words // len(sections))

    async def _write_one(sec: dict[str, Any]) -> tuple[str, str, float]:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        ask = (
            f"\n\n---\n### Section Writing Task:\n"
            f"Section: {title}\nFocus: {focus}\nTarget length: ~{per_sec_words} words.\n"
            f"Write ONLY the text for this section starting directly with markdown heading '## {title}'. "
            f"Cite sources [n] strictly."
        )
        res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx_content}{ask}"},
            ],
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(1200, per_sec_words * 3),
            timeout=90.0,
            deadline=90.0,
            attempts=1,
        )
        c = float(res.get("cost_usd", 0.0) or 0.0)
        cnt = (res.get("content") or "").strip()
        return title, cnt, c

    # Step 2: Parallel execution via asyncio.gather
    sec_results = await asyncio.gather(*(_write_one(s) for s in sections), return_exceptions=True)

    sec_parts: list[str] = []
    sec_cost = 0.0
    for _idx, r in enumerate(sec_results):
        if isinstance(r, tuple):
            title, cnt, c = r
            sec_cost += c
            if cnt:
                sec_parts.append(_clean_section_heading(cnt, title))

    if not sec_parts:
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + sec_cost + c, time.monotonic() - t0

    if unsettled:
        unsettled_text = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        sec_parts.append(unsettled_text)

    report = "\n\n".join(sec_parts)
    total_cost = plan_cost + sec_cost
    elapsed = time.monotonic() - t0
    return report, total_cost, elapsed


async def run_synth_anchored_3step(
    topic: str,
    context: list[dict[str, Any]],
    depth: str = "max",
) -> tuple[str, float, float]:
    """Arm D: Fact-Anchored Parallel Synthesis (3-step: Blueprint with must_cover -> parallel -> assembly)."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    base_msgs = deep_research.build_synthesis_messages(topic, context, depth=depth)
    system = base_msgs[0]["content"]
    ctx_content = base_msgs[1]["content"]

    t0 = time.monotonic()
    # Step 1: Blueprint with must_cover
    src_summary = [
        {"id": s["id"], "title": s["title"], "says": (s.get("content") or "")[:400]} for s in context[:16]
    ]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": BLUEPRINT_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nSources: {json.dumps(src_summary)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=2048,
        response_format={"type": "json_object"},
        timeout=35.0,
        deadline=35.0,
        attempts=1,
    )
    plan_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + c, time.monotonic() - t0

    per_sec_words = max(150, cfg.target_words // len(sections))

    async def _write_one(sec: dict[str, Any]) -> tuple[str, str, float]:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        must_cover = sec.get("must_cover") or []
        must_cover_text = ""
        if must_cover:
            must_cover_text = "Required findings to cover in this section:\n" + "\n".join(
                f"- {item}" for item in must_cover
            )
        ask = (
            f"\n\n---\n### Section Writing Task:\n"
            f"Section: {title}\nFocus: {focus}\n"
            f"{must_cover_text}\n\n"
            f"Target length: ~{per_sec_words} words.\n"
            f"Write ONLY the text for this section starting directly with markdown heading '## {title}'. "
            f"Cite sources [n] strictly."
        )
        res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx_content}{ask}"},
            ],
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(1200, per_sec_words * 3),
            timeout=90.0,
            deadline=90.0,
            attempts=1,
        )
        c = float(res.get("cost_usd", 0.0) or 0.0)
        cnt = (res.get("content") or "").strip()
        return title, cnt, c

    # Step 2: Parallel execution via asyncio.gather
    sec_results = await asyncio.gather(*(_write_one(s) for s in sections), return_exceptions=True)

    sec_parts: list[str] = []
    sec_cost = 0.0
    for _idx, r in enumerate(sec_results):
        if isinstance(r, tuple):
            title, cnt, c = r
            sec_cost += c
            if cnt:
                sec_parts.append(_clean_section_heading(cnt, title))

    if not sec_parts:
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + sec_cost + c, time.monotonic() - t0

    if unsettled:
        unsettled_text = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        sec_parts.append(unsettled_text)

    report = "\n\n".join(sec_parts)
    total_cost = plan_cost + sec_cost
    elapsed = time.monotonic() - t0
    return report, total_cost, elapsed


async def run_synth_anchored_4step(
    topic: str,
    context: list[dict[str, Any]],
    depth: str = "max",
) -> tuple[str, float, float]:
    """Arm E: Fact-Anchored 4-Step Synthesis (Blueprint -> parallel -> Executive Matrix/Audit -> assembly)."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    base_msgs = deep_research.build_synthesis_messages(topic, context, depth=depth)
    system = base_msgs[0]["content"]
    ctx_content = base_msgs[1]["content"]

    t0 = time.monotonic()
    # Step 1: Blueprint with must_cover
    src_summary = [
        {"id": s["id"], "title": s["title"], "says": (s.get("content") or "")[:400]} for s in context[:16]
    ]
    plan_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": BLUEPRINT_PLAN_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\nSources: {json.dumps(src_summary)}"},
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=2048,
        response_format={"type": "json_object"},
        timeout=35.0,
        deadline=35.0,
        attempts=1,
    )
    plan_cost = float(plan_res.get("cost_usd", 0.0) or 0.0)
    parsed, _ = sanitize_json(plan_res.get("content") or "{}")
    sections = parsed.get("sections") if isinstance(parsed, dict) else []
    unsettled = parsed.get("unsettled") if isinstance(parsed, dict) else []

    if not sections:
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + c, time.monotonic() - t0

    per_sec_words = max(150, cfg.target_words // len(sections))

    async def _write_one(sec: dict[str, Any]) -> tuple[str, str, float]:
        title = sec.get("title", "Section")
        focus = sec.get("focus", "")
        must_cover = sec.get("must_cover") or []
        must_cover_text = ""
        if must_cover:
            must_cover_text = "Required findings to cover in this section:\n" + "\n".join(
                f"- {item}" for item in must_cover
            )
        ask = (
            f"\n\n---\n### Section Writing Task:\n"
            f"Section: {title}\nFocus: {focus}\n"
            f"{must_cover_text}\n\n"
            f"Target length: ~{per_sec_words} words.\n"
            f"Write ONLY the text for this section starting directly with markdown heading '## {title}'. "
            f"Cite sources [n] strictly."
        )
        res = await runtime._chat_request(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx_content}{ask}"},
            ],
            thinking={"type": "enabled"},
            model=runtime.REWRITE_MODEL,
            max_tokens=min(1200, per_sec_words * 3),
            timeout=90.0,
            deadline=90.0,
            attempts=1,
        )
        c = float(res.get("cost_usd", 0.0) or 0.0)
        cnt = (res.get("content") or "").strip()
        return title, cnt, c

    # Step 2: Parallel execution via asyncio.gather
    sec_results = await asyncio.gather(*(_write_one(s) for s in sections), return_exceptions=True)

    sec_parts: list[str] = []
    sec_cost = 0.0
    for _idx, r in enumerate(sec_results):
        if isinstance(r, tuple):
            title, cnt, c = r
            sec_cost += c
            if cnt:
                sec_parts.append(_clean_section_heading(cnt, title))

    if not sec_parts:
        rep, c, _ = await run_synth_1pass(topic, context, depth)
        return rep, plan_cost + sec_cost + c, time.monotonic() - t0

    # Step 3 (The 4th Step): Executive Matrix & Audit Pass
    drafted_text = "\n\n".join(sec_parts)
    audit_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": AUDIT_MATRIX_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Blueprint Sections & Requirements:\n{json.dumps(sections)}\n\n"
                    f"Drafted Sections:\n{drafted_text[:12000]}"
                ),
            },
        ],
        thinking={"type": "disabled"},
        temperature=0.0,
        model=runtime.REWRITE_MODEL,
        max_tokens=1500,
        timeout=35.0,
        deadline=35.0,
        attempts=1,
    )
    audit_cost = float(audit_res.get("cost_usd", 0.0) or 0.0)
    matrix_content = (audit_res.get("content") or "").strip()

    # Step 4: Deterministic Assembly
    assembled_parts = []
    if matrix_content:
        assembled_parts.append(matrix_content)
    assembled_parts.extend(sec_parts)

    if unsettled:
        unsettled_text = "## What the sources do not settle\n\n" + "\n".join(
            f"- {u}" for u in unsettled if str(u).strip()
        )
        assembled_parts.append(unsettled_text)

    report = "\n\n".join(assembled_parts)
    total_cost = plan_cost + sec_cost + audit_cost
    elapsed = time.monotonic() - t0
    return report, total_cost, elapsed


def finalize_and_ground(
    report_text: str, context: list[dict[str, Any]], depth: str = "max"
) -> dict[str, Any]:
    """Finalize report: filter, normalize citations, sanitize, and verify numbers."""
    cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["max"])
    allowed_urls = {s["url"].strip() for s in context if isinstance(s.get("url"), str) and s["url"].strip()}
    filtered, injection_suspected, filter_notes = filter_synthesis_output(report_text, allowed_urls)
    report = normalize_citations(filtered, max_id=len(context))
    words = report_body_words(report)
    condense_notes = []
    if words > cfg.max_words:
        condense_notes = [
            f"Report has {words} words, over the {cfg.max_words}-word ceiling for depth {depth!r}; "
            "it was kept as written."
        ]
    report, sanitize_notes = sanitize_untrusted_text(report)
    report, unverified_numbers, unverified_count, unverified_notes = ground_report_numbers(
        report, context, drop=False
    )
    notes = filter_notes + condense_notes + sanitize_notes + unverified_notes
    return {
        "report": report,
        "words": words,
        "notes": notes,
        "unverified_count": unverified_count,
        "citation_count": len(citation_ids(report)),
    }


async def judge_report(
    topic_id: str, topic_text: str, report: str, context: list[dict[str, Any]], rubric: dict[str, Any]
) -> dict[str, Any]:
    """Judge report against rubric using deepseek-flash."""
    atoms_text = judge._rubric_text(topic_id, use_atoms=True)
    evidence_lines = []
    for s in context:
        s_id = s.get("id")
        content = s.get("content") or ""
        evidence_lines.append(f"[{s_id}] {content[:1000]}")

    prompt = (
        f"# Topic\n{topic_text}\n\n"
        f"# Key points\n{atoms_text}\n\n"
        f"# Report\n{report}\n\n"
        f"# Evidence (cited sources)\n" + "\n".join(evidence_lines)
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

    verdict, _ = sanitize_json(res.get("content") or "{}")
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


async def evaluate_topic(meta: dict[str, Any], arms: list[str], force: bool = False) -> dict[str, Any]:
    topic_id = meta["id"]
    topic_text = meta["topic"]
    rubric_path = ROOT / "evals" / "rubrics" / f"{topic_id}.json"
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))

    # Load context from LOO results
    loo_path = ROOT / "evals" / "results" / "loo" / f"{topic_id}__full_max.json"
    if not loo_path.exists():
        raise FileNotFoundError(f"Missing LOO run for {topic_id} at {loo_path}")
    loo_data = json.loads(loo_path.read_text(encoding="utf-8"))
    context = loo_data["dump"]["synthesis_context"]

    topic_results: dict[str, Any] = {}

    for arm in arms:
        out_file = RESULTS_DIR / f"{topic_id}__{arm}.json"
        if out_file.exists() and not force:
            print(f"  [cached] {arm}")
            topic_results[arm] = json.loads(out_file.read_text(encoding="utf-8"))
            continue

        print(f"  Running {arm}...", end="", flush=True)
        if arm == "synth_1pass":
            raw_report, cost, elapsed = await run_synth_1pass(topic_text, context)
        elif arm == "synth_sequential":
            raw_report, cost, elapsed = await run_synth_sequential(topic_text, context)
        elif arm == "synth_focused_parallel":
            raw_report, cost, elapsed = await run_synth_focused_parallel(topic_text, context)
        elif arm == "synth_anchored_3step":
            raw_report, cost, elapsed = await run_synth_anchored_3step(topic_text, context)
        elif arm == "synth_anchored_4step":
            raw_report, cost, elapsed = await run_synth_anchored_4step(topic_text, context)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        final = finalize_and_ground(raw_report, context)
        print(f" done ({elapsed:.1f}s, ${cost:.4f}, {final['words']} words). Judging...", end="", flush=True)
        judged = await judge_report(topic_id, topic_text, final["report"], context, rubric)
        print(f" supported {judged['supported_atoms']}/{judged['total_atoms']}")

        rec = {
            "topic_id": topic_id,
            "topic": topic_text,
            "arm": arm,
            "elapsed_s": round(elapsed, 2),
            "cost_usd": round(cost, 6),
            "words": final["words"],
            "notes": final["notes"],
            "unverified_count": final["unverified_count"],
            "citation_count": final["citation_count"],
            "eval": judged,
            "report": final["report"],
        }
        out_file.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        topic_results[arm] = rec

    return topic_results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Focused Parallel Synthesis Benchmark")
    parser.add_argument("--topic", help="Specific topic_id to run (default: all 4)")
    parser.add_argument(
        "--arm",
        help="Specific arm: synth_1pass, synth_sequential, synth_focused_parallel",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    topics = [t for t in TOPICS if not args.topic or t["id"] == args.topic]
    arms = [args.arm] if args.arm else ["synth_1pass", "synth_sequential", "synth_focused_parallel"]

    print(f"=== Focused Parallel Synthesis Benchmark ({len(topics)} topics x {len(arms)} arms) ===")
    print(f"Model: {os.environ.get('DEEPSEEK_CHAT_MODEL', 'default')}")
    print(f"Results dir: {RESULTS_DIR}\n")

    all_data = {}
    for meta in topics:
        print(f"\n[Topic] {meta['id']}")
        res = await evaluate_topic(meta, arms, force=args.force)
        all_data[meta["id"]] = res

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Topic':<26} | {'Arm':<24} | {'Supported':<12} | {'Words':<7} | {'Cost':<9} | {'Time':<7}")
    print("-" * 90)
    for topic_id, t_arms in all_data.items():
        for arm_name, r in t_arms.items():
            ev = r.get("eval", {})
            supp = f"{ev.get('supported_atoms', 0)}/{ev.get('total_atoms', 0)}"
            words = str(r.get("words", 0))
            cost = f"${r.get('cost_usd', 0.0):.4f}"
            sec = f"{r.get('elapsed_s', 0.0):.1f}s"
            print(f"{topic_id:<26} | {arm_name:<24} | {supp:<12} | {words:<7} | {cost:<9} | {sec:<7}")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
