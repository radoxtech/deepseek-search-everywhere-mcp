# ADR 0014 — Universal Multi-Perspective Decomposition, Verbatim Metric Preservation, and Dynamic Gap Detection

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** Empirical benchmark across 3 diagnostic domains (104 rubric atoms); experiment `E31` in `evals/EXPERIMENTS.md`
- **Related:** `0009-agent-framework.md`, `0013-synthesis-source-pruning-and-prompt-optimization.md`

## Context

Detailed analysis of missing atoms across diagnostic benchmarks (104 rubric atoms across medicine, politics, and IT)
revealed two distinct failure modes causing atom loss:

1. **Type I: Retrieval Loss (~70% of lost atoms):**
   Subquery decomposition frequently fell into narrow conceptual traps. For example, on `politics-eu-cbam` (EU Carbon
   Border Adjustment Mechanism), queries only searched general definitions and covered sectors, completely omitting
   international disputes (Russia WTO challenge, Poland's EU Court of Justice lawsuit, COP30 dialogues, and India FTA
   annexures). In scraped context, occurrences of these critical rubric atoms was literally 0.
2. **Type II: Metric Fading in Synthesis (~30% of lost atoms):**
   Numerical percentages and clinical trial outcomes existed verbatim in retrieved sources (e.g., 44% nausea, 30%
   diarrhea, 5–7% discontinuation, 2038 free allocation phase-out), but synthesis models frequently generalized them into
   imprecise qualitative descriptions such as "frequent adverse events", "substantial discontinuation", or "later dates".

### Strict Architectural Constraint

**All prompts and heuristics must remain 100% universal and domain-agnostic.** The server is a general-purpose MCP for
any topic (medicine, law, IT, politics, history, finance, physics). No hardcoded domain taxonomies or topic-specific
keywords are permitted.

## Benchmark Results

We evaluated the universal remedies across the 3 benchmark topics:

| Topic | Rubric Atoms | Baseline (Before) | Arm 1 (Metric Directive) | Arm 3 (Agent Dynamic Gaps) |
|---|:---:|:---:|:---:|:---:|
| `politics-eu-cbam` | 33 | 19 / 33 (57.6%) | 23 / 33 (69.7%) | **27 / 33 (81.8%)** |
| `it-pg-incremental-backup` | 40 | 36 / 40 (90.0%) | **39 / 40 (97.5%)** | 38 / 40 (95.0%) |
| `medicine-glp1` | 31 | 16 / 31 (51.6%) | **24 / 31 (77.4%)** | 11 / 31 (35.5%)* |
| **Combined** | **104** | **71 / 104 (68.3%)** | **86 / 104 (82.7%)** | — |

*\* Note on `medicine-glp1` in live search Arm 3:* Live search query 5 over-indexed on contemporary regulatory news
(2025/2026 epidemiological dispensing studies, SELECT cardiovascular approvals, SUMMIT HFpEF) which filled the 20-source
ceiling and crowded out foundational clinical trials (STEP 1 / SURMOUNT-1). This establishes an important design rule:
*aspect decomposition must anchor foundational primary benchmarks before branching into contemporary news.*

## Decisions

1. **Universal Multi-Perspective Aspect Decomposition (`DECOMPOSE_SYSTEM_PROMPT`):**
   Enforce that aspect planning distributes across 4 distinct universal factual dimensions:
   - Primary/mechanism: what the primary standard, law, trial, or documentation specifies.
   - Numbers: measured quantitative results, sizes, rates, prices, percentages, or dates with scope.
   - Limits: limitations, risks, criticism, adverse effects, legal challenges, or open disputes.
   - Current: latest status, revisions, official proposals, amendments, or deadlines as of today.
2. **Universal Verbatim Metric Preservation (Rule 14 & `REPORT_ASK`):**
   Add Rule 14 to `build_synthesis_system_prompt` and explicit metric preservation directives to `REPORT_ASK` in
   `agent_research.py`. Instruct the synthesis writer never to generalize numbers, percentages, rates, dates, sample
   sizes, doses, or statutory thresholds into descriptive words.
3. **LLM Dynamic Gap Inspector in Agent Research (`_gap_queries_model`):**
   In `agent_research.py`, replace static keyword heuristics in round 2+ with `_gap_queries_model` using
   `deepseek-flash` to inspect accumulated source summaries against planned aspects and generate targeted keyword
   queries for missing factual dimensions.
4. **Round-Robin Diversification and Subquery-Aware Salience Pruning:**
   Ensure `rank_and_diversify_sources` round-robins across each query's result list so gap queries are guaranteed
   representation in top synthesis sources. Ensure `prune_source_content` includes terms from all planned and executed
   subqueries so secondary aspect evidence is preserved.

## Consequences

- Factual atom yield improved from **68.3% to 82.7%** on frozen contexts, and reached **81.8%** on previously intractable
  political dispute retrieval.
- Verbatim numerical metrics (rates, dates, thresholds) are reliably preserved and cited rather than obscured by
  generalizing prose.
- The pipeline remains 100% domain-agnostic with zero domain-specific rules.
