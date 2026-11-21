# ADR 0013 — Pre-synthesis source salience pruning and prompt optimization

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** Empirical synthesis evaluation across 5 domains (174 rubric atoms); experiment `E30` in
  `evals/EXPERIMENTS.md`
- **Related:** `0005-multi-stage-synthesis-rejected.md`, `0011-elimination-of-post-synthesis-condensation-and-gap-fetch-streamlining.md`

## Context

In `deepseek_quick_research`, single-pass report synthesis operated over raw scraped page content
accumulating 450,000 to 800,000 characters across 15–20 sources. In addition, Rule 2 of the synthesis
system prompt instructed the reasoning model to list facts and sources sequentially in its reasoning trace.
This caused three major issues:

1. **High Latency:** Synthesis calls took 65–98 seconds, dominated by reasoning loops sequentially transcribing
   raw context.
2. **Attention Dilution:** Key metrics, edge-case warnings, and limits were frequently diluted by site
   boilerplate (navigation bars, footers, legal terms).
3. **High Token Costs:** Passing 500k–800k characters into chat completions billed excessive prompt tokens
   per research job.

We evaluated six candidate remedy architectures across five diverse domains (174 rubric atoms):
1. IT: PostgreSQL 17 incremental backup (40 atoms)
2. Medicine: GLP-1 receptor agonists & metabolic mechanisms (31 atoms)
3. Law: EU AI Act General Purpose AI regulatory obligations (35 atoms)
4. Security: Passkeys WebAuthn/FIDO2 credential lifecycle (35 atoms)
5. Politics: EU Carbon Border Adjustment Mechanism (33 atoms)

### Comparative Benchmark Results

| Arm | Architecture | Atoms (174) | Coverage | Avg Wall (s) | Cost ($) | $/Atom |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **`arm0_baseline`** | Monolith unpruned (800k) + verbose listing rule | 156 / 174 | 89.7% | 65.0s | $0.0944 | $0.00061 |
| **`arm1_prompt_surgery`** | Monolith + direct outline planning | 157 / 174 | 90.2% | 51.4s | $0.0890 | $0.00057 |
| **`arm2_focused_parallel`** | Map-Reduce parallel section generation | 150 / 174 | 86.2% | 19.1s | $0.0286 | $0.00019 |
| **`arm3_source_pruning`** | Salience pruning (<=8k/source) + prompt surgery | **158 / 174** | **90.8%** | **39.8s** | **$0.0144** | **$0.00009** |
| **`arm4_hybrid_plan`** | Thinking Blueprint + non-thinking Writer | 146 / 174 | 83.9% | 32.2s | $0.0350 | $0.00024 |
| **`arm5_hybrid_ab`** | Pruned Map-Reduce parallel sections | 142 / 174 | 81.6% | 21.8s | $0.0198 | $0.00014 |

### Architectural Insights

1. **The Partitioning Penalty of Map-Reduce and Hybrids (Arms 2 & 5):**
   Partitioning context across isolated parallel section writers produced significant factual losses
   (down to 68–77% in medicine and law). Complex sources spanning multiple dimensions (e.g., clinical
   efficacy vs adverse risk warnings, or regulatory thresholds vs compliance exemptions) cannot be cited
   by sections they were not assigned to. Single-pass synthesis with global context visibility is
   essential for cross-sectional synthesis.
2. **Salience Pruning + Prompt Surgery (Arm 3 / Option A):**
   By filtering page content to top-scoring paragraphs (favoring numbers, percentages, benchmark keywords,
   and aspect terms) capped at 8,000 characters per source, prompt characters were reduced by ~75% while
   factual yield actually increased (+2 atoms over baseline, 90.8% vs 89.7%). Latency fell from 65.0s
   to 39.8s (1.6x faster), and total synthesis cost dropped by **6.6x** ($0.0144 vs $0.0944).

## Decision

1. **Adopt Extractive Salience Pruning (`prune_source_content`):**
   In `research_jobs.py` (`_research_synthesize`), prune each source's text to its most salient paragraphs
   (up to 8,000 characters per source) based on term overlap with topic, subqueries, and aspect criteria,
   as well as number and metric density.
2. **Optimize Synthesis System Prompt (Rule 2):**
   Instruct the reasoning model to plan its outline directly rather than transcribing raw facts sequentially
   into its reasoning trace.
3. **Preserve Full Context for Grounding & Envelope:**
   Retain the full original scraped sources for `ground_report_numbers`, citation validation, and public
   envelope formatting.
4. **Reject Multi-Agent Sectional Partitioning for Default Quick Research:**
   Map-Reduce sectional synthesis is rejected due to its unacceptable partitioning penalty on cross-domain
   research topics.

## Consequences

- 1.6x–2.5x reduction in single-pass synthesis latency.
- 6.6x reduction in prompt token billing for synthesis ($0.00009/atom vs $0.00061/atom).
- Increased factual density and atom coverage without attention dilution from web page boilerplate.
