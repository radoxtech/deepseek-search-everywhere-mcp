# ADR 0011 — Elimination of post-synthesis condensation and gap fetch streamlining

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** Leave-One-Out (LOO) ablation study on 4 domains (117 atoms); experiments `E06`, `E20–E24`,
  `E26`, `E27` in `evals/EXPERIMENTS.md`; 24 live LOO runs and 12 controlled synthesis benchmark runs
- **Related:** `0001-report-word-budgets.md`, `0006-writer-at-ceiling.md`, `0010-tool-surface-pruning-and-taxonomy.md`

## Context

Two mechanisms in the research pipeline introduced hidden overhead and quality degradation without
producing measurable factual yield:

1. **Post-synthesis condensation (`_condense_report`):**
   DeepSeek chat models often exceed soft word targets specified in prompts. To enforce `max_words`,
   the pipeline previously dispatched a second chat completion call to compress over-length reports.
   Measurements across pilot topics (E06) and the 4-domain LOO study showed that this second call:
   - Increased synthesis latency and billed redundant output tokens.
   - Truncated verified factual atoms and numbers that were present in the first pass.
   - Dropped or mismatched citation indices (`[n]`), causing valid claims to fail downstream verification.
   Retaining reports as synthesized with a clear ceiling note in `notes` (the rule already used for
   client synthesis reports) preserves every verified fact without destructive edits.

2. **Redundant secondary gap fetching (`_fill_gaps` in research):**
   Research search requests set `need_content=True` (`content="full"`), which directs the DeepSeek
   Search API to extract and deliver complete Markdown text (averaging 12,500 characters per page)
   directly within the search response. The pipeline subsequently invoked `_fill_gaps`, which checked
   `RESEARCH_GAP_CHARS = 200`. Because search results already exceeded 200 characters in over 99% of cases,
   this phase incurred scheduling and progress-reporting overhead for an operation that performed zero fetches.

## Decision

1. **Eliminate post-synthesis condensation:**
   Remove `_condense_report` from `research_jobs.py`. When a synthesized report exceeds `max_words`, it is
   retained as written, and an informative notice is added to `notes`:
   `"Report has {words} words, over the {max_words}-word ceiling for depth {depth!r}; it was kept as written."`
   Word length control is managed structurally through section planning and token caps.

2. **Streamline research source fetching:**
   Rely directly on the DeepSeek Search API's native `need_content=True` full-text scraping. Retain
   `_fill_gaps` only as a safety fallback for rare search results that return truly empty content (<200 chars).

3. **Synthesis architecture decision (experiment E27):**
   - Sequential sectional synthesis is rejected (lowest factual yield at 72.6%, slowest, highest cost).
   - Single-pass synthesis (`_research_synthesize`) is retained as the production default across all tiers,
     delivering the highest overall factual quantum yield (80.3% supported atoms) at the lowest cost ($0.0048/report).
   - Focused Parallel Synthesis (`asyncio.gather` with hard token caps) is verified as a valid, high-speed
     alternative (2.5x faster, 17.7s avg), but remains experimental until planning prompts can guarantee
     zero partitioning penalty on broad clinical topics.

## Consequences

- No verified facts or valid citations are lost to a secondary compression LLM call.
- Reduced overall research latency and lower token billing per job.
- Clean, deterministic report finalization pipeline.
