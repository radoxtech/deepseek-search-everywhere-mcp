# ADR 0012 — Modern context window and scaled research limit architecture

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** Live empirical browser extraction of `llm-stats.com` (Pages 1 & 2, ranks #1–60), confirming
  that 1.0M–1.1M tokens is the universal standard context window across all top-tier foundation models
  (OpenAI GPT-6/5, Anthropic Claude Opus/Sonnet/Haiku, Google Gemini 4/3.8, DeepSeek V4.1/V4, Moonshot Kimi,
  Qwen 4/3.8, Zhipu GLM, Meta Muse, Mistral Large, Xiaomi MiMo); Experiment `E28` in `evals/EXPERIMENTS.md`.
- **Related:** `0001-report-word-budgets.md`, `0004-depth-tiers.md`, `0006-writer-at-ceiling.md`,
  `0007-where-the-ceiling-is.md`, `0011-elimination-of-post-synthesis-condensation-and-gap-fetch-streamlining.md`

## Context

The research pipeline previously operated under obsolete constraints designed for legacy LLMs with small
32k/64k context windows:

1. **Artificially suffocated word budgets (`target_words`, `max_words`):**
   Depth `max` gathered up to 24 rich sources (500,000 characters), yet forced synthesis into a rigid
   3,200-word target (3,840-word ceiling). Squeezing 500k characters into 3.2k words caused severe loss
   of factual atoms, clinical trial outcomes, and technical parameters, yielding terse, laconic reports.

2. **Rigid section straightjackets (`SECTION_COUNTS`):**
   Synthesis prompts mandated "exactly 6 sections of ~530 words". When researching broad multi-faceted
   topics covering 8–12 distinct clinical trials, legal regulations, or software modules, forcing 6
   sections compelled models to discard or merge critical research aspects.

3. **Constrained research brief inputs (`MAX_TOPIC = 500`):**
   Complex user briefs containing background, hypotheses, requirements, and constraints were rejected
   with `topic_too_long` whenever they exceeded 500 characters (~70 words).

4. **Suffocated output generation token headroom:**
   Previous pipeline configurations reserved small generation windows (e.g. 8k–16k tokens). For modern
   reasoning LLMs where chain-of-thought (`<think>`) and visible output tokens share the completion budget,
   deep thinking easily consumes 20k–50k reasoning tokens, causing synthesis requests to prematurely abort
   with `finish_reason="length"` before the actual report is fully generated.

5. **Client synthesis ceiling (`MAX_CLIENT_REPORT_CHARS = 60_000`):**
   Subagent orchestrators writing reports from paged context were arbitrarily capped at 60k chars (~9k words).

6. **Autonomous research round limits (`DEPTH_ROUNDS`):**
   Depth `max` in `deepseek_deep_agent_research` was capped at 3 rounds (identical to `high`), curtailing
   iterative discovery when subsequent rounds still yielded valuable numeric facts.

## Decision

1. **Scale topic input length:**
   Raise `MAX_TOPIC` from 500 to 4,000 characters across `tools/research.py` and `tools/agent.py`. User
   briefs may now provide complete research specifications and constraints.

2. **Modernize `DEPTH_CONFIG` source and word budgets:**
   - **`low`:** `target_words=1000`, `max_words=1200`, `per_page_char_cap=20_000`, `total_source_chars=80_000`,
     `max_tokens=16_384`, `deadline_s=60.0`.
   - **`medium`:** `target_words=2000` (was 1600), `max_words=2400` (was 1920), `k_sources=12` (was 10),
     `per_page_char_cap=30_000` (was 20_000), `total_source_chars=240_000` (was 150_000), `max_tokens=32_768`,
     `deadline_s=180.0` (was 150.0).
   - **`high`:** `target_words=3500` (was 2400), `max_words=4200` (was 2880), `subqueries=5` (was 4),
     `k_sources=20` (was 16), `per_page_char_cap=40_000` (was 25_000), `total_source_chars=450_000` (was 300_000),
     `max_tokens=65_536`, `deadline_s=300.0` (was 240.0).
   - **`max`:** `target_words=5000` (was 3200), `max_words=6000` (was 3840), `subqueries=6` (was 4),
     `count=12` (was 10), `k_sources=30` (was 24), `per_page_char_cap=50_000` (was 30_000),
     `total_source_chars=800_000` (was 500_000; ~200k tokens input, comfortably within 1M context),
     `max_tokens=131_072`, `deadline_s=420.0` (was 360.0).
   *Rationale for `max_tokens`:* In reasoning mode (`thinking=True`), API completion limits encapsulate both
   internal chain-of-thought tokens (which generate up to 20k–50k reasoning tokens under `reasoning_effort="high"`)
   and visible output report tokens. DeepSeek official API supports up to 384k generation tokens (defaulting to 64k–128k
   in thinking mode). Providing 32k, 64k, and 128k completion headroom ensures reasoning models never truncate mid-thought.

3. **Fix `deepseek_fetch` multi-URL total characters ceiling bug:**
   When callers specify `max_chars` on a URL list, scale `max_total_chars` to `max(DEFAULT_MAX_TOTAL_CHARS, max_chars * len(urls))`
   instead of hardcoding `DEFAULT_MAX_TOTAL_CHARS = 20_000`, preventing silent truncation to 20k chars.

3. **Expand section counts (`SECTION_COUNTS`):**
   Update to `{"low": 2, "medium": 4, "high": 5, "max": 8}`. This provides sufficient structural space
   for multi-aspect reporting (500–700 words per section) without artificial fact compression.

4. **Expand client synthesis ceiling:**
   Raise `MAX_CLIENT_REPORT_CHARS` from 60,000 to 120,000 characters (~18,000 words).

5. **Expand autonomous agent research depth:**
   Raise `DEPTH_ROUNDS["max"]` from 3 to 5 rounds, `MAX_ROUNDS = 5`, and `MAX_STEPS = 15`.

6. **Scale synthesis token guard:**
   Update `MAX_EST_SYNTHESIS_TOKENS` from 600,000 to 1,000,000 tokens in `deep_research.py`.

## Consequences

- Research reports on high and max tiers achieve comprehensive factual coverage without truncation.
- Users can provide multi-paragraph detailed research briefs without hitting 500-character errors.
- Completion token requests align with empirical frontier provider limits (128k–384k output tokens),
  giving reasoning models ample headroom for extensive thinking traces while generating up to 6,000 words.
- All 1260 unit and integration tests remain 100% green.
