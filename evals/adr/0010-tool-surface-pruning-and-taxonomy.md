# ADR 0010 — Tool surface pruning, research taxonomy, and multilingual guidance

- **Status:** accepted
- **Date:** 2026-11-20
- **Evidence:** Maintainer decisions on LLM cognitive load and parameter anchoring; `E25` in
  `evals/EXPERIMENTS.md`; commits `78bab47` and `279bd91`; 1264 passing unit tests; `test_param_descriptions.py`
- **Related:** `0004-depth-tiers.md`, `0009-agent-framework.md`, `plans/kontrakt-agent-deep-research.md`

## Context

The MCP server's public surface had accumulated numerous secondary parameters across search, fetch, extract,
and research tools (`exclude_terms`, `exact_phrases`, `after`, `before`, `include_domains`, `exclude_domains`,
`rewrite`, `section`, `offset`, `find_text`, `max_snippets`, `max_total_chars`, `deadline`, `max_rounds`,
`max_cost_usd`, `unverified`).

This parameter bloat caused three concrete issues in practice:
1. **Cognitive overload and anchor bias in calling LLMs:** Models calling the MCP tools second-guessed
   internal pagination, character caps, and dates rather than asking the user clear questions or issuing
   simple queries.
2. **Upstream impedance mismatch:** DeepSeek Search API enforces a strict 128-character limit on the raw
   `query` string. Operators synthesized from `include_domains` or `exclude_terms` consumed the query budget
   before actual search keywords could be evaluated.
3. **Ambiguous research tool naming:** The distinction between fast single-pass research and autonomous
   multi-round agents was blurred (`deepseek_research` vs `deepseek_agent_deep_research`), while
   `deepseek_check_research` was inconsistently documented.

## Decision

Execute a breaking change (v3.0.0) that prunes public tool contracts down to their essential parameters,
establishes clear tool personas, and introduces context-adaptive multilingual guidance.

### 1. Research Tool Taxonomy
- **`deepseek_quick_research`** (formerly `deepseek_research`): Fast single-pass cited deep research across
  several search angles (~30–60s). Reads 6–24 sources and returns a synthesized report with `[n]` citations.
- **`deepseek_deep_agent_research`** (formerly `deepseek_agent_deep_research`): Autonomous multi-round
  research agent with dynamic aspect planning, evidence matrix verification, gap analysis, and deterministic
  autostop (~60–180s).
- **`deepseek_check_research`**: Universal polling tool for all background research jobs, source context
  pagination, and client-synthesis answer verification.

### 2. Aggressive Parameter Pruning
Prune all non-essential and mutually exclusive parameters from public MCP signatures:
- **`deepseek_search`**: Pruned `exclude_terms`, `exact_phrases`, `after`, `before`, `include_domains`,
  `exclude_domains`, `rewrite`, `max_total_chars`, `deadline`. Kept: `query`, `count=5`,
  `content="snippets"`, `lang=None`, `refresh=False`.
- **`deepseek_fetch`**: Pruned `section`, `offset`, `find_text`, `max_snippets`, `max_total_chars`,
  `deadline`. Kept: `url`, `query=None`, `max_chars=None`, `refresh=False`.
- **`deepseek_extract`**: Pruned `max_chars`, `max_total_chars`, `deadline`. Kept: `url`, `fields=None`,
  `schema=None`, `instruction=None`, `refresh=False`.
- **`deepseek_quick_research`**: Pruned `unverified` (hardcoded to `"flag"` to prevent sentence deletion).
  Kept: `topic`, `depth="medium"`, `sources="any"`, `lang=None`, `synthesis="server"`, `refresh=False`.
- **`deepseek_deep_agent_research`**: Pruned `max_rounds` (derived directly from `depth`: low=1, med=2,
  high=3, max=3) and `max_cost_usd` (guarded by global `budget.py`). Kept: `topic`, `depth="medium"`,
  `sources="any"`, `lang=None`, `refresh=False`.

### 3. Preserved Engine Capabilities
The underlying retrieval and parsing engines (`search_core._search`, `fetch_core._fetch`,
`extract_core._extract`) retain full implementation of domain filters, selectors, pagination, and
character budgets for programmatic and internal use. Only public tool schemas exposed to calling LLMs are
streamlined.

### 4. Context-Adaptive Multilingual Guidance (E25)
- DeepSeek Search API evaluates one language per search query via `lang:<iso>` prepended to `query`.
- When `lang is None`, decomposition prompts adaptively generate queries: local language for domestic laws,
  entities, or regional context; English for global tech, specs, benchmarks, and academic papers. Standardized
  proper nouns and library names are kept untranslated.
- In `_gap_queries`, hardcoded English suffix words (`data`, `figures`, `report`) are eliminated, preventing
  query corruption on non-English topics.
- Synthesis prompts explicitly instruct the model to produce the final report in the language of the user's
  question, translating facts from foreign-language sources while preserving established industry terminology.

### 5. Client Synthesis Roadmap
`synthesis="client"` in `deepseek_quick_research` and answer verification in `deepseek_check_research` are
preserved for subagent orchestrators (e.g. Claude Code with Opus). Documentation explicitly gates client
synthesis to hosts with subagent capabilities to avoid single-agent context exhaustion.

## Consequences

- Breaking change to the public MCP API: tool names and signatures are modified.
- Calling LLMs experience substantially reduced cognitive friction and no longer anchor on secondary options.
- Zero extra latency or cost: multilingual targeting is integrated into existing decomposition prompts
  rather than an extra LLM judge call.
- All 1264 offline tests pass; parameter description length (<= 120 chars) and tool docstring length
  (<= 1200 chars) invariants are enforced.

## Alternatives Rejected

- **Separate LLM "language judge" call:** Adding an extra LLM call prior to retrieval adds 1.5–3s latency
  and token cost to every request. Handled instead inside the single decomposition prompt and agent gap loop.
- **Dropping `synthesis="client"`:** Rejected by maintainer decision; retained for frontier external
  synthesis models (Claude Opus).
- **Supporting comma-separated `lang:pl,en`:** DeepSeek upstream search treats multiple language operators
  as conjunction (AND) or drops one, returning 0 results. Multilingual search must be performed across
  separate subqueries.
