# Roadmap & Technical Specification: v1.2.0 & v1.3.0

## Status: Milestone 1 Released (v1.2.0) / Milestone 2 Implemented, Verified & Pushed (v1.3.0 Ready)
## Scope: Feature additions for DeepSeek Search Everywhere MCP

---

### Upstream DeepSeek API Realities (Verified 2026-11-01)
- **Domain filtering syntax:** DeepSeek Search API treats space-separated `site:a site:b` as union/OR. Pipe operator `(site:a | site:b)` is **not supported** and returns 0 results.
- **Exclusion syntax:** `-site:domain` works reliably.
- **Date filtering syntax:** `after:YYYY-MM-DD` and `before:YYYY-MM-DD` are natively parsed by DeepSeek's search index.
- **Payload limits:** Search endpoint has no separate filter parameters; all filtering must be compiled into the single `query` string with hard maximum length of 128 characters (`MAX_QUERY = 128`).
- **Model:** `deepseek-flash` is the standard fast/efficient model with `"thinking": {"type": "disabled"}`.

---

### Milestone 1 (v1.2.0) Summary (Committed)

1. **Explicit Search Filters:**
   - Tools: `deepseek_web_search`, `deepseek_search_and_read`, `deepseek_research`.
   - Keyword-only parameters: `after`, `before`, `include_domains`, `exclude_domains`.
   - Hard budget constraint: Combines query + operators strictly within DeepSeek's 128-character limit (`MAX_QUERY = 128`).
   - Query rewrite compatibility: Rewrites base query first (if `rewrite=True`), then attaches operators.

2. **Block-Level Lexical Extraction & Sectional Reading:**
   - Tools: `deepseek_fetch`, `deepseek_fetch_many`.
   - Keyword-only parameters: `query`, `section`, `max_snippets`.
   - Algorithm: Markdown header/block chunking + BM25-lite scoring with surrounding context window preservation.
   - Cache optimization: Raw document cached once; snippet selection occurs in-memory at 0 cost.

3. **In-Memory Session Observability & Auditing:**
   - Tool: `deepseek_balance`.
   - Telemetry block: `session` including uptime, upstream request count, cache hit rate, and estimated USD cost/savings.
   - Resettable state: Flushed in `reset_state()` for test isolation.

---

### Milestone 2 (v1.3.0) Specification & Hardening Improvements

1. **`deepseek_deep_research` (Macro-Tool Pattern):**
   - Multi-query decomposition via `deepseek-flash` into 2-4 subqueries.
   - Parallel retrieval across subqueries with single-flight deduplication.
   - **Context & Timeout Hardening:**
     - `SYNTHESIS_TIMEOUT = 25.0s`, overall deadline `35.0s`.
     - Caps synthesis context to max 6 highest-authority sources and max 24,000 characters to eliminate latency spikes and *Lost-in-the-Middle* degradation.
     - Automatically cleans hallucinated citation IDs (`[n]` where $n > \text{len(sources)}$).
     - Graceful fallback: If deadline is approached, returns gathered sources with `synthesized: False`.

2. **`deepseek_extract` (Structured Data Extractor):**
   - Extracts structured JSON without ingesting full page text into client context.
   - **Ergonomics & Hardening:**
     - Supports either `fields: list[str]` (resilient field list) or `schema: dict[str, Any]` (JSON Schema).
     - Fast-fail on empty/blocked pages or 404s (PDF/Reddit advice) without invoking LLM.
     - Strips markdown code fences (` ```json `), extracts between `{` and `}`, parses cleanly.
     - Populates missing requested fields with `None`.
     - `EXTRACT_TIMEOUT = 20.0s`.

3. **MCP Token Diet & Tool Boundary Enforcement:**
   - Strict front-loaded verbs and exclusionary guidance preventing *Tool Misselection*.
   - Tool descriptions budgeted strictly under 420 chars each, keeping total schema length across all 8 tools at ~6,200 chars (safely below the `< 6500` chars test threshold).
