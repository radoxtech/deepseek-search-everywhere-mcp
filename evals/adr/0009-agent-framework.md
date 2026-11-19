# ADR 0009 — Agent orchestration on Vercel `ai` (Python), no custom provider

- **Status:** accepted
- **Date:** 2026-11-19
- **Evidence:** `plans/plan-agent-faza7-vercel.md`, source reads of `ai` 0.10.1 / `openai` 3.55.1 / `langgraph` 1.2.23
  (2026-11-19), `E07` (noise floor), `0003-coverage-only-judging.md`
- **Related:** `plans/kontrakt-agent-deep-research.md`, `plans/plan-agent-role-fazy.md`,
  `plans/handoff-agent-faza6-2026-11-19.md`

## Context

The agent tool needs a real orchestration layer: a main loop, check steps as sub-agents with their own
context, parallel branches, approval/defer hooks, and structured output. Two candidate frameworks were
evaluated against the repository's hard invariants (`AGENTS.md`): one audited HTTP path, no secrets in
prompts, `thinking` explicit on every chat payload, errors never cached, no telemetry, Python floor.

Three options were on the table: a hand-written loop on `runtime._chat_request` (~200 lines, zero new
dependencies), LangGraph 1.2.23, and Vercel's Python `ai` SDK 0.10.1 (`vercel-labs/ai-python`).

## Decision

Build the agent on **Vercel `ai` 0.10.1** and **do not write a custom provider**. The DeepSeek routes through
the generic `OpenAICompatibleProvider`, which the library maintains.

Concretely:

1. **Provider: `ai[openai]`, no custom `ProviderProtocol`.** `ai.get_provider("deepseek")` resolves to
   `OpenAICompatibleProvider` → `base_url=https://api.deepseek.com`, protocol
   `OpenAIChatCompletionsProtocol`. This works because the bundled `modelsdotdev` dataset carries a
   `deepseek` record whose `npm` is `@ai-sdk/openai-compatible`, and `OpenAICompatibleProvider.handles`
   contains that npm id. There is **no** separate `ai-deepseek` Python package on PyPI; the dataset lookup
   is the supported mechanism, the same one TS reaches through `@ai-sdk/deepseek`.
2. **One HTTP library: migrate `transport.py` from `httpx` to `httpx2`.** `ai` and its `openai` dependency
   use `httpx2` (2.16.2), which is the successor to `httpx` by the same author (Tom Christie, BSD-3-Clause,
   `pydantic/httpx2`). `AsyncOpenAI` accepts a plain `httpx2.AsyncClient` in `http_client`, so the product
   shares **one** client, one connection pool, one TLS/proxy configuration.
3. **Retry authority stays ours.** `AsyncOpenAI(max_retries=0)`; `transport.py` is the only retry layer.
   The SDK validates the same way (`_validate_max_retries`: "Use 0 to disable retries").
4. **Redaction, response cap and logging go on `httpx2` `event_hooks`**, under the SDK and above the socket —
   one place for all traffic.
5. **Errors are caught, not wrapped.** `openai._exceptions` exposes `APIError` with `.status_code`, `.body`,
   `.code`; `except openai.APIError` feeds the existing `_map_http_error(status, body, path)` in `errors.py`.
   `APITimeoutError` / `APIConnectionError` map to `timeout` / `network_error`. Estimated ~15 lines, using
   code that already exists.
6. **Budget per request, from `event.usage`.** `ai` normalises `Usage(input_tokens, output_tokens,
   reasoning_tokens, cache_read_tokens, cache_write_tokens, raw)` on every event and accumulates with `+`.
   `deepseek-flash` reports reasoning tokens separately, which the pricing model needs.
7. **Cache stays ours.** `ai`'s `CacheParams` is *provider prompt caching* (`prompt_cache_key`), not a
   response cache. `cache.py` (TTL + LRU + single-flight, errors never cached) is unchanged and is not
   replaceable by the framework.
8. **The step limit is ours.** The library's `Agent.loop()` is unbounded: no `max_steps`, no deadline, no
   budget awareness (`keep_running()` only checks whether the last message needs a reply). `ResearchAgent`
   subclasses `ai.Agent`, overrides `loop()` using the documented skeleton, and adds `step < MAX_STEPS` and
   `budget.allows()`. `LOOP_BUFFER = 0` runs the loop in lockstep so the budget check has teeth.
9. **Python floor 3.12**, CI matrix `3.12 / 3.13 / 3.14 / 3.15`. 3.11 is dropped because `ai` declares
   `requires-python >=3.12`; this is a breaking change and takes a major bump from release-please.

## Evidence

Measured, not assumed (2026-11-19):

| Fact | How verified |
|---|---|
| `ai` 0.10.1 exists, `requires-python >=3.12`, Apache-2.0 | PyPI JSON, installed wheel |
| `ai.get_provider("deepseek")` → `OpenAICompatibleProvider`, `https://api.deepseek.com`, Chat Completions | executed against installed 0.10.1 |
| No `ai-deepseek` / `ai-sdk-deepseek` / `ai-provider-deepseek` on PyPI (404) | PyPI JSON |
| `AsyncOpenAI(http_client=httpx2.AsyncClient)` accepted | `openai/_client.py:829` |
| `max_retries=0` disables retries | `openai/_base_client.py:888-898`, `:1154-1158` |
| `APIError` carries `status_code`, `body`, `code` | `openai/_exceptions.py` |
| `Usage` on every event, `__add__` accumulates | `ai/types/usage.py`, `ai/types/events.py:94-100,349-351` |
| Cache is prompt caching only, no response cache | grep of `src/ai/` |
| No `max_steps` / turn limit in the core loop | grep of `src/ai/`; hits only in `external/claude_code` |
| Dependency delta `ai` (4) vs `ai[openai]` (14) vs LangGraph (~9 incl. mandatory `langsmith`) | installed wheels |

LangGraph was rejected on evidence: `langgraph` 1.2.23 depends mandatorily on `langchain-core`, which
depends mandatorily on `langsmith` (an OTel exporter), so "no telemetry" would have to be enforced by an
env assertion rather than by absence; `create_react_agent` exposes no `retry_policy`/`timeout` (they live on
`StateGraph.add_node`), and the default `recursion_limit` is 10007 — a spend risk roughly opposite to ours.

## Consequences

- **Two new runtime dependency families** enter the product: `ai` + `httpx2`. `httpx` is removed once
  `transport.py` is migrated. `openai` and `jiter` arrive with the `[openai]` extra.
- **`ai` is 0.x public beta** with a version cycle separate from TS. Pin `>=0.10,<0.11` and expect churn.
  Any upgrade re-runs the invariant audit (telemetry, retry, payload shape).
- **`json_schema` is rejected by DeepSeek** (HTTP 400; official docs document `{"type":"json_object"}` only).
  Structured output therefore goes through `structured.py`: send `json_object` via `extra_body`, validate in
  Python with `pydantic`, one retry on validation failure. This is the one place the framework leaks.
- **Budget granularity is per request** (from `event.usage`), not per loop turn — better than assumed while
  the framework's limits were not yet read.
- **The step cap is product code, not configuration.** `MAX_STEPS` needs calibration and becomes an
  experiment (P5), not a default.
- **Known unknowns to verify during implementation, not now:** whether `ai` maps DeepSeek's
  `prompt_cache_hit_tokens` into `cache_read_tokens` (otherwise read `Usage.raw`); whether `ai` propagates
  `openai.*` exceptions unwrapped (otherwise a thin adapter); `modelsdotdev`'s licence is undeclared on PyPI.

## Alternatives rejected

- **Hand-written loop on `runtime._chat_request`.** Zero dependencies and full control, but re-implements
  agent loops, tool scheduling, sub-agent streaming and hooks — the four things this ADR buys. Retained as
  the fallback if `ai` proves unworkable for the invariants.
- **LangGraph.** Rejected on telemetry in a mandatory dependency, on graph-shaped control flow where the
  problem is a loop with a stop rule, and on a 10007-step default with no spend awareness.
- **Custom `ProviderProtocol` calling `transport.py`.** Eliminates `openai` from the path and keeps one
  audited logical layer, but the user's requirement is not to own a provider. Mitigated instead by
  `max_retries=0`, `httpx2` event hooks and catching `openai.APIError` at the boundary.
- **Staying on `httpx` beside `httpx2`.** Zero migration work, but two HTTP libraries in one process for no
  benefit; `httpx2` arrives regardless.
- **Keeping Python 3.11.** `ai` does not support it; there is no version of this decision that keeps 3.11.
