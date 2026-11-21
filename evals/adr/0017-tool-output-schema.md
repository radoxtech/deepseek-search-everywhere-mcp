# ADR 0017 — Per-tool output schemas for `structuredContent`

- **Status:** Proposed
- **Date:** 2026-11-21
- **Evidence:** TDQS baseline evaluation (`npx mcp-tdqs`, spec 1.3, judge `deepseek-ai/DeepSeek-V4-Flash-0731`, two runs) recorded in `.sisyphus/plans/tdqs-improvements.md`; post-change measurements in `.sisyphus/evidence/task-15-*.json` and the post-remediation re-measurement at HEAD `5f380ee` in `.sisyphus/evidence/task-15b-{run1,run2,run3,lint,aggregate}.json`; SDK behavior in `mcp/server/mcpserver/utilities/func_metadata.py`; experiment `E07` in `evals/EXPERIMENTS.md` for the noise floor.
- **Related:** `0010-tool-surface-pruning-and-taxonomy.md`, `.sisyphus/plans/tdqs-improvements.md`

## Context

A TDQS evaluation of the seven tools produced an overall grade of **A 4.7**, with **A 4.6** on tool
definition quality and **A 4.8** on coherence. Every tool reached tier A (7/7) with zero flags and zero smells,
and the total definition payload measured **14,302 bytes**. That baseline is the reference scorer at spec 1.3
with the `deepseek-ai/DeepSeek-V4-Flash-0731` judge over two runs.

The one structural weak point is the published output schema. Each tool returns `dict[str, Any]`, and the SDK
derives the schema from the return annotation, so `tools/list` advertises the same shape for every tool:

```json
{"type": "object", "additionalProperties": true}
```

That schema claims "some object with unknown keys". It names no field, carries no type and no description, and
tells a client nothing about what a call actually returns. TDQS penalizes it under definition and contextual
completeness, which is where the 4.6 comes from.

Three SDK facts constrain any fix:

1. `_create_output_model` builds `outputSchema` from the return annotation. There is no `output_schema=`
   argument, so the only way to publish a real schema is to change the annotation.
2. `convert_result` validates the returned value and then calls
   `model_dump(mode="json", by_alias=True)` with **no** `exclude_none`. Nulls are unavoidable: an all-optional
   model emits every absent field as `null`, and that null padding reaches `structuredContent`.
3. `_convert_to_content` builds the text block from the original dict, not from the model. Unknown keys are
   dropped from `structuredContent` unless the model allows them, while the text block keeps them.

The runtime envelope, every tool name and every parameter are public API and stay unchanged. `structuredContent`
is a new response surface layered on top of that stable dict, so it needs its own decision record.

## Decision

1. **Publish per-tool pydantic models through return annotations.** Each tool's model builds on the shared
   `Envelope` (`ok`, `_meta`, `error`), and each model adds `elapsed_ms` plus the fields that tool emits. The
   return annotation in `tools/*.py` changes from `dict[str, Any]` to the model; every `return` statement stays
   untouched.
2. **`extra="ignore"` with exhaustive models.** The models are complete, and the contract tests assert that no
   returned key is unmodeled. `extra="allow"` would keep stray keys but re-publish
   `additionalProperties: true`, which is the exact defect being fixed.
3. **Only `ok` is required.** Every other field is optional. Status-conditional and error envelopes omit fields
   by design, and a required field would turn a valid return into a call-time validation failure.
4. **Accept `null` padding.** Because the SDK dumps without `exclude_none`, absent optional fields appear as
   `null` in `structuredContent`. This is accepted rather than mitigated, since no supported switch removes it.
5. **Add a top-level `title`.** All seven tools pass `title=` to `add_tool`, alongside the existing
   `annotations.title`, so clients that read the top-level field also get a human label.
6. **Leave nested `dict[str, Any]` fields open.** Result entries, extraction payloads and similar nested maps
   keep their `additionalProperties: true`. The fix targets the top-level schema only; closing nested maps
   would drop fields the code intentionally passes through.

## Consequences

- `structuredContent` becomes field-typed: clients and models see field names, types and descriptions instead
  of an opaque object. This is the measured gain the change exists for.
- The payload grows. Every optional field absent from a sparse envelope is materialized as `null`, and
  content-bearing fields such as `content` in `deepseek_fetch` or a full-text search result are duplicated
  between the text block and `structuredContent`. Measured in T15: for a 1,352-character page, a full-content
  `deepseek_search` call sent 1,911 characters in the text block and 1,945 in `structuredContent` (1.02x), and
  `deepseek_fetch` sent 1,707 and 1,859 (1.09x), so a content-bearing response now carries the page text twice
  (about 2.0x the text block in total).
- The models must track the runtime envelope. Adding, renaming or removing a returned key requires a matching
  model change; the contract tests catch drift, and until they are updated a new key is silently absent from
  `structuredContent` while the text block still carries it.
- Any unmodeled key is silently dropped from `structuredContent`. The text block is unaffected, so existing
  consumers that parse the text block see no change; consumers of `structuredContent` see only modeled fields.
- The top-level `title` duplicates `annotations.title`. That redundancy is deliberate for client compatibility.
- No tool name, parameter, error code or envelope construction changes. The published `tools/list` definitions
  change in `outputSchema` and `title` only.

## Alternatives Rejected

- **`extra="allow"`.** It preserves unknown keys, but the generated schema re-emits
  `additionalProperties: true` at the top level. That is precisely the defect, so it cannot be the fix.
- **`TypedDict`.** It carries the same null and serialization constraints as a pydantic model, but offers
  weaker field descriptions and no validation, which leaves the schema quality lower for no gain.
- **Discriminated union on `status`.** Error envelopes carry no `status` field, and the status vocabulary
  differs per tool, so a union keyed on `status` cannot represent the error branch.
- **Leaving the schemas empty.** Keeping `dict[str, Any]` holds Contextual Completeness at 4 and leaves the
  definition grade at 4.6, which is the outcome this decision is meant to change.

## Evidence

### Baseline (measured, two runs)

| Metric | Value |
|---|---|
| Overall TDQS | A 4.7 |
| Tool definition quality | A 4.6 |
| Coherence | A 4.8 |
| Tools at tier A | 7/7 |
| Flags / smells | 0 / 0 |
| Total definition payload | 14,302 bytes |
| Spec version | 1.3 |
| Judge model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Runs | 2 |

### Post-change measurement (three runs, post-remediation tree)

Three independent `npx mcp-tdqs@0.2.0 score` runs against the same spec (1.3) and judge
(`deepseek-ai/DeepSeek-V4-Flash-0731`) as the baseline, measured at HEAD `5f380ee` **after** the follow-up
remediation (research output models collapsed into one, per-field schema descriptions shortened). Each run
scored all seven tools and the server coherence, and all three reported zero flags and zero smells with 7/7
tools at tier A.

| Tool | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| `deepseek_search` | 4.7 | 4.7 | 4.9 | 4.7 |
| `deepseek_fetch` | 4.9 | 4.9 | 4.9 | 4.9 |
| `deepseek_extract` | 4.9 | 4.9 | 4.9 | 4.9 |
| `deepseek_quick_research` | 4.7 | 4.7 | 4.9 | 4.7 |
| `deepseek_deep_agent_research` | 4.6 | 4.6 | 4.6 | 4.6 |
| `deepseek_check_research` | 4.7 | 4.7 | 4.7 | 4.7 |
| `deepseek_check_balance` | 4.7 | 4.7 | 4.7 | 4.7 |
| Overall | 4.6 | 4.6 | 4.6 | 4.6 |

Server rollup per run:

| Metric | Run 1 | Run 2 | Run 3 | Median |
|---|---|---|---|---|
| Overall | A 4.6 | A 4.6 | A 4.6 | 4.6 |
| Tool definition quality | A 4.7 | A 4.7 | A 4.7 | 4.7 |
| Coherence | A 4.5 | A 4.3 | A 4.5 | 4.5 |
| Mean TDQS | 4.7 | 4.7 | 4.8 | 4.7 |
| Min TDQS | 4.6 | 4.6 | 4.6 | 4.6 |
| Tools at tier A | 7/7 | 7/7 | 7/7 | 7/7 |
| Flags / smells | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Definition-byte delta versus the 14,302-byte baseline: **57,612 bytes, +43,310 (+302.8%)**. This is 2,823
bytes below the first post-change measurement (60,435) and still roughly four times the baseline: the typed
`outputSchema` on every tool is the context cost of field-typed responses, and trimming descriptions reduced
but did not remove it.

**Measurement.** Three runs, median reported rather than the best run; overall spread 4.6–4.6 (range 0.0,
population stdev 0.0), with the only per-tool movement between runs being `deepseek_search` and
`deepseek_quick_research` at 4.7→4.9 in one run each. Definition bytes from `lint --format json` at HEAD
`5f380ee`; the payload-duplication test was re-run on the same 1,352-character page and reproduced the earlier
numbers exactly (`deepseek_search(content="full")` text 1,911 / structured 1,945, 1.02x;
`deepseek_fetch` text 1,707 / structured 1,859, 1.09x), because the search and fetch models changed descriptions
only, not their field sets. Judge `deepseek-ai/DeepSeek-V4-Flash-0731`, spec 1.3.

The median overall is **4.6, below the 4.8 threshold** (`Proposed` retained, shortfall: `median overall 4.6 < 4.8`).
The tool floor (every tool median ≥ 4.6, lowest `deepseek_deep_agent_research` at 4.6) and the zero-flag/zero-smell
conditions do hold. The remediation left the verdict unchanged: the overall median stayed at 4.6, definition
quality stayed at 4.7, coherence stayed at 4.5, and two tool medians moved by 0.1 in opposite directions
(`deepseek_quick_research` 4.6→4.7, `deepseek_deep_agent_research` 4.7→4.6), all inside the measured noise floor
(`E07`: 7.2 pp between identical runs). The baseline is a two-run cross-session figure, so this measurement
establishes neither a quality gain nor a regression; the decision is recorded but not accepted on score.
