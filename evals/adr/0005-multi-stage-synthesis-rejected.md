# ADR 0005 — Multi-stage synthesis is not shipped

- **Status:** accepted
- **Date:** 2026-11-18
- **Experiments:** `E05`, `E11` (and the earlier `notes-write` / `write-revise` runs)
- **Supersedes:** nothing; it gates phase 10 of `plans/plan-tryb-agent.md`

## Context

Phase 6 of the master plan proposed multi-stage synthesis (notes per source → sections → merge) and made
phase 10 (multi-stage writing at `max`) conditional on that variant winning. The diagnosis that motivated
it is real: 62% of the missed facts are "partial" — the report states the gist of an aspect and drops
its numbers, dates and conditions, even though the fact is in the context.

Four writer-side designs have now been measured against the single call:

| variant | coverage vs single call | verdict |
|---|---|---|
| `notes-write` (notes, then report) | 0.586 → 0.605 | noise, worse citations |
| `write-revise` (draft, then revise) | 0.586 → 0.602 | noise, worse citations |
| `specifics` (deterministic evidence checklist) | +2.9 pp at 960, +2.4 pp at 1920 | reject: not significant, worse faithfulness |
| `sections` (one call per aspect, then merge) | 0.501 → 0.527 (+2.6 pp, t 0.83) | **reject: 4× the cost** |

## Decision

Do not ship multi-stage synthesis. `_research_synthesize` stays one call. Phase 10 of the master plan is
cancelled on this evidence; re-opening it needs a new mechanism, not a new decomposition.

## Consequences

- Cost stays at one synthesis call per report; the medium report is $0.0081 with a 1920-word ceiling.
- The writer keeps dropping details. That is documented, not hidden: about half the rubric's facts are
  missing at medium, and the only lever that moved it was the word budget (+7.6 pp, `E06`).
- Future work on quality should start from the sources (what is fetched, how pages are cut) or from the
  model, not from the writing pipeline's shape.
- `evals/variants.py` keeps `sections`, `notes-write` and `write-revise` so a later model or prompt
  change can be re-tested cheaply on frozen contexts.

## Alternatives rejected

- **Ship `sections` at high/max only.** It costs 4× at every tier, and the gain is inside the noise.
- **Sections for the weakest topic only.** `medicine-glp1` gained +16.1 pp, but a per-topic switch is
  tuning on the test set.
- **A merge pass over the single call's output.** That is `write-revise`, already measured as noise.
