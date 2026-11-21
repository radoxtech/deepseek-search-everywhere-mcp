# ADR 0006 — The writer stays a single call; guidance does not help

- **Status:** accepted
- **Date:** 2026-11-18
- **Experiments:** `E05`, `E11`, `E12`, `E13` (plus the earlier `notes-write` and `write-revise` runs)
- **Related:** `0005-multi-stage-synthesis-rejected.md`, `0001-report-word-budgets.md`

## Context

Coverage at medium sits at ~0.50 of the rubric facts while ~0.77 of them appear on some fetched page. The
gap motivated seven attempts to change how the report is produced, all of them grounded in the
literature on long-form factuality (atomic claims, per-aspect coverage, evidence matrices, per-claim
verification).

| design | mechanism | result |
|---|---|---|
| `notes-write` | notes, then report | +1.9 pp, noise |
| `write-revise` | draft, then revise | +1.6 pp, noise |
| `specifics` | evidence block in the prompt | +2.4 pp, worse faithfulness |
| `sections` | one call per aspect, full context each | +2.6 pp for 4x cost |
| `matrix` | one call per aspect, **narrow** evidence only | **−37.5 pp** |
| `matrix-repair` | narrow evidence + gap-driven repair | **−33.5 pp** |
| `matrix-checklist` | full context + checklist + repair | **−3.2 pp** for +33% cost |

## Decision

Keep `_research_synthesize` as one call with the shipped prompt. Do not add checklists, per-aspect
sections, or repair passes. The only writer-side lever that ever moved coverage is the word budget
(+7.6 pp, `E06`), which is about room, not guidance.

## Consequences

- The model writes what it judges important. Telling it what to emphasise shifts attention away from
  the facts the rubric values, and narrowing its material destroys recall (E12).
- The remaining ~45 pp of missing facts are dominated by (a) facts the sources do not contain and
  (b) facts the model chooses not to state. Neither is reachable by restructuring the writing step.
- Future quality work must target retrieval (what is fetched, how pages are cut, how many results) or the
  model, and must be measured with the same protocol.
- `evals/variants.py` keeps all seven variants so a future model can be re-tested on frozen contexts for
  a few cents per arm.

## Alternatives rejected

- **Ship `matrix-checklist` anyway** (+33% cost, −3.2 pp).
- **Per-topic switches** (e.g. sections where the topic is weak): `sections` helped `medicine` (+16.1)
  while `matrix-checklist` hurt it (−9.7); that is tuning on the test set, not a mechanism.
- **Ask for longer reports again**: 1920 is already the saturation point at medium (`E06`).
