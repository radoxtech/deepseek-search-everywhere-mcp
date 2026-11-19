# ADR 0001 — Report word budgets per depth

- **Status:** accepted; medium measured and shipped, high and max scaled but unmeasured, low unchanged
- **Date:** 2026-11-18
- **Experiment:** `E06` in `evals/EXPERIMENTS.md`

## Context

The report length was never calibrated against coverage. Medium asked for 800 words with a 960-word
ceiling, and every report reached that ceiling: the model wrote 976–1002 words, then a second
"condense" call cut it back. 5 of 6 reports were condensed in one measurement, 9 of 12 in another.
At 960 words the report covered 45% of the rubric's core facts, and condensation silently removed
part of what the writer had produced.

## Decision

Shipped in `c761676`; the `low` tier was taken back in `c6e12a0`.

Double the budget for medium, high and max, and give the writing step room in the deadline:

| depth | target / max before | after | deadline before → after |
|---|---|---|---|
| low | 400 / 480 | **400 / 480 (unchanged)** | 60 s |
| medium | 800 / 960 | **1600 / 1920** | 90 s → 150 s |
| high | 1200 / 1440 | 2400 / 2880 | 150 s → 240 s |
| max | 1800 / 2160 | 3200 / 3840 | 240 s → 360 s |

The synthesis time reserve rises with the budget (`medium` 70 s, `high` 110 s, `max` 150 s) so the
earlier phases do not eat the time the writer now needs. `low` keeps its 20 s reserve and its 480-word
ceiling: it exists to be fast and cheap, and doubling it was never measured.

## Evidence

Frozen contexts, baseline writer, 6 tuning topics, 1 repeat per rung:

| ceiling | core | words written | condensed | wall | cost |
|---|---|---|---|---|---|
| 960 | 0.450 | 1040 | 5/6 | 49.3 s | $0.0079 |
| **1920** | **0.501** | 1690 | 1/6 | 56.9 s | $0.0081 |
| 3840 | 0.518 | 2954 | 1/6 | 83.7 s | $0.0105 |

960 → 1920: **+5.0 pp, sd 4.0, t 3.07** (df=5). 1920 → 3840: +1.7 pp, t 0.82 — past saturation.

Confirmation with 2 repeats per rung (6 topics): 960 → 0.434, 1920 → 0.509, **+7.6 pp, sd 5.1, t 3.63**,
all six topics up; words 996 → 1705, wall +29%, cost $0.0072 → $0.0080.

## Consequences

- Reports are roughly twice as long. This is a product change: the report goes back into the caller's
  context and costs output tokens.
- Condensation almost never runs at medium any more, which also saves a call.
- The medium figure is measured; high and max are scaled by the same factor and **not measured**.
  Saturation depends on how many sources the writer receives (10 at medium, 16 high, 24 max), so the
  prediction is that high (2880) sits near its saturation and max (3840) below it; an experiment should
  confirm before quoting either.
- `low` keeps 400/480, a 60 s deadline and a 20 s reserve.
- Medium is confirmed on two repeats (+7.6 pp). High and max rest on a scaled guess; `low` was doubled
  first and taken back, because nothing measured it and it is the quick tier.

## Alternatives rejected

- **Keep 960.** It condenses most reports and loses 5 pp of coverage.
- **3840 everywhere.** +1.7 pp (noise) for +47% wall clock and +30% cost.
- **Trim the prompt instead of raising the budget.** Prompt variants were already tried and produced
  noise (`notes-write` 0.605, `write-revise` 0.602 vs 0.586 baseline, with worse citations).
