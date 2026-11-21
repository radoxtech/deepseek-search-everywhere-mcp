# ADR 0007 — Where the ceiling is, after fourteen experiments

- **Status:** accepted
- **Date:** 2026-11-18
- **Experiments:** `E01`–`E15` (E14 withdrawn: it compared two aliases of one model)

## Context

Thirteen experiments later, every lever that was suspected of moving quality has been measured on the same
protocol: 6 tuning topics, frozen or live arms measured in the same session, coverage judged by the same
coverage-only pass, faithfulness from `evals/metrics.py`, noise floor measured at 7.2 pp (`E07`).

## Decision

Record the plateaus and stop looking for a lever inside the pipeline's configuration:

| lever | measurement | verdict |
|---|---|---|
| word budget 960 → 1920 (medium) | **+7.6 pp**, t 3.63 | **keep** |
| word budget 1920 → 3840 | +1.7 pp, t 0.82 | reject |
| retrieval breadth (5 queries × 12) | −3.4 pp, t −0.99 | revert |
| selection width K=14 | flat, +33% cost | revert |
| K=20 at high | +2.0 pp, t 1.43 | reject |
| per-page cap 25k → 40k at high | +2.5 pp, t 0.42 | inconclusive |
| seven writer designs (E05, E11–E13) | −3.2 to −37.5 pp | reject |

At medium, coverage sits at ~0.50 of the rubric facts while ~0.77 of them appear on some fetched page.
The gap is not closable by any parameter or restructuring that has been tried.

## Consequences

- Future quality work should start from **what the sources are**, not from the pipeline's shape: the
  23% of facts that appear on no page are the largest untouched block, and the remaining in-context
  losses are the model's editorial choices.
- Any new experiment must budget for **two repeats**: the per-topic spread in these arms reaches 15–19 pp,
  so a one-repeat arm can only detect effects larger than ~10 pp.
- The register (`E01`–`E15`) and the ADRs are the memory of this work: do not retry a rejected lever
  without a new mechanism, and check the register first.
- Subagent judging, coverage-only packs, `cost_per_point_usd` in `score`, and the noise floor are now part
  of the harness, so a future change costs cents to screen and one repeat to rule out.

## Alternatives rejected

- **Keep searching for a prompt or structure that works.** Seven designs, all measured, all negative.
- **Ship the inconclusive levers anyway** (big pages, K=20): they cost money and the evidence does not
  separate them from noise.
