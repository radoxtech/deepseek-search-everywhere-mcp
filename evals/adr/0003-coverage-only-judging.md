# ADR 0003 — How the evaluation judges a report

- **Status:** accepted
- **Date:** 2026-11-18
- **Applies to:** every experiment in `evals/EXPERIMENTS.md`

## Context

Coverage is judged by a model, not measured. Two things were learned the hard way:

1. **The judge must be identical across arms.** Numbers produced by a full evidence-based pack are not
   comparable with numbers produced by a coverage-only pass. The first baseline of this project read
   0.524 with evidence packs and 0.449–0.457 with coverage-only packs on the same reports.
2. **Judging subagents fail silently.** Given four packs of ~54 KB each, an agent spent its whole turn
   reasoning about the atoms and never wrote the verdict file: 3 of 12 files appeared. One pack per
   agent, a script-generated coverage-only pack of ~12 KB, the `quick` category, and an explicit
   "write the file, then verify it with `json.load`" instruction produced 36 of 36.

## Decision

Two-pass measurement, with different owners:

- **Coverage** — `evals/keypoints.py build --atoms` turns each report into a pack; the pack is reduced
  to topic + rubric atoms + report (no evidence), and one subagent per pack answers, per atom,
  `covered` and the sentence number. `supported` is left null.
- **Faithfulness** — taken from `evals/metrics.py`, which is deterministic: `citation_support_rate`
  and `unverified_numbers` are computed from the report and its cited passages without a model.

The full evidence-based pack stays available (`verdicts` with `supported`) for a final, expensive
confirmation of a winner, never for screening.

## Consequences

- Screening a 12-report arm costs a few cents and about ten minutes of judging.
- `core_supported` is not available in coverage-only mode; use coverage plus the deterministic support
  rate. When `keypoints.py score` reports `core_supported: 0`, that is the missing pass, not a result.
- Coverage numbers are comparable only inside one judge protocol. The register records which protocol
  produced each number.
- A single judge model is used within a comparison, so a systematic bias applies to both arms and
  cancels in the paired delta.

## Alternatives rejected

- **One big pack with evidence for everything.** 4× the tokens, 4× the failure rate, and the support
  signal it adds is already available deterministically.
- **Judging inside the main agent.** 12 packs of 54 KB do not fit in one context, and truncation
  produces silent partial verdicts.
- **Scoring with `core_supported` only.** It mixes two effects (does the report state it, does the
  citation back it) and doubles the judging cost for screening.
