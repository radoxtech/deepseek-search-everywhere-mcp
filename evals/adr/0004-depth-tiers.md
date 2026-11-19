# ADR 0004 — The four depth tiers, as measured

- **Status:** accepted; one repeat per tier
- **Date:** 2026-11-18
- **Experiment:** `E09` in `evals/EXPERIMENTS.md`

## Context

The tiers were designed by intuition: low is quick and cheap, max searches four angles with 24 sources
and writes the longest report. Until now nobody had measured all four on the same topics with the same
rubric and the same judge, so the price of a percentage point was unknown and `high` could not be
defended or removed.

## Decision

Keep all four tiers, but record what each one costs per point:

| tier | core | words | sources | wall | cost/report | cost per pp over the tier below |
|---|---|---|---|---|---|---|
| low | 0.300 | 481 | 5.3 | 10.9 s | $0.0043 | — |
| medium | 0.501 | 1690 | 9.7 | 56.9 s | $0.0104 | $0.0003 |
| high | 0.565 (2 repeats) | 2624 | 15.2 | 62.6 s | $0.0284 | **$0.0042** |
| max | 0.615 | 3390 | 22.0 | 70.4 s | $0.0388 | $0.0016 |

`medium` is the sweet spot and stays the default recommendation. `high` is the worst value of the four
(+4.0 pp over medium for 2.6× the cost); it is kept because it is a shipped tier and callers may want
its section depth, but nobody should be steered to it. `max` earns its price better than `high`
(+11.4 pp over medium), so "if you need more, go to max" is the honest advice.

The word budgets shipped in `c761676` are not binding at high (2576 of 2880 words, nothing condensed) or
at max (3390 of 3840). No further budget ladder is needed above medium.

## Consequences

- Tool descriptions and README should sell `medium` and `max`, and not present `high` as an upgrade
  worth its price. That is a documentation change, not a code change.
- `low` hits its own 480-word ceiling and condensed 3 of 6 reports, so the quick tier may be leaving
  coverage on the table; a small ladder (480 / 720) is the obvious follow-up if `low` matters.
- `low`, `medium` and `max` rest on one repeat and carry the 7.2 pp A/A noise from `E07`; `high` has two
  repeats (0.565) and sits between medium (0.509) and max (0.615) with gaps of +5.6 and +5.0 pp.
- A wider selection at `high` (K=20) was tried and rejected: +2.0 pp at t 1.43, `E10`. Do not raise it
  without a bundle that pays for the extra measurement.

## Alternatives rejected

- **Drop `high`.** It is a shipped tier; removing it is a breaking change to the public shape and needs
  a maintainer decision, not an experiment.
- **Rescale `high` to 2880 → 4600 words.** Nothing suggests the ceiling binds there (2576 written, no
  condensation); the gap to `max` comes from sources and reasoning, not from the word budget.
