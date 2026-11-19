# ADR 0002 — Retrieval breadth and selection width stay at 3 queries and K=10

- **Status:** accepted (reverts two earlier changes)
- **Date:** 2026-11-18
- **Experiments:** `E03`, `E04` in `evals/EXPERIMENTS.md`

## Context

The funnel said the largest leak was before reading: 23% of the core facts appeared on no page the
searches returned, and a quarter of those had no query aimed at them (medium planned exactly three
queries however many aspects the topic had). The second leak was the K cut: 13 pp of facts were on a
found page but never reached the writer.

Two changes were made and measured together:

1. plan `subqueries + 2` aspects (five queries at medium) and ask for 12 results per query;
2. hand the writer 14 pages instead of 10, and drop reprinted pages by a content fingerprint.

## Decision

Both changes are reverted: the selection width in `1140834`, the query breadth in `fa4f61c`. `medium`
reads 3 queries, 8 results per query and `k_sources = 10`, which is what the live config shows.

## Evidence

6 tuning topics, 2 repeats, live runs, at the 1920-word budget:

| arm | core | sources | support rate | cost |
|---|---|---|---|---|
| old retrieval | 0.501 | 9.5 | 0.840 | $0.0081 |
| new retrieval | 0.467 | 13.3 | 0.897 | $0.0087 |

Paired difference: **−3.4 pp, sd 8.3, t −0.99** — no coverage gain. The only benefit is +5.7 pp of
citation support on a deterministic metric, and that did not reproduce on the control topics
(−0.5 pp there).

At the older 960 budget the same comparison gave −1.3 pp (t −0.50) on 2 repeats and +3.5 pp on the
first repeat alone.

## Consequences

- Cost per report returns to the lower level (+33% was almost entirely the 14-page context).
- The support-rate gain is given up. If citation support later becomes the priority, test the two
  changes separately: they were always measured as a bundle, so we do not know which one moved it.
- The `found` ceiling (77% of facts on some page) is real but does not translate into report coverage:
  the writer ignores most of the extra context, so adding sources does not help until the writer can
  use them.
- Content-level dedupe was rejected with the bundle. It is cheap and independent; it can be revisited
  alone if duplicate pages are observed in a pool.

## Alternatives rejected

- **Keep K=14 only.** It is the expensive half (context size drives the LLM cost) and the coverage
  result was flat.
- **Raise `count` only.** Never measured alone; the bundle's failure does not rule it out, but nothing
  supports it yet, and search results are free of LLM cost only while K stays small.
