# ADR 0008 — The query space is exhausted; `aspects` is the best style measured

- **Status:** accepted
- **Date:** 2026-11-19
- **Evidence:** `evals/results/qlab/_score.json` (6 topics, 6 styles), `E03`, `E04`
- **Related:** `0002-retrieval-reverted.md`, `0007-where-the-ceiling-is.md`

## Context

After the writer work closed, the remaining lever was retrieval: 23% of rubric facts appear on no page at
all, and the weak topics (medicine, nutrition) reach only half the facts even in their best query set.
Before spending more, the query-lab results that already existed were read as a ranking.

## Decision

Do not revisit query formulation. The styles measured on the same six topics, judged by the same page
labelling, rank as follows (core fact yield, and the two weak topics separately):

| style | mean yield | weak topics | primary share | pages |
|---|---|---|---|---|
| **aspects (shipped)** | **0.771** | **0.500** | 42% | 24 |
| questions | 0.619 | 0.357 | 40% | 24 |
| old (3.3) | 0.610 | 0.143 | 40% | 23 |
| evidence (reviews, overviews) | 0.592 | 0.214 | 38% | 24 |
| site: authoritative domains | 0.470 | 0.286 | 63% | 24 |
| topic as one query | 0.562 | 0.000 | 31% | 8 |

Two consequences follow and both are counter-intuitive:

- **Authoritative sources are less fact-dense here.** The `site:` style has the highest primary share
  (63%) and the lowest yield (0.470). Preferring official domains is not a quality lever for this task.
- **Aspect-shaped queries are the mechanism that works**, by a wide margin, and most of all exactly where
  coverage is weak (0.500 against 0.143 for the 3.3 decomposition).

## Consequences

- "Prefer primary sources" and "restrict to authoritative domains" are rejected; both were already
  measured and are worse.
- The 23% of facts that never appear on a page are the ceiling of the open web for these topics, not a
  query bug. Remaining dials (results per query alone, page caps) were measured as null or inconclusive
  (`E03`, `E15`).
- Quality work on retrieval is closed until the source corpus changes (new index, new provider), and any
  future attempt must start from this table rather than from intuition.

## Alternatives rejected

- **Re-run the style comparison on today's config.** The pipeline's decomposition has not changed since
  the lab run; only the word budget has, and that is a writer setting.
- **Mix styles** (aspects plus a couple of site: queries). The `site:` yield is low enough that spending a
  query slot on it would need evidence; the per-style data argues against it.
