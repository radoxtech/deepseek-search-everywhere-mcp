# Experiments register

Every change to `deepseek_research` that claims to improve quality gets an entry here, with numbers
and a verdict. No entry, no change. **A change stays only if it beats the pre-registered criterion on
the tuning topics and does not lose on the control topics.**

Each entry answers three questions in order: **what changed** (the exact parameter or code path, before
→ after), **what it was supposed to buy** (the mechanism, not "better quality"), and **what it actually
bought** (mean, sd, t, cost, wall).

## How to run an experiment

1. Write the hypothesis and the success criterion **before** the run.
2. Pick arms. Prefer frozen contexts (`--frozen-from`) when only the writer changes: no search, no
   fetch, a twelfth of the price.
3. Run **both arms in the same batch**: the baseline next to the variant, same topics, repeats, depth
   and judge protocol. Never compare against a number stored from an earlier session — the model, its
   pricing and the judge drift. 6 tuning topics × 2 repeats, plus 3 control topics × 2 repeats when the
   change could be topic-specific. Paid calls off-peak only (Mon–Fri 01–04 and 06–10 UTC are peak).
4. Judge coverage with the pack + quick-agent protocol (`evals/keypoints.py build --atoms`, then the
   coverage-only instruction in `evals/adr/0003-coverage-only-judging.md`). Take faithfulness from
   `evals/metrics.py`, which is deterministic and needs no judge.
5. Record, then stop. Do not "tune the winner further" on the same data.

## Template

```
### E<nn> — <short title> (<date>)
Change:     <file / parameter, before → after>          <- what was actually done
Claim:      <the mechanism that should improve the metric, and why>
Arms:       <A vs B, frozen/live, topics, repeats, budget>
Criterion:  <e.g. core +3 pp on tuning and not worse on control>
Result:     <core A -> B (mean, sd, t); support; cost; wall; anything unexpected>
Verdict:    keep | revert | reject | inconclusive
Not shown:  <what this experiment cannot tell you>
```

## Log

### E01 — Per-stage trace in the debug dump (2026-11-18)

**Change.** `research_jobs.py`: a `_ResearchRun.trace` list, a `_stage()` context manager wrapping every
phase, per-stage counters, and four new dump fields (`subqueries`, `aspects`, `notes`, `stage_trace`).
`deep_research.py`: a keyword-only `stats` dict on `rank_and_diversify_sources` and
`select_synthesis_context`, filled with the counts that were computed and thrown away before (canonical
duplicates, domain-cap overflow, K cut, context chars, truncations).

**Claim.** A stage-level funnel cannot be trusted if the pipeline never records per-stage numbers; the
dump had only synthesis inputs and outputs, so every loss between search and report was invisible.

**Arms.** None (instrumentation). Verified with the offline suite and one synthetic job.

**Result.** Seven stages with duration and counters; the public envelope is unchanged (the trace lives
only in the debug dump). Five new tests.

**Verdict: keep.**

### E02 — Robustness: search timeouts, number grounding, dump on failure (2026-11-18)

**Change.**
- `_research_retrieve`: `asyncio.gather(..., return_exceptions=True)`; a timed-out query becomes a failed
  search plus a note instead of raising through the pipeline.
- `ground_report_numbers`: a paragraph that starts with a markdown heading used to be skipped whole; now
  only the heading lines are stripped and the body is checked (the heading is restored).
- `_research_source`: `chars` reported `0`; it now reports `len(content)`.
- `_maybe_dump_research_job`: called on the error path too, with `status: failed`, `error`, and the
  failing stage's `counters["error"]`.

**Claim.** Two artefacts were being measured as quality: a single slow query turned the whole job into
`internal_error`, and a report whose first line is a heading silently skipped number verification.

**Arms.** None; five new tests (timeout isolation, heading paragraph, dump on failure, `chars`).

**Result.** All green. `elapsed_ms` was investigated and left alone: `_timed` is not used anywhere in
`src/`, so nothing overwrites it — the suspected bug did not exist.

**Verdict: keep.**

**Not shown.** Failed or unsynthesised runs still write no dump, so failure diagnosis stays partial.

### E03 — Retrieval breadth: five queries, twelve results each (2026-11-18)

**Change.** `DEPTH_CONFIG["medium"].count` 8 → 12; the decomposition budget became `subqueries + 2`
(medium: 3 → 5 aspects, one query each), used both in the prompt and when parsing the answer.

**Claim.** 23% of the rubric's core facts appeared on no returned page, and a quarter of those had no
query aimed at them: medium always planned three queries however many aspects the topic had. More
queries and deeper result lists should raise `found`, and therefore the report.

**Arms.** Old `src` vs new `src`, live, 6 tuning topics × 2 repeats, at the 1920-word budget.

**Result.** core 0.501 → 0.467 (**−3.4 pp**, sd 8.3, t −0.99). Support rate +5.7 pp (deterministic).
Cost +7% on frozen synthesis; +33% on the live pipeline, because 14 pages reach the writer.

**Verdict: revert.**

**Not shown.** Which half caused the support-rate gain (queries or page count) — it was always measured
as a bundle. `found` itself was not re-measured: page labels for the new pool were not built.

### E04 — Selection width K=10 → 14, plus content-level dedupe (2026-11-18)

**Change.** `DEPTH_CONFIG["medium"].k_sources` 10 → 14; `rank_and_diversify_sources` gained
`_content_fingerprint` (first ~400 characters of a page, normalised, only for text of 150+ characters)
so the same article published on a second host no longer takes a slot.

**Claim.** 13 pp of facts were on a found page but never reached the writer; and a reprinted article
occupied one of the ten slots under a different URL. More, and less redundant, pages should lift the
report.

**Arms.** Bundled with E03, same run set.

**Result.** Pages handed to the writer 9.5 → 13.3; coverage did not follow (see E03). Dedupe removed
about one page per topic where a reprint existed.

**Verdict: revert.** The writer ignores most of the extra context, so the cut was not the bottleneck.

**Not shown.** The dedupe alone — it is cheap and independent, and may still be worth keeping if
duplicate pages show up in a pool.

### E05 — Writer variant `specifics`: a deterministic evidence checklist (2026-11-18)

**Change.** New harness variant (`evals/variants.py`, `--variant specifics`): before the writer runs, the
server scans the context for the sentences that carry each aspect's terms and numbers, picks up to three
per aspect, and appends them to the writer's user message as `KEY SPECIFICS FROM THE SOURCES` with
citation ids. No model call, no change to the shipped prompt.

**Claim.** 62% of the misses are "partial": the report states the gist and drops the specific number or
condition. The writer had the fact in the context, so handing it the exact sentences should convert
those partial hits into full ones.

**Arms.** Frozen old contexts, baseline writer vs `specifics`, 6 topics × 2 repeats, first at the 960
budget and again at 1920.

**Result.** At 960: +2.9 pp (t 1.47), but 9 of 12 reports needed condensation, support rate −1.7 pp,
wall +39%. At 1920: +2.4 pp (t 1.06), unverified numbers 0.33 → 1.17.

**Verdict: reject.** No significant gain, worse faithfulness, more tokens.

**Not shown.** A trimmed block (one sentence per aspect, numeric aspects only) was never run.

### E06 — Report word budget ladder: 960 / 1920 / 3840 (2026-11-18)

**Change.** `DEPTH_CONFIG` target/max words doubled for medium, high and max (medium 800/960 →
**1600/1920**, high 1200/1440 → 2400/2880, max 1800/2160 → 3200/3840), the deadlines raised so the longer
synthesis fits (medium 90 → 150 s), and the synthesis reserve raised with them (medium 30 → 70 s).
`low` was doubled in the first cut and taken back unmeasured in `c6e12a0`: it exists to be fast and
cheap. The harness got `--variant words` with `EVAL_MAX_WORDS` to run any ceiling without editing code.

**Claim.** The report always reached the 960-word ceiling, and a second "condense" call then removed
content: 5–9 of 12 reports were condensed. Raising the ceiling should stop the condensation and give the
writer room for the details it was dropping.

**Arms.** Frozen old contexts, baseline writer, 6 topics × 1 repeat per rung.

**Result.** First pass, 1 repeat per rung:

| ceiling | core | words | condensed | wall | cost |
|---|---|---|---|---|---|
| 960 | 0.450 | 1040 | 5/6 | 49.3 s | $0.0079 |
| **1920** | **0.501** | 1690 | 1/6 | 56.9 s | $0.0081 |
| 3840 | 0.518 | 2954 | 1/6 | 83.7 s | $0.0105 |

960 → 1920: +5.0 pp, sd 4.0, t 3.07 (df=5). 1920 → 3840: +1.7 pp, t 0.82.

**Confirmation pass, 2 repeats per rung (960 vs 1920 only):**

| topic | 960 | 1920 | delta |
|---|---|---|---|
| it-passkeys | 0.486 | 0.571 | +8.6 |
| it-pg-incremental-backup | 0.675 | 0.700 | +2.5 |
| law-eu-ai-act-gpai | 0.671 | 0.686 | +1.4 |
| medicine-glp1 | 0.290 | 0.355 | +6.5 |
| nutrition-time-restricted-eating | 0.161 | 0.290 | +12.9 |
| politics-eu-cbam | 0.318 | 0.455 | +13.6 |
| **mean** | **0.434** | **0.509** | **+7.6** |

**+7.6 pp, sd 5.1, t 3.63** (df=5); all six topics improve. Words 996 → 1705, wall 40.8 → 52.7 s
(+29%), cost $0.0072 → $0.0080 per report (+11%).

**Verdict: keep 1920 at medium** (`c761676`). Confirmed on 2 repeats and larger than the first estimate.
3840 is past saturation: +1.7 pp (noise) for +47% wall, +30% cost. `low` stays at 400/480 (`c6e12a0`);
high and max carry the doubled budget unmeasured.

**Not shown.** low/high/max were scaled by the same factor and never measured.

### E07 — A/A: how much noise does the judge and the model add? (2026-11-18)

**Change.** None. Two runs of the identical configuration (`cmp-old`, repeats 1 and 2) are compared
against each other, 9 topics, live, at the 960-word budget.

**Claim.** If the same configuration changes the score from one repeat to the next by more than the
win rule allows, then any threshold below that band is meaningless and single-report conclusions are
noise.

**Result.**

| statistic | value |
|---|---|
| mean abs difference between repeats | **7.2 pp** |
| sd of the difference | 8.4 pp |
| range | −14.3 pp … +9.7 pp |
| worst topic | it-passkeys 0.514 → 0.371 |

**Verdict: the win rule was too tight.** +3 pp sits below the noise of a single report; a change must be
judged on the paired mean over six or more topics (t > 2), not on one aggregate. It also explains E03,
which read +3.5 pp on one repeat and −1.3 pp on two.

**Not shown.** This is noise for one report. With 2 repeats per arm and 6 topics the paired mean is
much tighter, which is why E06's +7.6 pp (t 3.63) survives while smaller effects do not.

### E08 — The judge must make a deliberately worse variant lose (2026-11-18)

**Change.** None to the product. Harness variant `degraded-two-sources` routes only the first two
selected sources into the synthesis context (the rest of the pipeline is the live one).

**Claim.** If the judge cannot separate a report written from 2 sources from one written from 10, then
every number in this register is decoration.

**Arms.** Frozen old contexts, baseline (10 sources) vs degraded (2 sources), 6 tuning topics, 1 repeat,
same 1920-word budget.

**Result.**

| topic | 10 sources | 2 sources | delta |
|---|---|---|---|
| it-passkeys | 0.514 | 0.000 | −51.4 |
| it-pg-incremental-backup | 0.675 | 0.450 | −22.5 |
| law-eu-ai-act-gpai | 0.714 | 0.371 | −34.3 |
| medicine-glp1 | 0.323 | 0.194 | −12.9 |
| nutrition-time-restricted-eating | 0.323 | 0.226 | −9.7 |
| politics-eu-cbam | 0.455 | 0.091 | −36.4 |
| **mean** | **0.501** | **0.222** | **−27.9** |

Paired: **−27.9 pp, sd 15.8, t −4.31** (df=5); every topic falls.

**Verdict: the judge separates arms clearly.** With E07's noise floor of 7.2 pp, an effect of this size
is unambiguous, and smaller real effects (6–8 pp) remain detectable with 2 repeats on 6 topics.

**Not shown.** Nothing about which *extra* sources help — only that removing eight of them hurts.

### E09 — The four depth tiers, measured live (2026-11-18)

**Change.** None. Baseline of the four shipped tiers on the 6 tuning topics, 1 repeat each, live, with
the current budgets and rubrics.

**Claim.** The tiers should be monotonic in coverage, and each step up in price should buy a
proportional gain. A tier that costs 2.6× for 4 points is a tier nobody should choose.

**Result.**

| tier | core | words | sources | wall | cost/report |
|---|---|---|---|---|---|
| low | 0.300 | 481 | 5.3 | 10.9 s | $0.0043 |
| medium | 0.501 | 1690 | 9.7 | 56.9 s | $0.0104 (incl. the search a frozen run skipped) |
| high | 0.541 | 2576 | 15.2 | 58.2 s | $0.0273 |
| max | 0.615 | 3390 | 22.0 | 70.4 s | $0.0388 |

Cost per percentage point: low→medium $0.0003, medium→high **$0.0042**, high→max $0.0016.

**Verdict: monotonic, but `high` is poor value.** It buys +4.0 pp over medium for 2.6× the cost, while
`max` buys +11.4 pp over medium for 3.7×. The word ceilings are not binding at high (2576 of 2880) or
max (3390 of 3840), and nothing is condensed — the scaled budgets are adequate.

**Not shown.** 1 repeat per tier, so each number carries the 7.2 pp A/A noise; `low` at 481 words hits
its own 480 ceiling and condensed 3 of 6 reports, so it may deserve a budget test of its own.

**Tooling note.** `evals/metrics.py` recomputes cost from `prompt_tokens` and `completion_tokens` and
undercounts the thinking tiers by about 1.8× (high: $0.0151 vs $0.0273 in the envelope); reasoning
tokens are billed but not reported there. Use the envelope's `cost_usd` for the ledger.

### E10 — `high` calibrated by selection width: K=16 → 20 (2026-11-18, screening)

**Change.** `DEPTH_CONFIG["high"].k_sources` 16 → 20, via the harness `--variant k` and `EVAL_K=20`. The
queries, page cap, total character cap and word budget stay as shipped.

**Claim.** `high` uses only 49% of its 300k character budget (146k char, 15.2 sources), while `max`
reaches 22 sources and 0.615. The gap between them is selection width, not words: showing the writer
four more pages from the same 40-candidate pool should land `high` halfway between medium and max.

**Arms.** Live, 6 tuning topics, 1 repeat per arm: shipped `high` (K=16, 0.541, $0.0273) vs K=20
(0.575, $0.0284). Baseline measured in the same session.

**Result.**

| topic | K=16 | K=20 | delta |
|---|---|---|---|
| it-passkeys | 0.486 | 0.514 | +2.9 |
| it-pg-incremental-backup | 0.900 | 0.775 | −12.5 |
| law-eu-ai-act-gpai | 0.657 | 0.686 | +2.9 |
| medicine-glp1 | 0.484 | 0.516 | +3.2 |
| nutrition-time-restricted-eating | 0.323 | 0.355 | +3.2 |
| politics-eu-cbam | 0.394 | 0.606 | +21.2 |
| **mean** | **0.541** | **0.575** | **+3.5** |

Paired: +3.5 pp, sd 10.7, **t 0.80** (df=5). Five of six topics rise; `it-pg-incremental-backup` falls
from its outlying 0.900 to 0.775, which is exactly what `max` scores on that topic — the 0.900 looks
like a lucky draw, not a property of K=16.

**Confirmation, 2 repeats per arm (24 reports):**

| topic | K=16 | K=20 | delta |
|---|---|---|---|
| it-passkeys | 0.514 | 0.529 | +1.4 |
| it-pg-incremental-backup | 0.812 | 0.800 | −1.2 |
| law-eu-ai-act-gpai | 0.614 | 0.686 | +7.1 |
| medicine-glp1 | 0.532 | 0.516 | −1.6 |
| nutrition-time-restricted-eating | 0.387 | 0.403 | +1.6 |
| politics-eu-cbam | 0.530 | 0.576 | +4.5 |
| **mean** | **0.565** | **0.585** | **+2.0** |

Paired: **+2.0 pp, sd 3.4, t 1.43** (df=5); cost $0.0284 → $0.0289 per report (+2%).

**Verdict: reject.** The screening's +3.5 pp shrank to +2.0 pp on the second repeat — the same
shrinkage pattern as E03 — and t 1.43 does not clear the t > 2 bar. K stays at 16.

**What it did establish.** With two repeats, `high` measures 0.565: between medium (0.509) and max
(0.615), with gaps of +5.6 and +5.0 pp. The tier is properly placed after all; the one-repeat figure
(0.541) understated it.

**Not shown.** Whether K=20 helps at 3–4 repeats. The direction was consistent in both passes and costs
2%, so it is a candidate to bundle with any later change that touches `high`, and measure once.

### E11 — Phase 6: one writing call per aspect, then a merge (2026-11-18, screening)

**Change.** New harness variant `sections` (`evals/variants.py`): the writer is called once per aspect
(up to six), each call writing only that aspect's section with every specific the sources state, and
then one merge call stitches the sections into a report, keeping facts and citations as written.

**Claim.** 62% of the misses are "partial" — the report states the gist of an aspect and drops its
numbers, dates and conditions. A global call spreads attention over the whole topic; a call that owns
one aspect cannot "move on", so its specifics should survive. This is the phase-6 design from the
master plan (notes/sections → merge), not another prompt wording.

**Arms.** Frozen old contexts, single-call writer (0.501, `w-1920`) vs `sections`, 6 tuning topics,
1 repeat, 1920-word budget.

**Criterion.** core +7 pp (above the 7.2 pp A/A band) with t > 2; citation support not worse. The variant
costs about 4x the single call by construction, so the question is whether the gain justifies it.

**Result.**

| topic | single call | sections | delta |
|---|---|---|---|
| it-passkeys | 0.514 | 0.457 | −5.7 |
| it-pg-incremental-backup | 0.675 | 0.725 | +5.0 |
| law-eu-ai-act-gpai | 0.714 | 0.686 | −2.9 |
| medicine-glp1 | 0.323 | 0.484 | +16.1 |
| nutrition-time-restricted-eating | 0.323 | 0.323 | 0.0 |
| politics-eu-cbam | 0.455 | 0.485 | +3.0 |
| **mean** | **0.501** | **0.527** | **+2.6** |

Paired: **+2.6 pp, sd 7.7, t 0.83**. Cost $0.0081 → $0.0323 per report (4×), wall 57 → 90 s,
**$0.0093 per point**.

**Verdict: reject.** Four times the cost for 2.6 pp inside the noise band. The one real signal is
`medicine-glp1` (+16.1, the weakest topic at single-call), which suggests the decomposition helps where
the sources are thin — but not enough to pay for it everywhere.

**What it establishes.** With `notes-write`, `write-revise`, `specifics` and now `sections`, four
writer-side designs have failed to move coverage beyond noise. The writer is not the lever we can pull
with structure alone at this price; the remaining ~45 pp of missing facts are dominated by what the
sources contain and by what the model judges worth writing.

### E12 — Phase 6, research-grounded: evidence matrix per aspect (2026-11-18, reject)

**Change.** New harness module `evals/synthesis_lab.py` (deterministic: evidence matrix, coverage gaps,
guard, assembly) and two variants: `matrix` (one writing call per aspect on **only that aspect's
evidence block**) and `matrix-repair` (same, plus one targeted repair call fed the server's list of
missing facts). Designed after SAFE/LongFact, ICAT, protocol-constrained RAG and Retrieve-then-Verify.

**Claim.** Four earlier writer variants failed because none kept a per-aspect list of facts and checked
it after writing. A narrow evidence block makes each section focus; a deterministic gap check plus a
targeted repair closes the loop.

**Arms.** Frozen old contexts, baseline single call (0.501) vs `matrix` vs `matrix-repair`, 6 tuning
topics, 1 repeat, 1920-word budget.

**Result.**

| topic | baseline | matrix | matrix-repair |
|---|---|---|---|
| it-passkeys | 0.514 | 0.086 | 0.171 |
| it-pg-incremental-backup | 0.675 | 0.300 | 0.275 |
| law-eu-ai-act-gpai | 0.714 | 0.143 | 0.171 |
| medicine-glp1 | 0.323 | 0.129 | 0.161 |
| nutrition-time-restricted-eating | 0.323 | 0.065 | 0.065 |
| politics-eu-cbam | 0.455 | 0.030 | 0.152 |
| **mean** | **0.501** | **0.125** | **0.166** |

`matrix`: −37.5 pp, sd 13.5, t −6.83. `matrix-repair`: −33.5 pp, sd 13.0, t −6.30. Costs $0.0076 and
$0.0093 against $0.0081, wall 77 s and 85 s.

**Verdict: reject.** The reports are coherent and correctly cited, but the sections saw only ~2.5k
characters of lexically matched sentences per aspect. The writer was faithful to a narrow slice, so
recall collapsed: the rubric asks about facts that never entered the evidence block.

**What it establishes.** The literature's evidence matrix is built over a **high-recall** candidate pool;
this implementation cut it with top-k lexical matching, which is recall-poor by construction. Narrowing
the context is the wrong lever here: the baseline's 93k characters of context are what carry the facts.
The per-aspect checklist idea is still untested as a *checklist on top of the full context* (v2).

**Not shown.** Whether `matrix` (full context + per-aspect evidence checklist + targeted repair) improves
on the baseline. That is the v2 design.

### E13 — Phase 6 v2: full context plus a per-aspect checklist and a targeted repair (2026-11-18, reject)

**Change.** Variant `matrix-checklist`: the writer keeps the **full context** (E12's recall collapse was
caused by narrowing it), gets a deterministic must-cover checklist of up to six evidence sentences per
aspect, then one repair call fed the server's list of facts the report does not state.

**Claim.** E12 lost recall because the sections saw only 2.5k characters. Keeping the full context
restores recall, and the checklist plus the gap-driven repair supplies the closure the earlier variants
lacked. Designed after SAFE/ICAT (facts to cover) and Retrieve-then-Verify (check per claim).

**Arms.** Frozen old contexts, baseline single call (0.501, $0.0081) vs `matrix-checklist`, 6 tuning
topics, 1 repeat, 1920-word budget.

**Result.**

| topic | baseline | checklist | delta |
|---|---|---|---|
| it-passkeys | 0.514 | 0.543 | +2.9 |
| it-pg-incremental-backup | 0.675 | 0.675 | 0.0 |
| law-eu-ai-act-gpai | 0.714 | 0.686 | −2.9 |
| medicine-glp1 | 0.323 | 0.226 | −9.7 |
| nutrition-time-restricted-eating | 0.323 | 0.290 | −3.2 |
| politics-eu-cbam | 0.455 | 0.394 | −6.1 |
| **mean** | **0.501** | **0.469** | **−3.2** |

Paired: −3.2 pp, sd 4.4, t −1.76. Cost $0.0081 → $0.0107 (+33%), wall 57 → 58 s. Every report had a
checklist gap and the repair added it, so the loop closed — and coverage still fell.

**Verdict: reject.** The checklist pulls attention toward the facts the server selected for it, and those
are the ones lexically matched to the decomposition's aspects, not the ones the rubric asks about. The
model writes what it is told to emphasise, and the rubric is not what it was told.

**What it establishes.** Seven writer-side designs have now been measured against the single call:
`notes-write`, `write-revise`, `specifics`, `sections`, `matrix`, `matrix-repair`, `matrix-checklist`.
None beats it; two lose. The single call is at the model's ceiling for this prompt, and further gains
have to come from the sources or from the model itself.

### E15 — Does page truncation cost facts? Per-page cap 25k → 40k at `high` (2026-11-18, inconclusive)

**Change.** Harness variant `big-pages` raises `per_page_char_cap` from `EVAL_PAGE_CHARS`. At `high` the
cap was 25k and the pipeline truncated 6 pages across 6 reports; with 40k it truncated 2.

**Claim.** A fact cut by the per-page cap is invisible to the writer. Six truncated pages per six reports
means some facts never arrive, and widening the cap should recover them.

**Arms.** Live `high`, 6 tuning topics, 1 repeat, baseline `base-high` repeat 1.

**Result.** 0.541 → 0.566 (**+2.5 pp, sd 14.9, t 0.42**). Per topic: +17.1, −10.0, +2.9, −19.4, +6.5,
+18.2. Truncated pages 6 → 2; context 151k → 156k; cost $0.0273 → $0.0174.

**Verdict: inconclusive.** The direction is positive and truncation halved, but the per-topic spread is
the largest of any arm so far, which is the draw variance talking, not the cap.

## Where the work landed (2026-11-19)

Thirteen experiments, eight decision records, one shipped quality change.

Keep: the per-stage trace, the robustness fixes, the doubled word budget from medium up, the single-call
writer, the aspect-shaped queries. Reverted or rejected: retrieval breadth, selection width, seven writer
designs, wider pages at high, a longer `max` budget.

The plateaus are recorded so they are not re-measured: coverage is ~0.50 at medium against ~0.77 of facts
on some fetched page; the query style is already the best of six measured; the writer does not use more
material than it already has. The next quality step needs a change of sources or of model, not of the
pipeline's shape.

## Ledger

| date | what | deepseek $ |
|---|---|---|
| 2026-11-18 | baseline + new retrieval, 2 × 12 reports | 0.258 |
| 2026-11-18 | control topics, 2 × 6 reports | 0.131 |
| 2026-11-18 | `specifics` at 960, 12 reports | 0.097 |
| 2026-11-18 | budget ladder, 18 reports | 0.158 |
| 2026-11-18 | cross-tests at 1920, 12 reports | 0.104 |
| 2026-11-18 | confirmation 960 vs 1920, 2 reps, 12 reports | 0.086 |
| 2026-11-18 | degraded variant, 6 reports | 0.022 |
| 2026-11-18 | tier baselines: low 6, high 6, max 6 | 0.423 |
| 2026-11-18 | high K=20 screening, 6 reports | 0.170 |
| 2026-11-18 | high K=20 confirmation, 12 + 12 reports | 0.354 |
| 2026-11-18 | sections variant, 6 reports | 0.194 |
| 2026-11-18 | matrix / matrix-repair, 13 reports | 0.110 |
| 2026-11-18 | matrix-checklist, 7 reports | 0.076 |
| 2026-11-18 | big pages at high, 6 reports | 0.104 |
| 2026-11-18 | smoke tests (2) | 0.024 |
| **total** | | **2.311** |

## Traps worth remembering

- **A single repeat lies.** E03 measured +3.5 pp on one repeat and −1.3 pp on two. Never decide on one.
- **A stored baseline is not a baseline.** The same configuration read 52.4% in one session and 45.7% in
  the next, because the judge and the model had moved. Always re-run the baseline arm next to the
  variant; the pair from one day is the only comparable unit.
- **A changed judge breaks comparisons.** Numbers from a full evidence-based judge pack are not
  comparable with coverage-only numbers. Compare only runs judged by the same protocol.
- **Frozen vs live is not comparable across arms** unless the frozen arm replays the exact context of
  the live arm (it does when you pass `--frozen-from <run>`).
- **Condensation hides the result.** A variant that pushes the report over the word ceiling triggers a
  second call that removes content, so the gain cancels itself. Check `condense_called` in the trace.
- **A lexical coverage matcher is lenient.** On 205 atoms it agreed with the judge 75% of the time, but
  it counted 45 of the judge's 100 misses as covered. Use `synthesis_lab.is_covered` for stage-by-stage
  *structure* (is the fact in the context at all), never as a level: it reads ~20 pp high.
- **A prefix match can merge two arms.** `"w-matrix-r"` starts with `"w-matrix"`, so a `startswith`
  filter put both arms in one column and produced identical numbers for every topic. Compare sides by
  the exact string before the depth suffix, and treat identical per-topic values as a bug until proven
  otherwise.
- **Subagent judging fails silently.** With 4 packs per agent the judge reasons until its budget runs
  out and never writes the file. One pack per agent, `quick`, and a "write then verify with json.load"
  requirement fixed it (36 of 36).
