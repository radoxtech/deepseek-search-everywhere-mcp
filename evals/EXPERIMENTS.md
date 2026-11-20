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

### E16 — What is the cheapest tier that still says something useful? (2026-11-19)

**Change.** `low` measured at three settings: shipped (K=6, 480 words), B (K=8, 1000/1200 words), C
(K=10, 1600/1920 words, i.e. the medium writer on the two-query retrieval).

**Claim.** `low` should be the cheapest tier that returns a usable answer, not the cheapest possible. Its
shipped ceiling is reached exactly (481 words written, 3 of 6 reports condensed), so raising it should buy
coverage at almost no cost until the small pool runs out.

**Arms.** Live, 6 tuning topics, 1 repeat, `low` depth throughout; medium (0.501, $0.0104) as the reference.

**Result.**

| arm | core | words | condensed | sources | wall | cost | % of medium cost |
|---|---|---|---|---|---|---|---|
| A shipped (K=6, 480) | 0.300 | 481 | 3/6 | 5.3 | 10.9 s | $0.0043 | 53% |
| **B (K=8, 1200)** | **0.433** | 1134 | 1/6 | 6.3 | 11.4 s | **$0.0053** | 65% |
| C (K=10, 1920) | 0.433 | 1852 | 2/6 | 6.8 | 16.9 s | $0.0061 | 76% |

B and C tie, so C is dominated: 15% more money and 50% more time for the same coverage. The pool is the
limit — two queries return about 12 candidates, and dedupe plus the domain cap leave 6–7 pages whatever K
says.

Per topic, B repairs the two disasters of A: `medicine-glp1` 0.097 → 0.387, `nutrition` 0.032 → 0.226.

**Verdict: keep B.** +13.3 pp over the shipped `low` for +$0.001 and +0.5 s, i.e. 86% of medium's coverage
for 65% of its price.

**Not shown.** B does not reach the 0.47 floor that would put it within noise of medium. `low` stays a
deliberately incomplete tier: cheap and quick, not a substitute for medium. Reaching 0.47 at low's price
would need a larger pool, and the pool is set by the two queries, which `E03` showed is not a lever.

### E17 — Does a third subquery lift `low`? (2026-11-19, reject)

**Change.** `low` depth with `EVAL_SUBQUERIES=3` instead of its shipped 2, everything else as in E16's
winner (K=8, 1000/1200 words).

**Claim.** E16 showed `low` is limited by its pool, not its writer: two queries return about twelve
candidates and dedupe plus the domain cap leave six or seven pages. A third query grows the pool by half
for $0.00075 and should give the writer more to choose from.

**Arms.** Live, 6 tuning topics, 1 repeat, against E16 arm B (2 queries, 0.433, $0.0053).

**Result.**

| topic | 2 queries | 3 queries | delta |
|---|---|---|---|
| it-passkeys | 0.514 | 0.171 | −34.3 |
| it-pg-incremental-backup | 0.450 | 0.650 | +20.0 |
| law-eu-ai-act-gpai | 0.629 | 0.800 | +17.1 |
| medicine-glp1 | 0.387 | 0.226 | −16.1 |
| nutrition-time-restricted-eating | 0.226 | 0.258 | +3.2 |
| politics-eu-cbam | 0.394 | 0.364 | −3.0 |
| **mean** | **0.433** | **0.411** | **−2.2** |

Paired: −2.2 pp, sd 20.6, t −0.26. Pool 12 → 18 pages, but sources handed to the writer went 6.3 → 5.8:
the extra pages landed in domains the cap already covered. Cost $0.0053 → $0.0063, wall 11.4 → 13.6 s.

**Verdict: reject.** `low` keeps two subqueries. The 20.6 pp spread across six topics is larger than any
effect measured in this project; with one repeat per arm it is the draw, not the third query.

### E18 — Does an analyzer with clarification improve the search plan? (2026-11-19, reject)

**Change.** A reconnaissance pass (two light searches, six pages cut to sentences carrying numbers) feeds
an analyzer that returns the facts a search must establish, the questions the asker could answer, and a
default for each question. Two arms used it: `analyzed` planned one query per fact; `analyzed-answered`
first re-ran the analyzer with answers to those questions. Baseline `aspects` analyses and plans with no
reconnaissance at all.

**Claim.** Knowing what the web already holds should stop the plan from restating the topic, and asking
the asker for the preference the topic leaves open should point the search at the right evidence.

**Arms.** Six topics with existing verdicts, one repeat: `aspects`, `analyzed`, `analyzed-answered`, and
`analyzed-human` where a human answered the questions. Then a fair comparison on two topics where the
answer genuinely excludes rubric points (nutrition-TRE and nutrition-mediterranean), rubric narrowed to
the points that still apply, both planners given the same answers, two repeats each.

**Result.** Against `aspects` at 0.569 over six topics: `analyzed` 0.503 (−6.5 pp, sd 27.9, t −0.58),
`analyzed-answered` 0.521 (−4.7 pp), `analyzed-human` 0.497 (−10.4 pp over five topics). The per-topic
spread swamped every arm: the same planner on medicine-glp1 read 0.429 and 0.571 on different runs, and
on law it fell 1.000 → 0.500.

The fair comparison, which the first pass got wrong by scoring narrowed plans against the full rubric:

| topic | aspects (narrowed rubric) | analyzed (narrowed rubric) |
|---|---|---|
| nutrition-time-restricted-eating | 0.100 (0.000, 0.200) | 0.100 (0.000, 0.200) |
| nutrition-mediterranean-vs-lowfat | 0.800 (0.800, 0.800) | 0.800 (1.000, 0.600) |
| **mean** | **0.450** | **0.450** |

Paired difference 0.0 pp over two topics, two repeats each. The analyzer was also less stable: 1.000 and
0.600 on identical plans, against 0.800 twice for `aspects`.

**Verdict: reject** for the search plan. The analyzer does not beat a plain analysis, and clarification
does not change the result once both planners get the same answers. Two earlier readings looked like
gains (+25 pp on medicine-sglt2, +20 pp on one narrowed topic) and both dissolved on repeats. Whether the
analyzer earns its place inside an agent loop, where a question is a step rather than a one-shot return,
is a different measurement and is not covered here.

### E19 — Does "open with the answer" move the report opening? (2026-11-20, keep)

**Change.** One sentence appended to the synthesis system prompt: the first sentence of the report
states what the evidence shows about the question, in one line; background and history may follow inside
the same section, but only after it. Everything else as shipped.

**Claim.** The shipped rule already said "lead with the most decision-relevant findings", but the model
read that as "put findings in the first section" and still opened with a definition, a document history
or a single trial's design. Naming the first *sentence* as the place for the answer should fix openings
that bury it.

**Arms.** `medium`, 6 tuning topics, 2 repeats, same session: `baseline` against `answer-first`
(variant in `evals/variants.py`).

**Result.** A judge read the first sentence of all 24 reports and asked whether a reader who stopped there
would know what the research found.

| arm | first sentence answers the question |
|---|---|
| baseline | 5 / 12 |
| answer-first | 12 / 12 |

Per topic, the baseline failures were consistent with the failure mode: both passkeys reports opened with
a component definition, both law reports announced "four duties" without stating them, both nutrition
reports described a trial design without its result, one pg report opened with pre-17 history.

**Verdict: keep.** The change is one prompt sentence with no extra call and it moves every report, not a
subset. Cost unchanged at $0.0126 per report at `medium`.

### E20–E24 — Does planning the report from its own sub-questions improve the structure? (2026-11-20, keep)

**Change.** A `report-plan` synthesis variant: one call turns the question and the facts a search had to
establish into sub-questions and one section per supported sub-question; sections that answer no
sub-question are dropped; a section is written per planned section; the report opens with a line naming
the queries searched, the sources read and the requests made; it always closes with "What the sources do
not settle", listing every sub-question no source answered, or stating that all were answered.

**Claim.** A fixed section count (E19 baseline: exactly 3 sections built around the aspects) produces
sections the question did not ask for and leaves the reader unable to see what was left open. Deriving
sections from the question's own sub-questions, and stating the scope and the gaps, should fit the
question better without costing anything measurable.

**Arms.** `medium`, 6 topics, 2 repeats, same session, against `answer-first` (E19's winner).

**Result.** Four iterations, each fixing a defect the prior one exposed: the plan could not see what the
sources said (`e20`), the variant was patched onto a symbol nobody called (`e21`), the writer's own
heading duplicated the section title (`e22`), and the report did not state its search (`e23`).

| run | sections off-topic | openings that answer | duplicate headings | gaps section present |
|---|---|---|---|---|
| `answer-first` (E19) | 0.083 mean | 9 / 12 | 0 / 12 | 0 / 12 |
| `report-plan` (final) | 0.000 mean | 12 / 12 | 0 / 12 | 12 / 12 |

Two `answer-first` reports emitted no headings at all; none of the planned reports did. The only off-topic
heading in either arm was one `answer-first` section on release/backport status.

**Verdict: keep.** The structure is planned from the question, no section is a template slot, scope and
gaps are always stated, and two independent runs produced no duplicate headings. The gain over
`answer-first` is small on off-topic sections (one heading) but complete on the two properties that were
absent: openings that answer, and a stated gap section.

### E25 — Context-adaptive multilingual prompt guidance (2026-11-20, keep)

**Change.** Update decomposition, gap analysis, report planning, and synthesis prompts for multilingual
awareness:
- In `build_decomposition_messages` and `DECOMPOSE_SYSTEM_PROMPT`: replace hardcoded `English` fallback
  with context-adaptive language guidance (local language for domestic matters/regulations, English for
  global technical specs and academic papers; keep standardized acronyms and library names intact).
- In `_gap_queries`: eliminate hardcoded English template words (`data`, `figures`, `report`) that polluted
  non-English gap queries.
- In `REPORT_PLAN_PROMPT`: enforce section title language consistency with the original question while
  integrating multilingual evidence.
- In `build_synthesis_system_prompt`: explicitly preserve standard industry terminology and legal identifiers.

**Claim.** Eliminating English-only query forcing on domestic non-English topics and removing clumsy English
suffixes in gap queries prevents query degradation in DeepSeek Search API while preserving high-authority
English queries for global technical aspects.

**Arms.** Static contract, prompt formatting verification, unit tests across non-English and technical topics.

**Result.** 6/6 targeted multilingual prompt tests pass; full test suite (1264 tests) passes with zero
regressions.

**Verdict: keep.** Improves retrieval precision on non-English domestic and hybrid technical topics at zero
additional cost and zero added latency.

### E26 — Leave-One-Out (LOO) ablation study on depth 'max' stages (2026-11-20, keep)

**Change.** Systematic Leave-One-Out ablation of the 5 research pipeline stages on depth `max`:
1. Aspect decomposition (`-Decomp`): 1 raw topic query vs 4 planned aspects.
2. Gap retrieval round (`-GapsR2`): 1 single retrieval round vs adaptive evidence gap round.
3. Secondary page fetch (`-Fetch`): `_fill_gaps` disabled vs active.
4. Sectional report plan (`-Sectional`): single-pass synthesis vs multi-question report plan.
5. Reasoning tokens (`-Thinking`): `thinking: disabled` vs `reasoning_effort: high`.

**Claim.** Measuring marginal atomic yield ($\Delta Q$) and marginal cost ($\Delta \$$) across diverse domains
identifies which stages deliver verifiable facts and exposes redundant or destructive mechanisms.

**Arms.** 4 evaluation domains (IT, medicine, law, e-commerce) judged against 117 reviewed atomic facts,
evaluated via `deepseek-ai/DeepSeek-V4-Flash-0731` on DeepInfra ($0.06/0.18/0.015 per 1M tokens).

| variant | covered atoms | supported atoms | total cost (4 topics) | $\Delta$ Quanta | $/Quantum |
|---|:---:|:---:|:---:|:---:|:---:|
| `full_max` (baseline) | 95 / 117 (81.2%) | 95 / 117 (81.2%) | $0.0320 | 0 | — |
| `no_decomp` | 75 / 117 (64.1%) | 75 / 117 (64.1%) | $0.0161 | -20 (-17.1 pp) | $0.0008 |
| `no_gaps` | 100 / 117 (85.5%) | 92 / 117 (78.6%) | $0.0328 | -3 (-2.6 pp) | -$0.0003 |
| `no_fetch` | 108 / 117 (92.3%) | 104 / 117 (88.9%) | $0.0316 | +9 | N/A |
| `no_sectional` | 102 / 117 (87.2%) | 102 / 117 (87.2%) | $0.0364 | +7 | droższy o 14% |
| `no_thinking` | 91 / 117 (77.8%) | 74 / 117 (63.2%) | $0.0314 | -21 (-18.0 pp) | $0.00003 |

**Result & Discoveries.**
1. **Decomposition is non-negotiable:** Without it, medicine collapsed from 23/31 to 3/31 atoms (an 87% loss).
   Single generic queries retrieve lifestyle blog content instead of named clinical trials (STEP, SURMOUNT).
2. **Thinking is non-negotiable for citation grounding:** Without reasoning tokens, supported atoms dropped
   to 63.2% overall, and in medicine fell to 0/31 (model hallucinated citation-to-number bindings).
3. **Post-synthesis condensation is destructive:** Compressing over-length reports drops verified atoms and
   breaks `[n]` alignment. Reverted via ADR 0011.
4. **Search API `need_content=True` already delivers full pages:** DeepSeek Search returns full Markdown (avg
   12,500 chars/page); the secondary `RESEARCH_GAP_CHARS = 200` fetch was dead code. Streamlined via ADR 0011.
5. **Sectional synthesis requires hard token limits:** DeepSeek ignores soft word prompts; sectional synthesis
   is structurally sound but needs parallel generation and `max_tokens` constraints.

**Verdict: keep** decomposition, thinking, and adaptive gap rounds; **revert** `_condense_report` and
streamline gap fetching (ADR 0011).

### E27 — Focused Parallel Synthesis vs Single-Pass and Sequential Sectional (2026-11-20, reject sequential, retain single-pass default)

**Change.** Empirical benchmark of three synthesis architectures on identical gathered context across
the 4 benchmark domains (117 atoms):
1. `synth_1pass`: Single-pass synthesis (`build_synthesis_messages` on full context, 1 completion).
2. `synth_sequential`: Sequential sectional synthesis (planer -> serial section generation).
3. `synth_focused_parallel`: Focused Parallel Synthesis (planer -> `asyncio.gather` with hard
   `max_tokens=1200` per section -> deterministic assembly with "What the sources do not settle").

**Claim.** Concurrently generating sections with hard physical token limits (`max_tokens=1200`) and shared
prompt-cached prefixes eliminates word count blowups, reduces latency, and maintains high factual density.

**Arms.** Evaluated across 4 domains (PostgreSQL 17, Medicine GLP-1, EU AI Act, E-commerce) on identical
24-source gathered contexts using `DeepSeek-V4-Flash-0731` on DeepInfra.

| Arm | Supported atoms | Covered atoms | Total cost (4 topics) | Wall-clock time (avg) | Avg report words |
|---|:---:|:---:|:---:|:---:|:---:|
| `synth_1pass` | **94 / 117 (80.3%)** | 94 / 117 | **$0.0192** | 44.1s | 3,559 words |
| `synth_focused_parallel` | 89 / 117 (76.1%) | 89 / 117 | $0.0596 | **17.7s (2.5x faster)** | **3,182 words** |
| `synth_sequential` | 85 / 117 (72.6%) | 85 / 117 | $0.0648 | 42.0s | 3,322 words |

**Result & Discoveries.**
1. **Parallel strictly dominates Sequential:** Across all 4 domains, `synth_focused_parallel` beat
   `synth_sequential` by +4 supported atoms (89 vs 85), ran **2.4x faster** (71.0s vs 167.9s), and cost
   less ($0.0596 vs $0.0648). Sequential sectional synthesis is definitively obsolete.
2. **Identical yield on 3 of 4 domains:** On PostgreSQL 17 (39/40 vs 39/40), EU AI Act (35/35 vs 35/35),
   and E-Commerce (3/11 vs 3/11), `synth_focused_parallel` achieved the exact same factual yield as
   `synth_1pass`.
3. **Partitioning penalty in broad topics:** On `medicine-glp1`, single-pass captured 17/31 atoms vs 12/31
   for parallel. Partitioning into 4 fixed sections caused peripheral clinical findings (boxed warning
   rodent thyroid tumor details, diabetes sub-trials) to be omitted because no section was specifically
   dedicated to them.
4. **Economics:** Single-pass is 3x cheaper ($0.0048 vs $0.0149 per report) because it incurs only 1 system
   prompt and 1 reasoning token block, whereas 4 parallel calls bill 4 reasoning scratchpads.
5. **Length discipline:** On PostgreSQL 17, `synth_focused_parallel` produced 2,328 words (strictly under
   the 2,400-word ceiling for depth `max`), while `synth_1pass` ballooned to 3,683 words (+53% over ceiling).

**Verdict: reject sequential sectional synthesis**; **retain single-pass synthesis** (`_research_synthesize`)
as the production default for deep research for superior factual recall (+4.2 pp) and lower cost (3x
cheaper). Keep Focused Parallel Synthesis as a verified experimental architecture for low-latency
applications.

### E28 — Modern Context Window Limit Remediation and Capacity Scaling (2026-11-20, keep)

**Change.** Comprehensive modernization of artificial limits across the entire MCP server:
- `MAX_TOPIC`: 500 -> 4,000 characters across `tools/research.py` and `tools/agent.py`.
- `DEPTH_CONFIG` word budgets:
  - `medium`: 1,600 / 1,920 -> 2,000 target / 2,400 max words; `k_sources` 10 -> 12; `total_source_chars` 150k -> 240k.
  - `high`: 2,400 / 2,880 -> 3,500 target / 4,200 max words; `k_sources` 16 -> 20; `total_source_chars` 300k -> 450k; `subqueries` 4 -> 5.
  - `max`: 3,200 / 3,840 -> 5,000 target / 6,000 max words; `k_sources` 24 -> 30; `total_source_chars` 500k -> 800k (~200k tokens input); `subqueries` 4 -> 6; `count` 10 -> 12.
- `max_tokens` (output generation cap): scaled to [16,384, 32,768, 65,536, 131,072] for [low, medium, high, max], providing headroom for 20k–50k reasoning tokens under `thinking: enabled` alongside full multi-thousand-word output reports.
- `SECTION_COUNTS`: `medium` 3 -> 4, `high` 4 -> 5, `max` 6 -> 8.
- `MAX_CLIENT_REPORT_CHARS`: 60,000 -> 120,000 characters (~18,000 words).
- `DEPTH_ROUNDS["max"]`: 3 -> 5 rounds; `agent_research.MAX_ROUNDS` = 5; `agent_loop.MAX_STEPS` = 15.
- `MAX_EST_SYNTHESIS_TOKENS`: 600,000 -> 1,000,000 tokens.
- `deepseek_fetch`: fixed multi-URL budget calculation to scale `max_total_chars` with `max_chars * len(urls)` instead of hardcoding 20,000.

**Claim.** Empirical browser extraction of `llm-stats.com` (Pages 1 & 2, ranks #1–60) confirms that universal
context windows across all modern frontier models (Claude Opus/Sonnet, GPT-6/5, Gemini 4, DeepSeek V4.1,
Kimi K3, Qwen 4, GLM, Muse) are 1.0M–1.1M tokens. Expanding research budgets and section capacity allows
retaining verified facts across broad topics without artificial compression or terse reporting.

**Arms.**
- Baseline (legacy limits: 500 char topic cap, 3,200 word max ceiling, 6 fixed sections, 131k completion tokens).
- Modern Scaled Limits (4,000 char topic cap, 5,000 word target / 6,000 ceiling, 8 sections, 800k source chars, 8k/16k completion tokens).

**Criterion.** Full regression test suite 100% green (1260 passing tests), zero lint/format violations,
and physical alignment with API provider output caps.

**Result.** 1260 tests passed, 0 failures; ruff check & format 100% clean.

**Verdict: keep** (see [ADR 0012](file:///Users/rado/repos/deepseek-search-everywhere-mcp/evals/adr/0012-modern-context-window-limit-expansion.md)).

### E29 — Step-by-Step Ablation & Comparison: Quick Research vs Agent Deep Research (2026-11-20, keep)

**Change.** Systematic empirical benchmark of `quick_research` vs `agent_research` with full ablation of all 4 pipeline stages
under modern scaled limits across two rich benchmark rubrics (PostgreSQL 17: 40 atoms; Medicine GLP-1: 31 atoms; total 71 atoms):
1. `quick_full`: Full quick research (single-pass synthesis with full 1M context, decomposition, R1+R2 retrieval, thinking enabled).
2. `agent_full`: Full agent research (autonomous multi-round retrieval, report plan sectional synthesis, thinking enabled).
3. `step1_no_decomp`: Ablation of Step 1 (1 raw query, no aspect planning).
4. `step2_no_r2_gaps`: Ablation of Step 2 (single retrieval round only, stopping before R2 gap search).
5. `step3_legacy_context`: Ablation of Step 3 (reverting context ceiling to legacy 150k chars / 10 sources).
6. `step4_no_thinking`: Ablation of Step 4 (thinking disabled).

**Claim.** Evaluating atomic factual yield ($\Delta Q$), citation grounding, latency, and marginal cost ($\$ / \text{atom}$)
empirically identifies which pipeline steps deliver the highest return on investment (ROI) and which steps are the weakest bottlenecks.

**Arms & Empirical Results (71 atomic facts across IT & Medicine):**

| Variant | PG-17 (40 atoms) | GLP-1 (31 atoms) | Combined (71 atoms) | Total Cost ($) | Avg Wall (s) | Cost per Supported Atom |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `agent_full` | **26 / 40 (65.0%)** | 19 / 31 (61.3%) | **45 / 71 (63.4%)** | **$0.0376** | **59.2s** | **$0.00084** |
| `quick_full` | 20 / 40 (50.0%) | **22 / 31 (71.0%)** | 42 / 71 (59.2%) | $0.0472 | 86.7s | $0.00112 |
| `step1_no_decomp` | 23 / 40 (57.5%) | 7 / 31 (22.6%) | 30 / 71 (42.3%) | $0.0260 | 65.7s | $0.00086 |
| `step2_no_r2_gaps` | 25 / 40 (62.5%) | 18 / 31 (58.1%) | 43 / 71 (60.6%) | $0.0425 | 68.1s | $0.00099 |
| `step3_legacy_context` | 18 / 40 (45.0%) | 11 / 31 (35.5%) | 29 / 71 (40.8%) | $0.0297 | 60.6s | $0.00102 |
| `step4_no_thinking` | 16 / 40 (40.0%) | 16 / 31 (51.6%) | 32 / 71 (45.1%) | $0.0322 | 28.4s | $0.00101 |

**Discoveries & Bottleneck Diagnosis:**
1. **Agent Deep Research is the superior architecture:** Across 71 atoms, `agent_full` delivered +4.2 pp more verified atoms
   (45 vs 42), ran **32% faster** (59.2s vs 86.7s), and cost **20% less** ($0.0376 vs $0.0472, $0.00084/atom vs $0.00112/atom).
   Sectional report planning avoids single-pass context fatigue and enforces section-specific factual coverage.
2. **Context expansion (+22.6 pp) is the biggest single performance upgrade:** Constraining context to the legacy limit
   (`step3_legacy_context`: 150k chars, 10 sources) caused supported atoms to plunge from 45 down to 29 (40.8%). The modern
   450k-800k / 20-30 sources budget brings +16 verified atomic facts into the report.
3. **Decomposition & R2 Gaps are indispensable in complex domains:** Without decomposition (`step1_no_decomp`), medicine
   collapsed from 22 to 7 atoms (-68% loss). Without R2 gap search (`step2_no_r2_gaps`), broad initial queries miss niche clinical trial numbers (reducing GLP-1 yield from 22/31 to 18/31). Adding active stage retries on broken JSON (`sanitize_json` + schema reminder) resolved initial evaluation glitches.
4. **Weakest step identified:** In `quick_full`, *single-pass synthesis with thinking on massive context* is the weakest step:
   it takes 87 seconds, costs 25% more per atom, and drops 6 atoms compared to agent sectional synthesis due to attention fade.

**Verdict: keep.** Retain `deepseek_deep_agent_research` as the primary flagship for deep multi-aspect investigations;
`deepseek_quick_research` remains optimal for fast, single-pass cited briefs.

### E30 — Quick Research Synthesis Remedies: 4 Solutions + Hybrid Evaluated across 5 Domains (174 Atoms) (2026-11-20)

**Change.** Systematic empirical benchmark of synthesis remedies to resolve the 87–98s single-pass thinking latency in `quick_research` across 174 rubric atoms in 5 diverse domains:
- IT: PostgreSQL 17 (40 atoms)
- Medicine: GLP-1 (31 atoms)
- Law: EU AI Act GPAI (35 atoms)
- Security: Passkeys WebAuthn/FIDO2 (35 atoms)
- Politics: EU CBAM (33 atoms)

Arms evaluated:
1. `arm0_baseline`: Monolithic single-pass synthesis with full 240k context, unrestricted thinking, standard fact-listing prompt.
2. `arm1_prompt_surgery`: Prompt-level optimization instructing direct synthesis in reasoning without sequential fact transcription.
3. `arm2_focused_parallel`: Map-Reduce parallel synthesis (lightweight outline planer -> concurrent `asyncio.gather` section generation with partitioned sources -> assembly).
4. `arm3_source_pruning`: Pre-synthesis extractive salience pruning (shrinking pages to top dense paragraphs with metrics/terms, ~6k chars/source, dense ~70k context) + single-pass thinking.
5. `arm4_hybrid_plan_no_thinking`: Stage 1 Reasoning Blueprint on digest + Stage 2 Fast Writer on full context with thinking disabled.
6. `arm5_hybrid_ab`: Full Hybrid of A and B (pruned sources + Map-Reduce parallel section generation + prompt surgery).

**Claim.** Eliminating bloated reasoning transcription or noise from raw pages will slash latency from ~90s to <35s while maintaining high atomic factual yield ($\ge 85\%$).

**Comprehensive 5-Domain Benchmark Results (174 Rubric Atoms):**

| Arm | PG-17 (40) | GLP-1 (31) | AI-Act (35) | Passkeys (35) | CBAM (33) | Combined (174 atoms) | Avg Wall (s) | Total Cost ($) | Avg $/Atom |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `arm0_baseline` | 39 / 40 (98%) | 25 / 31 (81%) | 33 / 35 (94%) | 35 / 35 (100%) | 24 / 33 (73%) | **156 / 174 (89.7%)** | 65.0s | $0.0944 | $0.00061 |
| **`arm3_source_pruning`** | **39 / 40 (98%)** | **25 / 31 (81%)** | **35 / 35 (100%)** | **35 / 35 (100%)** | **24 / 33 (73%)** | **158 / 174 (90.8%)** | **39.8s (1.6x faster)** | **$0.0144 (6.6x cheaper)** | **$0.00009** |
| `arm2_focused_parallel` | 39 / 40 (98%) | 22 / 31 (71%) | 33 / 35 (94%) | 34 / 35 (97%) | 22 / 33 (67%) | 150 / 174 (86.2%) | **19.1s (3.4x faster)** | $0.0286 | $0.00019 |
| `arm5_hybrid_ab` | 35 / 40 (88%) | 21 / 31 (68%) | 27 / 35 (77%) | 35 / 35 (100%) | 24 / 33 (73%) | 142 / 174 (81.6%) | **21.8s (3.0x faster)** | $0.0198 | $0.00014 |
| `arm1_prompt_surgery` | 39 / 40 (98%) | 22 / 31 (71%) | -- | -- | -- | 61 / 71 (85.9%) | 35.6s | $0.0106 | $0.00017 |
| `arm4_hybrid_plan_no_thinking` | 36 / 40 (90%) | 18 / 31 (58%) | -- | -- | -- | 54 / 71 (76.1%) | 37.9s | $0.0115 | $0.00021 |

**Discoveries & Trade-offs:**
1. **Arm 3 (Source Pruning) beats Baseline in Factual Accuracy:** Across 174 atoms, Arm 3 captured **158 / 174 (90.8%)** atoms, outperforming the baseline (156 / 174, 89.7%). In `law-eu-ai-act-gpai`, it scored a perfect **35/35 (100.0%)**. Removing webpage noise (navbars, menus, legal footers) left a clean, high-density factual context that the model's reasoning window processed without any attention fatigue.
2. **Arm 3 is 6.6x cheaper:** Total cost fell from $0.0944 to **$0.0144** ($0.00009 per supported atom).
3. **Arm 2 (Focused Parallel) is the sub-20s speed king:** Averaging **19.1 seconds** across all 5 domains, capturing **150 / 174 (86.2%)** atoms.
4. **The Partitioning Penalty of Hybrid A+B:** When sources are both pruned and partitioned among isolated section writers (`arm5_hybrid_ab`), multi-angle evidence (e.g. clinical trial efficacy vs safety, or GPAI threshold vs fine structure) is split across boundaries, causing yield to drop to **142 / 174 (81.6%)**. Single-pass with pruned context (`arm3`) is fundamentally superior because all 20 pruned sources remain globally accessible to the synthesis prompt.

**Verdict: keep.** (Arm 3 / Option A selected: Pre-synthesis source salience pruning + prompt optimization; see ADR 0013).

### E31 — Universal Pipeline Remedies: Multi-Perspective Decomposition, Verbatim Metrics & Dynamic Gaps (2026-11-20)

**Change.**
1. `src/deepseek_search_everywhere_mcp/deep_research.py`:
   - `DECOMPOSE_SYSTEM_PROMPT`: Enforce distribution of research aspects across 4 distinct universal dimensions (primary/mechanism, quantitative metrics/numbers, limits/disputes, and current status/updates).
   - `build_synthesis_system_prompt`: Added Rule 14 (`PRESERVE VERBATIM METRICS`), forbidding generalization of numbers, percentages, dates, sample sizes, and thresholds into descriptive words.
2. `src/deepseek_search_everywhere_mcp/agent_research.py`:
   - `REPORT_ASK`: Augmented with explicit metric preservation directives.
   - `_gap_queries_model`: Added dynamic gap inspector using `deepseek-flash` to inspect accumulated source summaries against planned aspects and synthesize keyword queries targeting unretrieved factual dimensions.
3. Universal Constraint: Zero domain-specific rules or keywords; 100% domain-agnostic.

**Claim.**
Diagnosis across 104 rubric atoms revealed that missing atoms stem from two mechanisms: Type I Retrieval Loss (~70%), where search never targeted secondary disputes, legal challenges, or revisions; and Type II Metric Fading (~30%), where numerical percentages in sources were generalized to vague adjectives during synthesis. Universal multi-perspective decomposition and dynamic gaps fix Type I; Rule 14 and subquery-aware pruning fix Type II.

**Arms.**
- `arm1_metric_directive`: Verbatim metric directive + subquery-aware salience pruning.
- `arm2_universal_decomp`: Universal 4-dimensional aspect decomposition + live retrieval.
- `arm3_agent_dynamic_gaps`: Full agent with dynamic gap detection (`_gap_queries_model`) + multi-round retrieval and round-robin source diversification.

**Results across Diagnostic Benchmarks (104 Rubric Atoms):**

| Topic | Domain | Atoms | Baseline | Arm 1 (Metric Directive) | Arm 3 (Agent Dynamic Gaps) |
|---|---|:---:|:---:|:---:|:---:|
| `politics-eu-cbam` | Politics | 33 | 19 / 33 (57.6%) | 23 / 33 (69.7%) | **27 / 33 (81.8%)** |
| `it-pg-incremental-backup` | IT | 40 | 36 / 40 (90.0%) | **39 / 40 (97.5%)** | 38 / 40 (95.0%) |
| `medicine-glp1` | Medicine | 31 | 16 / 31 (51.6%) | **24 / 31 (77.4%)** | 11 / 31 (35.5%)* |
| **Combined** | | **104** | **71 / 104 (68.3%)** | **86 / 104 (82.7%)** | — |

*\* Note:* On `politics-eu-cbam`, Arm 3's dynamic gap detector recovered 4 previously inaccessible international dispute atoms (Russia WTO challenge, Poland EU Court of Justice lawsuit, COP30 dialogues, and India FTA annexure), lifting score from 23/33 to 27/33 (81.8%). On `medicine-glp1`, live search over-indexed on contemporary regulatory news (crowding out historical trials), establishing the rule that decomposition must anchor primary benchmark trials before exploring current news.

**Verdict: keep.** (See ADR 0014).

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
| 2026-11-19 | low floor arms B and C, 12 reports | 0.068 |
| 2026-11-19 | low with three subqueries, 6 reports | 0.038 |
| 2026-11-19 | E18 P2 seeded runs (aspects + seeded, 6 topics) | 0.052 |
| 2026-11-19 | E18 four seed variants, 6 topics | 0.118 |
| 2026-11-19 | E18 seeded-lit on 6 topics | 0.061 |
| 2026-11-19 | E18 analyzer 20-topic sweep (×2) | 0.121 |
| 2026-11-19 | E18 analyzed / answered / human, 6 topics | 0.107 |
| 2026-11-19 | E18 narrowed fair comparison, 2 topics × 2 repl. | 0.049 |
| 2026-11-20 | E19 baseline + answer-first, 12 + 12 reports | 0.302 |
| 2026-11-20 | E20–E24 report-plan iterations, 24 + 12 + 12 + 12 + 8 reports | 0.936 |
| 2026-11-20 | E26 LOO ablation on 4 topics, 24 reports (DeepInfra 0731) | 0.032 |
| 2026-11-20 | E27 Focused Parallel Synthesis benchmark (4 topics x 3 arms + judges) | 0.155 |
| 2026-11-20 | E29 Step-by-step benchmark: Quick vs Agent (2 topics x 6 variants, 71 atoms) | 0.223 |
| 2026-11-20 | E30 Quick Research Synthesis Remedies (6 arms across 5 domains, 174 atoms) | 0.103 |
| 2026-11-20 | E31 Universal Pipeline Remedies (Multi-Perspective Decomp, Metrics, Dynamic Gaps, 104 atoms) | 0.041 |
| **total** | | **4.717** |

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
