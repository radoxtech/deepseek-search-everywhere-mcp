# Evaluation harness

Development-only harness that measures research quality, cost and time. It is not part of the package
and not part of the default test run; nothing here runs unless started by hand.

- `topics.json`: 12 topics (4 IT; one each for nutrition, medicine, sport, e-commerce, science,
  politics, law, business).
- `rubrics/<topic_id>.json`: 8-12 key points per topic with supporting sources, prepared with web
  research. Only rubrics with `"status": "approved"` should be used for a real comparison.
- `run.py` / `run_one.py`: run jobs, each in a fresh process with `refresh=True`, saving the full
  envelope, the `DEEPSEEK_MCP_DEBUG_DUMP_DIR` dump and the token usage of every chat call.
- `variants.py`: harness-side patches that turn the server into a variant under test.
- `metrics.py`: code metrics (citation validity and lexical support, unverified numbers, source count
  and diversity, off-peak cost, wall time).
- `judge.py`: blind pairwise judge packs (both orders) and scoring with the win rule.

Results go to `evals/results/` (ignored by git).

## Flow

```bash
set -a; . ./.env; set +a                                   # DEEPSEEK_API_KEY; runs cost money
python evals/run.py --estimate --depths low,medium,high,max --repeats 2
python evals/run.py --name baseline --depths low,medium,high,max --repeats 2
python evals/run.py --name aa --depths medium --repeats 4
python evals/run.py --name degraded --variant degraded-two-sources --depths medium --repeats 2
python evals/metrics.py evals/results/baseline             # per-depth summary

python evals/judge.py build --aa aa:medium --out cmp-aa
# judge every pack in evals/results/cmp-aa/packs/ with a blind Claude subagent (no network);
# each verdict goes to evals/results/cmp-aa/verdicts/<pack_id>.json
python evals/judge.py score cmp-aa --write-tolerance
python evals/judge.py build --a baseline:medium --b degraded:medium --out cmp-degraded
python evals/judge.py score cmp-degraded --tolerance evals/results/cmp-aa/_tolerance.json
```

## Win rule

12 topics x 2 repeats = 24 comparisons. Each comparison is judged in both orders and counts as a win
only when both orders agree; otherwise it is a tie. A candidate passes with at least 16 wins and when it
is not worse on citation support and unverified numbers than the reference, beyond the A/A spread. Cost
and time are reported next to the result; a cost ratio above 3x needs a maintainer decision.

## Self-check

```bash
pytest evals/tests
```
