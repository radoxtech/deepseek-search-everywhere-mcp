"""Blind pairwise judging: build judge packs, then score the verdicts with the win rule.

Build packs (A is the reference, B the candidate)::

    python evals/judge.py build --a baseline:medium --b degraded:medium --out cmp-degraded
    python evals/judge.py build --aa aa:medium --out cmp-aa          # repeats 1,2 vs 3,4 of one run

Every comparison (same topic and repeat on both sides) becomes two packs, one per order, so position
bias cancels out. A pack holds the topic, the approved rubric, both reports and, for every cited
sentence, an excerpt of the cited source: the judge needs no network. Pack files never name the
variants; ``_pairs.json`` maps packs back to sides. The judge (a Claude subagent, run by hand) writes
``verdicts/<pack_id>.json``::

    {"winner": "1" | "2" | "tie", "rubric_hits": {"1": ["k1", ...], "2": [...]},
     "unsupported": {"1": [sentence numbers], "2": [...]}, "reason": "one or two sentences"}

Score::

    python evals/judge.py score cmp-degraded [--tolerance evals/results/cmp-aa/_tolerance.json]

Win rule: a comparison is a win only when both orders pick the same side, otherwise a tie. The candidate
passes with at least 16 wins out of 24 comparisons and when it is not worse than the reference on
citation support and unverified numbers beyond the A/A tolerance. Cost and time are reported next to it;
a cost ratio above 3x is flagged for a maintainer decision.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402

RESULTS = HERE / "results"
RUBRICS = HERE / "rubrics"
WINS_TO_PASS = 16
COST_RATIO_LIMIT = 3.0
EXCERPT_CHARS = 450
EXCERPT_WINDOWS = 2

JUDGE_INSTRUCTIONS = """You are a strict, impartial judge comparing two research reports on the same topic.
You have no web access and need none: judge only from this pack.

Score each report against the rubric (key points a good report must cover; "core" points matter most),
then check faithfulness: for each numbered cited sentence, compare it with the excerpt of the source it
cites and list the sentence numbers the excerpt does not support. Prefer the report that covers more core
points correctly and makes fewer unsupported claims; length alone is not quality. Ignore formatting
differences. Answer "tie" when neither is clearly better.

Write only this JSON object as your answer:
{"winner": "1" | "2" | "tie", "rubric_hits": {"1": ["k1", ...], "2": [...]},
 "unsupported": {"1": [sentence numbers], "2": [...]}, "reason": "one or two sentences"}
"""


def _parse_side(spec: str) -> tuple[str, str]:
    name, _, depth = spec.partition(":")
    return name, depth


def _records(name: str, depth: str) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for record in metrics.load_records(RESULTS / name):
        if record.get("depth") == depth:
            out[(record["topic_id"], int(record["repeat"]))] = record
    return out


def _window_score(window: str, words: set[str], numbers: set[str]) -> int:
    lowered = window.lower()
    return sum(1 for w in words if w in lowered) + 3 * len(numbers & metrics._numbers(window))


def _excerpt(sentence: str, text: str, windows: int = EXCERPT_WINDOWS) -> str:
    """The ``windows`` best non-overlapping passages of ``text`` for the sentence, joined by " … ".

    A passage scores one point per content word of the sentence it contains and three per number of the
    sentence it contains, so the passage that holds the cited figure wins.
    """
    if not text:
        return "(source text not available)"
    words = metrics._content_words(sentence)
    numbers = metrics._numbers(sentence)
    step = EXCERPT_CHARS // 3
    starts = range(0, max(1, len(text) - EXCERPT_CHARS + step), step)
    scored = sorted(
        ((_window_score(text[i : i + EXCERPT_CHARS], words, numbers), -i) for i in starts), reverse=True
    )
    picked: list[int] = []
    for score_, neg_start in scored:
        start = -neg_start
        if score_ <= 0 and picked:
            break
        if all(abs(start - other) >= EXCERPT_CHARS for other in picked):
            picked.append(start)
        if len(picked) == windows:
            break
    return " … ".join(text[i : i + EXCERPT_CHARS].replace("\n", " ") for i in sorted(picked))


def _evidence(record: dict[str, Any]) -> list[str]:
    env = record.get("envelope") or {}
    texts = {
        c.get("id"): c.get("content") or ""
        for c in ((record.get("dump") or {}).get("synthesis_context") or [])
    }
    lines = []
    for n, (sentence, ids) in enumerate(metrics.cited_sentences(env.get("report") or ""), start=1):
        lines.append(f"S{n}. {sentence}")
        for i in ids:
            lines.append(f"    [{i}] {_excerpt(sentence, texts.get(i, ''))}")
    return lines


def atoms(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    """The atomic facts of a rubric's key points; a point without atoms is its own single atom."""
    out = []
    for k in rubric["key_points"]:
        for a in k.get("atoms") or [{"id": k["id"], "fact": k["point"], "main": True}]:
            out.append({**a, "point": k["id"], "importance": k["importance"]})
    return out


def _rubric_text(topic_id: str, *, use_atoms: bool = False) -> str:
    path = RUBRICS / f"{topic_id}.json"
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if not use_atoms:
        return "\n".join(f"- {k['id']} ({k['importance']}): {k['point']}" for k in rubric["key_points"])
    lines = []
    for a in atoms(rubric):
        accept = f" (also accept: {'; '.join(a['also_accept'])})" if a.get("also_accept") else ""
        lines.append(f"- {a['id']} ({a['importance']}): {a['fact']}{accept}")
    return "\n".join(lines)


def _pack(topic: str, topic_id: str, first: dict[str, Any], second: dict[str, Any]) -> str:
    parts = [JUDGE_INSTRUCTIONS, f"\n# Topic\n{topic}\n", f"# Rubric\n{_rubric_text(topic_id)}\n"]
    for label, record in (("1", first), ("2", second)):
        env = record.get("envelope") or {}
        report = env.get("report") if env.get("ok") and env.get("status") == "done" else None
        parts.append(f"# Report {label}\n{report or '(the job failed and returned no report)'}\n")
        parts.append(f"# Evidence for report {label} (cited sentences and cited source excerpts)\n")
        parts.append("\n".join(_evidence(record)) + "\n")
    return "\n".join(parts)


def build(args: argparse.Namespace) -> None:
    if args.aa:
        name, depth = _parse_side(args.aa)
        recs = _records(name, depth)
        pairs = [((t, 1), (t, 3)) for t, r in recs if r == 1] + [((t, 2), (t, 4)) for t, r in recs if r == 2]
        side_a, side_b = recs, recs
        labels = {"a": f"{name}:{depth}:r1,r2", "b": f"{name}:{depth}:r3,r4"}
    else:
        side_a, side_b = _records(*_parse_side(args.a)), _records(*_parse_side(args.b))
        pairs = [(key, key) for key in sorted(set(side_a) & set(side_b))]
        labels = {"a": args.a, "b": args.b}
    out = RESULTS / args.out
    (out / "packs").mkdir(parents=True, exist_ok=True)
    (out / "verdicts").mkdir(exist_ok=True)
    mapping: dict[str, Any] = {"sides": labels, "packs": {}}
    for key_a, key_b in sorted(pairs):
        if key_a not in side_a or key_b not in side_b:
            continue
        rec_a, rec_b = side_a[key_a], side_b[key_b]
        comparison = f"{key_a[0]}__r{key_a[1]}"
        for order, first, second in (("AB", rec_a, rec_b), ("BA", rec_b, rec_a)):
            pack_id = f"{comparison}__{order}"
            (out / "packs" / f"{pack_id}.md").write_text(
                _pack(rec_a["topic"], key_a[0], first, second), encoding="utf-8"
            )
            mapping["packs"][pack_id] = {"comparison": comparison, "order": order}
        mapping.setdefault("comparisons", {})[comparison] = {
            "a": metrics.record_metrics(rec_a),
            "b": metrics.record_metrics(rec_b),
        }
    (out / "_pairs.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"{len(mapping['packs'])} packs for {len(mapping.get('comparisons', {}))} comparisons in {out}")


def _side_of(winner: str, order: str) -> str:
    if winner not in ("1", "2"):
        return "tie"
    return {"AB": {"1": "a", "2": "b"}, "BA": {"1": "b", "2": "a"}}[order][winner]


def _read_verdict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def decide(votes: list[str]) -> str:
    """One comparison: a side wins only when both orders picked it."""
    return votes[0] if len(votes) == 2 and votes[0] == votes[1] and votes[0] != "tie" else "tie"


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def score(args: argparse.Namespace) -> None:
    cmp_dir = RESULTS / args.cmp
    mapping = json.loads((cmp_dir / "_pairs.json").read_text(encoding="utf-8"))
    votes: dict[str, list[str]] = {}
    missing = []
    for pack_id, info in mapping["packs"].items():
        verdict = _read_verdict(cmp_dir / "verdicts" / f"{pack_id}.json")
        if verdict is None:
            missing.append(pack_id)
            continue
        votes.setdefault(info["comparison"], []).append(_side_of(str(verdict.get("winner")), info["order"]))
    outcomes = {c: decide(v) for c, v in votes.items()}
    wins_b = sum(1 for o in outcomes.values() if o == "b")
    wins_a = sum(1 for o in outcomes.values() if o == "a")

    comps = mapping.get("comparisons", {}).values()
    side_metrics: dict[str, dict[str, float]] = {}
    for side in ("a", "b"):
        ok = [c[side] for c in comps if c[side].get("ok")]
        side_metrics[side] = {
            "citation_support_rate": _mean([m["citation_support_rate"] for m in ok]),
            "unverified_numbers": _mean([m["unverified_numbers"] for m in ok]),
            "cost_offpeak_usd": _mean([c[side]["cost_offpeak_usd"] for c in comps]),
            "wall_s": _mean([c[side]["wall_s"] or 0.0 for c in comps]),
            "failed": float(sum(1 for c in comps if not c[side].get("ok"))),
        }
    delta = {k: round(side_metrics["b"][k] - side_metrics["a"][k], 4) for k in side_metrics["a"]}
    tolerance = {"citation_support_rate": 0.0, "unverified_numbers": 0.0}
    if args.tolerance:
        tolerance.update(json.loads(Path(args.tolerance).read_text(encoding="utf-8")))
    not_worse = (
        delta["citation_support_rate"] >= -tolerance["citation_support_rate"]
        and delta["unverified_numbers"] <= tolerance["unverified_numbers"]
    )
    cost_ratio = (
        round(side_metrics["b"]["cost_offpeak_usd"] / side_metrics["a"]["cost_offpeak_usd"], 2)
        if side_metrics["a"]["cost_offpeak_usd"]
        else None
    )
    summary = {
        "sides": mapping["sides"],
        "comparisons": len(outcomes),
        "missing_verdicts": missing,
        "wins_b": wins_b,
        "wins_a": wins_a,
        "ties": len(outcomes) - wins_a - wins_b,
        "passes_win_rule": wins_b >= WINS_TO_PASS,
        "metrics": side_metrics,
        "delta_b_minus_a": delta,
        "tolerance": tolerance,
        "not_worse_on_faithfulness": not_worse,
        "cost_ratio_b_over_a": cost_ratio,
        "cost_needs_maintainer_decision": cost_ratio is not None and cost_ratio > COST_RATIO_LIMIT,
        "outcomes": outcomes,
    }
    if args.write_tolerance:
        # An A/A comparison: the spread between identical variants becomes the tolerance of later ones.
        (cmp_dir / "_tolerance.json").write_text(
            json.dumps(
                {
                    "citation_support_rate": abs(delta["citation_support_rate"]),
                    "unverified_numbers": abs(delta["unverified_numbers"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    (cmp_dir / "_score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json.dump({k: v for k, v in summary.items() if k != "outcomes"}, sys.stdout, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--a")
    b.add_argument("--b")
    b.add_argument("--aa")
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("cmp")
    s.add_argument("--tolerance")
    s.add_argument("--write-tolerance", action="store_true")
    args = parser.parse_args()
    if args.cmd == "build":
        if not args.aa and not (args.a and args.b):
            parser.error("build needs --a and --b, or --aa")
        build(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
