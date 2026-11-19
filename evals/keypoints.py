"""Binary key-point judging: does each report state each rubric key point, and does its citation hold?

The main metric of the harness. One pack per report (no pairing, no order bias); pack names are
random so the judge cannot tell variants apart, and ``_packs.json`` maps them back. Build::

    python evals/keypoints.py build --side pilot:medium --side new-medium:medium --out kp-new-medium

The judge (a Claude subagent) reads ``packs/<id>.md`` and writes ``verdicts/<id>.json``::

    {"points": {"k1": {"covered": true, "sentence": 3, "supported": true}, ...},
     "unsupported": [sentence numbers whose cited excerpt does not back them]}

Score (per side: core and all coverage, coverage with a supporting citation, unsupported sentences;
then the per-topic difference of every side against the first)::

    python evals/keypoints.py score kp-new-medium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
import metrics  # noqa: E402

RESULTS = HERE / "results"

KEYPOINT_INSTRUCTIONS = """You are a strict, impartial fact checker. You get one research report, the key
points a good report on its topic must state, and for every numbered cited sentence the passages of the
sources it cites. You have no web access and need none: judge only from this pack.

For each key point decide "covered": true only when the report itself states the substance of the point:
its specific fact, and its figures where it has figures (same value at the precision the point gives,
same population, period or condition). A vaguer statement ("weight loss was large") or a different figure
is not covered. A report may word it differently or cover it across two adjacent sentences.
When covered, give "sentence": the number of the cited sentence that states it (S-numbers in the evidence),
or null when only an uncited sentence states it, and "supported": whether the cited passages back that
sentence's fact and figures (null when sentence is null).
Then list in "unsupported" the numbers of all cited sentences whose cited passages do not back them.
Judge every point independently; length and style do not matter.

Write only this JSON object as your answer:
{"points": {"k1": {"covered": true, "sentence": 3, "supported": true}, "k2": {"covered": false,
 "sentence": null, "supported": null}}, "unsupported": [5, 9]}
"""


def _rubric(topic_id: str) -> dict[str, Any]:
    return json.loads((judge.RUBRICS / f"{topic_id}.json").read_text(encoding="utf-8"))


def _pack(record: dict[str, Any], use_atoms: bool = False) -> str:
    env = record.get("envelope") or {}
    report = env.get("report") if env.get("ok") and env.get("status") == "done" else None
    return "\n".join(
        [
            KEYPOINT_INSTRUCTIONS,
            f"\n# Topic\n{record['topic']}\n",
            f"# Key points\n{judge._rubric_text(record['topic_id'], use_atoms=use_atoms)}\n",
            f"# Report\n{report or '(the job failed and returned no report)'}\n",
            "# Evidence (cited sentences and passages of the cited sources)\n",
            "\n".join(judge._evidence(record)) + "\n",
        ]
    )


def pack_id(side: str, topic_id: str, repeat: int) -> str:
    return hashlib.sha256(f"{side}|{topic_id}|{repeat}".encode()).hexdigest()[:12]


def build(args: argparse.Namespace) -> None:
    out = RESULTS / args.out
    (out / "packs").mkdir(parents=True, exist_ok=True)
    (out / "verdicts").mkdir(exist_ok=True)
    topics = set(args.topic or [])
    mapping: dict[str, Any] = {"sides": args.side, "atoms": args.atoms, "packs": {}}
    for side in args.side:
        for (topic_id, repeat), record in sorted(judge._records(*judge._parse_side(side)).items()):
            if topics and topic_id not in topics:
                continue
            if args.repeats and repeat > args.repeats:
                continue
            pid = pack_id(side, topic_id, repeat)
            (out / "packs" / f"{pid}.md").write_text(_pack(record, args.atoms), encoding="utf-8")
            mapping["packs"][pid] = {
                "side": side,
                "topic_id": topic_id,
                "repeat": repeat,
                "metrics": metrics.record_metrics(record),
            }
    (out / "_packs.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"{len(mapping['packs'])} packs in {out}")


def atom_scores(verdict: dict[str, Any], rubric: dict[str, Any]) -> dict[str, float]:
    """Atom-level scores: each key point counts by the share of its atoms the report states."""
    points = verdict.get("points") or {}
    by_point: dict[str, list[dict[str, Any]]] = {}
    for a in judge.atoms(rubric):
        by_point.setdefault(a["point"], []).append(a)

    def share(atoms_: list[dict[str, Any]], *, supported: bool = False) -> float:
        hits = [
            bool((points.get(a["id"]) or {}).get("covered"))
            and (not supported or (points.get(a["id"]) or {}).get("supported") is True)
            for a in atoms_
        ]
        return sum(hits) / len(hits)

    core = [k["id"] for k in rubric["key_points"] if k["importance"] == "core"]
    every = [k["id"] for k in rubric["key_points"]]

    def mean(ids: list[str], **kw: bool) -> float:
        return statistics.fmean(share(by_point[i], **kw) for i in ids) if ids else 0.0

    main = [next((a for a in by_point[i] if a.get("main")), by_point[i][0]) for i in core]
    return {
        "core": mean(core),
        "all": mean(every),
        "core_supported": mean(core, supported=True),
        "unsupported": float(len(verdict.get("unsupported") or [])),
        "core_main": statistics.fmean(share([a]) for a in main) if main else 0.0,
        "core_full": statistics.fmean(float(share(by_point[i]) == 1.0) for i in core) if core else 0.0,
    }


def verdict_scores(verdict: dict[str, Any], rubric: dict[str, Any]) -> dict[str, float]:
    """Coverage of core and all points, coverage backed by the citation, and unsupported sentences."""
    points = verdict.get("points") or {}
    core = [k["id"] for k in rubric["key_points"] if k["importance"] == "core"]
    every = [k["id"] for k in rubric["key_points"]]

    def covered(kid: str, *, supported: bool = False) -> bool:
        p = points.get(kid) or {}
        return bool(p.get("covered")) and (not supported or p.get("supported") is True)

    def rate(ids: list[str], **kw: bool) -> float:
        return sum(covered(k, **kw) for k in ids) / len(ids) if ids else 0.0

    return {
        "core": rate(core),
        "all": rate(every),
        "core_supported": rate(core, supported=True),
        "unsupported": float(len(verdict.get("unsupported") or [])),
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def score(args: argparse.Namespace) -> None:
    out = RESULTS / args.cmp
    mapping = json.loads((out / "_packs.json").read_text(encoding="utf-8"))
    per: dict[str, dict[str, list[dict[str, float]]]] = {}
    missing = []
    for pid, info in mapping["packs"].items():
        verdict = judge._read_verdict(out / "verdicts" / f"{pid}.json")
        if verdict is None:
            missing.append(pid)
            continue
        scorer = atom_scores if mapping.get("atoms") else verdict_scores
        scores = scorer(verdict, _rubric(info["topic_id"]))
        scores["cost_offpeak_usd"] = info["metrics"]["cost_offpeak_usd"]
        scores["wall_s"] = info["metrics"]["wall_s"] or 0.0
        scores["words"] = float(info["metrics"].get("words") or 0)
        per.setdefault(info["side"], {}).setdefault(info["topic_id"], []).append(scores)
    keys: tuple[str, ...] = (
        "core",
        "all",
        "core_supported",
        "unsupported",
        "cost_offpeak_usd",
        "wall_s",
        "words",
    )
    if mapping.get("atoms"):
        keys += ("core_main", "core_full")
    sides: dict[str, Any] = {}
    for side, topics in per.items():
        flat = [s for runs in topics.values() for s in runs]
        sides[side] = {
            "reports": len(flat),
            **{k: _mean([s[k] for s in flat]) for k in keys},
            "by_topic": {t: {k: _mean([s[k] for s in runs]) for k in keys[:4]} for t, runs in topics.items()},
        }
    reference = mapping["sides"][0]
    deltas = {}
    for side in mapping["sides"][1:]:
        if side not in sides or reference not in sides:
            continue
        common = set(sides[side]["by_topic"]) & set(sides[reference]["by_topic"])
        deltas[side] = {
            k: _mean([sides[side]["by_topic"][t][k] - sides[reference]["by_topic"][t][k] for t in common])
            for k in keys[:4]
        }
        # What one extra percentage point of coverage costs, in US dollars per report.
        delta_core = deltas[side]["core"] * 100
        delta_cost = sides[side]["cost_offpeak_usd"] - sides[reference]["cost_offpeak_usd"]
        deltas[side]["cost_per_point_usd"] = round(delta_cost / delta_core, 6) if delta_core > 0 else None
        deltas[side]["delta_cost_usd"] = round(delta_cost, 6)
    summary = {
        "reference": reference,
        "missing_verdicts": missing,
        "sides": sides,
        "delta_vs_reference": deltas,
    }
    (out / "_score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    json.dump(summary, sys.stdout, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--side", action="append", required=True, help="results dir and depth, e.g. pilot:medium")
    b.add_argument("--topic", action="append", help="limit to these topic ids")
    b.add_argument("--repeats", type=int, default=0, help="only repeats up to this number (0 = all)")
    b.add_argument("--out", required=True)
    b.add_argument("--atoms", action="store_true", help="judge the atomic facts of the key points")
    s = sub.add_parser("score")
    s.add_argument("cmp")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
