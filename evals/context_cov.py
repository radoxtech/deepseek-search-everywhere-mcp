"""Context coverage: how many key facts reach the synthesis model's context, per selection method?

Works offline on the pages a query-lab run saved (``evals/query_lab.py run``). For every topic it builds
the synthesis context with each selection method at the same character budget, writes one blind judge
pack per context (the pack name does not reveal the method), and after the verdicts scores the share of
core and all key points the context states::

    python evals/context_cov.py packs --name qlab --style aspects
    # a judge writes <topic>/ctx_verdicts/<pack>.json for every <topic>/ctx_packs/<pack>.md
    python evals/context_cov.py score --name qlab

Methods: ``pages`` (current: round-robin page ranking, top ``k`` whole pages) and ``passages``
(passage windows chosen per aspect with BM25 from every candidate page, see ``passages.py``).
The budget is the character size of the ``pages`` context, so both contexts are the same size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import judge  # noqa: E402
import query_lab  # noqa: E402
from deepseek_search_everywhere_mcp.passages import select_passages  # noqa: E402

from deepseek_search_everywhere_mcp import deep_research as dr  # noqa: E402

METHODS = ("pages", "passages", "passages_num", "hybrid", "hybrid_num")
VARIANTS = {
    "passages": {},
    "passages_num": {"number_lane": True},
    "hybrid": {"whole_pages": 3},
    "hybrid_num": {"whole_pages": 3, "number_lane": True},
}
UNITS_PER_PACK = 14
CONTEXT_INSTRUCTIONS = """You check a research context for an evaluation. Below are a topic, its key points,
and the source text a writer will receive. For each key point decide whether the source text itself states its
substance (its specific fact, and its figures where the key point has figures, at the precision the point
gives). Mentioning the subject, or a different figure, does not count. Judge only from the text shown; do
not use the web or your own knowledge.

Write only this JSON object: {"points": {"k1": true, "k2": false, ...}} with every key point id.
"""


def build_contexts(data: dict[str, Any], topic: str, k: int) -> dict[str, list[dict[str, Any]]]:
    by_query: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for r in data["results"]:
        by_query.setdefault(r["query"], []).append(dict(r))
    lists = list(by_query.values())
    pages, _ = dr.rank_and_diversify_sources(lists, k)
    budget = sum(len(p.get("content") or "") for p in pages)
    pool, _ = dr.rank_and_diversify_sources(lists, len(data["results"]))
    out = {"pages": pages}
    for method, options in VARIANTS.items():
        out[method] = select_passages(pool, data["queries"], topic, budget, **options)
    return out


def _context_text(entries: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"## Source {n}\nTitle: {e.get('title') or ''}\nURL: {e.get('url') or ''}\n\n"
        f"{e.get('content') or ''}\n"
        for n, e in enumerate(entries, 1)
    )


def _pack_name(topic_id: str, method: str) -> str:
    return "c" + hashlib.sha256(f"{topic_id}:{method}".encode()).hexdigest()[:8]


def packs(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = topic_dir / f"{args.style}.json"
        if not f.exists():
            continue
        rubric = query_lab._rubric(topic_dir.name)
        contexts = build_contexts(json.loads(f.read_text()), rubric["topic"], args.k)
        out = topic_dir / "ctx_packs"
        out.mkdir(exist_ok=True)
        stats = {}
        for method, entries in contexts.items():
            text = _context_text(entries)
            body = [
                CONTEXT_INSTRUCTIONS,
                f"\n# Topic\n{rubric['topic']}\n",
                f"# Key points\n{judge._rubric_text(topic_dir.name)}\n",
                f"# Source text\n{text}",
            ]
            (out / f"{_pack_name(topic_dir.name, method)}.md").write_text("\n".join(body), "utf-8")
            stats[method] = {
                "sources": len(entries),
                "chars": sum(len(e.get("content") or "") for e in entries),
            }
        print(topic_dir.name, json.dumps(stats))


def _unit_id(entry: dict[str, Any]) -> str:
    return "u" + hashlib.sha256(f"{entry.get('url')}\n{entry.get('content')}".encode()).hexdigest()[:9]


def _page_labels(topic_dir: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for v in sorted((topic_dir / "verdicts").glob("*.json")):
        labels.update((judge._read_verdict(v) or {}).get("pages", {}))
    return labels


def _known_units(
    topic_dir: Path, entries: list[dict[str, Any]], data: dict[str, Any]
) -> dict[str, list[str]]:
    """Labels that need no judge: a whole page keeps its page label; a part of an empty page is empty."""
    full = {r["url"]: r.get("content") or "" for r in data["results"]}
    labels = _page_labels(topic_dir)
    known: dict[str, list[str]] = {}
    for e in entries:
        page = labels.get(query_lab._page_id(e["url"]))
        if page is None:
            continue
        if not page or e.get("content") == full.get(e["url"]):
            known[_unit_id(e)] = page
    return known


def unit_packs(args: argparse.Namespace) -> None:
    """Judge packs of context entries (pages or their windows), like the query-lab page packs."""
    root = HERE / "results" / args.name
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        f = topic_dir / f"{args.style}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        rubric = query_lab._rubric(topic_dir.name)
        contexts = build_contexts(data, rubric["topic"], args.k)
        units: dict[str, dict[str, Any]] = {}
        for entries in contexts.values():
            for e in entries:
                units.setdefault(_unit_id(e), e)
        known = _known_units(topic_dir, list(units.values()), data)
        done: set[str] = set()
        for v in (topic_dir / "unit_verdicts").glob("*.json"):
            done |= set((judge._read_verdict(v) or {}).get("pages", {}))
        todo = sorted(u for u in units if u not in known and u not in done)
        out = topic_dir / "unit_packs"
        out.mkdir(exist_ok=True)
        start = len(list(out.glob("*.md")))
        for i in range(0, len(todo), UNITS_PER_PACK):
            body = [
                query_lab.PAGE_INSTRUCTIONS,
                f"\n# Topic\n{rubric['topic']}\n",
                f"# Key points\n{judge._rubric_text(topic_dir.name)}\n",
            ]
            for uid in todo[i : i + UNITS_PER_PACK]:
                e = units[uid]
                body.append(
                    f"## Page {uid}\nTitle: {e.get('title') or ''}\nURL: {e['url']}\n\n{e['content']}\n"
                )
            (out / f"u{start + i // UNITS_PER_PACK:02d}.md").write_text("\n".join(body), "utf-8")
        print(topic_dir.name, len(units), "units", len(known), "known", len(todo), "to judge")


def unit_score(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    table: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in METHODS}
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = json.loads((topic_dir / f"{args.style}.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        core = {k["id"] for k in rubric["key_points"] if k["importance"] == "core"}
        every = {k["id"] for k in rubric["key_points"]}
        contexts = build_contexts(data, rubric["topic"], args.k)
        labels = _known_units(topic_dir, [e for es in contexts.values() for e in es], data)
        for v in (topic_dir / "unit_verdicts").glob("*.json"):
            labels.update((judge._read_verdict(v) or {}).get("pages", {}))
        for method, entries in contexts.items():
            missing = [e for e in entries if _unit_id(e) not in labels]
            if missing:
                print(topic_dir.name, method, len(missing), "unjudged units")
            got = set().union(*[set(labels.get(_unit_id(e), [])) for e in entries] or [set()])
            table[method][topic_dir.name] = (len(got & core) / len(core), len(got & every) / len(every))
    for method, rows in table.items():
        core_mean = statistics.fmean(v[0] for v in rows.values())
        all_mean = statistics.fmean(v[1] for v in rows.values())
        per = " ".join(f"{t[:5]}={v[0]:.2f}" for t, v in rows.items())
        print(f"{method:13} core={core_mean:.3f} all={all_mean:.3f}  {per}")


def score(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    table: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rubric = query_lab._rubric(topic_dir.name)
        core = {k["id"] for k in rubric["key_points"] if k["importance"] == "core"}
        every = {k["id"] for k in rubric["key_points"]}
        for method in ("pages", "passages"):
            verdict = judge._read_verdict(
                topic_dir / "ctx_verdicts" / f"{_pack_name(topic_dir.name, method)}.json"
            )
            if not verdict:
                continue
            got = {k for k, v in verdict.get("points", {}).items() if v is True}
            table[method][topic_dir.name] = (len(got & core) / len(core), len(got & every) / len(every))
    for method, rows in table.items():
        if rows:
            core_mean = statistics.fmean(v[0] for v in rows.values())
            all_mean = statistics.fmean(v[1] for v in rows.values())
            per = " ".join(f"{t.split('-')[0]}={v[0]:.2f}" for t, v in rows.items())
            print(f"{method:9} core={core_mean:.3f} all={all_mean:.3f}  {per}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packs")
    p.add_argument("--name", required=True)
    p.add_argument("--style", default="aspects")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(func=packs)
    for name, func in (("unit-packs", unit_packs), ("unit-score", unit_score)):
        u = sub.add_parser(name)
        u.add_argument("--name", required=True)
        u.add_argument("--style", default="aspects")
        u.add_argument("--k", type=int, default=10)
        u.set_defaults(func=func)
    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
