"""Key-point funnel: at which stage of a research job does each key fact get lost?

For every rubric key point of every topic it follows the fact through the stages of a frozen run made by
``evals/full_context.py`` and judged by ``evals/keypoints.py``::

    found     some page the searches returned states it (full page, up to the per-page cap)
    selected  one of the top-ranked pages handed to synthesis states it
    context   the synthesis context (after per-page and total caps) states it
    report    the report states it (share of repeats)
    supported the report states it in a sentence whose cited passages back it

Page labels come from the query-lab verdicts (first ``query_lab.CONTENT_CHARS`` of each page) plus
labels for the rest of each fetched page::

    python evals/funnel.py packs --name qlab          # packs for the page remainders
    # a judge writes <topic>/rest_verdicts/<pack>.json for every <topic>/rest_packs/<pack>.md
    python evals/funnel.py score --name qlab --kp kp-fullctx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import full_context  # noqa: E402
import judge  # noqa: E402
import query_lab  # noqa: E402

from deepseek_search_everywhere_mcp import deep_research as dr  # noqa: E402

REST_CHARS = 56_000  # characters of page text per remainder pack
STAGES = ("found", "selected", "context", "report", "supported")


def _cap() -> int:
    return dr.DEPTH_CONFIG["medium"].per_page_char_cap


def _rest(page: dict[str, Any]) -> str:
    return (page.get("content") or "")[query_lab.CONTENT_CHARS : _cap()]


def packs(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    for topic_dir in sorted(p for p in root.iterdir() if (p / "full_pages.json").exists()):
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        out = topic_dir / "rest_packs"
        out.mkdir(exist_ok=True)
        chunks: list[list[dict[str, Any]]] = [[]]
        size = 0
        for page in pool:
            rest = _rest(page)
            if not rest.strip():
                continue
            if chunks[-1] and size + len(rest) > REST_CHARS:
                chunks.append([])
                size = 0
            chunks[-1].append(page)
            size += len(rest)
        for i, chunk in enumerate(c for c in chunks if c):
            body = [
                query_lab.PAGE_INSTRUCTIONS,
                f"\n# Topic\n{rubric['topic']}\n",
                f"# Key points\n{judge._rubric_text(topic_dir.name)}\n",
            ]
            for page in chunk:
                pid = query_lab._page_id(page["url"])
                body.append(
                    f"## Page {pid}\nTitle: {page.get('title') or ''}\nURL: {page['url']}\n\n…{_rest(page)}\n"
                )
            (out / f"r{i:02d}.md").write_text("\n".join(body), "utf-8")
        print(topic_dir.name, sum(1 for c in chunks if c), "packs")


def atom_packs(args: argparse.Namespace) -> None:
    """Packs of whole fetched pages (up to the per-page cap) to label with the atomic facts."""
    root = HERE / "results" / args.name
    for topic_dir in sorted(p for p in root.iterdir() if (p / "full_pages.json").exists()):
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        out = topic_dir / "atom_packs"
        out.mkdir(exist_ok=True)
        chunks: list[list[dict[str, Any]]] = [[]]
        size = 0
        for page in pool:
            text = (page.get("content") or "")[: _cap()]
            if not text.strip():
                continue
            if chunks[-1] and size + len(text) > REST_CHARS:
                chunks.append([])
                size = 0
            chunks[-1].append(page)
            size += len(text)
        for i, chunk in enumerate(c for c in chunks if c):
            body = [
                query_lab.PAGE_INSTRUCTIONS,
                f"\n# Topic\n{rubric['topic']}\n",
                f"# Key points\n{judge._rubric_text(topic_dir.name, use_atoms=True)}\n",
            ]
            for page in chunk:
                pid = query_lab._page_id(page["url"])
                text = (page.get("content") or "")[: _cap()]
                body.append(
                    f"## Page {pid}\nTitle: {page.get('title') or ''}\nURL: {page['url']}\n\n{text}\n"
                )
            (out / f"a{i:02d}.md").write_text("\n".join(body), "utf-8")
        print(topic_dir.name, sum(1 for c in chunks if c), "packs")


def _labels(topic_dir: Path, subs: tuple[str, ...] = ("verdicts", "rest_verdicts")) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for sub in subs:
        for v in sorted((topic_dir / sub).glob("*.json")):
            for pid, ks in ((judge._read_verdict(v) or {}).get("pages") or {}).items():
                labels.setdefault(pid, set()).update(ks)
    return labels


def _report_verdicts(kp: str, side: str) -> dict[str, list[dict[str, Any]]]:
    root = HERE / "results" / kp
    meta = json.loads((root / "_packs.json").read_text())["packs"]
    out: dict[str, list[dict[str, Any]]] = {}
    for pack_id, info in meta.items():
        if not info["side"].startswith(side):
            continue
        verdict = judge._read_verdict(root / "verdicts" / f"{pack_id}.json") or {}
        out.setdefault(info["topic_id"], []).append(verdict.get("points") or {})
    return out


def _write_json(
    args: argparse.Namespace,
    mode: str,
    table: dict[str, list[float]],
    topics: list[str],
    losses: list[str],
    detail: list[dict[str, str]],
    atoms_total: int,
) -> None:
    """Write the funnel numbers for the dashboard (average, per-topic and the loss list)."""
    if not args.json_out:
        return
    payload = {
        "mode": mode,
        "context": args.context,
        "points": "all" if args.all else "core",
        "stages": list(STAGES),
        "atoms_total": atoms_total,
        "average": {s: round(statistics.fmean(table[s]), 4) for s in STAGES},
        "by_topic": {t: {s: round(table[s][i], 4) for s in STAGES} for i, t in enumerate(topics)},
        "losses": losses,
        "loss_detail": detail,
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.json_out}")


def score(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    side = f"fullctx-{args.context}"
    reports = _report_verdicts(args.kp, side)
    if args.atoms:
        _score_atoms(args, root, reports)
        return
    table: dict[str, list[float]] = {s: [] for s in STAGES}
    losses: list[str] = []
    detail: list[dict[str, str]] = []
    points_total = 0
    topics: list[str] = []
    for topic_dir in sorted(p for p in root.iterdir() if (p / "full_pages.json").exists()):
        topics.append(topic_dir.name)
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        labels = _labels(topic_dir)
        context = full_context.build_contexts(pool, rubric["topic"])[args.context]
        rank = {query_lab._page_id(p["url"]): n for n, p in enumerate(pool, 1)}
        selected = pool if args.context == "all" else pool[: dr.DEPTH_CONFIG["medium"].k_sources]

        def has(pages: list[dict[str, Any]], kid: str, labels: dict[str, set[str]] = labels) -> bool:
            return any(kid in labels.get(query_lab._page_id(p["url"]), set()) for p in pages)

        def where(kid: str, labels: dict[str, set[str]] = labels, rank: dict[str, int] = rank) -> str:
            return ",".join(str(rank[pid]) for pid, ks in labels.items() if kid in ks and pid in rank)

        verdicts = reports.get(topic_dir.name, [])
        core = [k for k in rubric["key_points"] if args.all or k["importance"] == "core"]
        points_total += len(core)
        per = {s: 0.0 for s in STAGES}
        for k in core:
            kid = k["id"]
            found, sel, ctx = has(pool, kid), has(selected, kid), has(context, kid)
            rep = (
                statistics.fmean(bool((v.get(kid) or {}).get("covered")) for v in verdicts)
                if verdicts
                else 0.0
            )
            sup = (
                statistics.fmean(
                    bool((v.get(kid) or {}).get("covered") and (v.get(kid) or {}).get("supported"))
                    for v in verdicts
                )
                if verdicts
                else 0.0
            )
            for stage, value in zip(STAGES, (found, sel, ctx, rep, sup), strict=True):
                per[stage] += float(value)
            stage_lost = (
                "not found"
                if not found
                else "not selected"
                if not sel
                else "cut from context"
                if not ctx
                else "not written"
                if rep == 0
                else "partly written"
                if rep < 1
                else "unsupported"
                if sup < rep
                else ""
            )
            if stage_lost:
                losses.append(
                    f"{topic_dir.name:34} {kid:4} {stage_lost:17} pages={where(kid) or '-':14} "
                    f"{k['point'][:70]}"
                )
                detail.append(
                    {"topic": topic_dir.name, "point": kid, "stage": stage_lost, "fact": k["point"]}
                )
        for stage in STAGES:
            table[stage].append(per[stage] / max(1, len(core)))
    print(f"context={args.context}  points={'all' if args.all else 'core'}")
    for stage in STAGES:
        print(
            f"  {stage:10} {statistics.fmean(table[stage]):.3f}  "
            + " ".join(f"{v:.2f}" for v in table[stage])
        )
    print("\n".join(losses))
    _write_json(args, "keypoints", table, topics, losses, detail, points_total)


DIAGNOSE_INSTRUCTIONS = """You diagnose why research reports missed key facts that their writer had in its
sources. For each case below you get a key point, excerpts of the source pages in the writer's context
that state it, and the search queries the job planned. The reports follow at the end (one per
repeat). For every case and every report pick one cause:
  a  omitted: the report says nothing about the subject of this point
  b  vague: the report touches the subject but without the point's specific fact or figures
  c  other figure: the report states it with a different figure, study or source than the key point
  d  partial: the report states part of the point but misses a distinct element of it
  e  stated: the report does state the point (the earlier check missed it) or the point is debatable
Also answer "planned": does any search query target the subject of this point (true/false), and give a
short "note" (at most 20 words) on what the report says instead, if anything.

Write only this JSON object:
{"cases": {"k5": {"r1": "a", "r2": "b", "planned": true, "note": "..."}, ...}}
"""
LOSS_STAGES = ("not written", "partly written", "unsupported")


def _cases(args: argparse.Namespace) -> dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    """Per topic: the key points that were in the context but not (fully) in the reports, with their pages."""
    root = HERE / "results" / args.name
    side = f"fullctx-{args.context}"
    reports = _report_verdicts(args.kp, side)
    out: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    for topic_dir in sorted(p for p in root.iterdir() if (p / "full_pages.json").exists()):
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        labels = _labels(topic_dir)
        context = full_context.build_contexts(pool, rubric["topic"])[args.context]
        verdicts = reports.get(topic_dir.name, [])
        for k in rubric["key_points"]:
            if not (args.all or k["importance"] == "core"):
                continue
            pages = [p for p in context if k["id"] in labels.get(query_lab._page_id(p["url"]), set())]
            if not pages or not verdicts:
                continue
            covered = [bool((v.get(k["id"]) or {}).get("covered")) for v in verdicts]
            supported = [bool((v.get(k["id"]) or {}).get("supported")) for v in verdicts]
            if all(covered) and all(supported):
                continue
            out.setdefault(topic_dir.name, []).append((k, pages))
    return out


def diagnose_packs(args: argparse.Namespace) -> None:
    out = HERE / "results" / args.kp / "diagnose"
    out.mkdir(exist_ok=True)
    side = HERE / "results" / f"fullctx-{args.context}"
    for topic_id, cases in _cases(args).items():
        records = [json.loads(f.read_text()) for f in sorted(side.glob(f"{topic_id}__*.json"))]
        body = [
            DIAGNOSE_INSTRUCTIONS,
            f"\n# Topic\n{query_lab._rubric(topic_id)['topic']}\n",
            "# Search queries",
        ]
        for rec in records:
            body.append(
                f"- r{rec['repeat']}: " + " | ".join((rec.get("envelope") or {}).get("subqueries") or [])
            )
        body.append("\n# Cases")
        for k, pages in cases:
            body.append(f"\n## Case {k['id']}\nKey point: {k['point']}\n")
            for page in pages[:2]:
                excerpt = judge._excerpt(k["point"], page.get("content") or "", windows=3)
                body.append(f"Source ({page.get('title') or page['url']}): {excerpt}\n")
        for rec in records:
            report = (rec.get("dump") or {}).get("report") or (rec.get("envelope") or {}).get("report") or ""
            body.append(f"\n# Report r{rec['repeat']}\n{report}\n")
        (out / f"{args.context}__{topic_id}.md").write_text("\n".join(body), "utf-8")
        print(topic_id, len(cases), "cases")


def diagnose_score(args: argparse.Namespace) -> None:
    from collections import Counter

    root = HERE / "results" / args.kp / "diagnose"
    causes: Counter[str] = Counter()
    planned: Counter[bool] = Counter()
    for v in sorted(root.glob(f"{args.context}__*.json")):
        topic = v.stem.split("__", 1)[1]
        for kid, case in ((judge._read_verdict(v) or {}).get("cases") or {}).items():
            per = [case[r] for r in sorted(case) if r.startswith("r") and len(r) <= 3]
            causes.update(per)
            planned[bool(case.get("planned"))] += 1
            print(
                f"{topic:34} {kid:4} {' '.join(per):6} planned={case.get('planned')!s:5} "
                f"{case.get('note', '')}"
            )
    total = sum(causes.values()) or 1
    names = {"a": "omitted", "b": "vague", "c": "other figure", "d": "partial", "e": "stated / debatable"}
    for cause, n in causes.most_common():
        print(f"  {cause} {names.get(cause, cause):20} {n:3}  {n / total:.0%}")
    print("  planned:", dict(planned))


def _score_atoms(args: argparse.Namespace, root: Path, reports: dict[str, list[dict[str, Any]]]) -> None:
    """The funnel at atom level: each key point counts by the share of its atoms past each stage."""
    table: dict[str, list[float]] = {s: [] for s in STAGES}
    lines: list[str] = []
    detail: list[dict[str, str]] = []
    atoms_total = 0
    topics: list[str] = []
    for topic_dir in sorted(p for p in root.iterdir() if (p / "full_pages.json").exists()):
        topics.append(topic_dir.name)
        pool = json.loads((topic_dir / "full_pages.json").read_text())
        rubric = query_lab._rubric(topic_dir.name)
        labels = _labels(topic_dir, ("atom_verdicts",))
        context = full_context.build_contexts(pool, rubric["topic"])[args.context]
        selected = pool if args.context == "all" else pool[: dr.DEPTH_CONFIG["medium"].k_sources]
        verdicts = reports.get(topic_dir.name, [])
        ids = {query_lab._page_id(p["url"]) for p in pool}
        sel_ids = {query_lab._page_id(p["url"]) for p in selected}
        ctx_ids = {query_lab._page_id(p["url"]) for p in context}
        points: dict[str, list[dict[str, Any]]] = {}
        for a in judge.atoms(rubric):
            if args.all or a["importance"] == "core":
                points.setdefault(a["point"], []).append(a)
        per = {s: [] for s in STAGES}
        atoms_total += sum(len(v) for v in points.values())
        for atoms_ in points.values():
            shares = {s: [] for s in STAGES}
            for a in atoms_:
                on = {p for p, ks in labels.items() if a["id"] in ks}
                rep = (
                    statistics.fmean(bool((v.get(a["id"]) or {}).get("covered")) for v in verdicts)
                    if verdicts
                    else 0.0
                )
                sup = (
                    statistics.fmean(
                        bool(
                            (v.get(a["id"]) or {}).get("covered") and (v.get(a["id"]) or {}).get("supported")
                        )
                        for v in verdicts
                    )
                    if verdicts
                    else 0.0
                )
                vals = (bool(on & ids), bool(on & sel_ids), bool(on & ctx_ids), rep, sup)
                for stage, value in zip(STAGES, vals, strict=True):
                    shares[stage].append(float(value))
                lost = (
                    "not found"
                    if not vals[0]
                    else "not selected"
                    if not vals[1]
                    else "cut"
                    if not vals[2]
                    else "not written"
                    if rep == 0
                    else "partly written"
                    if rep < 1
                    else "unsupported"
                    if sup < rep
                    else ""
                )
                if lost:
                    lines.append(f"{topic_dir.name:34} {a['id']:5} {lost:15} {a['fact'][:80]}")
                    detail.append(
                        {"topic": topic_dir.name, "point": a["id"], "stage": lost, "fact": a["fact"]}
                    )
            for stage in STAGES:
                per[stage].append(statistics.fmean(shares[stage]))
        for stage in STAGES:
            table[stage].append(statistics.fmean(per[stage]))
    print(f"atoms  context={args.context}  points={'all' if args.all else 'core'}")
    for stage in STAGES:
        print(
            f"  {stage:10} {statistics.fmean(table[stage]):.3f}  "
            + " ".join(f"{v:.2f}" for v in table[stage])
        )
    if args.verbose:
        print("\n".join(lines))
    _write_json(args, "atoms", table, topics, lines, detail, atoms_total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packs")
    p.add_argument("--name", required=True)
    p.set_defaults(func=packs)
    ap = sub.add_parser("atom-packs")
    ap.add_argument("--name", required=True)
    ap.set_defaults(func=atom_packs)
    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.add_argument("--kp", required=True)
    s.add_argument("--context", default="top10", choices=full_context.CONTEXTS)
    s.add_argument("--all", action="store_true", help="all key points, not only core")
    s.add_argument(
        "--atoms", action="store_true", help="atom-level funnel (atom_verdicts, an --atoms kp run)"
    )
    s.add_argument("--verbose", action="store_true")
    s.add_argument("--json-out", help="write the stage numbers and the loss list to this JSON file")
    s.set_defaults(func=score)
    for name, func in (("diagnose-packs", diagnose_packs), ("diagnose-score", diagnose_score)):
        d = sub.add_parser(name)
        d.add_argument("--name", default="qlab")
        d.add_argument("--kp", required=True)
        d.add_argument("--context", default="top10", choices=full_context.CONTEXTS)
        d.add_argument("--all", action="store_true")
        d.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
