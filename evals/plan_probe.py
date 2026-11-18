"""Cheap probe of query planning: decomposition plus one search per query, no synthesis.

For many topics it measures what the planned queries bring back, with no judge::

    python evals/plan_probe.py --name probe-new                      # current prompt
    python evals/plan_probe.py --name probe-old --variant old-decompose

Per topic: the queries, how much they overlap each other and the topic, and for the results the
share of primary source types, of pages that look like reference lists or citing pages, of pages in
another language, of repeated hosts, and the mean upstream rerank score. Cost per topic is one
decomposition call and one search per query (about $0.003 at medium).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import variants  # noqa: E402

# A page that mostly lists or cites studies: many "et al." or DOIs per thousand characters, or a
# letter/correspondence or reference-list title.
_REFERENCE_TITLE_RE = re.compile(
    r"(?i)\b(?:references|bibliography|correspondence|letter to the editor|citations?)\b"
)


def looks_bibliographic(title: str, url: str, content: str) -> bool:
    text = content[:6000]
    per_k = 1000 / max(1, len(text))
    density = (
        len(re.findall(r"(?i)\bet al\b", text)) + len(re.findall(r"(?i)\bdoi\b|10\.\d{4,9}/", text))
    ) * per_k
    return density > 4 or bool(_REFERENCE_TITLE_RE.search(title)) or "/NEJMc" in url


async def probe_topic(topic: str, depth: str) -> dict[str, Any]:
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import research_jobs, runtime, search_core
    from deepseek_search_everywhere_mcp.filters import MAX_QUERY

    cfg = dr.DEPTH_CONFIG[depth]
    res = await runtime._chat_request(
        messages=research_jobs.build_decomposition_messages(
            topic, max_queries=cfg.subqueries, sources="any", lang=None
        ),
        thinking={"type": "disabled"},
        model=runtime.REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=20.0,
        deadline=20.0,
        attempts=2,
    )
    queries, aspects = dr.parse_research_plan(res.get("content") or "", topic, cfg.subqueries)
    searches = await asyncio.gather(
        *[
            search_core._search(
                q,
                cfg.count,
                topic[:MAX_QUERY],
                "full",
                None,
                cfg.per_page_char_cap,
                cfg.per_page_char_cap * cfg.count,
                True,
                False,
                refresh=True,
            )
            for q in queries
        ]
    )
    results = [r for s in searches if s.get("ok") for r in s.get("results", [])]
    hosts = [(r.get("url") or "").split("/")[2] if "//" in (r.get("url") or "") else "" for r in results]
    words = [dr._query_words(q) for q in queries]
    n = max(1, len(results))
    return {
        "topic": topic,
        "queries": queries,
        "aspect_kinds": [a.kind for a in aspects],
        "decompose_cost": res.get("cost_usd", 0.0),
        "query_overlap": max((dr._overlap(a, b) for a, b in combinations(words, 2)), default=0.0),
        "topic_overlap": max((dr._overlap(w, dr._query_words(topic)) for w in words), default=0.0),
        "results": len(results),
        "primary": sum(dr._is_primary(r) for r in results) / n,
        "bibliographic": sum(
            looks_bibliographic(r.get("title") or "", r.get("url") or "", r.get("content") or "")
            for r in results
        )
        / n,
        "foreign": sum((dr.text_language(r.get("content") or "") or "en") != "en" for r in results) / n,
        "repeated_host": 1 - len(set(hosts)) / n,
        "rerank": statistics.fmean([float(r.get("rerank_score") or 0) for r in results]) if results else 0.0,
    }


async def main_async(args: argparse.Namespace) -> None:
    variants.apply(args.variant)
    topics = json.loads((HERE / "probe_topics.json").read_text(encoding="utf-8"))["topics"][: args.limit]
    out = HERE / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(p.read_text())["topic"] for p in out.glob("t*.json")}
    sem = asyncio.Semaphore(args.parallel)

    async def one(i: int, topic: str) -> None:
        if topic in done:
            return
        async with sem:
            row = await probe_topic(topic, args.depth)
        (out / f"t{i:02d}.json").write_text(json.dumps(row, indent=1), encoding="utf-8")

    await asyncio.wait_for(
        asyncio.gather(*[one(i, t) for i, t in enumerate(topics)]), timeout=args.time_budget
    )


def summarize(name: str) -> dict[str, Any]:
    rows = [json.loads(p.read_text()) for p in sorted((HERE / "results" / name).glob("t*.json"))]
    keys = (
        "query_overlap",
        "topic_overlap",
        "results",
        "primary",
        "bibliographic",
        "foreign",
        "repeated_host",
        "rerank",
    )
    return (
        {"topics": len(rows), **{k: round(statistics.fmean([r[k] for r in rows]), 3) for k in keys}}
        if rows
        else {}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--variant", default="baseline", choices=sorted(variants.VARIANTS))
    parser.add_argument("--depth", default="medium")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--time-budget", type=float, default=160.0)
    parser.add_argument("--summary", action="store_true", help="only print the summary of saved rows")
    args = parser.parse_args()
    if not args.summary:
        try:
            asyncio.run(main_async(args))
        except TimeoutError:
            print("time budget reached; run again to continue", file=sys.stderr)
    print(json.dumps(summarize(args.name), indent=1))


if __name__ == "__main__":
    main()
