"""Query lab: which way of writing search queries brings back the key facts?

For each rubric topic and query style it plans queries, runs the searches a medium research job would
run, and saves the raw results. A judge then labels every unique page once (which key points it states),
so each style's fact yield is the union of key points over its results::

    python evals/query_lab.py run --name qlab --topics it-passkeys,medicine-glp1
    python evals/query_lab.py packs --name qlab          # one pack per topic slice of pages
    python evals/query_lab.py score --name qlab          # yield per style, after the verdicts

Styles: ``old`` (3.3 decomposition), ``aspects`` (current), ``questions`` (natural-language
questions), ``evidence`` (reviews, meta-analyses, official overviews), ``site`` (authoritative
domains with site:), ``topic`` (the topic itself as one query). Cost: one search per query (about
$0.00075) plus one tiny planning call per style.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
import variants  # noqa: E402

STYLE_PROMPTS = {
    "questions": (
        "Write {n} different questions that a careful researcher would type into a web search engine to "
        "answer the research topic. Each is a natural-language question of at most 120 characters, and "
        "together they cover different parts of the topic (results and numbers, how it works, limitations, "
        'current status). Return only a json object: {{"queries": ["...", "..."]}}'
    ),
    "evidence": (
        "Write {n} web search queries that find pages summarising the evidence on the research topic: "
        "systematic reviews, meta-analyses, official guidance or documentation, authoritative overviews "
        "and reports. Each query covers a different part of the topic and has 4 to 12 words. "
        'Return only a json object: {{"queries": ["...", "..."]}}'
    ),
    "site": (
        "Write {n} web search queries that go straight to the most authoritative sources for the research "
        "topic, each with the site: operator and the domain of the body that publishes them (a regulator, "
        "standards body, official documentation, a journal). Use only domains you are certain exist, and a "
        "different part of the topic in each query. "
        'Return only a json object: {{"queries": ["site:example.org ...", "..."]}}'
    ),
}
TOPIC_ANALYST_PROMPT = (
    "You analyse a research question before anyone searches. Name what is ambiguous in it, what traps a "
    "searcher falls into (paywalled results, outdated figures, several bodies sharing one name, vague "
    "definitions), what a wrong answer would look like, and which parts a good answer must cover. "
    'Return only json: {"ambiguities": [], "traps": [], "wrong_answers": [], "must_cover": []}'
)
PLAN_BUILDER_PROMPT = (
    "You plan the searches for a research question, given an analysis of it. Write {n} aspects. Each "
    "aspect needs: a short name, what the reader must learn from it, one web search query of 4 to 12 words, "
    "and the success criteria that would tell a later check that the aspect is covered (figures, dates, "
    "conditions, named mechanisms). Queries must be distinct and must not reuse the topic verbatim. "
    'Return only json: {"aspects": [{"aspect": "", "looking_for": "", "query": "", "success_criteria": []}]}'
)
QUERY_SCREENER_PROMPT = (
    "You review search queries before they are sent. For each query say whether it will return generic "
    "pages, repeated pages or the wrong kind of source, and if so give a better query of 4 to 12 words. "
    'Return only json: {"queries": ["..."]} with the final list, improved where needed.'
)
STYLES = ("old", "aspects", "questions", "evidence", "site", "topic", "planned", "seeded")
SEED_SNIPPETS = 6
SEED_CHARS = 500
SEED_PROMPT = (
    "Write {n} short web search queries that would show what the open web already says about this research "
    "question. They are reconnaissance, not the final search: broad enough to surface the main sources. "
    'Return only json: {{"queries": ["...", "..."]}}'
)
CONTENT_CHARS = 6000
PRIMARY = frozenset(
    {"official", "specification", "documentation", "guideline", "rct", "meta-analysis", "systematic_review"}
    | {"study", "preprint", "repository"}
)
PAGES_PER_PACK = 14


def _rubric(topic_id: str) -> dict[str, Any]:
    return json.loads((judge.RUBRICS / f"{topic_id}.json").read_text(encoding="utf-8"))


async def plan_queries(style: str, topic: str, n: int) -> list[str]:
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import research_jobs, runtime

    if style == "topic":
        return [topic[:128]]
    if style == "planned":
        return await _planned_queries(topic, n)
    if style == "seeded":
        return await _seeded_queries(topic, n)
    if style == "old":
        messages = [
            {"role": "system", "content": variants._OLD_DECOMPOSE_SYSTEM},
            {"role": "user", "content": f"Decompose this research topic into {n} search queries: {topic}"},
        ]
    elif style == "aspects":
        messages = research_jobs.build_decomposition_messages(topic, max_queries=n, sources="any", lang=None)
    else:
        messages = [
            {"role": "system", "content": STYLE_PROMPTS[style].format(n=n)},
            {"role": "user", "content": f"Research topic: {topic}\nReturn {n} queries as json."},
        ]
    res = await runtime._chat_request(
        messages=messages,
        thinking={"type": "disabled"},
        model=runtime.REWRITE_MODEL,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        timeout=20.0,
        deadline=20.0,
        attempts=2,
    )
    if style in ("old", "aspects"):
        return dr.parse_subqueries(res.get("content") or "", topic, n)
    parsed, _ = dr.sanitize_json(res.get("content") or "")
    raw = parsed.get("queries") if isinstance(parsed, dict) else None
    return [q.strip()[:128] for q in raw or [] if isinstance(q, str) and q.strip()][:n] or [topic[:128]]


async def _ask(system: str, user: str) -> dict[str, Any]:
    """One reasoning call for the planning roles; returns the parsed JSON object."""
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import runtime

    res = await runtime._chat_request(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=2048,
        response_format={"type": "json_object"},
        timeout=45.0,
        deadline=45.0,
        attempts=1,
    )
    parsed, _ = dr.sanitize_json(res.get("content") or "")
    return parsed if isinstance(parsed, dict) else {}


async def _planned_queries(topic: str, n: int) -> list[str]:
    """Plan in three steps: analyse the question, build aspects with criteria, screen the queries."""
    brief = await _ask(TOPIC_ANALYST_PROMPT, f"Research question: {topic}")
    plan = await _ask(
        PLAN_BUILDER_PROMPT.replace("{n}", str(n)),
        f"Research question: {topic}\nAnalysis: {json.dumps(brief)[:2000]}",
    )
    aspects = [a for a in plan.get("aspects") or [] if isinstance(a, dict)]
    queries = [str(a.get("query") or "").strip() for a in aspects if a.get("query")]
    if not queries:
        return [topic[:128]]
    screened = await _ask(
        QUERY_SCREENER_PROMPT,
        f"Research question: {topic}\nQueries: {json.dumps(queries)}",
    )
    final = [str(q).strip()[:128] for q in screened.get("queries") or [] if str(q).strip()]
    return (final or queries)[:n]


async def _seed_snippets(topic: str) -> list[dict[str, str]]:
    """Two light searches, flattened into the finite list of snippets the plan builder gets to see."""
    seed = await _ask(SEED_PROMPT.replace("{n}", "2"), f"Research question: {topic}")
    queries = [str(q).strip()[:128] for q in seed.get("queries") or [] if str(q).strip()][:2] or [topic[:128]]
    results = await search_all(queries, topic, "low")
    snippets: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in results:
        url = str(r.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        body = str(r.get("content") or r.get("snippet") or "").strip()
        if not body:
            continue
        snippets.append({"title": str(r.get("title") or "")[:200], "url": url, "text": body[:SEED_CHARS]})
        if len(snippets) >= SEED_SNIPPETS:
            break
    return snippets


async def _seeded_queries(topic: str, n: int) -> list[str]:
    """Analyse, then plan with what the web already says in view, then screen the queries."""
    brief = await _ask(TOPIC_ANALYST_PROMPT, f"Research question: {topic}")
    snippets = await _seed_snippets(topic)
    plan = await _ask(
        PLAN_BUILDER_PROMPT.replace("{n}", str(n)),
        (
            f"Research question: {topic}\n"
            f"Analysis: {json.dumps(brief)[:1500]}\n"
            f"Pages already found on the web (title, url, text): {json.dumps(snippets)[:6000]}\n"
            "Use them to see what exists and where the gaps are; the queries must still stand on their own."
        ),
    )
    aspects = [a for a in plan.get("aspects") or [] if isinstance(a, dict)]
    queries = [str(a.get("query") or "").strip() for a in aspects if a.get("query")]
    if not queries:
        return [topic[:128]]
    screened = await _ask(
        QUERY_SCREENER_PROMPT,
        f"Research question: {topic}\nQueries: {json.dumps(queries)}",
    )
    final = [str(q).strip()[:128] for q in screened.get("queries") or [] if str(q).strip()]
    return (final or queries)[:n]


async def search_all(queries: list[str], topic: str, depth: str) -> list[dict[str, Any]]:
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import search_core
    from deepseek_search_everywhere_mcp.filters import MAX_QUERY

    cfg = dr.DEPTH_CONFIG[depth]
    answers = await asyncio.gather(
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
    out = []
    for q, a in zip(queries, answers, strict=True):
        for rank, r in enumerate(a.get("results", []) if a.get("ok") else [], 1):
            out.append(
                {
                    "query": q,
                    "rank": rank,
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "content": (r.get("content") or "")[:CONTENT_CHARS],
                    "rerank": r.get("rerank_score"),
                    "authority": r.get("authority"),
                    "source_type": dr.classify_source_type(
                        r.get("url") or "", r.get("title") or "", r.get("content") or ""
                    ),
                }
            )
    return out


async def run(args: argparse.Namespace) -> None:
    from deepseek_search_everywhere_mcp import deep_research as dr

    n = dr.DEPTH_CONFIG[args.depth].subqueries
    for topic_id in args.topics.split(","):
        topic = _rubric(topic_id)["topic"]
        for style in args.styles.split(","):
            path = HERE / "results" / args.name / topic_id / f"{style}.json"
            if path.exists():
                continue
            queries = await plan_queries(style, topic, n)
            results = await search_all(queries, topic, args.depth)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"style": style, "queries": queries, "results": results}, indent=1), "utf-8"
            )
            print(topic_id, style, len(queries), "queries", len(results), "results", flush=True)


def _page_id(url: str) -> str:
    return hashlib.sha256((url or "").encode()).hexdigest()[:10]


PAGE_INSTRUCTIONS = """You label web pages for a research evaluation. For each page below, list the ids of the
key points whose substance the page itself states (its specific fact, and its figures where the key
point has figures, at the precision the point gives). A page that only mentions the subject, or states a
different figure, does not count. Judge only from the page text shown; do not use the web.

Write only this JSON object: {"pages": {"<page id>": ["k1", "k4"], "<page id>": []}}
"""


def packs(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pages: dict[str, dict[str, Any]] = {}
        for f in sorted(topic_dir.glob("*.json")):
            for r in json.loads(f.read_text())["results"]:
                pages.setdefault(_page_id(r["url"]), r)
        rubric = judge._rubric_text(topic_dir.name)
        ids = sorted(pages)
        (topic_dir / "packs").mkdir(exist_ok=True)
        for i in range(0, len(ids), PAGES_PER_PACK):
            chunk = ids[i : i + PAGES_PER_PACK]
            body = [
                PAGE_INSTRUCTIONS,
                f"\n# Topic\n{_rubric(topic_dir.name)['topic']}\n",
                f"# Key points\n{rubric}\n",
            ]
            for pid in chunk:
                p = pages[pid]
                body.append(f"## Page {pid}\nTitle: {p['title']}\nURL: {p['url']}\n\n{p['content']}\n")
            (topic_dir / "packs" / f"p{i // PAGES_PER_PACK:02d}.md").write_text("\n".join(body), "utf-8")
        print(topic_dir.name, len(ids), "unique pages", -(-len(ids) // PAGES_PER_PACK), "packs")


def score(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    table: dict[str, dict[str, Any]] = {}
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        labels: dict[str, list[str]] = {}
        for v in sorted((topic_dir / "verdicts").glob("*.json")):
            labels.update((judge._read_verdict(v) or {}).get("pages", {}))
        rubric = _rubric(topic_dir.name)
        core = {k["id"] for k in rubric["key_points"] if k["importance"] == "core"}
        every = {k["id"] for k in rubric["key_points"]}
        for f in sorted(topic_dir.glob("*.json")):
            data = json.loads(f.read_text())
            hits = [set(labels.get(_page_id(r["url"]), [])) for r in data["results"]]
            got = set().union(*hits) if hits else set()
            n = max(1, len(data["results"]))
            row = table.setdefault(data["style"], {"topics": {}})
            row["topics"][topic_dir.name] = {
                "queries": len(data["queries"]),
                "pages": len(data["results"]),
                "core_yield": len(got & core) / len(core),
                "all_yield": len(got & every) / len(every),
                "useful_pages": sum(1 for h in hits if h) / n,
                "primary": sum(
                    r["source_type"]
                    in (
                        "official",
                        "specification",
                        "documentation",
                        "guideline",
                        "rct",
                        "meta-analysis",
                        "systematic_review",
                        "study",
                        "preprint",
                        "repository",
                    )
                    for r in data["results"]
                )
                / n,
            }
    for row in table.values():
        vals = list(row["topics"].values())
        for k in ("core_yield", "all_yield", "useful_pages", "primary", "pages"):
            row[k] = round(statistics.fmean(v[k] for v in vals), 3)
    (root / "_score.json").write_text(json.dumps(table, indent=1), "utf-8")
    for style in STYLES:
        if style in table:
            r = table[style]
            print(
                f"{style:10} core={r['core_yield']:.2f} all={r['all_yield']:.2f} "
                f"useful_pages={r['useful_pages']:.2f} primary={r['primary']:.2f} pages={r['pages']:.0f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--topics", required=True)
    r.add_argument("--styles", default=",".join(STYLES))
    r.add_argument("--depth", default="medium")
    for name in ("packs", "score"):
        s = sub.add_parser(name)
        s.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "packs":
        packs(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
