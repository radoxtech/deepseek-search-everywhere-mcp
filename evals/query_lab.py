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
import re
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
SEED_ROLE = (
    "The pages listed are reconnaissance, not material to plan from. Use them only to see which "
    "sub-topics are already crowded and which are missing. Never let them widen a query: when a page "
    "covers a sub-topic, your query for it must be narrower than the page. Plan from the analysis; if "
    "the two disagree about what matters, follow the analysis. Avoid queries that would return a "
    "general review of the topic."
)
QUERY_SCREENER_PROMPT = (
    "You review search queries before they are sent. For each query say whether it will return generic "
    "pages, repeated pages or the wrong kind of source, and if so give a better query of 4 to 12 words. "
    "Return exactly the same number of queries you were given, one for each input query, in the same "
    'order, so the plan keeps its coverage. Return only json: {{"queries": ["..."]}}'
)
STYLES = (
    "old",
    "aspects",
    "questions",
    "evidence",
    "site",
    "topic",
    "planned",
    "seeded",
    "seeded-neg",
    "seeded-guard",
    "seeded-twostage",
    "seeded-blend",
    "seeded-roled",
    "seeded-lit",
    "analyzed",
    "analyzed-answered",
)
SEEDED_VARIANTS = ("seeded", "seeded-neg", "seeded-guard", "seeded-twostage", "seeded-blend")
SEED_COVERED_PROMPT = (
    "You list what a search already covered. For each page give one short line: the topic it actually "
    "answers and its main claim. No titles, no URLs, no marketing. These lines are a do-not-repeat list. "
    'Return only json: {{"covered": ["...", "..."]}}'
)
SEED_GAP_PROMPT = (
    "You are given a draft search plan and a list of pages already found. Add only the gaps the plan "
    "misses that those pages do not cover. Do not rewrite or reorder the existing queries. "
    'Return only json: {{"added": ["...", "..."]}}'
)
SEED_SNIPPETS = 6
SEED_CHARS = 840
MAX_QUESTIONS = 5
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
    if style in ("seeded-neg", "seeded-guard", "seeded-twostage", "seeded-blend"):
        return await _seeded_variant(style, topic, n)
    if style == "seeded-roled":
        return await _seeded_roled(topic, n)
    if style == "seeded-lit":
        return await _seeded_lit(topic, n)
    if style in ("analyzed", "analyzed-answered"):
        return await _analyzed_queries(style, topic, n)
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


async def _ask(system: str, user: str, max_tokens: int = 16384) -> dict[str, Any]:
    """One reasoning call for the planning roles; returns the parsed JSON object.

    ``thinking`` is enabled, and on this model the reasoning tokens come out of ``max_tokens``
    together with the answer. A small budget returns an empty body with ``finish_reason: "length"``,
    so the budget here matches the low depth tier rather than the answer size alone.
    """
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import runtime

    res = await runtime._chat_request(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        thinking={"type": "enabled"},
        model=runtime.REWRITE_MODEL,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=60.0,
        deadline=60.0,
        attempts=1,
    )
    content = res.get("content") or ""
    if not content and res.get("finish_reason") == "length":
        raise RuntimeError("planning call ran out of output tokens; raise max_tokens")
    parsed, _ = dr.sanitize_json(content)
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
    return await _screen_queries(topic, queries, n)


async def _seeded_roled(topic: str, n: int) -> list[str]:
    """Plan from the analysis, with the seed presented by its role rather than as material to plan from."""
    brief = await _ask(TOPIC_ANALYST_PROMPT, f"Research question: {topic}")
    snippets = await _seed_snippets(topic)
    pages = json.dumps(
        [{"title": s["title"], "url": s["url"], "opening": s["text"][:300]} for s in snippets]
    )[:4000]
    plan = await _ask(
        PLAN_BUILDER_PROMPT.replace("{n}", str(n)),
        "Research question: "
        + topic
        + chr(10)
        + "Analysis of the question: "
        + json.dumps(brief)[:1500]
        + chr(10)
        + chr(10)
        + SEED_ROLE
        + chr(10)
        + chr(10)
        + "Pages already found: "
        + pages,
    )
    aspects = [a for a in plan.get("aspects") or [] if isinstance(a, dict)]
    queries = [str(a.get("query") or "").strip() for a in aspects if a.get("query")]
    if not queries:
        return [topic[:128]]
    return await _screen_queries(topic, queries, n)


COVERAGE_PROMPT = (
    "You decide what a set of pages actually answers. For each key question of the research topic, say "
    "whether those pages answer it and how (a figure, a date, a named mechanism, an official rule). "
    'Return only json: {"answered": [{"need": "...", "how": "..."}], "missing": ["..."]}'
)
ANCHORED_PLAN_PROMPT = (
    "You plan the searches for a research question. Write {n} aspects, each with a short name, what the "
    "reader must learn, one query of 4 to 12 words, and the success criteria (figures, dates, conditions). "
    "You are given what existing pages already answer and what they leave open. Write one aspect per "
    "missing need, aimed at a specific figure, date, condition or named mechanism, never at the topic as "
    "a whole. If the existing pages already answer a need, do not write an aspect for it. "
    'Return only json: {"aspects": [{"aspect": "", "looking_for": "", "query": "", "success_criteria": []}]}'
)


async def _seeded_lit(topic: str, n: int) -> list[str]:
    """The literature's version: verified seed facts, a coverage call, and queries aimed only at the gaps.

    EviReform keeps the original need visible while adding a residual query; HERO selects follow-ups for
    uncovered information rather than restating the topic; the LLM-assisted PRF line filters the seed
    before it can steer anything. This style applies all three together.
    """
    brief = await _ask(TOPIC_ANALYST_PROMPT, f"Research question: {topic}")
    snippets = await _seed_snippets(topic)
    needs = [str(x) for x in brief.get("must_cover") or [] if str(x).strip()][:8]
    coverage = await _ask(
        COVERAGE_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research question: "
        + topic
        + chr(10)
        + "What the answer must cover: "
        + json.dumps(needs)
        + chr(10)
        + "Pages already found: "
        + json.dumps(snippets)[:4000],
    )
    missing = [str(x).strip() for x in coverage.get("missing") or [] if str(x).strip()][:8]
    answered = [
        str(x.get("need") or "").strip() for x in coverage.get("answered") or [] if isinstance(x, dict)
    ]
    plan = await _ask(
        ANCHORED_PLAN_PROMPT.replace("{n}", str(n)),
        "Research question: "
        + topic
        + chr(10)
        + "Analysis: "
        + json.dumps(brief)[:1200]
        + chr(10)
        + "Still missing (plan for these): "
        + json.dumps(missing or needs)
        + chr(10)
        + "Already answered (do not re-plan these): "
        + json.dumps(answered),
    )
    aspects = [a for a in plan.get("aspects") or [] if isinstance(a, dict)]
    queries = [str(a.get("query") or "").strip() for a in aspects if a.get("query")]
    if not queries:
        return [topic[:128]]
    return await _screen_queries(topic, queries, n)


MUST_FIND_PLAN_PROMPT = (
    "You plan web searches for a research question. You are given the facts a search must establish. "
    "Write one query per fact, 4 to 12 words, aimed at that fact rather than at the topic: name the "
    "document, the measure, the body or the specific thing the fact asks about. Do not merge two facts "
    "into one query and do not add queries for facts that are not listed. "
    'Return only json: {"queries": ["..."]}'
)
ANSWER_PROMPT = (
    "You answer clarifying questions on behalf of whoever asked a research topic. Answer as a careful "
    "person with a specific situation would: pick the branch that fits a typical case, in one short "
    "sentence, and never invent a figure the question does not ask for. "
    'Return only json: {"answers": ["..."]}'
)


async def _analyzed_queries(style: str, topic: str, n: int) -> list[str]:
    """Plan from the analyzer's must_find list, with or without answers to the clarifying questions.

    ``analyzed`` takes the defaults as they stand, which is what an asker who does not answer would
    get. ``analyzed-answered`` first has the questions answered by a stand-in reader, so the pair
    separates the value of covering the branches from the value of the answers themselves.
    """
    snippets = await _seed_snippets(topic)
    analysis = await _analyzer(topic, snippets)
    asked = analysis.get("askable") or []
    must = [str(x).strip() for x in analysis.get("must_find") or [] if str(x).strip()]

    if style == "analyzed-answered" and asked:
        answered = await _ask(
            ANSWER_PROMPT.replace("{{", "{").replace("}}", "}"),
            "Research topic: "
            + topic
            + chr(10)
            + "Questions: "
            + json.dumps(asked, ensure_ascii=False)[:3000],
        )
        pairs = [
            {"question": q["question"], "answer": str(a).strip()}
            for q, a in zip(asked, answered.get("answers") or [], strict=False)
        ]
        if pairs:
            refreshed = await _analyzer_with_answers(topic, snippets, pairs)
            must = [str(x).strip() for x in refreshed.get("must_find") or [] if str(x).strip()] or must

    if not must:
        return [topic[:128]]
    plan = await _ask(
        MUST_FIND_PLAN_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research question: "
        + topic
        + chr(10)
        + "Facts to establish: "
        + json.dumps(must[:n], ensure_ascii=False),
    )
    queries = [str(q).strip()[:128] for q in plan.get("queries") or [] if str(q).strip()]
    return (queries or must)[:n]


async def _analyzer_with_answers(
    topic: str, snippets: list[dict[str, str]], pairs: list[dict[str, str]]
) -> dict[str, Any]:
    """Re-run the analyzer with the questions already answered, so must_find targets the chosen branch."""
    pages = json.dumps([{"title": s["title"], "url": s["url"], "facts": s["text"][:400]} for s in snippets])[
        :4000
    ]
    return await _ask(
        ANALYZER_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research topic: "
        + topic
        + chr(10)
        + "Answers already given: "
        + json.dumps(pairs, ensure_ascii=False)[:2000]
        + chr(10)
        + "What the web already holds: "
        + pages,
    )


async def _screen_queries(topic: str, queries: list[str], n: int) -> list[str]:
    """Screen the plan's queries, but never let the screener thin the plan out.

    A screener that returns fewer queries than it was given silently costs the style pages:
    one query fetches ``count`` results, so three queries collapse to one third of the budget.
    The repair tries once more for the same number, then falls back to the unscreened queries
    padded from the plan, so every style is measured on the same number of searches.
    """
    asked = (
        f"Research question: {topic}\nQueries ({len(queries)} of them, return {len(queries)}): "
        f"{json.dumps(queries)}"
    )
    final: list[str] = []
    for attempt in range(2):
        screened = await _ask(QUERY_SCREENER_PROMPT, asked)
        answered = [str(q).strip()[:128] for q in screened.get("queries") or [] if str(q).strip()]
        final = list(dict.fromkeys(final + answered))
        if len(final) >= min(len(queries), n):
            return final[:n]
        if attempt == 0:
            asked = (
                f"Research question: {topic}\nReturn exactly {len(queries)} queries. "
                f"Queries: {json.dumps(queries)}"
            )
    kept = list(dict.fromkeys(final + queries))
    while len(kept) < n:
        kept.append(topic[:128])
    return kept[:n]


def _evidence_lines(text: str, limit: int = 3) -> list[str]:
    """Sentences that carry numbers, so the seed shows facts instead of page headers."""
    clean = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    scored = []
    for part in parts:
        if len(part) < 40:
            continue
        digits = sum(ch.isdigit() for ch in part)
        latin = sum(1 for ch in part if ch.isascii() and ch.isalpha())
        if digits < 2 or latin < 0.5 * max(1, len(part)):
            continue
        scored.append((digits, part))
    scored.sort(key=lambda item: -item[0])
    return [part[:280] for _, part in scored[:limit]]


ANALYZER_PROMPT = (
    "You decide whether a research topic can be searched as it stands, and what a search must establish.\n"
    "You are given the topic and reconnaissance pages showing what the open web already holds.\n"
    "The pages are the result of two broad searches, not a map of the field: they show what those two "
    "searches found, not what exists. Take them as a sample, never as the whole.\n"
    "Write in English unless the topic is clearly written in another language, in which case write in "
    "that language. Never mix the two.\n"
    "Before writing a question, check whether its answer is already present: in the topic itself, in "
    "the way it is phrased, or in what the pages show.\n"
    "\n"
    "Answer three separate questions.\n"
    "\n"
    "1. Does more than one reading of this topic lead to a different search? Name each reading briefly, "
    "or report none when the topic points at one thing.\n"
    "2. Is any missing fact able to change the answer? Name the fact, not the subject area. Leave out "
    "what the reconnaissance pages already settle.\n"
    "3. Can a search settle what is missing, or does it depend on a preference only the asker holds? "
    "A price, a current status, a specification or a named document can be found; a priority, an "
    "acceptable trade-off or an intended use cannot.\n"
    "\n"
    "Ask only about what the second kind leaves open. Ask in the words the person used, about things "
    "they can answer without knowing the field: what happened to them, what they will do with it, what "
    "they already have. Never ask about a distinction only a specialist would draw, or one the answer "
    "itself has to settle; those belong in what to find, and the search then covers every branch.\n"
    "\n"
    "The asker may not answer. Give every question a default, addressed to whoever reads the result: "
    "which branches the search will cover if nobody answers. The default names branches, never a value: "
    "it may not invent a budget, a size, a threshold, a location or a preference the topic does not "
    "state. It describes coverage, and the search itself is driven by what to find, not by the default.\n"
    "\n"
    "Ask only the questions that change what gets searched. When nothing of the second kind is missing, "
    "questions is empty.\n"
    "\n"
    "Before anything else, say what the pages cover and what they leave out of the topic as asked. Name "
    "the branches the sample missed, so the plan can cover them.\n"
    "Then list what must be established from the web: one line per thing to find, phrased as the fact, "
    "not as a subject area, leaving out what the pages already answer. Order the list so the fact that "
    "matters most for answering the topic comes first; a fact that only matters for one narrow case "
    "goes last, however interesting it is.\n"
    "\n"
    "When you can search both branches instead of asking, do that and state the assumption, but only "
    "when the branches point the same way and the reader can tell which one an answer applies to. If "
    "the branches could lead to opposite conclusions, the choice belongs to the asker: ask instead of "
    "assuming. Beyond that, an assumption is allowed only when it is supported by the topic or by what "
    "the pages show rather than by your own preference, and when it stays harmless if wrong. Never "
    "assume what the asker alone can decide. When in doubt, ask: a question costs a turn, a wrong "
    "assumption costs the whole report.\n"
    "\n"
    'Return only json: {{"seed_covered": ["..."], "seed_missing": ["..."], '
    '"readings": ["..."], "askable": [{"question": "...", "default": "..."}], '
    '"assumptions": ["..."], "must_find": ["..."]}}'
)


SCORE_PROMPT = (
    "You judge questions that were prepared for the person who asked a research topic. Score each "
    "question from 1 to 5 on three axes, using only what the topic and the pages show:\n"
    "- known: can the asker answer this from their own situation or task? 5 means they certainly can, "
    "1 means they cannot. A question about a distinction only a specialist draws scores 1.\n"
    "- matters: how much does the answer change what gets searched? 5 means it changes which facts are "
    "looked for, 1 means the search would be the same either way.\n"
    "- answerable: how naturally can it be answered in a sentence? 5 is a plain everyday question, "
    "1 is an essay question or a demand for a number nobody has.\n"
    'Return only json: {"scores": [{"n": 1, "known": 4, "matters": 5, "answerable": 5}]}'
)
SCORE_FLOOR = 3


async def _score_questions(topic: str, asked: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Score every question on its own, in a call that only judges and never rewrites.

    The scoring is separated from generation because a model that has just invented a question tends
    to defend it; here it sees nothing but the topic and the list.
    """
    if not asked:
        return []
    numbered = [{"n": i + 1, "question": a["question"], "default": a["default"]} for i, a in enumerate(asked)]
    judged = await _ask(
        SCORE_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research topic: "
        + topic
        + chr(10)
        + "Questions: "
        + json.dumps(numbered, ensure_ascii=False)[:3000],
    )
    scores = {
        int(row["n"]): row
        for row in judged.get("scores") or []
        if isinstance(row, dict) and str(row.get("n", "")).isdigit()
    }
    out: list[dict[str, Any]] = []
    for i, row in enumerate(asked, start=1):
        got = scores.get(i, {})
        axes = {k: int(got.get(k) or 0) for k in ("known", "matters", "answerable")}
        out.append({**row, "scores": axes, "keep": min(axes.values()) >= SCORE_FLOOR})
    return out


TRIM_PROMPT = (
    "You are given a research topic and the questions prepared for the person who asked it. Keep the "
    "five that change what gets searched most. Drop the ones whose answer would barely alter the "
    'search, and the ones a default covers just as well. Return only json: {"keep": [1, 3, 4]}'
)


async def _trim_questions(topic: str, asked: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Cut an over-long question list by asking which ones matter, not by dropping the tail."""
    if len(asked) <= limit:
        return asked
    numbered = [{"n": i + 1, "question": a["question"], "default": a["default"]} for i, a in enumerate(asked)]
    picked = await _ask(
        TRIM_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research topic: "
        + topic
        + chr(10)
        + "Questions: "
        + json.dumps(numbered, ensure_ascii=False)[:3000],
    )
    keep = [int(n) for n in picked.get("keep") or [] if isinstance(n, int) or str(n).isdigit()]
    chosen = [asked[n - 1] for n in keep if 1 <= n <= len(asked)]
    if len(chosen) != limit:
        return asked[:limit]
    return chosen


def _normalise_question(raw: Any) -> dict[str, str]:
    """Keep the question and its fallback, whichever shape the model returned them in."""
    if isinstance(raw, dict):
        return {
            "question": str(raw.get("question") or "").strip(),
            "default": str(raw.get("default") or "").strip(),
        }
    return {"question": str(raw).strip(), "default": ""}


async def _analyzer(topic: str, snippets: list[dict[str, str]]) -> dict[str, Any]:
    """Decide, with the reconnaissance in hand, what to ask and what to establish.

    Splitting this from the planner matters: a model asked to check its own plan inside the planning
    turn does worse than a separate step that only judges, so the judgement arrives here as data.
    """
    pages = json.dumps([{"title": s["title"], "url": s["url"], "facts": s["text"][:400]} for s in snippets])[
        :4000
    ]
    result = await _ask(
        ANALYZER_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research topic: " + topic + chr(10) + "What the web already holds: " + pages,
    )
    ranked = [_normalise_question(q) for q in (result.get("askable") or [])]
    scored = await _score_questions(topic, ranked)
    kept = [row for row in scored if row["keep"]] or scored
    result["askable"] = await _trim_questions(topic, kept, MAX_QUESTIONS)
    return result


async def _seed_snippets(topic: str) -> list[dict[str, str]]:
    """Two light searches, cut down to the numbered sentences a planner can use as evidence of what exists."""
    seed = await _ask(SEED_PROMPT.replace("{n}", "2"), f"Research question: {topic}")
    queries = [str(q).strip()[:128] for q in seed.get("queries") or [] if str(q).strip()][:2] or [topic[:128]]
    results = await search_all(queries, topic, "low")
    snippets: list[dict[str, str]] = []
    seen: set[str] = set()
    for r in results:
        url = str(r.get("url") or "")
        if not url or url in seen:
            continue
        body = str(r.get("content") or "").strip()
        lines = _evidence_lines(body)
        if not lines:
            continue
        seen.add(url)
        snippets.append(
            {"title": str(r.get("title") or "")[:200], "url": url, "text": " ".join(lines)[:SEED_CHARS]}
        )
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
    return await _screen_queries(topic, queries, n)


def _overlap(query: str, covered: list[str]) -> float:
    """Jaccard overlap between a query and the covered lines, so a rephrased duplicate still counts."""

    def tokens(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())}

    q = tokens(query)
    if not q:
        return 0.0
    best = 0.0
    for line in covered:
        c = tokens(line)
        if c:
            best = max(best, len(q & c) / len(q | c))
    return best


async def _seeded_variant(style: str, topic: str, n: int) -> list[str]:
    """Four ways of letting the seed inform a plan without becoming the plan."""
    NL = chr(10)
    brief = await _ask(TOPIC_ANALYST_PROMPT, f"Research question: {topic}")
    snippets = await _seed_snippets(topic)
    covered = await _ask(
        SEED_COVERED_PROMPT.replace("{{", "{").replace("}}", "}"),
        "Research question: "
        + topic
        + NL
        + "Pages: "
        + json.dumps([s["title"] + " | " + s["text"] for s in snippets])[:6000],
    )
    lines = [str(x).strip() for x in covered.get("covered") or [] if str(x).strip()][:SEED_SNIPPETS]

    analysis = "Analysis: " + json.dumps(brief)[:1500]
    plan_messages = {
        "seeded-neg": (
            "Research question: "
            + topic
            + NL
            + analysis
            + NL
            + "Already covered by existing pages, do not write queries for these topics: "
            + json.dumps(lines)
        ),
        "seeded-guard": (
            "Research question: "
            + topic
            + NL
            + analysis
            + NL
            + "Pages already found: "
            + json.dumps(snippets)[:6000]
            + NL
            + "Plan from the analysis; use the pages only to avoid duplicates and to spot gaps."
        ),
        "seeded-blend": (
            "Research question: "
            + topic
            + NL
            + "Analysis (primary): "
            + json.dumps(brief)[:1500]
            + NL
            + "Web hints (weak signal, about a fifth of the plan at most): "
            + json.dumps(lines)[:1500]
        ),
    }

    if style == "seeded-twostage":
        aspects = (
            await _ask(
                PLAN_BUILDER_PROMPT.replace("{n}", str(n)),
                "Research question: " + topic + NL + "Analysis: " + json.dumps(brief)[:2000],
            )
        ).get("aspects") or []
        base = [str(a.get("query") or "").strip() for a in aspects if isinstance(a, dict) and a.get("query")]
        if not base:
            return [topic[:128]]
        gaps = await _ask(
            SEED_GAP_PROMPT.replace("{{", "{").replace("}}", "}"),
            "Research question: "
            + topic
            + NL
            + "Plan: "
            + json.dumps(base)
            + NL
            + "Pages found: "
            + json.dumps(lines),
        )
        added = [str(x).strip()[:128] for x in gaps.get("added") or [] if str(x).strip()]
        return await _screen_queries(topic, _merge_plan(base, added, n), n)

    plan = await _ask(PLAN_BUILDER_PROMPT.replace("{n}", str(n)), plan_messages[style])
    aspects = [a for a in plan.get("aspects") or [] if isinstance(a, dict)]
    queries = [str(a.get("query") or "").strip() for a in aspects if a.get("query")]
    if not queries:
        return [topic[:128]]

    if style == "seeded-guard":
        duplicates = [q for q in queries if _overlap(q, lines) > 0.34]
        if len(duplicates) * 3 > len(queries):
            retry = await _ask(
                PLAN_BUILDER_PROMPT.replace("{n}", str(n)),
                "Research question: "
                + topic
                + NL
                + analysis
                + NL
                + "Do not write queries for topics already covered; find the gaps instead.",
            )
            retry_aspects = [a for a in retry.get("aspects") or [] if isinstance(a, dict)]
            retry_queries = [str(a.get("query") or "").strip() for a in retry_aspects if a.get("query")]
            if retry_queries:
                queries = retry_queries
    return await _screen_queries(topic, queries, n)


def _merge_plan(base: list[str], added: list[str], n: int) -> list[str]:
    """Keep the analysis plan, let the seed add only what it genuinely missed."""
    return list(dict.fromkeys(base + added))[:n]


async def search_all(
    queries: list[str], topic: str, depth: str, n: int | None = None
) -> list[dict[str, Any]]:
    """Search every query, padding to ``n`` distinct queries so each style gets the same budget.

    ``queries`` is a plan, and a plan that returns fewer queries would otherwise fetch fewer
    pages, which compares budgets rather than plans. Padding repeats the topic itself, so the
    extra searches are honest but carry no planning signal.
    """
    from deepseek_search_everywhere_mcp import deep_research as dr
    from deepseek_search_everywhere_mcp import search_core
    from deepseek_search_everywhere_mcp.filters import MAX_QUERY

    planned = list(dict.fromkeys(q for q in queries if q.strip()))
    if n is not None:
        while len(planned) < n:
            planned.append(topic[:128])
    queries = planned[: n or len(planned)]

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
            results = await search_all(queries, topic, args.depth, n)
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


def _labelled_page_ids(results_dir: Path) -> set[str]:
    """Every page id that already has a verdict under ``results_dir``, so a pack run can reuse it."""
    ids: set[str] = set()
    for verdict in results_dir.glob("*/verdicts/*.json"):
        pages = json.loads(verdict.read_text(encoding="utf-8")).get("pages")
        if isinstance(pages, dict):
            ids.update(pages)
    return ids


def packs(args: argparse.Namespace) -> None:
    root = HERE / "results" / args.name
    already = _labelled_page_ids(HERE / "results" / args.skip_labelled) if args.skip_labelled else set()
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pages: dict[str, dict[str, Any]] = {}
        for f in sorted(topic_dir.glob("*.json")):
            for r in json.loads(f.read_text())["results"]:
                pages.setdefault(_page_id(r["url"]), r)
        rubric = judge._rubric_text(topic_dir.name)
        ids = [pid for pid in sorted(pages) if pid not in already]
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
        if name == "packs":
            s.add_argument(
                "--skip-labelled",
                default=None,
                help="another results dir whose verdicts already cover some pages; those stay unpacked",
            )
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "packs":
        packs(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
