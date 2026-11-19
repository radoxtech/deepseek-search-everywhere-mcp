"""Each seed variant keeps the analysis as the plan source and the seed as a limited signal."""

from __future__ import annotations

import asyncio

import query_lab


def _plan(*queries: str) -> dict:
    return {"aspects": [{"aspect": "a", "query": q, "success_criteria": []} for q in queries]}


def _wire(monkeypatch, *, covered: list[str], plan: dict, gaps: dict | None = None) -> None:
    async def fake_ask(system: str, user: str) -> dict:
        if system.startswith("You list what a search already covered"):
            return {"covered": covered}
        if system.startswith("You are given a draft search plan"):
            return gaps or {"added": []}
        if system.startswith("You analyse a research question"):
            return {"ambiguities": [], "traps": [], "wrong_answers": [], "must_cover": []}
        if system.startswith("You review search queries"):
            return {"queries": []}
        return plan

    async def fake_seed(_topic: str) -> list[dict[str, str]]:
        return [{"title": "t", "url": "https://x.test/1", "text": "seed text"}]

    monkeypatch.setattr(query_lab, "_ask", fake_ask)
    monkeypatch.setattr(query_lab, "_seed_snippets", fake_seed)


def test_negative_variant_forbids_covered_topics(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def capture(system: str, user: str) -> dict:
        if system.startswith("You plan the searches"):
            seen["plan"] = user
            return _plan("q1", "q2", "q3")
        return await _fallback(system)

    async def _fallback(system: str) -> dict:
        if system.startswith("You list what a search"):
            return {"covered": ["covered topic"]}
        if system.startswith("You analyse"):
            return {}
        if system.startswith("You review"):
            return {"queries": ["q1", "q2", "q3"]}
        return {}

    async def fake_seed(_t: str) -> list[dict[str, str]]:
        return [{"title": "t", "url": "u", "text": "body text"}]

    monkeypatch.setattr(query_lab, "_ask", capture)
    monkeypatch.setattr(query_lab, "_seed_snippets", fake_seed)

    got = asyncio.run(query_lab.plan_queries("seeded-neg", "topic", 3))

    assert got == ["q1", "q2", "q3"]
    assert "do not write queries" in seen["plan"].lower()
    assert "body text" not in seen["plan"], "the body of a page must not reach the plan as source material"


def test_guard_retries_when_the_plan_duplicates_the_seed(monkeypatch) -> None:
    calls = {"plan": 0}

    async def fake_ask(system: str, user: str) -> dict:
        if system.startswith("You plan the searches"):
            calls["plan"] += 1
            return (
                _plan("postgresql incremental backup wal summary")
                if calls["plan"] == 1
                else _plan("p2", "p3", "p4")
            )
        if system.startswith("You list what a search"):
            return {"covered": ["postgresql incremental backup wal summary"]}
        if system.startswith("You analyse"):
            return {}
        if system.startswith("You review"):
            return {"queries": []}
        return {}

    async def fake_seed(_t: str) -> list[dict[str, str]]:
        return [{"title": "t", "url": "u", "text": "b"}]

    monkeypatch.setattr(query_lab, "_ask", fake_ask)
    monkeypatch.setattr(query_lab, "_seed_snippets", fake_seed)

    got = asyncio.run(query_lab.plan_queries("seeded-guard", "topic", 3))

    assert calls["plan"] == 2
    assert got == ["p2", "p3", "p4"]


def test_twostage_keeps_the_analysis_plan_and_adds_gaps(monkeypatch) -> None:
    _wire(monkeypatch, covered=["x"], plan=_plan("a1", "a2"), gaps={"added": ["g1"]})

    got = asyncio.run(query_lab.plan_queries("seeded-twostage", "topic", 3))

    assert got[:2] == ["a1", "a2"]
    assert "g1" in got


def test_overlap_sees_a_rephrased_duplicate() -> None:
    assert (
        query_lab._overlap(
            "postgres incremental backup wal summary", ["PostgreSQL incremental backups via WAL summary"]
        )
        > 0.3
    )
    assert query_lab._overlap("nutrition timing fasting", ["postgres backup"]) < 0.1
