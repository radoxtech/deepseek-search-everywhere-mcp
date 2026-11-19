"""The seeded style: two light searches feed the plan builder, then the screener trims to n."""

from __future__ import annotations

import asyncio

import query_lab


def test_seeded_style_puts_snippets_in_front_of_the_plan_builder(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def fake_ask(system: str, user: str) -> dict:
        if system == query_lab.SEED_PROMPT.format(n=2) or system.startswith("Write 2 short web search"):
            seen["seed"] = user
            return {"queries": ["seed one", "seed two"]}
        if system.startswith("You analyse a research question"):
            return {"ambiguities": [], "traps": [], "wrong_answers": [], "must_cover": []}
        if system.startswith("You plan the searches"):
            seen["plan"] = user
            return {
                "aspects": [
                    {"aspect": "a", "looking_for": "", "query": "q1", "success_criteria": []},
                    {"aspect": "b", "looking_for": "", "query": "q2", "success_criteria": []},
                ]
            }
        if system.startswith("You review search queries"):
            return {"queries": ["q1", "q2"]}
        raise AssertionError(system[:40])

    calls: list[tuple[list[str], str, str]] = []

    async def fake_search_all(queries, topic, depth):
        calls.append((list(queries), topic, depth))
        return [
            {"query": queries[0], "rank": 1, "url": "https://a.test/1", "title": "A", "content": "first"},
            {"query": queries[0], "rank": 2, "url": "https://a.test/1", "title": "A dup", "content": "dup"},
            {"query": queries[1], "rank": 1, "url": "https://b.test/2", "title": "B", "content": "second"},
        ]

    monkeypatch.setattr(query_lab, "_ask", fake_ask)
    monkeypatch.setattr(query_lab, "search_all", fake_search_all)

    got = asyncio.run(query_lab.plan_queries("seeded", "some topic", 2))

    assert got == ["q1", "q2"]
    assert len(calls) == 1 and calls[0][2] == "low"
    assert "https://a.test/1" in seen["plan"] and "https://b.test/2" in seen["plan"]
    assert "dup" not in seen["plan"], "duplicate URLs must reach the plan builder once"


def test_seed_snippets_stop_at_the_cap(monkeypatch) -> None:
    async def fake_ask(system, user):
        return {"queries": ["a", "b"]}

    async def fake_search_all(queries, topic, depth):
        return [
            {"query": "a", "rank": i, "url": f"https://x.test/{i}", "title": "t", "content": "c"}
            for i in range(20)
        ]

    monkeypatch.setattr(query_lab, "_ask", fake_ask)
    monkeypatch.setattr(query_lab, "search_all", fake_search_all)

    snippets = asyncio.run(query_lab._seed_snippets("topic"))

    assert len(snippets) == query_lab.SEED_SNIPPETS
    assert all(len(s["text"]) <= query_lab.SEED_CHARS for s in snippets)
