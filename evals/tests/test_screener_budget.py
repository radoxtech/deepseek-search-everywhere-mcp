"""The screener may improve queries but never reduce their number, and search pads to n."""

from __future__ import annotations

import asyncio

import query_lab


def test_screener_retries_then_falls_back_to_the_plan(monkeypatch) -> None:
    answers = [{"queries": ["only one"]}, {"queries": ["still one"]}]

    async def fake_ask(_system: str, _user: str) -> dict:
        return answers.pop(0)

    monkeypatch.setattr(query_lab, "_ask", fake_ask)

    got = asyncio.run(query_lab._screen_queries("topic", ["q1", "q2", "q3"], 3))

    assert got == ["only one", "still one", "q1"][:3]
    assert len(got) == 3


def test_screener_accepts_a_full_answer(monkeypatch) -> None:
    async def fake_ask(_system: str, _user: str) -> dict:
        return {"queries": ["better one", "better two", "better three"]}

    monkeypatch.setattr(query_lab, "_ask", fake_ask)

    assert asyncio.run(query_lab._screen_queries("topic", ["q1", "q2", "q3"], 3)) == [
        "better one",
        "better two",
        "better three",
    ]


def test_search_pads_a_short_plan_to_n(monkeypatch) -> None:
    from deepseek_search_everywhere_mcp import search_core

    asked: list[str] = []

    async def fake_search(query, *args, **kwargs):
        asked.append(query)
        return {"ok": False, "results": []}

    monkeypatch.setattr(search_core, "_search", fake_search)

    asyncio.run(query_lab.search_all(["q1"], "the topic", "low", 3))

    assert len(asked) == 3
    assert asked[0] == "q1" and asked[1] == "the topic"[:128]
