"""Integration tests for deepseek_deep_research tool."""

from __future__ import annotations

import json
from typing import Any

from deepseek_search_everywhere_mcp import server


class DeepResearchUpstream:
    def __init__(
        self,
        subqueries: list[str] | None = None,
        search_results: list[dict[str, Any]] | None = None,
        report_text: str = "",
        fail_synthesis: bool = False,
        fail_decomposition: bool = False,
    ) -> None:
        self.subqueries = subqueries or ["subquery 1", "subquery 2"]
        self.search_results = search_results or [
            {
                "title": "Quantum Error Correction Guide",
                "url": "https://physics.test/qec",
                "content": "Surface codes provide fault tolerance in quantum computing.",
            },
            {
                "title": "Qubit Fidelity Benchmarks",
                "url": "https://physics.test/benchmarks",
                "content": "Two-qubit gate fidelities exceed 99.9% in recent trials.",
            },
        ]
        self.report_text = report_text or (
            "Fault tolerance is achieved via surface codes [1]. "
            "Recent trials demonstrate gate fidelities over 99.9% [2]. "
            "Hallucinated citation [99] should be cleaned."
        )
        self.fail_synthesis = fail_synthesis
        self.fail_decomposition = fail_decomposition
        self.searches: list[dict[str, Any]] = []
        self.chats: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.searches.append(payload)
            return {"ok": True, "data": {"results": self.search_results}}

        if path == server.FETCH_PATH:
            return {
                "ok": True,
                "data": {"title": "Fetched", "content": "Fetched page content " * 20},
            }

        if path == "/v1/chat/completions":
            self.chats.append(payload)
            # First chat is decomposition
            if len(self.chats) == 1:
                if self.fail_decomposition:
                    return {
                        "ok": False,
                        "error": {"code": "internal_error", "message": "Decomposition broke"},
                    }
                return {
                    "ok": True,
                    "data": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps({"queries": self.subqueries}),
                                }
                            }
                        ]
                    },
                }
            # Second chat is synthesis
            if self.fail_synthesis:
                return {"ok": False, "error": {"code": "rate_limit", "message": "Synthesis limited"}}
            return {
                "ok": True,
                "data": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": self.report_text,
                            }
                        }
                    ]
                },
            }

        return {"ok": False, "error": {"code": "not_found", "message": path}}


async def test_deep_research_validations(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())

    # Empty topic
    res = await server.deepseek_deep_research("   ")
    assert res["ok"] is False
    assert res["error"]["code"] == "empty_query"

    # Too long topic
    res = await server.deepseek_deep_research("x" * 501)
    assert res["ok"] is False
    assert res["error"]["code"] == "topic_too_long"


async def test_deep_research_end_to_end_success(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_deep_research("quantum computing scaling", max_queries=2)
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert res["subqueries"] == ["subquery 1", "subquery 2"]
    assert "[1]" in res["report"]
    assert "[2]" in res["report"]
    assert "[99]" not in res["report"]  # Hallucinated citation stripped
    assert len(res["sources"]) == 2
    assert res["sources"][0]["id"] == 1
    assert res["sources"][1]["id"] == 2
    assert len(upstream.searches) == 2
    assert len(upstream.chats) == 2  # Decomposition + Synthesis


async def test_deep_research_decomposition_fallback(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream(fail_decomposition=True)
    fake_upstream(upstream)

    res = await server.deepseek_deep_research("quantum topic", max_queries=3)
    assert res["ok"] is True
    assert res["synthesized"] is True
    # Fell back to the original topic for search
    assert res["subqueries"] == ["quantum topic"]
    assert len(upstream.searches) == 1


async def test_deep_research_synthesis_fallback(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream(fail_synthesis=True)
    fake_upstream(upstream)

    res = await server.deepseek_deep_research("quantum topic", max_queries=2)
    assert res["ok"] is True
    assert res["synthesized"] is False
    assert res["report"] is None
    assert "sources" in res
    assert any("Synthesis model call failed" in note for note in res["notes"])
