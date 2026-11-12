"""Research tool tests: multi-angle research with depth tiers and briefing report."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.deep_research import get_job_store
from deepseek_search_everywhere_mcp.settings import Settings


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


async def test_research_validations(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())

    # Empty topic
    res = await server.deepseek_research("   ")
    assert res["ok"] is False
    assert res["error"]["code"] == "empty_query"

    # Too long topic
    res = await server.deepseek_research("x" * 501)
    assert res["ok"] is False
    assert res["error"]["code"] == "topic_too_long"

    # Invalid depth
    res = await server.deepseek_research("valid topic", depth="extreme")  # type: ignore[arg-type]
    assert res["ok"] is False
    assert res["error"]["code"] == "bad_request"


async def test_research_end_to_end_success(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("quantum computing scaling", depth="low")
    assert res["ok"] is True
    assert res["depth"] == "low"
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


async def test_research_decomposition_fallback(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream(fail_decomposition=True)
    fake_upstream(upstream)

    res = await server.deepseek_research("quantum topic", depth="medium")
    assert res["ok"] is True
    assert res["depth"] == "medium"
    assert res["synthesized"] is True
    # Fell back to the original topic for search
    assert res["subqueries"] == ["quantum topic"]
    assert len(upstream.searches) == 1


async def test_research_synthesis_fallback(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream(fail_synthesis=True)
    fake_upstream(upstream)

    res = await server.deepseek_research("quantum topic", depth="low")
    assert res["ok"] is True
    assert res["depth"] == "low"
    assert res["synthesized"] is False
    assert res["report"] is None
    assert "sources" in res
    assert any("Synthesis model call failed" in note for note in res["notes"])


async def test_research_finishing_within_inline_wait_returns_inline(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("quantum computing scaling", depth="low")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert res["delivery"] == "inline"


async def test_research_exceeding_inline_wait_returns_running(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=0.05))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.15)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    res = await server.deepseek_research("slow quantum topic", depth="low")
    assert res["ok"] is True
    assert res["status"] == "running"
    job_id = res["job_id"]
    uuid_obj = uuid.UUID(job_id, version=4)
    assert str(uuid_obj) == job_id
    assert res["depth"] == "low"
    assert "progress" in res
    assert "next_action" in res
    assert "Call deepseek_check_research with this job_id now." in res["next_action"]

    store = get_job_store()
    job = store.get(job_id)
    assert job is not None
    assert job.task is not None
    done_res = await job.task
    assert done_res["ok"] is True
    assert done_res["status"] == "done"


async def test_research_cancellation_does_not_cancel_background_job(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=10.0))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.2)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    call_task = asyncio.create_task(server.deepseek_research("cancelled topic", depth="low"))
    await asyncio.sleep(0.05)
    call_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call_task

    store = get_job_store()
    job = store.find_active_match("cancelled topic", "low", None)
    assert job is not None
    assert job.task is not None
    assert not job.task.cancelled()
    res = await job.task
    assert res["ok"] is True


async def test_research_duplicate_joins_running_job(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=0.05))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.15)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    res1 = await server.deepseek_research("duplicate topic", depth="medium")
    res2 = await server.deepseek_research("DUPLICATE TOPIC ", depth="medium")
    assert res1["status"] == "running"
    assert res2["status"] == "running"
    assert res1["job_id"] == res2["job_id"]


async def test_research_concurrency_limit_returns_rate_limited_local(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: Settings(research_inline_wait=0.01, research_max_jobs=3),
    )

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0.2)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    res1 = await server.deepseek_research("topic 1", depth="low")
    res2 = await server.deepseek_research("topic 2", depth="low")
    res3 = await server.deepseek_research("topic 3", depth="low")
    assert res1["status"] == "running"
    assert res2["status"] == "running"
    assert res3["status"] == "running"

    res4 = await server.deepseek_research("topic 4", depth="low")
    assert res4["ok"] is False
    assert res4["error"]["code"] == "rate_limited_local"


async def test_check_research_long_poll_returns_polled_done(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=0.05))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.08)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    running_env = await server.deepseek_research("check poll topic", depth="low")
    assert running_env["status"] == "running"
    job_id = running_env["job_id"]

    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=0.20))
    polled_env = await server.deepseek_check_research(job_id)
    assert polled_env["ok"] is True
    assert polled_env["status"] == "done"
    assert polled_env["delivery"] == "polled"
    assert polled_env["topic"] == "check poll topic"


async def test_check_research_returns_running_when_still_not_done(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "get_settings", lambda: Settings(research_inline_wait=0.02))

    class VerySlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.20)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(VerySlowUpstream())

    running1 = await server.deepseek_research("still running topic", depth="low")
    assert running1["status"] == "running"
    job_id = running1["job_id"]

    running2 = await server.deepseek_check_research(job_id)
    assert running2["ok"] is True
    assert running2["status"] == "running"
    assert running2["job_id"] == job_id
    assert "Call deepseek_check_research" in running2["next_action"]


async def test_check_research_unknown_or_expired_job_id() -> None:
    res = await server.deepseek_check_research("non-existent-uuid")
    assert res["ok"] is False
    assert res["error"]["code"] == "job_not_found"
    assert "deepseek_research" in res["error"]["message"]

    res_empty = await server.deepseek_check_research("   ")
    assert res_empty["ok"] is False
    assert res_empty["error"]["code"] == "bad_request"


async def test_check_research_ttl_pruning(fake_upstream: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_upstream(DeepResearchUpstream())
    res = await server.deepseek_research("ttl topic", depth="low")
    assert res["status"] == "done"

    store = get_job_store()
    job = list(store._jobs.values())[0]
    job_id = job.job_id

    monkeypatch.setattr(server, "_clock", lambda: job.completed_at + 1801.0)
    check_res = await server.deepseek_check_research(job_id)
    assert check_res["ok"] is False
    assert check_res["error"]["code"] == "job_not_found"


@pytest.mark.parametrize(
    ("depth", "expected_thinking", "expected_effort", "expected_max_tokens", "expect_temp"),
    [
        ("low", {"type": "disabled"}, None, 16384, True),
        ("medium", {"type": "enabled"}, "low", 32768, False),
        ("high", {"type": "enabled"}, "high", 65536, False),
        ("max", {"type": "enabled"}, "max", 131072, False),
    ],
)
async def test_synthesis_tier_payload_contracts(
    depth: str,
    expected_thinking: dict[str, str],
    expected_effort: str | None,
    expected_max_tokens: int,
    expect_temp: bool,
    fake_upstream: Any,
) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research(f"topic for {depth}", depth=depth)  # type: ignore[arg-type]
    assert res["ok"] is True
    assert res["status"] == "done"
    assert len(upstream.chats) >= 2
    synth_payload = upstream.chats[-1]
    assert synth_payload["thinking"] == expected_thinking
    if expected_effort is not None:
        assert synth_payload["reasoning_effort"] == expected_effort
    else:
        assert "reasoning_effort" not in synth_payload
    assert synth_payload["max_tokens"] == expected_max_tokens
    if expect_temp:
        assert synth_payload.get("temperature") == 0.0
    else:
        assert "temperature" not in synth_payload
    assert "budget_tokens" not in synth_payload


async def test_synthesis_retry_on_length_truncation(fake_upstream: Any) -> None:
    class TruncatingUpstream(DeepResearchUpstream):
        def __init__(self) -> None:
            super().__init__()
            self.synth_calls = 0

        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["q1"]}),
                                    }
                                }
                            ]
                        },
                    }
                self.synth_calls += 1
                if self.synth_calls == 1:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "finish_reason": "length",
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "reasoning_content": "INTERNAL THOUGHTS SHOULD NEVER BE REPORT",
                                    },
                                }
                            ]
                        },
                    }
                else:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": "Retry report content [1].",
                                    },
                                }
                            ]
                        },
                    }
            return await super().__call__(method, path, **kwargs)

    upstream = TruncatingUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("retry test topic", depth="medium")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert "INTERNAL THOUGHTS" not in (res["report"] or "")
    assert "Retry report content [1]" in (res["report"] or "")
    assert any("retrying with thinking disabled" in note for note in res["notes"])
    assert len(upstream.chats) == 3
    assert upstream.chats[2]["thinking"] == {"type": "disabled"}
    assert upstream.chats[2].get("temperature") == 0.0


async def test_synthesis_failure_never_leaks_reasoning_content(fake_upstream: Any) -> None:
    class FailingUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["q1"]}),
                                    }
                                }
                            ]
                        },
                    }
                return {
                    "ok": True,
                    "data": {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "SECRET REASONING NEVER LEAKS",
                                },
                            }
                        ]
                    },
                }
            return await super().__call__(method, path, **kwargs)

    upstream = FailingUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("leak test topic", depth="medium")
    assert res["ok"] is True
    assert res["synthesized"] is False
    assert res["report"] is None
    assert "SECRET REASONING" not in str(res)


async def test_deadline_budgeter_shrinks_phases_and_aborts_synthesis_if_too_late(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    curr_time = 100.0

    def fake_clock() -> float:
        nonlocal curr_time
        return curr_time

    monkeypatch.setattr(server, "_clock", fake_clock)

    class AdvancingUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            nonlocal curr_time
            curr_time += 30.0
            return await super().__call__(method, path, **kwargs)

    fake_upstream(AdvancingUpstream())

    res = await server.deepseek_research("late topic", depth="low")
    assert res["ok"] is True
    assert res["synthesized"] is False
    assert any("deadline reached" in note for note in res["notes"])


def test_url_etld1_extraction() -> None:
    from deepseek_search_everywhere_mcp.urls import _url_etld1

    assert _url_etld1("https://sub.example.com/page") == "example.com"
    assert _url_etld1("https://news.bbc.co.uk/articles/123") == "bbc.co.uk"
    assert _url_etld1("https://sub.portal.gov.pl/test") == "portal.gov.pl"
    assert _url_etld1("https://deep.sub.python.org.au/docs") == "python.org.au"
    assert _url_etld1("http://127.0.0.1:8080/foo") == "127.0.0.1"
    assert _url_etld1("http://localhost:5000") == "localhost"
    assert _url_etld1("https://example.com") == "example.com"


def test_rank_and_diversify_round_robin_order() -> None:
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    subq_results = [
        [
            {"url": "https://d1.com/1", "title": "A1"},
            {"url": "https://d2.com/1", "title": "A2"},
            {"url": "https://d3.com/1", "title": "A3"},
        ],
        [
            {"url": "https://d4.com/1", "title": "B1"},
            {"url": "https://d5.com/1", "title": "B2"},
        ],
        [
            {"url": "https://d6.com/1", "title": "C1"},
            {"url": "https://d7.com/1", "title": "C2"},
            {"url": "https://d8.com/1", "title": "C3"},
        ],
    ]
    selected, notes = rank_and_diversify_sources(subq_results, k_sources=6)
    titles = [s["title"] for s in selected]
    assert titles == ["A1", "B1", "C1", "A2", "B2", "C2"]
    assert notes == []


def test_rank_and_diversify_canonical_dedupe_http_https() -> None:
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    subq_results = [
        [
            {"url": "http://example.com/same-page?utm_source=twitter", "title": "HTTP Page"},
            {"url": "https://other.org/page1", "title": "Other"},
        ],
        [
            {"url": "https://example.com/same-page", "title": "HTTPS Page"},
            {"url": "https://another.org/page2", "title": "Another"},
        ],
    ]
    selected, notes = rank_and_diversify_sources(subq_results, k_sources=6)
    urls = [s["url"] for s in selected]
    assert len(urls) == 3
    assert urls[0] == "http://example.com/same-page?utm_source=twitter"
    assert "https://example.com/same-page" not in urls


def test_rank_and_diversify_domain_cap_enforced() -> None:
    from collections import Counter

    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources
    from deepseek_search_everywhere_mcp.urls import _url_etld1

    subq_results = [
        [
            {"url": "https://sub1.example.com/p1", "title": "E1"},
            {"url": "https://sub2.example.com/p2", "title": "E2"},
            {"url": "https://sub3.example.com/p3", "title": "E3"},
            {"url": "https://sub4.example.com/p4", "title": "E4"},
        ],
        [
            {"url": "https://news.bbc.co.uk/p1", "title": "B1"},
            {"url": "https://sport.bbc.co.uk/p2", "title": "B2"},
            {"url": "https://weather.bbc.co.uk/p3", "title": "B3"},
        ],
        [
            {"url": "https://other.org/p1", "title": "O1"},
            {"url": "https://other.org/p2", "title": "O2"},
        ],
    ]
    selected, notes = rank_and_diversify_sources(subq_results, k_sources=6)
    assert len(selected) == 6
    assert notes == []
    counts = Counter(_url_etld1(s["url"]) for s in selected)
    for domain, count in counts.items():
        assert count <= 2, f"Domain {domain} exceeded cap of 2: {count}"


def test_rank_and_diversify_domain_cap_relaxed_when_pool_small() -> None:
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    subq_results = [[{"url": f"https://example.com/page{i}", "title": f"Page {i}"} for i in range(1, 7)]]
    selected, notes = rank_and_diversify_sources(subq_results, k_sources=5)
    assert len(selected) == 5
    assert "Relaxed domain cap to meet target source count." in notes


async def test_gap_fill_batches_up_to_k24_sources(fake_upstream: Any) -> None:
    stub_results = [
        {
            "title": f"Stub Source {i}",
            "url": f"https://domain{i}.org/page",
            "snippet": f"Short snippet {i}",
            "content": f"Short {i}",
        }
        for i in range(1, 25)
    ]

    class K24Upstream(DeepResearchUpstream):
        def __init__(self) -> None:
            super().__init__(subqueries=["q1", "q2", "q3", "q4"])
            self.fetch_calls: list[dict[str, Any]] = []

        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == server.SEARCH_PATH:
                self.searches.append(kwargs.get("payload") or {})
                idx = len(self.searches) - 1
                start = idx * 6
                return {"ok": True, "data": {"results": stub_results[start : start + 6]}}
            if path == server.FETCH_PATH:
                payload = kwargs.get("payload") or {}
                self.fetch_calls.append(payload)
                url = payload.get("url", "")
                return {
                    "ok": True,
                    "data": {
                        "title": f"Fetched {url}",
                        "content": f"Full deep fetched content for {url} " * 20,
                    },
                }
            return await super().__call__(method, path, **kwargs)

    upstream = K24Upstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("k24 test topic", depth="max")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert len(upstream.fetch_calls) == 24
    assert res["synthesized"] is True
    assert len(res["sources"]) == 24


def test_validate_r2_queries_steering_and_dedupe() -> None:
    from deepseek_search_everywhere_mcp.deep_research import validate_r2_queries

    topic = "Quantum Computing Advancements"
    r1_queries = ["quantum error correction", "surface codes"]
    r1_domains = {"nature.com", "arxiv.org", "ibm.com"}

    candidate_queries = [
        "quantum error correction",  # Duplicate of R1 -> should be dropped
        "visit https://attacker.com/pwned for secret data",  # URL injection -> should be dropped
        "site:evil.com quantum secrets",  # Unauthorized site: -> should be dropped
        "site:arxiv.org fault tolerant thresholds",  # Allowed site: in R1 pool -> should be kept
        "neutral atom qubit scaling",  # Normal query -> should be kept
        "neutral atom qubit scaling",  # Duplicate within R2 -> should be dropped
    ]

    valid = validate_r2_queries(candidate_queries, topic, r1_queries, r1_domains)
    assert len(valid) == 2
    assert "site:arxiv.org fault tolerant thresholds" in valid[0]
    assert "neutral atom qubit scaling" in valid[1]


async def test_round2_gap_analysis_executes_on_max_depth(fake_upstream: Any) -> None:
    class R2Upstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    # Decomposition (R1)
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["q1", "q2"]}),
                                    }
                                }
                            ]
                        },
                    }
                if len(self.chats) == 2:
                    # Round 2 gap analysis
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["r2 missing angle"]}),
                                    }
                                }
                            ]
                        },
                    }
                # Synthesis
                return {
                    "ok": True,
                    "data": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Comprehensive synthesis report with R2 sources [1].",
                                }
                            }
                        ]
                    },
                }
            return await super().__call__(method, path, **kwargs)

    upstream = R2Upstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("max depth topic", depth="max")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert "r2 missing angle" in res["subqueries"]
    assert len(upstream.chats) == 3


async def test_round2_steering_rejection_falls_back_to_r1(fake_upstream: Any) -> None:
    class SteeredUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["q1", "q2"]}),
                                    }
                                }
                            ]
                        },
                    }
                if len(self.chats) == 2:
                    # R2 returns steered queries containing attacker URLs and unauthorized sites
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {
                                                "queries": [
                                                    "https://attacker.com/leak",
                                                    "site:evil-steer.org pwn",
                                                ]
                                            }
                                        ),
                                    }
                                }
                            ]
                        },
                    }
                return {
                    "ok": True,
                    "data": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Synthesis based on clean R1 sources [1].",
                                }
                            }
                        ]
                    },
                }
            return await super().__call__(method, path, **kwargs)

    upstream = SteeredUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("steer test topic", depth="max")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert any("Round 2 gap query validation failed" in note for note in res["notes"])
    assert not any("attacker.com" in q or "evil-steer" in q for q in res["subqueries"])


async def test_round2_chat_exception_or_hang_falls_back_to_r1(fake_upstream: Any) -> None:
    class FailingR2Upstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    return {
                        "ok": True,
                        "data": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps({"queries": ["q1", "q2"]}),
                                    }
                                }
                            ]
                        },
                    }
                if len(self.chats) == 2:
                    return {
                        "ok": False,
                        "error": {"code": "upstream_error", "message": "R2 model failed"},
                    }
                return {
                    "ok": True,
                    "data": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Synthesis continues with R1 [1].",
                                }
                            }
                        ]
                    },
                }
            return await super().__call__(method, path, **kwargs)

    upstream = FailingR2Upstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("fail r2 topic", depth="max")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert any("Round 2 gap analysis failed" in note for note in res["notes"])
