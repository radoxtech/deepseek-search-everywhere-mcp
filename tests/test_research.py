"""Research tool tests: multi-angle research with depth tiers and briefing report."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepseek_search_everywhere_mcp import runtime, server
from deepseek_search_everywhere_mcp.deep_research import get_job_store
from deepseek_search_everywhere_mcp.settings import LLMPricing, LLMRates, Settings


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
                "url": "https://physics.org/qec",
                "content": "Surface codes provide fault tolerance in quantum computing. " * 5,
            },
            {
                "title": "Qubit Fidelity Benchmarks",
                "url": "https://physics.org/benchmarks",
                "content": "Two-qubit gate fidelities exceed 99.9% in recent trials. " * 5,
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
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.05))

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
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=10.0))

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


async def test_running_research_progress_carries_timing(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.05))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.15)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())
    res = await server.deepseek_research("timing topic", depth="high")
    assert res["status"] == "running"
    progress = res["progress"]
    assert {"phase", "step", "of"} <= progress.keys()
    assert progress["deadline_ms"] == int(server.DEPTH_CONFIG["high"].deadline_s * 1000)
    assert progress["phase_elapsed_ms"] >= 0


async def test_progress_reports_a_synthesis_sub_step(fake_upstream: Any) -> None:
    job = server.ResearchJob(job_id="j", topic="t", depth="low", lang=None, created_at=0.0)
    await server._report_job_progress(job, "synthesis", 4, 4, detail="retry")
    view = server._progress_view(job)
    assert (view["phase"], view["detail"]) == ("synthesis", "retry")
    assert "phase_elapsed_ms" in view


async def test_research_duplicate_joins_running_job(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.05))

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
        runtime,
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
    assert "At most 3 research jobs" in res4["error"]["message"]
    assert "deepseek_check_research" in res4["error"]["message"]


async def test_check_research_long_poll_returns_polled_done(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.05))

    class SlowUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == "/v1/chat/completions":
                await asyncio.sleep(0.08)
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SlowUpstream())

    running_env = await server.deepseek_research("check poll topic", depth="low")
    assert running_env["status"] == "running"
    job_id = running_env["job_id"]

    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.20))
    polled_env = await server.deepseek_check_research(job_id)
    assert polled_env["ok"] is True
    assert polled_env["status"] == "done"
    assert polled_env["delivery"] == "polled"
    assert polled_env["topic"] == "check poll topic"


async def test_check_research_returns_running_when_still_not_done(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.02))

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

    monkeypatch.setattr(runtime, "_clock", lambda: job.completed_at + 1801.0)
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

    monkeypatch.setattr(runtime, "_clock", fake_clock)

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


async def test_synthesis_cost_ceiling_aborts_and_returns_sources(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(research_cost_ceiling_usd=0.0005)
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)

    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("ceiling test topic", depth="low")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert res["synthesized"] is False
    assert res["report"] is None
    assert any("exceeds ceiling" in n for n in res["notes"])
    assert res.get("derived_from_untrusted") is True
    assert res.get("injection_suspected") == 0


async def test_research_report_has_no_references_section(fake_upstream: Any) -> None:
    report = "Findings [1][2].\n\n## Sources\n\n[1] Fake - https://fake.test\n"
    fake_upstream(DeepResearchUpstream(report_text=report))
    res = await server.deepseek_research("no references topic", depth="low")
    assert res["synthesized"] is True
    assert "References" not in res["report"]
    assert "fake.test" not in res["report"]
    assert res["report"].strip() == "Findings [1][2]."
    assert [s["id"] for s in res["sources"]] == [1, 2]
    assert all(s["url"] for s in res["sources"])


def test_research_tool_description_describes_sources_not_references() -> None:
    desc = server.deepseek_research.__doc__ or ""
    assert "References" not in desc
    assert "sources" in desc and "ids" in desc


async def test_lean_envelope_contains_no_page_content_on_success(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research("quantum computing scaling", depth="low")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert "raw_sources" not in res
    assert res["partial"] is False
    assert len(res["sources"]) > 0
    for s in res["sources"]:
        assert "content" not in s
        assert "snippet" not in s
        assert set(s.keys()) == {
            "id",
            "title",
            "url",
            "date",
            "doi",
            "pmid",
            "pmcid",
            "access",
            "source_type",
            "authors",
        }

    serialized = json.dumps(res)
    assert len(serialized) < 50_000


async def test_lean_envelope_contains_no_page_content_on_all_fallbacks(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Synthesis model call failure fallback
    upstream = DeepResearchUpstream(fail_synthesis=True)
    fake_upstream(upstream)

    res1 = await server.deepseek_research("quantum topic", depth="low")
    assert res1["ok"] is True
    assert res1["synthesized"] is False
    assert res1["partial"] is True
    assert "raw_sources" not in res1
    for s in res1["sources"]:
        assert "content" not in s
        assert "snippet" in s
        assert len(s["snippet"]) <= 400
    assert len(json.dumps(res1)) < 50_000

    # 2. Cost ceiling exceeded fallback
    settings = Settings(research_cost_ceiling_usd=0.0005)
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)
    upstream2 = DeepResearchUpstream()
    fake_upstream(upstream2)

    res2 = await server.deepseek_research("quantum ceiling topic", depth="low")
    assert res2["ok"] is True
    assert res2["synthesized"] is False
    assert res2["partial"] is True
    assert "raw_sources" not in res2
    for s in res2["sources"]:
        assert "content" not in s
        assert "snippet" in s
        assert len(s["snippet"]) <= 400
    assert len(json.dumps(res2)) < 50_000


async def test_research_progress_reporting_with_context(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    ctx = MagicMock()
    ctx.report_progress = AsyncMock()

    res = await server.deepseek_research("quantum computing scaling", depth="low", ctx=ctx)
    assert res["ok"] is True
    assert ctx.report_progress.await_count == 4
    calls = ctx.report_progress.await_args_list
    assert calls[0].args == (1.0, 4.0)
    assert calls[1].args == (2.0, 4.0)
    assert calls[2].args == (3.0, 4.0)
    assert calls[3].args == (4.0, 4.0)


async def test_research_progress_reporting_with_context_max_depth(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream(subqueries=["subq1", "subq2", "subq3", "subq4"])
    fake_upstream(upstream)

    ctx = MagicMock()
    ctx.report_progress = AsyncMock()

    res = await server.deepseek_research("quantum max topic", depth="max", ctx=ctx)
    assert res["ok"] is True
    assert ctx.report_progress.await_count == 5
    calls = ctx.report_progress.await_args_list
    phases = [c.kwargs.get("message") for c in calls]
    assert any("Decomposing" in p for p in phases)
    assert any("Searching" in p for p in phases)
    assert any("Analyzing gaps" in p for p in phases)
    assert any("Fetching gap pages" in p for p in phases)
    assert any("Synthesizing" in p for p in phases)


async def test_research_progress_reporting_resilient_to_context_exceptions(fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    ctx = MagicMock()
    ctx.report_progress = AsyncMock(side_effect=RuntimeError("Client stream closed"))

    res = await server.deepseek_research("quantum resilient progress", depth="low", ctx=ctx)
    assert res["ok"] is True
    assert res["synthesized"] is True


@pytest.mark.parametrize("depth", ["low", "medium", "high", "max"])
async def test_fake_clock_44s_inline_for_every_depth(depth: str, fake_upstream: Any) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    res = await server.deepseek_research(f"inline test {depth}", depth=depth)
    assert res["ok"] is True
    assert res["status"] == "done"
    assert res["delivery"] == "inline"
    assert res["depth"] == depth


@pytest.mark.parametrize("depth", ["low", "medium", "high", "max"])
async def test_fake_clock_46s_running_for_every_depth(
    depth: str, fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    async def mock_wait_timeout(tasks: Any, timeout: float | None = None) -> tuple[set[Any], set[Any]]:
        return set(), set(tasks)

    monkeypatch.setattr(asyncio, "wait", mock_wait_timeout)

    res = await server.deepseek_research(f"running test {depth}", depth=depth)
    assert res["ok"] is True
    assert res["status"] == "running"
    assert res["depth"] == depth
    assert "job_id" in res
    assert "Call deepseek_check_research" in res["next_action"]

    job = get_job_store().get(res["job_id"])
    assert job is not None and job.task is not None
    await job.task


def _install_counting_pipeline(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_pipeline(*, job: Any, clean_topic: str, depth: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"topic": clean_topic, "depth": depth, **kwargs})
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "report": f"report {len(calls)}",
            "synthesized": True,
            "sources": [],
            "cached": False,
            "cost_usd": 0.0123,
            "cost_breakdown": {"search": 0.001, "fetch": 0.001, "llm": 0.0103},
            "_meta": {"cost_usd": 0.0123, "untrusted_content": True},
        }

    monkeypatch.setattr(server, "_execute_research_pipeline", fake_pipeline)
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=5.0))
    return calls


async def test_research_finished_reuse_returns_cached_zero_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_counting_pipeline(monkeypatch)

    first = await server.deepseek_research("Reuse topic", depth="low")
    assert first["status"] == "done"
    assert first["cost_usd"] == 0.0123
    assert first.get("cached") is False

    second = await server.deepseek_research("reuse topic ", depth="low")
    assert len(calls) == 1
    assert second["ok"] is True
    assert second["status"] == "done"
    assert second["delivery"] == "inline"
    assert second["cached"] is True
    assert second["cost_usd"] == 0
    assert second["cost_breakdown"] == {"search": 0, "fetch": 0, "llm": 0}
    assert second["_meta"]["cost_usd"] == 0
    assert second["report"] == first["report"]
    # the stored result itself is untouched
    assert first["cost_usd"] == 0.0123

    other_depth = await server.deepseek_research("reuse topic", depth="medium")
    assert other_depth["report"] == "report 2"
    assert len(calls) == 2


async def test_research_refresh_true_starts_new_job(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_counting_pipeline(monkeypatch)

    first = await server.deepseek_research("refresh topic", depth="low")
    refreshed = await server.deepseek_research("refresh topic", depth="low", refresh=True)
    assert len(calls) == 2
    assert calls[1]["refresh"] is True
    assert refreshed["report"] == "report 2"
    assert refreshed["cost_usd"] == 0.0123
    assert refreshed.get("cached") is False
    assert first["report"] == "report 1"


async def test_research_finished_result_not_reused_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_counting_pipeline(monkeypatch)
    monkeypatch.setattr(
        runtime, "get_settings", lambda: Settings(research_inline_wait=5.0, research_job_ttl=100.0)
    )
    now = [1000.0]
    monkeypatch.setattr(runtime, "_clock", lambda: now[0])

    await server.deepseek_research("ttl topic", depth="low")
    now[0] += 99.0
    reused = await server.deepseek_research("ttl topic", depth="low")
    assert reused["cached"] is True
    assert len(calls) == 1

    now[0] += 200.0
    fresh = await server.deepseek_research("ttl topic", depth="low")
    assert len(calls) == 2
    assert fresh["report"] == "report 2"


async def test_synthesis_retry_cost_includes_failed_first_attempt(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_upstream(DeepResearchUpstream())
    synth_calls: list[dict[str, Any]] = []

    async def fake_chat_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("max_tokens") == 1024:
            # Decomposition call: zero-cost, valid subqueries.
            return {
                "ok": True,
                "content": json.dumps({"queries": ["q1"]}),
                "reasoning_content": None,
                "finish_reason": "stop",
                "usage": {},
                "cost_usd": 0.0,
                "pricing_period": "peak",
                "notes": [],
                "raw": {},
            }
        synth_calls.append(kwargs)
        if len(synth_calls) == 1:
            return {
                "ok": True,
                "content": "",
                "reasoning_content": "internal",
                "finish_reason": "length",
                "usage": {},
                "cost_usd": 0.01,
                "pricing_period": "peak",
                "notes": [],
                "raw": {},
            }
        return {
            "ok": True,
            "content": "Retry report [1].",
            "reasoning_content": None,
            "finish_reason": "stop",
            "usage": {},
            "cost_usd": 0.002,
            "pricing_period": "peak",
            "notes": [],
            "raw": {},
        }

    monkeypatch.setattr(runtime, "_chat_request", fake_chat_request)
    res = await server.deepseek_research("cost retry topic", depth="medium")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert len(synth_calls) == 2
    assert res["cost_breakdown"]["llm"] == pytest.approx(0.012, abs=1e-6)
    breakdown = res["cost_breakdown"]
    assert res["cost_usd"] == pytest.approx(breakdown["search"] + breakdown["fetch"] + 0.012, abs=1e-6)


async def test_research_job_error_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"

    async def failing_pipeline(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"upstream rejected key {secret}")

    monkeypatch.setattr(server, "_execute_research_pipeline", failing_pipeline)
    res = await server.deepseek_research("redaction topic", depth="low")
    assert res["ok"] is False
    assert secret not in json.dumps(res)


async def test_research_cost_ceiling_uses_configured_pricing(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    expensive = LLMRates(cache_hit_per_1m=0.0, cache_miss_per_1m=1_000_000.0, output_per_1m=1_000_000.0)
    settings = Settings(
        research_cost_ceiling_usd=1.0,
        llm_pricing=LLMPricing(peak=expensive, off_peak=expensive),
    )
    monkeypatch.setattr(runtime, "get_settings", lambda: settings)
    fake_upstream(DeepResearchUpstream())

    res = await server.deepseek_research("configured pricing topic", depth="low")
    assert res["ok"] is True
    assert res["synthesized"] is False
    assert any("exceeds ceiling" in n for n in res["notes"])


async def test_research_job_drops_ctx_when_call_returns(fake_upstream: Any) -> None:
    fake_upstream(DeepResearchUpstream())

    class FakeCtx:
        async def report_progress(self, *args: Any, **kwargs: Any) -> None:
            return None

    res = await server.deepseek_research("ctx release topic", depth="low", ctx=FakeCtx())
    assert res["ok"] is True
    jobs = list(get_job_store(ttl=1800.0, max_active=3)._jobs.values())
    assert jobs and all(job.ctx is None for job in jobs)


def _chat_ok(content: str, cost: float = 0.0, finish: str = "stop") -> dict[str, Any]:
    return {
        "ok": True,
        "content": content,
        "reasoning_content": None,
        "finish_reason": finish,
        "usage": {},
        "cost_usd": cost,
        "pricing_period": "peak",
        "notes": [],
        "raw": {},
    }


def _install_condense_chat(
    monkeypatch: pytest.MonkeyPatch, long_report: str, condense_result: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Fake chat: decomposition, then synthesis (``long_report``), then the condense call."""
    calls: list[dict[str, Any]] = []

    async def fake_chat_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs.get("max_tokens") == 1024:
            return _chat_ok(json.dumps({"queries": ["q1"]}))
        if len(calls) == 2:
            return _chat_ok(long_report, cost=0.01)
        assert condense_result is not None, "unexpected extra chat call"
        return condense_result

    monkeypatch.setattr(runtime, "_chat_request", fake_chat_request)
    return calls


def _long_report(words: int) -> str:
    return "## Findings\n\n" + ("alpha beta gamma delta " * (words // 4)).strip() + " [1][2]."


async def test_overlength_report_is_condensed(fake_upstream: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_upstream(DeepResearchUpstream())
    short = "## Findings\n\nShort and dense [1][2]."
    calls = _install_condense_chat(monkeypatch, _long_report(600), _chat_ok(short, cost=0.003))
    res = await server.deepseek_research("condense topic", depth="low")  # ceiling 480 words
    assert res["ok"] is True and res["synthesized"] is True
    assert "Short and dense [1][2]." in res["report"]
    assert len(calls) == 3
    cond = calls[2]
    assert cond["thinking"] == {"type": "disabled"}
    assert cond["temperature"] == 0.0
    assert "response_format" not in cond
    system = cond["messages"][0]["content"]
    assert "400" in system and "[n]" in system
    assert any("condensed" in n.lower() for n in res["notes"])
    assert res["cost_breakdown"]["llm"] == pytest.approx(0.013, abs=1e-6)


async def test_report_within_ceiling_is_not_condensed(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_upstream(DeepResearchUpstream())
    calls = _install_condense_chat(monkeypatch, _long_report(440), None)
    res = await server.deepseek_research("no condense topic", depth="low")
    assert res["synthesized"] is True
    assert len(calls) == 2
    assert not any("condensed" in n.lower() for n in res["notes"])


async def test_condense_failure_keeps_original_with_note(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_upstream(DeepResearchUpstream())
    failed = {"ok": False, "error": {"code": "internal_error", "message": "boom"}, "cost_usd": 0.0}
    _install_condense_chat(monkeypatch, _long_report(600), failed)
    res = await server.deepseek_research("condense fail topic", depth="low")
    assert res["synthesized"] is True
    assert "alpha beta gamma" in res["report"]
    assert any("condens" in n.lower() and "kept" in n.lower() for n in res["notes"])


@pytest.mark.parametrize(
    "bad",
    [
        _chat_ok("", cost=0.002),
        _chat_ok("Invented a new citation [7].", cost=0.002),
        _chat_ok(_long_report(700), cost=0.002),
    ],
    ids=["empty", "new-citation-id", "not-shorter"],
)
async def test_condense_rejected_results_keep_original_and_bill_cost(
    bad: dict[str, Any], fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_upstream(DeepResearchUpstream())
    _install_condense_chat(monkeypatch, _long_report(600), bad)
    res = await server.deepseek_research("condense reject topic", depth="low")
    assert "alpha beta gamma" in res["report"]
    assert any("kept" in n.lower() for n in res["notes"])
    assert res["cost_breakdown"]["llm"] == pytest.approx(0.012, abs=1e-6)


async def test_research_sources_have_clean_urls_titles_and_dates(fake_upstream: Any) -> None:
    results = [
        {
            "title": "<table><tr><td>Target</td><td>Properties</td></tr></table>",
            "url": "https://dl.acm.org/doi/pdf/10.1145/1?download=true&__cf_chl_tk=abc&x=1#5#6#5",
            "content": "Surface codes provide fault tolerance in quantum computing. " * 5,
            "date": "2026-02-03",
        },
        {
            "title": "**Refinement based safety proofs",
            "url": "https://browse-export.arxiv.org/pdf/2509.07157#4#11#4",
            "content": "Two-qubit gate fidelities exceed 99.9% in recent trials. " * 5,
            "date": "2026-01-10",
        },
    ]
    fake_upstream(DeepResearchUpstream(search_results=results))
    res = await server.deepseek_research("clean sources topic", depth="low")
    assert res["synthesized"] is True
    by_id = {s["id"]: s for s in res["sources"]}
    urls = {s["url"] for s in by_id.values()}
    assert urls == {
        "https://dl.acm.org/doi/pdf/10.1145/1?download=true&x=1",
        "https://browse-export.arxiv.org/pdf/2509.07157",
    }
    titles = {s["title"] for s in by_id.values()}
    assert titles == {"dl.acm.org", "Refinement based safety proofs"}
    assert {s["date"] for s in by_id.values()} == {"2026-02-03", "2026-01-10"}


async def test_research_pipeline_skips_denylisted_sources_and_honours_empty_denylist(
    fake_upstream: Any,
) -> None:
    from deepseek_search_everywhere_mcp.settings import Settings, set_settings

    results = [
        {"title": "Feed", "url": "https://lib.ezproxy.uni.edu/entry/1", "content": "Proxy text. " * 20},
        {"title": "Real", "url": "https://physics.org/qec", "content": "Surface codes text. " * 20},
    ]
    set_settings(Settings(source_denylist=("*.ezproxy.*",)))
    fake_upstream(DeepResearchUpstream(search_results=results))
    res = await server.deepseek_research("denylist topic", depth="low")
    assert [s["url"] for s in res["sources"]] == ["https://physics.org/qec"]
    assert any("denylist" in n.lower() for n in res["notes"])

    set_settings(Settings(source_denylist=()))
    fake_upstream(DeepResearchUpstream(search_results=results))
    res = await server.deepseek_research("denylist topic two", depth="low")
    assert {s["url"] for s in res["sources"]} == {
        "https://lib.ezproxy.uni.edu/entry/1",
        "https://physics.org/qec",
    }


async def test_check_research_done_reports_elapsed_since_job_start(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(runtime, "_clock", lambda: clock["now"])
    gate = asyncio.Event()

    async def gated_pipeline(*, job: Any, clean_topic: str, depth: str, **kwargs: Any) -> dict[str, Any]:
        await gate.wait()
        return {
            "ok": True,
            "status": "done",
            "topic": clean_topic,
            "depth": depth,
            "report": "polled report",
            "synthesized": True,
            "sources": [],
            "cached": False,
            "cost_usd": 0.0,
            "cost_breakdown": {"search": 0.0, "fetch": 0.0, "llm": 0.0},
            "_meta": {"cost_usd": 0.0, "untrusted_content": True},
        }

    monkeypatch.setattr(server, "_execute_research_pipeline", gated_pipeline)
    monkeypatch.setattr(runtime, "get_settings", lambda: Settings(research_inline_wait=0.01))

    started = await server.deepseek_research("Elapsed polled topic", depth="low")
    assert started["status"] == "running"

    clock["now"] += 7.5
    gate.set()
    job = get_job_store().get(started["job_id"])
    assert job is not None and job.task is not None
    await job.task
    clock["now"] += 0.5

    res = await server.deepseek_check_research(started["job_id"])
    assert res["status"] == "done"
    assert res["delivery"] == "polled"
    assert res["elapsed_ms"] == 8000


async def test_research_sources_expose_identifiers_and_access(fake_upstream: Any) -> None:
    results = [
        {
            "title": "PubMed paper",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38123456/",
            "content": "PubMed abstract text. " * 20,
        },
        {
            "title": "PMC article",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10123456/",
            "content": "PMC full text body. " * 20,
        },
        {
            "title": "Plain",
            "url": "https://plain.example/page",
            "content": "Short.",
        },
    ]
    fake_upstream(DeepResearchUpstream(search_results=results))
    res = await server.deepseek_research("identifiers topic", depth="low")
    by_title = {s["title"]: s for s in res["sources"]}

    pubmed = by_title["PubMed paper"]
    assert (pubmed["doi"], pubmed["pmid"], pubmed["pmcid"]) == (None, "38123456", None)
    assert pubmed["access"] == "abstract"

    pmc = by_title["PMC article"]
    assert (pmc["doi"], pmc["pmid"], pmc["pmcid"]) == (None, None, "PMC10123456")
    assert pmc["access"] == "abstract"

    plain = by_title["Plain"]
    assert (plain["doi"], plain["pmid"], plain["pmcid"]) == (None, None, None)
    assert plain["access"] == "full_text"


async def test_research_flags_unverified_numbers_in_envelope(fake_upstream: Any) -> None:
    results = [
        {
            "title": "Bremer 2025 Study",
            "url": "https://example.com/1",
            "content": "Bremer evaluated 4 RCTs with 200 participants. " * 5,
        },
        {
            "title": "Garcia 2024 Study",
            "url": "https://example.com/2",
            "content": "Garcia included 11 studies with 934 participants. " * 5,
        },
    ]
    report = (
        "Bremer reported 11 studies with 934 participants [1]. "
        "Garcia evaluated 11 studies with 934 participants [2]."
    )
    fake_upstream(DeepResearchUpstream(search_results=results, report_text=report))
    res = await server.deepseek_research("numbers topic", depth="low")
    assert res["ok"] is True
    assert res["synthesized"] is True
    assert res["unverified_number_count"] == 2
    assert len(res["unverified_numbers"]) == 2
    assert "[unverified]" in res["report"]
    assert any("unverified number(s)" in n for n in res["notes"])


async def test_research_drops_unverified_sentences_on_request(fake_upstream: Any) -> None:
    results = [
        {
            "title": "Bremer 2025 Study",
            "url": "https://example.com/1",
            "content": "Bremer evaluated 4 RCTs with 200 participants. " * 5,
        },
    ]
    report = "Bremer reported 11 studies [1]. Bremer evaluated 4 RCTs with 200 participants [1]."
    fake_upstream(DeepResearchUpstream(search_results=results, report_text=report))
    res = await server.deepseek_research("drop topic", depth="low", unverified="drop")
    assert res["ok"] is True
    assert "11 studies" not in res["report"]
    assert "200 participants" in res["report"]
    assert "[unverified]" not in res["report"]
    assert res["unverified_number_count"] == 1
    assert res["unverified_numbers"][0]["number"] == "11"
    assert any("Removed 1 sentence(s)" in n for n in res["notes"])


async def test_research_unverified_mode_is_part_of_the_job_key(fake_upstream: Any) -> None:
    results = [{"title": "T", "url": "https://example.com/1", "content": "Value is 4. " * 30}]
    fake_upstream(DeepResearchUpstream(search_results=results, report_text="Value is 9 [1]."))
    flagged = await server.deepseek_research("key topic", depth="low")
    dropped = await server.deepseek_research("key topic", depth="low", unverified="drop")
    assert "[unverified]" in flagged["report"]
    assert dropped.get("cached") is not True
    assert "9" not in (dropped["report"] or "")


async def test_research_rejects_an_unknown_unverified_mode() -> None:
    res = await server.deepseek_research("topic", unverified="hide")  # type: ignore[arg-type]
    assert res["error"]["code"] == "bad_request"


async def test_research_sources_preference_and_type(fake_upstream: Any) -> None:
    results = [
        {
            "title": "Randomized controlled trial of therapy",
            "url": "https://pubmed.ncbi.nlm.nih.gov/38123456/",
            "content": "A randomized controlled trial evaluating recovery. " * 5,
        },
    ]
    fake_upstream(DeepResearchUpstream(search_results=results))
    res = await server.deepseek_research("therapy", depth="low", sources="academic")
    assert res["ok"] is True
    assert len(res["sources"]) > 0
    s0 = res["sources"][0]
    assert "source_type" in s0
    assert s0["source_type"] == "rct"

    res_invalid = await server.deepseek_research("therapy", depth="low", sources="invalid")
    assert res_invalid["ok"] is False
    assert res_invalid["error"]["code"] == "bad_request"


async def test_research_debug_dump_hook(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    # 1. When DEEPSEEK_MCP_DEBUG_DUMP_DIR is unset: no dump created
    monkeypatch.delenv("DEEPSEEK_MCP_DEBUG_DUMP_DIR", raising=False)
    res = await server.deepseek_research("dump test unset", depth="low")
    assert res["ok"] is True
    assert list(tmp_path.iterdir()) == []

    # 2. When DEEPSEEK_MCP_DEBUG_DUMP_DIR is set: dump created
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DEEPSEEK_MCP_DEBUG_DUMP_DIR", str(dump_dir))
    res2 = await server.deepseek_research("dump test set", depth="low")
    assert res2["ok"] is True
    dump_files = list(dump_dir.glob("*.json"))
    assert len(dump_files) == 1
    dump_data = json.loads(dump_files[0].read_text(encoding="utf-8"))
    assert dump_data["topic"] == "dump test set"
    assert dump_data["depth"] == "low"
    assert isinstance(dump_data["sources"], list)
    assert isinstance(dump_data["synthesis_context"], list)
    assert len(dump_data["synthesis_context"]) > 0
    sc0 = dump_data["synthesis_context"][0]
    assert "id" in sc0 and "url" in sc0 and "title" in sc0 and "content" in sc0
    assert "report" in dump_data
    assert "raw_report" in dump_data or "raw_model_report" in dump_data
    assert "unverified_numbers" in dump_data

    # 3. When dump write fails: job still succeeds (failure tolerated)
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("block", encoding="utf-8")
    invalid_dump_dir = blocking_file / "sub"
    monkeypatch.setenv("DEEPSEEK_MCP_DEBUG_DUMP_DIR", str(invalid_dump_dir))
    res3 = await server.deepseek_research("dump test fail", depth="low")
    assert res3["ok"] is True


def _chat_reply(content: str) -> dict[str, Any]:
    return {"ok": True, "data": {"choices": [{"message": {"role": "assistant", "content": content}}]}}


async def test_research_without_readable_content_returns_sources_only(fake_upstream: Any) -> None:
    class EmptyPages(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == server.FETCH_PATH:
                return {"ok": False, "error": {"code": "not_found", "message": "gone", "status": 404}}
            return await super().__call__(method, path, **kwargs)

    fake_upstream(EmptyPages(search_results=[{"title": "Empty", "url": "https://a.test/1", "content": ""}]))
    res = await server.deepseek_research("empty pages topic", depth="low")
    assert res["ok"] is True and res["synthesized"] is False and res["report"] is None
    assert res["partial"] is True
    assert res["notes"][-1] == "No readable source content available for synthesis."


async def test_research_stops_before_synthesis_when_the_deadline_is_near(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server._ResearchRun, "remaining", lambda self: 1.0)
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)
    res = await server.deepseek_research("deadline topic", depth="low")
    assert res["synthesized"] is False
    assert res["notes"][-1] == "Synthesis deadline reached; returning gathered sources."
    assert len(upstream.chats) == 1  # decomposition only, no synthesis call


async def test_round_two_is_skipped_without_deadline_budget(
    fake_upstream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server._ResearchRun, "remaining", lambda self: 30.0)
    fake_upstream(DeepResearchUpstream())
    res = await server.deepseek_research("skip round two topic", depth="max")
    assert res["synthesized"] is True
    assert "Round 2 skipped due to deadline budget." in res["notes"]


async def test_round_two_without_search_results_keeps_round_one(fake_upstream: Any) -> None:
    class FailingR2Search(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            payload = kwargs.get("payload") or {}
            if path == server.SEARCH_PATH and "gap angle" in payload.get("query", ""):
                return {"ok": False, "error": {"code": "upstream_error", "message": "down", "status": 502}}
            if path == "/v1/chat/completions" and len(self.chats) == 1:
                self.chats.append(payload)
                return _chat_reply(json.dumps({"queries": ["gap angle one", "gap angle two"]}))
            return await super().__call__(method, path, **kwargs)

    upstream = FailingR2Search()
    fake_upstream(upstream)
    res = await server.deepseek_research("round two search topic", depth="max")
    assert res["synthesized"] is True
    assert "Round 2 search produced no results; proceeding with Round 1 sources." in res["notes"]
    assert not any("gap angle" in q for q in res["subqueries"])
