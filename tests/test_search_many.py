"""Unit and integration tests for deepseek_search_many."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server


def _install_fake_search(fake_upstream: Any, results_map: dict[str, list[dict[str, Any]]]) -> list[dict]:
    calls: list[dict[str, Any]] = []

    async def custom_request(
        method: str, path: str, *, payload: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append({"method": method, "path": path, "payload": payload})
        q = (payload or {}).get("query", "")
        for k, v in results_map.items():
            if k in q:
                return {"ok": True, "data": {"results": v}}
        return {"ok": True, "data": {"results": []}}

    fake_upstream(custom_request)
    return calls


@pytest.mark.parametrize(
    ("queries", "expected_code"),
    [
        ([], "bad_request"),
        (None, "bad_request"),
        (["q1", "q2", "q3", "q4", "q5", "q6"], "bad_request"),
        ([123], "bad_request"),
        (["   ", "  "], "bad_request"),
    ],
)
def test_search_many_validation_errors(queries, expected_code):
    result = asyncio.run(server.deepseek_search_many(queries))
    assert result["ok"] is False
    assert result["error"]["code"] == expected_code


def test_search_many_deduplicates_queries_preserving_order(fake_upstream):
    _install_fake_search(fake_upstream, {"a": [{"title": "A", "url": "https://a.test"}]})
    result = asyncio.run(server.deepseek_search_many(["a", "a", "a"]))
    assert result["ok"] is True
    assert result["count"] == 1
    assert any("deduped 2 duplicate query/queries" in note for note in result["notes"])
    assert result["results"][0]["query"] == "a"


def test_search_many_concurrent_success(fake_upstream):
    results_map = {
        "python": [{"title": "Python Docs", "url": "https://python.org"}],
        "asyncio": [{"title": "Asyncio Docs", "url": "https://asyncio.org"}],
    }
    _install_fake_search(fake_upstream, results_map)

    res = asyncio.run(server.deepseek_search_many(["python", "asyncio"], count=3))
    assert res["ok"] is True
    assert res["count"] == 2
    assert res["total_results"] == 2
    assert res["failed_count"] == 0
    assert res["partial"] is False
    assert res["requests_used"] == 2
    assert res["cached"] is False
    assert res["cost_usd"] == 0.0015
    assert res["untrusted_content"] is True
    assert "cost_usd" in res["_meta"]

    # searches is alias to results
    assert res["searches"] == res["results"]
    assert res["results"][0]["query"] == "python"
    assert res["results"][0]["count"] == 1
    assert res["results"][1]["query"] == "asyncio"
    assert res["results"][1]["count"] == 1


def test_search_many_partial_failure(fake_upstream):
    # One query exceeds MAX_QUERY (128 chars)
    long_q = "x" * 130
    results_map = {"valid": [{"title": "Valid", "url": "https://valid.test"}]}
    _install_fake_search(fake_upstream, results_map)

    res = asyncio.run(server.deepseek_search_many(["valid", long_q]))
    assert res["ok"] is True
    assert res["count"] == 2
    assert res["failed_count"] == 1
    assert res["partial"] is True

    valid_res = res["results"][0]
    assert valid_res["ok"] is True
    assert valid_res["count"] == 1

    failed_res = res["results"][1]
    assert failed_res["ok"] is False
    assert failed_res["error"]["code"] == "query_too_long"


def test_search_many_cache_and_refresh(fake_upstream):
    server.reset_state()
    results_map = {"cached_q": [{"title": "Cached", "url": "https://c.test"}]}
    _install_fake_search(fake_upstream, results_map)

    # 1st call -> upstream
    r1 = asyncio.run(server.deepseek_search_many(["cached_q"]))
    assert r1["ok"] is True
    assert r1["cached"] is False
    assert r1["requests_used"] == 1

    # 2nd call -> cached
    r2 = asyncio.run(server.deepseek_search_many(["cached_q"]))
    assert r2["ok"] is True
    assert r2["cached"] is True
    assert r2["cached_count"] == 1
    assert r2["requests_used"] == 0
    assert r2["cost_usd"] == 0.0

    # 3rd call with refresh=True -> upstream again
    r3 = asyncio.run(server.deepseek_search_many(["cached_q"], refresh=True))
    assert r3["ok"] is True
    assert r3["cached"] is False
    assert r3["requests_used"] == 1


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    [
        ({"count": 0}, "bad_count"),
        ({"count": 51}, "bad_count"),
        ({"max_chars_per_result": 0}, "bad_request"),
        ({"max_total_chars": 0}, "bad_request"),
    ],
)
def test_search_many_invalid_count_and_caps(kwargs, expected_code):
    res = asyncio.run(server.deepseek_search_many(["q"], **kwargs))
    assert res["ok"] is False
    assert res["error"]["code"] == expected_code


def test_search_many_need_content_true(fake_upstream):
    results_map = {"full": [{"title": "Full", "url": "https://f.test", "content": "Full markdown body"}]}
    _install_fake_search(fake_upstream, results_map)

    res = asyncio.run(server.deepseek_search_many(["full"], need_content=True))
    assert res["ok"] is True
    assert res["results"][0]["results"][0]["content"] == "Full markdown body"


def test_search_many_deadline_timeout(fake_upstream):
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(0.5)
        return {"ok": True, "data": {"results": []}}

    fake_upstream(slow_request)
    res = asyncio.run(server.deepseek_search_many(["slow_q"], deadline=0.01))
    assert res["ok"] is False
    assert res["error"]["code"] == "timeout"
