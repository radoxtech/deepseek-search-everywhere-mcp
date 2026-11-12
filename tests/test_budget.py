"""Unit tests for DailyBudgetTracker and spend cap enforcement."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.budget import DailyBudgetTracker
from deepseek_search_everywhere_mcp.settings import Settings, set_settings


def _search_response(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"ok": True, "data": {"results": results or []}}


def _fetch_response() -> dict[str, Any]:
    return {"ok": True, "data": {"title": "Title", "content": "Content"}}


def test_budget_tracker_records_spend(tmp_path):
    state_file = tmp_path / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=0.01, state_path=state_file)

    allowed, spent, limit = asyncio.run(tracker.check_budget(0.00075))
    assert allowed is True
    assert spent == 0.0
    assert limit == 0.01

    asyncio.run(tracker.record_spend(0.00075))

    status = asyncio.run(tracker.get_status())
    assert status["spent_usd"] == 0.00075
    assert status["requests"] == 1
    assert status["remaining_usd"] == 0.00925

    # File should exist and be valid JSON
    assert state_file.exists()
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["spent_usd"] == 0.00075
    assert saved["requests"] == 1


def test_budget_tracker_enforces_limit(tmp_path):
    state_file = tmp_path / "budget.json"
    # Budget of $0.001 allows only 1 request at $0.00075 (second would be $0.0015 > $0.001)
    tracker = DailyBudgetTracker(limit_usd=0.001, state_path=state_file)

    allowed, _, _ = asyncio.run(tracker.check_budget(0.00075))
    assert allowed is True
    asyncio.run(tracker.record_spend(0.00075))

    # Second check should be rejected
    allowed, spent, limit = asyncio.run(tracker.check_budget(0.00075))
    assert allowed is False
    assert spent == 0.00075
    assert limit == 0.001


def test_budget_tracker_date_rollover(tmp_path):
    state_file = tmp_path / "budget.json"
    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    state_file.write_text(
        json.dumps({"date": yesterday, "spent_usd": 10.0, "requests": 50}), encoding="utf-8"
    )

    tracker = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)
    allowed, spent, _ = asyncio.run(tracker.check_budget(0.00075))
    assert allowed is True
    assert spent == 0.0  # Rolled over to today, so spent is 0


def test_budget_tracker_handles_corrupt_file(tmp_path):
    state_file = tmp_path / "budget.json"
    state_file.write_text("not json content", encoding="utf-8")

    tracker = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)
    allowed, spent, _ = asyncio.run(tracker.check_budget(0.00075))
    assert allowed is True
    assert spent == 0.0


def test_budget_tracker_handles_write_failure(tmp_path, caplog):
    # Make a file in place of the parent directory so mkdir/write raises OSError
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not directory", encoding="utf-8")
    state_file = blocker / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)

    with caplog.at_level(logging.WARNING):
        asyncio.run(tracker.record_spend(0.00075))

    # State still tracked in-memory
    status = asyncio.run(tracker.get_status())
    assert status["spent_usd"] == 0.00075
    assert "Could not persist budget" in caplog.text


def test_server_enforces_budget_exceeded(fake_upstream, tmp_path):
    fake_upstream(responses={"/v1/tools/search": _search_response()})
    state_file = tmp_path / "budget.json"

    # Set budget to $0.001 in settings
    settings = Settings(daily_budget_usd=0.001)
    set_settings(settings)
    server.reset_state()

    # Wire tracker to temporary state file
    tracker = DailyBudgetTracker(limit_usd=0.001, state_path=state_file)
    server._budget_tracker = tracker

    # First search succeeds
    res1 = asyncio.run(server.deepseek_search("test1"))
    assert res1["ok"] is True
    assert res1["cached"] is False

    # Second search exceeds budget
    res2 = asyncio.run(server.deepseek_search("test2"))
    assert res2["ok"] is False
    assert res2["error"]["code"] == "budget_exceeded"
    assert res2["error"]["status"] == 402
    assert "Daily spend limit" in res2["error"]["message"]

    # Reset server state
    set_settings(None)
    server.reset_state()


async def test_fifty_concurrent_reservations_never_exceed_limit(tmp_path: Any) -> None:
    state_file = tmp_path / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=0.05, state_path=state_file)

    # 50 tasks attempt to reserve $0.002 each ($0.10 total, limit is $0.05)
    results = await asyncio.gather(*(tracker.reserve(0.002) for _ in range(50)))
    successful = [r for r in results if r is not None]

    assert len(successful) == 25
    assert len(successful) * 0.002 <= 0.05

    status = await tracker.get_status()
    assert status["reserved_usd"] == 0.05
    assert status["spent_usd"] == 0.0
    assert status["remaining_usd"] == 0.0

    # Release 10 reservations
    for res in successful[:10]:
        await res.release()

    status_after_release = await tracker.get_status()
    assert round(status_after_release["reserved_usd"], 6) == 0.03
    assert round(status_after_release["remaining_usd"], 6) == 0.02

    # Commit remaining 15 reservations at $0.0015 actual cost
    for res in successful[10:]:
        await res.commit(0.0015)

    status_after_commit = await tracker.get_status()
    assert status_after_commit["reserved_usd"] == 0.0
    assert round(status_after_commit["spent_usd"], 6) == round(15 * 0.0015, 6)
    assert status_after_commit["requests"] == 15


async def test_two_trackers_on_one_file_lose_no_updates(tmp_path: Any) -> None:
    state_file = tmp_path / "budget.json"
    tracker1 = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)
    tracker2 = DailyBudgetTracker(limit_usd=1.0, state_path=state_file)

    async def run_worker(tracker: DailyBudgetTracker, cost: float) -> None:
        res = await tracker.reserve(cost * 2)
        assert res is not None
        await asyncio.sleep(0.005)
        await res.commit(cost)

    tasks = []
    # 10 tasks on tracker1 committing $0.001 each
    for _ in range(10):
        tasks.append(run_worker(tracker1, 0.001))
    # 10 tasks on tracker2 committing $0.002 each
    for _ in range(10):
        tasks.append(run_worker(tracker2, 0.002))

    await asyncio.gather(*tasks)

    # Both trackers should observe 20 requests and $0.030 total spent
    status1 = await tracker1.get_status()
    status2 = await tracker2.get_status()

    assert status1["requests"] == 20
    assert status2["requests"] == 20
    assert round(status1["spent_usd"], 6) == 0.03
    assert round(status2["spent_usd"], 6) == 0.03
    assert status1["reserved_usd"] == 0.0
    assert status2["reserved_usd"] == 0.0


async def test_concurrency_mixed_search_fetch_chat_never_exceeds_limit(
    fake_upstream: Any, tmp_path: Any
) -> None:
    # Set limit to allow exactly 5 operations ($0.00075 * 5 = $0.00375)
    limit = 0.00375
    state_file = tmp_path / "budget.json"
    tracker = DailyBudgetTracker(limit_usd=limit, state_path=state_file)

    settings = Settings(daily_budget_usd=limit)
    set_settings(settings)
    server.reset_state()
    server._budget_tracker = tracker

    # Mock responses with a small delay to induce concurrency race in check_budget
    async def delayed_handler(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        if path == server.SEARCH_PATH:
            res_item = {"url": "http://example.com/1", "title": "T", "content": "C"}
            return {"ok": True, "data": {"results": [res_item]}}
        if path == server.FETCH_PATH:
            return {"ok": True, "data": {"url": "http://example.com/2", "title": "T", "content": "C"}}
        if path == "/v1/chat/completions":
            return {
                "ok": True,
                "data": {
                    "choices": [{"message": {"role": "assistant", "content": "rewritten query"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                },
            }
        return {"ok": False, "error": {"code": "not_found", "message": "Unknown path"}}

    fake_upstream(delayed_handler)

    # Launch 20 mixed concurrent requests (search, fetch)
    tasks = []
    for i in range(10):
        tasks.append(server.deepseek_search(f"query_{i}", refresh=True, rewrite=False))
        tasks.append(server.deepseek_fetch(f"http://example.com/page_{i}", refresh=True))

    results = await asyncio.gather(*tasks)
    status = await tracker.get_status()

    # The total recorded spend + remaining reservations must NEVER exceed the limit
    assert (status["spent_usd"] + status["reserved_usd"]) <= limit + 1e-9, (
        f"Budget breached! Spent {status['spent_usd']}, limit {limit}"
    )
    # Check that budget_exceeded occurred on excess requests
    failures = [r for r in results if not r.get("ok") and r.get("error", {}).get("code") == "budget_exceeded"]
    assert len(failures) > 0

    set_settings(None)
    server.reset_state()
