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
