"""Local daily spend budget tracking and enforcement.

Enforces an optional daily spending limit (in USD) configured via ``DEEPSEEK_DAILY_BUDGET_USD``.
State is persisted to a local JSON file (configured via ``DEEPSEEK_STATE_DIR`` or defaulting to
``~/.cache/deepseek-search-everywhere-mcp/budget.json``) with atomic writes and fallback to
in-memory tracking if filesystem writes fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _utc_today() -> str:
    """Return today's date in UTC formatted as YYYY-MM-DD."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def default_state_path() -> pathlib.Path:
    """Return the default budget state file path."""
    state_dir = os.environ.get("DEEPSEEK_STATE_DIR", "").strip()
    if state_dir:
        return pathlib.Path(state_dir).expanduser() / "budget.json"
    return pathlib.Path.home() / ".cache" / "deepseek-search-everywhere-mcp" / "budget.json"


class DailyBudgetTracker:
    """Thread-safe and async-safe tracker for daily spend limits."""

    def __init__(self, limit_usd: float, state_path: pathlib.Path | None = None) -> None:
        self.limit_usd = limit_usd
        self.state_path = state_path or default_state_path()
        self._lock = asyncio.Lock()
        self._date = _utc_today()
        self._spent_usd = 0.0
        self._requests = 0
        self._loaded = False
        self._disk_warning_logged = False

    def _load_locked(self) -> None:
        """Load state from disk if available, resetting if the date has changed."""
        today = _utc_today()
        if not self.state_path.exists():
            self._date = today
            self._spent_usd = 0.0
            self._requests = 0
            self._loaded = True
            return

        try:
            raw = self.state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            file_date = data.get("date")
            if file_date == today:
                self._date = today
                self._spent_usd = float(data.get("spent_usd", 0.0))
                self._requests = int(data.get("requests", 0))
            else:
                self._date = today
                self._spent_usd = 0.0
                self._requests = 0
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read budget file %s (%s); resetting tracker.", self.state_path, exc)
            self._date = today
            self._spent_usd = 0.0
            self._requests = 0
        self._loaded = True

    def _save_locked(self) -> None:
        """Atomically persist state to disk; falls back to in-memory on error."""
        data = {
            "date": self._date,
            "spent_usd": round(self._spent_usd, 6),
            "requests": self._requests,
            "limit_usd": self.limit_usd,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(f".tmp.{os.getpid()}")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(self.state_path)
        except OSError as exc:
            if not self._disk_warning_logged:
                logger.warning(
                    "Could not persist budget to %s (%s); keeping in-memory state.",
                    self.state_path,
                    exc,
                )
                self._disk_warning_logged = True

    async def check_budget(self, projected_cost: float = 0.00075) -> tuple[bool, float, float]:
        """Check whether ``projected_cost`` fits within today's remaining budget.

        Returns ``(allowed, current_spent_usd, limit_usd)``.
        """
        async with self._lock:
            today = _utc_today()
            if not self._loaded:
                self._load_locked()
            elif self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._requests = 0
                self._save_locked()

            if self._spent_usd + projected_cost > self.limit_usd:
                return False, self._spent_usd, self.limit_usd
            return True, self._spent_usd, self.limit_usd

    async def record_spend(self, cost: float = 0.00075) -> None:
        """Record an incurred spend after a successful upstream call."""
        async with self._lock:
            today = _utc_today()
            if not self._loaded:
                self._load_locked()
            elif self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._requests = 0

            self._spent_usd += cost
            self._requests += 1
            self._save_locked()

    async def get_status(self) -> dict[str, Any]:
        """Return the current budget tracker status."""
        async with self._lock:
            if not self._loaded:
                self._load_locked()
            return {
                "date": self._date,
                "limit_usd": self.limit_usd,
                "spent_usd": round(self._spent_usd, 6),
                "requests": self._requests,
                "remaining_usd": max(0.0, round(self.limit_usd - self._spent_usd, 6)),
            }
