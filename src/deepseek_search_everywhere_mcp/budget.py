"""Local daily spend budget tracking and enforcement.

Enforces an optional daily spending limit (in USD) configured via ``DEEPSEEK_DAILY_BUDGET_USD``.
State is persisted to a local JSON file (configured via ``DEEPSEEK_STATE_DIR`` or defaulting to
``~/.cache/deepseek-search-everywhere-mcp/budget.json``) with atomic writes, cross-process file
locking, and fallback to in-memory tracking if filesystem writes fail.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import pathlib
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

RESERVATION_TTL_SECONDS = 900.0  # 15 minutes

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


def _utc_today() -> str:
    """Return today's date in UTC formatted as YYYY-MM-DD."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def default_state_path() -> pathlib.Path:
    """Return the default budget state file path."""
    state_dir = os.environ.get("DEEPSEEK_STATE_DIR", "").strip()
    if state_dir:
        return pathlib.Path(state_dir).expanduser() / "budget.json"
    return pathlib.Path.home() / ".cache" / "deepseek-search-everywhere-mcp" / "budget.json"


def _is_pid_alive(pid: int) -> bool:
    """Check if process with ``pid`` is currently running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _file_lock(lock_path: pathlib.Path) -> Iterator[None]:
    """Acquire a cross-process advisory lock on ``lock_path``."""
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    except OSError:
        yield
        return

    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif msvcrt is not None:
            try:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            except OSError:
                pass
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            os.close(fd)


class Reservation:
    """An active spend reservation against a daily budget."""

    __slots__ = ("_amount", "_reservation_id", "_settled", "_tracker")

    def __init__(self, tracker: DailyBudgetTracker, amount: float, reservation_id: str) -> None:
        self._tracker = tracker
        self._amount = amount
        self._reservation_id = reservation_id
        self._settled = False

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def reservation_id(self) -> str:
        return self._reservation_id

    async def commit(self, actual_cost: float) -> None:
        if self._settled:
            return
        self._settled = True
        await self._tracker._commit_reservation(self._reservation_id, self._amount, actual_cost)

    async def release(self) -> None:
        if self._settled:
            return
        self._settled = True
        await self._tracker._release_reservation(self._reservation_id, self._amount)

    async def __aenter__(self) -> Reservation:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._settled:
            await self.release()


class DailyBudgetTracker:
    """Thread-safe, process-safe and async-safe tracker for daily spend limits."""

    def __init__(self, limit_usd: float, state_path: pathlib.Path | None = None) -> None:
        self.limit_usd = limit_usd
        self.state_path = state_path or default_state_path()
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self._lock = asyncio.Lock()
        self._date = _utc_today()
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._reservations: dict[str, dict[str, Any]] = {}
        self._requests = 0
        self._loaded = False
        self._disk_warning_logged = False

    def _prune_reservations_locked(self, now: float | None = None) -> None:
        """Drop stale reservations whose TTL expired or whose PID no longer exists."""
        if now is None:
            now = time.time()
        pruned: dict[str, dict[str, Any]] = {}
        for res_id, r in self._reservations.items():
            created_at = float(r.get("created_at", 0.0))
            pid = int(r.get("pid", 0))
            if (now - created_at) < RESERVATION_TTL_SECONDS and _is_pid_alive(pid):
                pruned[res_id] = r
        self._reservations = pruned
        self._reserved_usd = sum(float(r.get("amount", 0.0)) for r in self._reservations.values())

    def _load_locked(self) -> None:
        """Load state from disk if available, resetting if the date has changed."""
        today = _utc_today()
        if not self.state_path.exists():
            if not self._loaded:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
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
                raw_reservations = data.get("reservations")
                if isinstance(raw_reservations, dict):
                    self._reservations = {k: v for k, v in raw_reservations.items() if isinstance(v, dict)}
                else:
                    self._reservations = {}
                self._requests = int(data.get("requests", 0))
                self._prune_reservations_locked()
            else:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0
            self._loaded = True
        except (OSError, ValueError, TypeError) as exc:
            if not self._loaded:
                logger.warning("Could not read budget file %s (%s); resetting tracker.", self.state_path, exc)
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0
                self._loaded = True

    def _save_locked(self) -> None:
        """Atomically persist state to disk; falls back to in-memory on error."""
        data = {
            "date": self._date,
            "spent_usd": round(self._spent_usd, 6),
            "reserved_usd": round(self._reserved_usd, 6),
            "reservations": self._reservations,
            "requests": self._requests,
            "limit_usd": self.limit_usd,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(f".tmp.{os.getpid()}.{id(self)}")
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

    def _sync_reserve(self, amount: float) -> str | None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0
                self._save_locked()

            if (self._spent_usd + self._reserved_usd + amount) > self.limit_usd:
                return None

            res_id = uuid.uuid4().hex
            self._reservations[res_id] = {
                "pid": os.getpid(),
                "amount": amount,
                "created_at": time.time(),
            }
            self._reserved_usd += amount
            self._save_locked()
            return res_id

    async def reserve(self, amount: float) -> Reservation | None:
        """Atomically reserve ``amount`` USD against today's limit.

        Returns a :class:`Reservation` on success, or ``None`` if the budget would be exceeded.
        """
        async with self._lock:
            res_id = await asyncio.to_thread(self._sync_reserve, amount)
            if res_id is None:
                return None
            return Reservation(self, amount, res_id)

    def _sync_commit_reservation(self, res_id: str, reserved: float, actual: float) -> None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0

            self._reservations.pop(res_id, None)
            self._reserved_usd = sum(float(r.get("amount", 0.0)) for r in self._reservations.values())
            self._spent_usd += actual
            self._requests += 1
            self._save_locked()

    async def _commit_reservation(self, res_id: str, reserved: float, actual: float) -> None:
        """Settle a reservation, deducting reserved amount and adding actual incurred spend."""
        async with self._lock:
            await asyncio.to_thread(self._sync_commit_reservation, res_id, reserved, actual)

    def _sync_release_reservation(self, res_id: str, reserved: float) -> None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0

            self._reservations.pop(res_id, None)
            self._reserved_usd = sum(float(r.get("amount", 0.0)) for r in self._reservations.values())
            self._save_locked()

    async def _release_reservation(self, res_id: str, reserved: float) -> None:
        """Release an unused reservation, restoring available budget."""
        async with self._lock:
            await asyncio.to_thread(self._sync_release_reservation, res_id, reserved)

    def _sync_check_budget(self, projected_cost: float) -> tuple[bool, float, float]:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0
                self._save_locked()

            if (self._spent_usd + self._reserved_usd + projected_cost) > self.limit_usd:
                return False, self._spent_usd, self.limit_usd
            return True, self._spent_usd, self.limit_usd

    async def check_budget(self, projected_cost: float = 0.00075) -> tuple[bool, float, float]:
        """Check whether ``projected_cost`` fits within today's remaining budget.

        Returns ``(allowed, current_spent_usd, limit_usd)``.
        """
        async with self._lock:
            return await asyncio.to_thread(self._sync_check_budget, projected_cost)

    def _sync_record_spend(self, cost: float) -> None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_usd = 0.0
                self._reserved_usd = 0.0
                self._reservations = {}
                self._requests = 0

            self._spent_usd += cost
            self._requests += 1
            self._save_locked()

    async def record_spend(self, cost: float = 0.00075) -> None:
        """Record an incurred spend after a successful upstream call."""
        async with self._lock:
            await asyncio.to_thread(self._sync_record_spend, cost)

    def _sync_get_status(self) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            self._load_locked()
            return {
                "date": self._date,
                "limit_usd": self.limit_usd,
                "spent_usd": round(self._spent_usd, 6),
                "reserved_usd": round(self._reserved_usd, 6),
                "requests": self._requests,
                "remaining_usd": max(0.0, round(self.limit_usd - self._spent_usd - self._reserved_usd, 6)),
            }

    async def get_status(self) -> dict[str, Any]:
        """Return the current budget tracker status."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_status)
