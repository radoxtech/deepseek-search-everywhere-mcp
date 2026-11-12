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


def to_micros(usd: float) -> int:
    """Convert a USD dollar float to integer micro-dollars ($1 = 1,000,000 micros)."""
    return round(usd * 1_000_000)


def to_usd(micros: int) -> float:
    """Convert integer micro-dollars to a USD dollar float rounded to 6 decimal places."""
    return round(micros / 1_000_000.0, 6)


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

    __slots__ = ("_amount", "_amount_micros", "_reservation_id", "_settled", "_tracker")

    def __init__(
        self,
        tracker: DailyBudgetTracker,
        amount: float,
        amount_micros: int,
        reservation_id: str,
    ) -> None:
        self._tracker = tracker
        self._amount = amount
        self._amount_micros = amount_micros
        self._reservation_id = reservation_id
        self._settled = False

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def amount_micros(self) -> int:
        return self._amount_micros

    @property
    def reservation_id(self) -> str:
        return self._reservation_id

    async def commit(self, actual_cost: float) -> None:
        if self._settled:
            return
        self._settled = True
        actual_micros = to_micros(actual_cost)
        await self._tracker._commit_reservation(self._reservation_id, self._amount_micros, actual_micros)

    async def release(self) -> None:
        if self._settled:
            return
        self._settled = True
        await self._tracker._release_reservation(self._reservation_id, self._amount_micros)

    async def __aenter__(self) -> Reservation:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self._settled:
            await self.release()


class DailyBudgetTracker:
    """Thread-safe, process-safe and async-safe tracker for daily spend limits."""

    def __init__(self, limit_usd: float, state_path: pathlib.Path | None = None) -> None:
        self.limit_usd = limit_usd
        self.limit_micros = to_micros(limit_usd)
        self.state_path = state_path or default_state_path()
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self._lock = asyncio.Lock()
        self._date = _utc_today()
        self._spent_micros = 0
        self._reserved_micros = 0
        self._reservations: dict[str, dict[str, Any]] = {}
        self._requests = 0
        self._loaded = False
        self._disk_warning_logged = False

    @property
    def _spent_usd(self) -> float:
        return to_usd(self._spent_micros)

    @property
    def _reserved_usd(self) -> float:
        return to_usd(self._reserved_micros)

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
        self._reserved_micros = sum(
            int(r.get("amount_micros", to_micros(float(r.get("amount", 0.0)))))
            for r in self._reservations.values()
        )

    def _load_locked(self) -> None:
        """Load state from disk if available, resetting if the date has changed."""
        today = _utc_today()
        if not self.state_path.exists():
            if not self._loaded:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
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
                if "spent_micros" in data:
                    self._spent_micros = int(data["spent_micros"])
                else:
                    self._spent_micros = to_micros(float(data.get("spent_usd", 0.0)))
                raw_reservations = data.get("reservations")
                if isinstance(raw_reservations, dict):
                    self._reservations = {k: v for k, v in raw_reservations.items() if isinstance(v, dict)}
                else:
                    self._reservations = {}
                self._requests = int(data.get("requests", 0))
                self._prune_reservations_locked()
            else:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0
            self._loaded = True
        except (OSError, ValueError, TypeError) as exc:
            if not self._loaded:
                logger.warning("Could not read budget file %s (%s); resetting tracker.", self.state_path, exc)
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0
                self._loaded = True

    def _save_locked(self) -> None:
        """Atomically persist state to disk; falls back to in-memory on error."""
        data = {
            "date": self._date,
            "spent_micros": self._spent_micros,
            "spent_usd": to_usd(self._spent_micros),
            "reserved_micros": self._reserved_micros,
            "reserved_usd": to_usd(self._reserved_micros),
            "reservations": self._reservations,
            "requests": self._requests,
            "limit_micros": self.limit_micros,
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

    def _sync_reserve(self, amount: float) -> tuple[str, int] | None:
        amount_micros = to_micros(amount)
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0
                self._save_locked()

            if (self._spent_micros + self._reserved_micros + amount_micros) > self.limit_micros:
                return None

            res_id = uuid.uuid4().hex
            self._reservations[res_id] = {
                "pid": os.getpid(),
                "amount_micros": amount_micros,
                "amount": to_usd(amount_micros),
                "created_at": time.time(),
            }
            self._reserved_micros += amount_micros
            self._save_locked()
            return res_id, amount_micros

    async def reserve(self, amount: float) -> Reservation | None:
        """Atomically reserve ``amount`` USD against today's limit.

        Returns a :class:`Reservation` on success, or ``None`` if the budget would be exceeded.
        """
        async with self._lock:
            result = await asyncio.to_thread(self._sync_reserve, amount)
            if result is None:
                return None
            res_id, amount_micros = result
            return Reservation(self, amount, amount_micros, res_id)

    def _sync_commit_reservation(self, res_id: str, reserved_micros: int, actual_micros: int) -> None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0

            self._reservations.pop(res_id, None)
            self._reserved_micros = sum(
                int(r.get("amount_micros", to_micros(float(r.get("amount", 0.0)))))
                for r in self._reservations.values()
            )
            self._spent_micros += actual_micros
            self._requests += 1
            self._save_locked()

    async def _commit_reservation(self, res_id: str, reserved_micros: int, actual_micros: int) -> None:
        """Settle a reservation, deducting reserved amount and adding actual incurred spend."""
        async with self._lock:
            await asyncio.to_thread(self._sync_commit_reservation, res_id, reserved_micros, actual_micros)

    def _sync_release_reservation(self, res_id: str, reserved_micros: int) -> None:
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0

            self._reservations.pop(res_id, None)
            self._reserved_micros = sum(
                int(r.get("amount_micros", to_micros(float(r.get("amount", 0.0)))))
                for r in self._reservations.values()
            )
            self._save_locked()

    async def _release_reservation(self, res_id: str, reserved_micros: int) -> None:
        """Release an unused reservation, restoring available budget."""
        async with self._lock:
            await asyncio.to_thread(self._sync_release_reservation, res_id, reserved_micros)

    def _sync_check_budget(self, projected_cost: float) -> tuple[bool, float, float]:
        projected_micros = to_micros(projected_cost)
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0
                self._save_locked()

            spent_usd = to_usd(self._spent_micros)
            if (self._spent_micros + self._reserved_micros + projected_micros) > self.limit_micros:
                return False, spent_usd, self.limit_usd
            return True, spent_usd, self.limit_usd

    async def check_budget(self, projected_cost: float = 0.00075) -> tuple[bool, float, float]:
        """Check whether ``projected_cost`` fits within today's remaining budget.

        Returns ``(allowed, current_spent_usd, limit_usd)``.
        """
        async with self._lock:
            return await asyncio.to_thread(self._sync_check_budget, projected_cost)

    def _sync_record_spend(self, cost: float) -> None:
        cost_micros = to_micros(cost)
        with _file_lock(self.lock_path):
            self._load_locked()
            today = _utc_today()
            if self._date != today:
                self._date = today
                self._spent_micros = 0
                self._reserved_micros = 0
                self._reservations = {}
                self._requests = 0

            self._spent_micros += cost_micros
            self._requests += 1
            self._save_locked()

    async def record_spend(self, cost: float = 0.00075) -> None:
        """Record an incurred spend after a successful upstream call."""
        async with self._lock:
            await asyncio.to_thread(self._sync_record_spend, cost)

    def _sync_get_status(self) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            self._load_locked()
            spent_usd = to_usd(self._spent_micros)
            reserved_usd = to_usd(self._reserved_micros)
            remaining_micros = max(0, self.limit_micros - self._spent_micros - self._reserved_micros)
            return {
                "date": self._date,
                "limit_usd": self.limit_usd,
                "limit_micros": self.limit_micros,
                "spent_usd": spent_usd,
                "spent_micros": self._spent_micros,
                "reserved_usd": reserved_usd,
                "reserved_micros": self._reserved_micros,
                "requests": self._requests,
                "remaining_usd": to_usd(remaining_micros),
                "remaining_micros": remaining_micros,
            }

    async def get_status(self) -> dict[str, Any]:
        """Return the current budget tracker status."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_status)
