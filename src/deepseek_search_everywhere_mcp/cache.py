"""In-process response cache: a TTL + LRU store with a byte budget, plus single-flight.

The cache is asyncio-only and process-local. Values are deep-copied on the way in and on
the way out, so callers may reshape what they get without corrupting stored entries.
Failures are never cached: ``None``, empty containers, ``{"ok": False, ...}`` envelopes,
mappings with an empty ``results`` list and exceptions all bypass the store.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
import weakref
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any, NamedTuple


def entry_size(value: Any) -> int:
    """Return the size in bytes of ``value`` as canonical UTF-8 JSON."""
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def is_cacheable(value: Any) -> bool:
    """Tell whether ``value`` looks like a successful, non-empty result worth caching."""
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, str)) and not value:
        return False
    if isinstance(value, dict):
        if value.get("ok") is False:
            return False
        if "results" in value and not value["results"]:
            return False
    return True


@dataclass(slots=True)
class _Entry:
    value: Any
    size: int
    expires_at: float


class TTLCache:
    """Least-recently-used cache whose entries expire ``ttl`` seconds after being stored.

    Both ``max_entries`` and ``max_bytes`` are hard limits: expired entries are dropped
    first, then the least recently used ones, until a new entry fits. An entry larger
    than ``max_bytes`` on its own is not stored at all.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int,
        *,
        ttl: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries!r}")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes!r}")
        if ttl < 0:
            raise ValueError(f"ttl must be >= 0, got {ttl!r}")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl = ttl
        self._clock = clock
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        _registry.add(self)

    def get(self, key: Hashable) -> Any | None:
        """Return a copy of the live value for ``key``, or ``None`` on a miss or expiry."""
        entry = self._entries.get(key)
        if entry is not None and self._clock() >= entry.expires_at:
            self._remove(key)
            entry = None
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return copy.deepcopy(entry.value)

    def set(self, key: Hashable, value: Any) -> bool:
        """Store a copy of ``value`` under ``key``; return whether it was cached.

        Any previous value for ``key`` is dropped even when the new one is rejected, so a
        stale entry never outlives a newer upstream answer.
        """
        self.delete(key)
        if self.ttl <= 0 or not is_cacheable(value):
            return False
        try:
            size = entry_size(value)
        except (TypeError, ValueError):
            return False
        if size > self.max_bytes:
            return False
        self._make_room(size)
        self._entries[key] = _Entry(copy.deepcopy(value), size, self._clock() + self.ttl)
        self._bytes += size
        return True

    def delete(self, key: Hashable) -> bool:
        """Remove ``key``; return whether it was present."""
        if key not in self._entries:
            return False
        self._remove(key)
        return True

    def clear(self) -> None:
        """Drop every entry and reset the hit/miss counters."""
        self._entries.clear()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return current occupancy, limits and hit/miss counters."""
        return {
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "hits": self._hits,
            "misses": self._misses,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def _remove(self, key: Hashable) -> None:
        self._bytes -= self._entries.pop(key).size

    def _make_room(self, size: int) -> None:
        now = self._clock()
        for key in [k for k, e in self._entries.items() if now >= e.expires_at]:
            self._remove(key)
        while self._entries and (
            len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes
        ):
            self._remove(next(iter(self._entries)))


class FlightResult(NamedTuple):
    """Outcome of :meth:`SingleFlight.run`: the value and whether it came from the cache."""

    value: Any
    cached: bool


@dataclass(slots=True)
class _Flight:
    task: asyncio.Task[Any]
    waiters: int = 0


class SingleFlight:
    """Collapse concurrent identical requests into one upstream call, backed by a cache.

    The first caller for a key (the leader) starts the factory as a task; every caller,
    the leader included, awaits it through :func:`asyncio.shield`. Cancelling one caller
    therefore never cancels the shared call while others still wait; when the last
    caller leaves, an unfinished call is cancelled. The task is created inside the
    running loop and forgotten as soon as it finishes, so nothing is bound to a loop
    across calls. Pass ``cache=None`` to collapse calls without storing results.
    """

    def __init__(self, cache: TTLCache | None) -> None:
        self.cache = cache
        self._inflight: dict[Hashable, _Flight] = {}
        _registry.add(self)

    async def run(
        self,
        key: Hashable,
        factory: Callable[[], Awaitable[Any]],
        *,
        refresh: bool = False,
    ) -> FlightResult:
        """Return the cached value for ``key`` or the result of a (shared) ``factory()`` call.

        With ``refresh=True`` the cache is not read, but a call already in flight for the
        same key is joined rather than duplicated. Exceptions from the factory propagate
        to every caller and are never cached.
        """
        if not refresh and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return FlightResult(hit, True)
        flight = self._inflight.get(key)
        if flight is None or flight.task.get_loop() is not asyncio.get_running_loop():
            flight = _Flight(asyncio.create_task(self._produce(key, factory)))
            self._inflight[key] = flight
            flight.task.add_done_callback(lambda task: self._finished(key, flight, task))
        flight.waiters += 1
        try:
            value = await asyncio.shield(flight.task)
        finally:
            flight.waiters -= 1
            if flight.waiters == 0 and not flight.task.done():
                flight.task.cancel()
                self._forget(key, flight)
        return FlightResult(copy.deepcopy(value), False)

    def in_flight(self) -> int:
        """Return how many distinct keys currently have a call in progress."""
        return len(self._inflight)

    def clear(self) -> None:
        """Forget every in-flight call (their tasks are left to finish on their own)."""
        self._inflight.clear()

    async def _produce(self, key: Hashable, factory: Callable[[], Awaitable[Any]]) -> Any:
        value = await factory()
        if self.cache is not None:
            self.cache.set(key, value)
        return value

    def _finished(self, key: Hashable, flight: _Flight, task: asyncio.Task[Any]) -> None:
        self._forget(key, flight)
        if not task.cancelled():
            task.exception()  # mark as retrieved even if every caller has gone

    def _forget(self, key: Hashable, flight: _Flight) -> None:
        if self._inflight.get(key) is flight:
            del self._inflight[key]


_registry: weakref.WeakSet[TTLCache | SingleFlight] = weakref.WeakSet()


def reset_state() -> None:
    """Clear every live cache and in-flight map in the process (used between tests)."""
    for owner in list(_registry):
        owner.clear()
