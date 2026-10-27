"""Cache tests: TTL expiry, LRU eviction, byte budget and single-flight (offline, fake clock)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import cache
from deepseek_search_everywhere_mcp.cache import SingleFlight, TTLCache, entry_size, is_cacheable
from deepseek_search_everywhere_mcp.settings import MIB, Settings

RESULT = {"ok": True, "results": [{"url": "https://example.com", "title": "Example"}]}


def make_cache(clock, *, max_entries: int = 128, max_bytes: int = MIB, ttl: float = 300) -> TTLCache:
    return TTLCache(max_entries, max_bytes, ttl=ttl, clock=clock.monotonic)


# --- TTL ---------------------------------------------------------------------------------


def test_ttl_search_expires_while_fetch_survives(fake_clock):
    settings = Settings()
    search = TTLCache(
        settings.cache_search_entries,
        settings.cache_search_bytes,
        ttl=settings.cache_search_ttl,
        clock=fake_clock.monotonic,
    )
    fetch = TTLCache(
        settings.cache_fetch_entries,
        settings.cache_fetch_bytes,
        ttl=settings.cache_fetch_ttl,
        clock=fake_clock.monotonic,
    )
    assert search.set("k", RESULT) and fetch.set("k", RESULT)

    fake_clock.advance(301)

    assert search.get("k") is None
    assert search.stats()["entries"] == 0
    assert fetch.get("k") == RESULT


def test_ttl_entry_alive_until_deadline(fake_clock):
    store = make_cache(fake_clock, ttl=10)
    store.set("k", RESULT)
    fake_clock.advance(9.9)
    assert store.get("k") == RESULT
    fake_clock.advance(0.1)
    assert store.get("k") is None


def test_ttl_zero_disables_storage(fake_clock):
    store = make_cache(fake_clock, ttl=0)
    assert store.set("k", RESULT) is False
    assert store.get("k") is None


def test_ttl_expired_entries_are_evicted_before_live_ones(fake_clock):
    store = make_cache(fake_clock, max_entries=2, ttl=10)
    store.set("old", RESULT)
    fake_clock.advance(5)
    store.set("live", RESULT)
    fake_clock.advance(6)  # "old" expired, "live" still valid
    store.set("new", RESULT)
    assert store.get("live") == RESULT
    assert store.get("new") == RESULT
    assert store.stats()["entries"] == 2


# --- LRU / budget --------------------------------------------------------------------------


def test_evict_least_recently_used_first(fake_clock):
    store = make_cache(fake_clock, max_entries=2)
    store.set("a", {"v": 1})
    store.set("b", {"v": 2})
    assert store.get("a") == {"v": 1}  # "b" is now least recently used
    store.set("c", {"v": 3})
    assert store.get("b") is None
    assert store.get("a") == {"v": 1}
    assert store.get("c") == {"v": 3}


def test_evict_to_fit_byte_budget(fake_clock):
    value = {"blob": "x" * 100}
    size = entry_size(value)
    store = make_cache(fake_clock, max_bytes=size * 2)
    store.set("a", value)
    store.set("b", value)
    store.set("c", value)
    assert store.get("a") is None
    assert store.stats()["bytes"] == size * 2


def test_budget_rejects_oversized_entry(fake_clock):
    settings = Settings()
    store = make_cache(fake_clock, max_bytes=settings.cache_fetch_bytes)
    store.set("small", RESULT)
    huge = {"content": "x" * (33 * MIB)}

    assert store.set("huge", huge) is False

    assert store.get("huge") is None
    assert store.get("small") == RESULT
    assert store.stats()["bytes"] == entry_size(RESULT)


def test_budget_never_exceeded_after_many_inserts(fake_clock):
    store = make_cache(fake_clock, max_entries=64, max_bytes=16 * 1024)
    for i in range(1000):
        store.set(f"k{i}", {"i": i, "pad": "y" * (i % 300)})
        stats = store.stats()
        assert stats["bytes"] <= stats["max_bytes"]
        assert stats["entries"] <= stats["max_entries"]
    assert store.get("k999") is not None


def test_budget_counts_utf8_bytes():
    assert entry_size({"t": "zażółć"}) == len('{"t": "zażółć"}'.encode())


def test_replacing_key_updates_byte_count(fake_clock):
    store = make_cache(fake_clock)
    store.set("k", {"v": "a" * 50})
    store.set("k", {"v": "b"})
    assert store.stats() == {
        "entries": 1,
        "bytes": entry_size({"v": "b"}),
        "max_entries": 128,
        "max_bytes": MIB,
        "hits": 0,
        "misses": 0,
    }


def test_rejected_value_drops_previous_entry(fake_clock):
    store = make_cache(fake_clock)
    store.set("k", RESULT)
    assert store.set("k", {"ok": False, "error": "boom"}) is False
    assert store.get("k") is None


# --- cacheability and copying ------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, {}, [], "", {"ok": False, "error": "x"}, {"ok": True, "results": []}],
)
def test_failures_and_empty_results_are_not_cached(fake_clock, value):
    store = make_cache(fake_clock)
    assert is_cacheable(value) is False
    assert store.set("k", value) is False
    assert store.stats()["entries"] == 0


def test_unserializable_value_is_not_cached(fake_clock):
    store = make_cache(fake_clock)
    assert store.set("k", {"obj": object()}) is False
    assert store.stats()["entries"] == 0


def test_values_are_copied_at_both_boundaries(fake_clock):
    store = make_cache(fake_clock)
    original = {"results": [{"title": "a"}]}
    store.set("k", original)
    original["results"][0]["title"] = "mutated before read"
    first = store.get("k")
    first["results"].append({"title": "mutated after read"})
    assert store.get("k") == {"results": [{"title": "a"}]}


def test_delete_clear_and_counters(fake_clock):
    store = make_cache(fake_clock)
    store.set("a", RESULT)
    store.set("b", RESULT)
    store.get("a")
    store.get("missing")
    assert (store.stats()["hits"], store.stats()["misses"]) == (1, 1)
    assert store.delete("a") is True
    assert store.delete("a") is False
    store.clear()
    assert store.stats()["entries"] == store.stats()["bytes"] == store.stats()["hits"] == 0


def test_reset_state_clears_live_caches(fake_clock):
    store = make_cache(fake_clock)
    store.set("k", RESULT)
    cache.reset_state()
    assert len(store) == 0


# --- single-flight -----------------------------------------------------------------------


class Upstream:
    """Factory double that counts calls and blocks until released."""

    def __init__(self, value: Any = RESULT) -> None:
        self.calls = 0
        self.value = value
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def __call__(self) -> Any:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


async def test_singleflight_collapses_concurrent_misses(fake_clock):
    flight = SingleFlight(make_cache(fake_clock))
    upstream = Upstream()

    calls = [asyncio.create_task(flight.run("k", upstream)) for _ in range(10)]
    await upstream.started.wait()
    assert flight.in_flight() == 1
    upstream.release.set()
    results = await asyncio.gather(*calls)

    assert upstream.calls == 1
    assert all(r.value == RESULT and r.cached is False for r in results)
    assert len({id(r.value) for r in results}) == 10  # every caller gets its own copy
    assert flight.in_flight() == 0

    again = await flight.run("k", upstream)
    assert again.cached is True and again.value == RESULT
    assert upstream.calls == 1


async def test_singleflight_error_propagates_and_is_not_cached(fake_clock):
    store = make_cache(fake_clock)
    flight = SingleFlight(store)
    upstream = Upstream(RuntimeError("upstream down"))

    calls = [asyncio.create_task(flight.run("k", upstream)) for _ in range(3)]
    await upstream.started.wait()
    upstream.release.set()
    results = await asyncio.gather(*calls, return_exceptions=True)

    assert upstream.calls == 1
    assert all(isinstance(r, RuntimeError) for r in results)
    assert len(store) == 0
    assert flight.in_flight() == 0


async def test_singleflight_error_envelope_is_returned_but_not_cached(fake_clock):
    store = make_cache(fake_clock)
    flight = SingleFlight(store)
    upstream = Upstream({"ok": False, "error": "rate_limited"})
    upstream.release.set()

    first = await flight.run("k", upstream)
    second = await flight.run("k", upstream)

    assert first.value == second.value == {"ok": False, "error": "rate_limited"}
    assert upstream.calls == 2
    assert len(store) == 0


async def test_singleflight_leader_cancellation_does_not_hang_followers(fake_clock):
    flight = SingleFlight(make_cache(fake_clock))
    upstream = Upstream()

    leader = asyncio.create_task(flight.run("k", upstream))
    await upstream.started.wait()
    followers = [asyncio.create_task(flight.run("k", upstream)) for _ in range(3)]
    await asyncio.sleep(0)

    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    upstream.release.set()
    results = await asyncio.wait_for(asyncio.gather(*followers, return_exceptions=True), timeout=1)

    # The shared call is shielded, so followers still get the upstream result.
    assert [r.value for r in results] == [RESULT] * 3
    assert upstream.calls == 1
    assert flight.in_flight() == 0


async def test_singleflight_last_caller_cancellation_cancels_upstream(fake_clock):
    store = make_cache(fake_clock)
    flight = SingleFlight(store)
    upstream = Upstream()

    callers = [asyncio.create_task(flight.run("k", upstream)) for _ in range(2)]
    await upstream.started.wait()
    for caller in callers:
        caller.cancel()
    results = await asyncio.wait_for(asyncio.gather(*callers, return_exceptions=True), timeout=1)

    assert all(isinstance(r, asyncio.CancelledError) for r in results)
    assert flight.in_flight() == 0
    assert len(store) == 0

    upstream.release.set()
    retry = await asyncio.wait_for(flight.run("k", upstream), timeout=1)
    assert retry.value == RESULT and upstream.calls == 2


async def test_singleflight_refresh_skips_cache_but_joins_in_flight(fake_clock):
    flight = SingleFlight(make_cache(fake_clock))
    upstream = Upstream()
    upstream.release.set()
    await flight.run("k", upstream)
    assert (await flight.run("k", upstream)).cached is True

    upstream.release.clear()
    upstream.started.clear()
    first = asyncio.create_task(flight.run("k", upstream, refresh=True))
    await upstream.started.wait()
    second = asyncio.create_task(flight.run("k", upstream, refresh=True))
    await asyncio.sleep(0)
    upstream.release.set()
    results = await asyncio.gather(first, second)

    assert upstream.calls == 2  # one initial call, one shared refresh
    assert all(r.cached is False for r in results)


async def test_singleflight_without_cache_only_collapses(fake_clock):
    flight = SingleFlight(None)
    upstream = Upstream()
    upstream.release.set()
    await flight.run("k", upstream)
    result = await flight.run("k", upstream)
    assert result.cached is False and upstream.calls == 2


def test_singleflight_works_across_event_loops(fake_clock):
    flight = SingleFlight(make_cache(fake_clock))

    async def produce() -> dict[str, Any]:
        await asyncio.sleep(0)
        return RESULT

    assert asyncio.run(flight.run("k", produce)).cached is False
    assert asyncio.run(flight.run("k", produce, refresh=True)).value == RESULT
    assert flight.in_flight() == 0


async def test_reset_state_clears_in_flight_map(fake_clock):
    flight = SingleFlight(make_cache(fake_clock))
    upstream = Upstream()
    caller = asyncio.create_task(flight.run("k", upstream))
    await upstream.started.wait()
    cache.reset_state()
    assert flight.in_flight() == 0
    upstream.release.set()
    assert (await caller).value == RESULT
