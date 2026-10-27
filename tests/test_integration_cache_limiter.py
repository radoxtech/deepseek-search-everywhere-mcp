"""Cross-cutting behaviour of the response caches, single-flight, rate limiter and rewrite cache.

Per-tool wiring is covered by ``test_search_cache.py`` and ``test_fetch_cache.py``; these
tests exercise the pieces together. Everything runs offline on a fake clock: the limiter
gets an injected ``sleep`` and the upstream is a stub, so no test waits for real time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server, transport
from deepseek_search_everywhere_mcp.ratelimit import FETCH, SEARCH, RateLimiter, Ticket
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

CHAT_PATH = "/v1/chat/completions"
URL = "https://example.com/article"
RESULTS = [
    {"title": "One", "url": "https://example.com/one", "site_name": "Example", "content": "a" * 300},
    {"title": "Two", "url": "https://example.com/two", "site_name": "Example", "content": "b" * 300},
]
PAGE = {"title": "Article", "content": "Body text " * 50}


class Upstream:
    """Fake ``server._request`` serving search, fetch and flash rewrites, counted per path."""

    def __init__(self, rewrite_to: str = "how sleep works") -> None:
        self.rewrite_to = rewrite_to
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.gate: asyncio.Event | None = None

    def count(self, path: str) -> int:
        return sum(1 for called, _ in self.calls if called == path)

    async def __call__(
        self, method: str, path: str, *, payload: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        self.calls.append((path, payload))
        if path == CHAT_PATH:
            return {"ok": True, "data": {"choices": [{"message": {"content": self.rewrite_to}}]}}
        if self.gate is not None:
            await self.gate.wait()
        if path == server.SEARCH_PATH:
            return {"ok": True, "data": {"results": RESULTS}}
        if path == server.FETCH_PATH:
            return {"ok": True, "data": PAGE}
        raise AssertionError(f"unexpected upstream path {path!r}")


class CountingLimiter(RateLimiter):
    """A real limiter that also records every ``acquire`` call per endpoint."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.acquired: list[str] = []

    async def acquire(self, endpoint: str) -> Ticket:
        self.acquired.append(endpoint)
        return await super().acquire(endpoint)


class RecordingSleep:
    """Injected limiter ``sleep``: records the delay and advances the fake clock instead of waiting."""

    def __init__(self, clock: Any) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self.clock.advance(delay)


def _install_limiter(
    monkeypatch: pytest.MonkeyPatch,
    clock: Any,
    settings: Settings | None = None,
    sleep: RecordingSleep | None = None,
) -> CountingLimiter:
    built = RateLimiter.from_settings(
        settings or Settings(), clock=clock.monotonic, sleep=sleep or RecordingSleep(clock)
    )
    limiter = CountingLimiter(built.buckets)
    monkeypatch.setattr(server, "_rate_limiter", limiter)
    return limiter


@pytest.fixture
def upstream(fake_upstream: Any) -> Upstream:
    return fake_upstream(Upstream())


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch, fake_clock: Any) -> CountingLimiter:
    return _install_limiter(monkeypatch, fake_clock)


async def _search(
    query: str = "test query", *, rewrite: bool = False, refresh: bool = False
) -> dict[str, Any]:
    return await server._search(query, 5, None, True, None, 1000, 4000, True, rewrite, refresh=refresh)


async def _spin(rounds: int = 5) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def test_ten_concurrent_searches_share_one_call_token_and_envelope(
    upstream: Upstream, limiter: CountingLimiter
) -> None:
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(_search()) for _ in range(10)]
    await _spin()
    upstream.gate.set()
    envelopes = await asyncio.gather(*tasks)
    assert upstream.count(server.SEARCH_PATH) == 1
    assert limiter.acquired == [SEARCH]
    assert all(envelope == envelopes[0] for envelope in envelopes)
    assert envelopes[0]["ok"] is True and envelopes[0]["cached"] is False
    assert server._get_search_flight().in_flight() == 0


async def test_cancelled_leader_leaves_waiters_served_and_no_flight_behind(
    upstream: Upstream, limiter: CountingLimiter
) -> None:
    upstream.gate = asyncio.Event()
    leader = asyncio.create_task(_search())
    await _spin()
    waiters = [asyncio.create_task(_search()) for _ in range(3)]
    await _spin()
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    upstream.gate.set()
    results = await asyncio.wait_for(asyncio.gather(*waiters), timeout=1)
    assert all(r["ok"] and r["cached"] is False for r in results)
    assert upstream.count(server.SEARCH_PATH) == 1
    assert limiter.acquired == [SEARCH]
    assert server._get_search_flight().in_flight() == 0
    assert (await _search())["cached"] is True


async def test_last_caller_leaving_cancels_the_call_and_refunds_the_token(
    upstream: Upstream, limiter: CountingLimiter
) -> None:
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(_search()) for _ in range(3)]
    await _spin()
    bucket = limiter.buckets[SEARCH]
    assert bucket.tokens == bucket.burst - 1
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await _spin()
    assert server._get_search_flight().in_flight() == 0
    assert bucket.tokens == bucket.burst
    upstream.gate.set()
    again = await _search()
    assert again["ok"] is True and again["cached"] is False
    assert upstream.count(server.SEARCH_PATH) == 2


async def test_retried_429_takes_one_token_and_is_cached_once_it_succeeds(
    monkeypatch: pytest.MonkeyPatch, limiter: CountingLimiter
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    attempts: list[str] = []

    async def rate_limited_once(
        method: str, url: str, path: str, **_: Any
    ) -> tuple[dict[str, Any], float | None]:
        attempts.append(path)
        if len(attempts) == 1:
            return {"ok": False, "error": {"code": "http_429", "message": "slow down"}}, 0.0
        return {"ok": True, "data": PAGE}, None

    monkeypatch.setattr(transport, "_attempt", rate_limited_once)
    first = await server.deepseek_fetch(URL)
    second = await server.deepseek_fetch(URL)
    assert first["ok"] is True and second["cached"] is True
    assert attempts == [server.FETCH_PATH, server.FETCH_PATH]
    assert limiter.acquired == [FETCH]


async def test_exhausted_429_retries_are_one_token_and_not_cached(
    monkeypatch: pytest.MonkeyPatch, limiter: CountingLimiter
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    attempts: list[str] = []

    async def always_429(method: str, url: str, path: str, **_: Any) -> tuple[dict[str, Any], float | None]:
        attempts.append(path)
        return {"ok": False, "error": {"code": "http_429", "message": "slow down"}}, 0.0

    monkeypatch.setattr(transport, "_attempt", always_429)
    first = await _search()
    assert first["ok"] is False and first["error"]["code"] == "http_429"
    assert len(attempts) == 2
    assert limiter.acquired == [SEARCH]
    second = await _search()
    assert second["ok"] is False
    assert len(attempts) == 4
    assert limiter.acquired == [SEARCH, SEARCH]


async def test_wait_within_budget_sleeps_and_beyond_budget_fails_fast_uncached(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch, fake_clock: Any
) -> None:
    class FrozenSleep(RecordingSleep):
        async def __call__(self, delay: float) -> None:
            self.delays.append(delay)

    sleep = FrozenSleep(fake_clock)
    settings = Settings(search_qps=1, search_burst=1, limiter_max_wait=1.5)
    _install_limiter(monkeypatch, fake_clock, settings, sleep)

    assert (await _search("first"))["ok"] is True
    assert sleep.delays == []
    assert (await _search("second"))["ok"] is True
    assert sleep.delays == [1.0]
    limited = await _search("third")
    assert limited["error"]["code"] == "rate_limited_local"
    assert sleep.delays == [1.0]
    assert upstream.count(server.SEARCH_PATH) == 2

    assert (await _search("second"))["cached"] is True
    fake_clock.advance(10)
    retried = await _search("third")
    assert retried["ok"] is True and retried["cached"] is False
    assert upstream.count(server.SEARCH_PATH) == 3


async def test_search_and_fetch_use_separate_buckets(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch, fake_clock: Any
) -> None:
    settings = Settings(search_qps=0.01, search_burst=1, fetch_qps=0.01, fetch_burst=1, limiter_max_wait=1)
    limiter = _install_limiter(monkeypatch, fake_clock, settings)
    search, fetch = await asyncio.gather(_search(), server.deepseek_fetch(URL))
    assert search["ok"] is True and fetch["ok"] is True
    assert sorted(limiter.acquired) == [FETCH, SEARCH]

    assert (await server.deepseek_fetch(URL, refresh=True))["error"]["code"] == "rate_limited_local"
    assert (await _search("another"))["error"]["code"] == "rate_limited_local"
    assert (await _search())["cached"] is True
    assert (await server.deepseek_fetch(URL))["cached"] is True


async def test_search_and_fetch_caches_have_separate_budgets(upstream: Upstream) -> None:
    set_settings(Settings(cache_search_entries=1, cache_fetch_entries=1))
    await server.deepseek_fetch(URL)
    await _search("one")
    await _search("two")
    assert (await _search("one"))["cached"] is False
    assert (await server.deepseek_fetch(URL))["cached"] is True
    assert len(server._get_search_cache()) == 1
    assert len(server._get_fetch_cache()) == 1


async def test_rewrite_and_search_cache_skip_flash_and_upstream(
    upstream: Upstream, limiter: CountingLimiter
) -> None:
    first = await _search("jak działa sen", rewrite=True)
    second = await _search("jak działa sen", rewrite=True)
    assert upstream.count(CHAT_PATH) == 1
    assert upstream.count(server.SEARCH_PATH) == 1
    assert limiter.acquired == [SEARCH]
    assert second["cached"] is True
    assert second["query_used"] == first["query_used"] == "how sleep works"
    assert second["query_original"] == "jak działa sen"


async def test_concurrent_rewritten_searches_share_one_upstream_call(
    upstream: Upstream, limiter: CountingLimiter
) -> None:
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(_search("jak działa sen", rewrite=True)) for _ in range(5)]
    await _spin()
    upstream.gate.set()
    results = await asyncio.gather(*tasks)
    assert all(r["query_used"] == "how sleep works" for r in results)
    assert upstream.count(server.SEARCH_PATH) == 1
    assert limiter.acquired == [SEARCH]
    flash_calls = upstream.count(CHAT_PATH)
    await _search("jak działa sen", rewrite=True)
    assert upstream.count(CHAT_PATH) == flash_calls


async def test_cache_disabled_by_env_keeps_single_flight_but_stores_nothing(
    upstream: Upstream, limiter: CountingLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_MCP_CACHE", "0")
    set_settings(None)
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(_search(rewrite=True)) for _ in range(4)]
    tasks += [asyncio.create_task(server.deepseek_fetch(URL)) for _ in range(4)]
    await _spin()
    upstream.gate.set()
    await asyncio.gather(*tasks)
    assert upstream.count(server.SEARCH_PATH) == 1
    assert upstream.count(server.FETCH_PATH) == 1

    rewrites = upstream.count(CHAT_PATH)
    assert (await _search(rewrite=True))["cached"] is False
    assert (await server.deepseek_fetch(URL))["cached"] is False
    assert upstream.count(server.SEARCH_PATH) == 2
    assert upstream.count(server.FETCH_PATH) == 2
    assert upstream.count(CHAT_PATH) == rewrites + 1
    assert sorted(limiter.acquired) == [FETCH, FETCH, SEARCH, SEARCH]
    assert server._get_search_cache() is None and server._get_fetch_cache() is None


def test_state_survives_a_cancelled_call_on_a_previous_event_loop(
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    limiter = _install_limiter(monkeypatch, fake_clock)
    bucket = limiter.buckets[SEARCH]

    async def abandon() -> None:
        upstream.gate = asyncio.Event()
        asyncio.get_running_loop().create_task(_search())
        await _spin()

    async def search_again() -> dict[str, Any]:
        upstream.gate = None
        return await _search()

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        asyncio.run(abandon())
        assert bucket.tokens == bucket.burst
        assert server._get_search_flight().in_flight() == 0
        result = asyncio.run(search_again())
    assert result["ok"] is True and result["cached"] is False
    assert upstream.count(server.SEARCH_PATH) == 2
    assert bucket.tokens == bucket.burst - 1
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "destroyed" not in messages and "different loop" not in messages
