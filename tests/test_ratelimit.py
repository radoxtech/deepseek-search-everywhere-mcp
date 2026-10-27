"""Tests for the per-endpoint token buckets; time is faked, nothing really sleeps."""

from __future__ import annotations

import asyncio

import pytest

from deepseek_search_everywhere_mcp import ratelimit
from deepseek_search_everywhere_mcp.ratelimit import (
    FETCH,
    SEARCH,
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
)
from deepseek_search_everywhere_mcp.settings import Settings


class FakeSleep:
    """Records requested waits and advances the fake clock instead of sleeping."""

    def __init__(self, clock) -> None:
        self.clock = clock
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.clock.advance(seconds)


class BlockingSleep:
    """A sleep that never finishes on its own, so a test can cancel the waiter."""

    def __init__(self) -> None:
        self.started = 0

    async def __call__(self, seconds: float) -> None:
        self.started += 1
        await asyncio.Future()


def make_bucket(fake_clock, qps: float = 10, burst: int = 20, max_wait: float = 5.0, sleep=None):
    sleep = sleep if sleep is not None else FakeSleep(fake_clock)
    bucket = TokenBucket(qps, burst, clock=fake_clock.monotonic, sleep=sleep, max_wait=max_wait, name=SEARCH)
    return bucket, sleep


async def test_burst_passes_then_next_acquire_waits(fake_clock):
    bucket, sleep = make_bucket(fake_clock)
    for _ in range(20):
        await bucket.acquire()
    assert sleep.waits == []
    assert bucket.tokens == pytest.approx(0)

    await bucket.acquire()
    assert sleep.waits == [pytest.approx(0.1)]
    assert fake_clock.monotonic() == pytest.approx(0.1)


async def test_tokens_refill_over_time_up_to_burst(fake_clock):
    bucket, _ = make_bucket(fake_clock, qps=5, burst=10)
    for _ in range(10):
        await bucket.acquire()
    fake_clock.advance(1.0)
    assert bucket.tokens == pytest.approx(5)
    fake_clock.advance(60.0)
    assert bucket.tokens == pytest.approx(10)


async def test_wait_beyond_max_wait_fails_without_sleeping_or_consuming(fake_clock):
    bucket, sleep = make_bucket(fake_clock, qps=1, burst=1, max_wait=0.5)
    await bucket.acquire()

    with pytest.raises(RateLimitExceeded) as excinfo:
        await bucket.acquire()

    assert sleep.waits == []
    assert bucket.tokens == pytest.approx(0)
    assert excinfo.value.code == "rate_limited_local"
    assert excinfo.value.endpoint == SEARCH
    assert excinfo.value.wait == pytest.approx(1.0)
    envelope = excinfo.value.to_envelope()
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "rate_limited_local"
    assert "status" not in envelope["error"]


async def test_wait_equal_to_max_wait_is_allowed(fake_clock):
    bucket, sleep = make_bucket(fake_clock, qps=2, burst=1, max_wait=0.5)
    await bucket.acquire()
    await bucket.acquire()
    assert sleep.waits == [pytest.approx(0.5)]


async def test_concurrent_waiters_queue_in_order_and_overflow_fails(fake_clock):
    sleep = BlockingSleep()
    bucket, _ = make_bucket(fake_clock, qps=10, burst=1, max_wait=0.25, sleep=sleep)
    await bucket.acquire()

    waiters = [asyncio.create_task(bucket.acquire()) for _ in range(2)]
    await asyncio.sleep(0)
    assert sleep.started == 2
    assert bucket.tokens == pytest.approx(-2)

    with pytest.raises(RateLimitExceeded) as excinfo:
        await bucket.acquire()
    assert excinfo.value.wait == pytest.approx(0.3)

    for task in waiters:
        task.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    assert bucket.tokens == pytest.approx(0)


async def test_cancelled_wait_refunds_the_reserved_token(fake_clock):
    sleep = BlockingSleep()
    bucket, _ = make_bucket(fake_clock, qps=10, burst=1, sleep=sleep)
    await bucket.acquire()
    before = bucket.tokens

    waiter = asyncio.create_task(bucket.acquire())
    await asyncio.sleep(0)
    assert sleep.started == 1
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert bucket.tokens == pytest.approx(before)


async def test_refund_is_idempotent_and_capped_at_burst(fake_clock):
    bucket, _ = make_bucket(fake_clock, qps=10, burst=3)
    first = await bucket.acquire()
    second = await bucket.acquire()
    assert bucket.tokens == pytest.approx(1)

    assert first.refund() is True
    assert first.refund() is False
    assert bucket.tokens == pytest.approx(2)

    fake_clock.advance(10)
    assert second.refund() is True
    assert bucket.tokens == pytest.approx(3)


async def test_reset_state_refills_buckets_and_voids_old_tickets(fake_clock):
    bucket, _ = make_bucket(fake_clock, qps=1, burst=2)
    ticket = await bucket.acquire()
    await bucket.acquire()

    ratelimit.reset_state()

    assert bucket.tokens == pytest.approx(2)
    assert ticket.refund() is False
    assert bucket.tokens == pytest.approx(2)


async def test_limiter_routes_endpoints_and_leaves_others_unlimited(fake_clock):
    sleep = FakeSleep(fake_clock)
    limiter = RateLimiter.from_settings(Settings(), clock=fake_clock.monotonic, sleep=sleep)

    assert set(limiter.buckets) == {SEARCH, FETCH}
    search, fetch = limiter.buckets[SEARCH], limiter.buckets[FETCH]
    assert (search.qps, search.burst, search.max_wait) == (10, 20, 5)
    assert (fetch.qps, fetch.burst, fetch.max_wait) == (5, 10, 5)

    for _ in range(50):
        ticket = await limiter.acquire("chat")
        assert ticket.refund() is False
    await limiter.acquire("balance")

    await limiter.acquire(FETCH)
    assert fetch.tokens == pytest.approx(9)
    assert search.tokens == pytest.approx(20)
    assert sleep.waits == []


async def test_limiter_uses_settings_overrides(fake_clock):
    settings = Settings(fetch_qps=2, fetch_burst=1, limiter_max_wait=0.1)
    limiter = RateLimiter.from_settings(settings, clock=fake_clock.monotonic, sleep=FakeSleep(fake_clock))
    await limiter.acquire(FETCH)
    with pytest.raises(RateLimitExceeded) as excinfo:
        await limiter.acquire(FETCH)
    assert excinfo.value.endpoint == FETCH


def test_bucket_works_across_separate_event_loops():
    bucket = TokenBucket(1000, 2, max_wait=1.0)

    async def take_three() -> None:
        for _ in range(3):
            await bucket.acquire()

    asyncio.run(take_three())
    asyncio.run(take_three())


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"qps": 0, "burst": 1}, "qps"),
        ({"qps": 1, "burst": 0}, "burst"),
        ({"qps": 1, "burst": 1, "max_wait": -1}, "max_wait"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, field):
    with pytest.raises(ValueError, match=field):
        TokenBucket(**kwargs)
