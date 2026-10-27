"""Process-local, per-endpoint token buckets that pace calls to the DeepSeek tool API.

The limiter is **asyncio-only** (both the stdio and the uvicorn transports run on
asyncio; trio is not supported) and deliberately process-local. It holds no asyncio
primitives, so a bucket can be shared by consecutive event loops.

Waiting uses reservations, in the spirit of Go's ``x/time/rate``: a caller takes its
token up front, possibly driving the balance negative, and sleeps until the refill
covers it. That keeps waiters in arrival order without a lock held across the sleep,
lets a caller fail at once when its wait would exceed ``max_wait`` instead of sleeping
for nothing, and makes cancellation a plain refund of the reserved token.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .errors import _fail
from .settings import Settings

SEARCH = "search"
FETCH = "fetch"

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[Any]]


class RateLimitExceeded(Exception):
    """Raised when a token cannot be obtained within ``max_wait`` seconds."""

    code = "rate_limited_local"

    def __init__(self, endpoint: str, wait: float, max_wait: float) -> None:
        self.endpoint = endpoint
        self.wait = wait
        self.max_wait = max_wait
        name = endpoint or "endpoint"
        super().__init__(
            f"Local rate limit for {name} reached: the next slot is {wait:.2f}s away, "
            f"over the {max_wait:g}s wait budget. Retry shortly."
        )

    def to_envelope(self) -> dict[str, Any]:
        """Return the tool error envelope for this failure."""
        return _fail(self.code, str(self))


class Ticket:
    """Proof of one token taken from a bucket; :meth:`refund` gives it back once.

    A ticket issued before its bucket was reset is stale and refunds nothing. A ticket
    from an unlimited endpoint has no bucket and refunds nothing either.
    """

    __slots__ = ("_bucket", "_epoch", "_refunded")

    def __init__(self, bucket: TokenBucket | None = None, epoch: int = 0) -> None:
        self._bucket = bucket
        self._epoch = epoch
        self._refunded = False

    def refund(self) -> bool:
        """Return the token to its bucket; return whether anything was returned."""
        if self._refunded or self._bucket is None:
            return False
        self._refunded = True
        return self._bucket._give_back(self._epoch)


class TokenBucket:
    """A token bucket refilled at ``qps`` tokens per second, holding at most ``burst``.

    The bucket starts full. ``clock`` must be monotonic and ``sleep`` must behave like
    :func:`asyncio.sleep`; both are injectable so tests run without real waiting.
    """

    def __init__(
        self,
        qps: float,
        burst: int,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        max_wait: float = 5.0,
        name: str = "",
    ) -> None:
        if not qps > 0:
            raise ValueError(f"qps must be > 0, got {qps!r}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst!r}")
        if max_wait < 0:
            raise ValueError(f"max_wait must be >= 0, got {max_wait!r}")
        self.qps = qps
        self.burst = burst
        self.max_wait = max_wait
        self.name = name
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(burst)
        self._updated = clock()
        self._epoch = 0
        _registry.add(self)

    @property
    def tokens(self) -> float:
        """Current balance after refill; negative while callers hold reservations."""
        self._refill()
        return self._tokens

    async def acquire(self) -> Ticket:
        """Take one token, sleeping until it is due; raise :class:`RateLimitExceeded`.

        The wait never exceeds ``max_wait``: a caller whose slot lies further out fails
        immediately without consuming anything. If the wait is cancelled, the reserved
        token is refunded before :class:`asyncio.CancelledError` propagates.
        """
        self._refill()
        wait = max(0.0, (1.0 - self._tokens) / self.qps)
        if wait > self.max_wait:
            raise RateLimitExceeded(self.name, wait, self.max_wait)
        self._tokens -= 1.0
        ticket = Ticket(self, self._epoch)
        if wait > 0:
            try:
                await self._sleep(wait)
            except BaseException:
                ticket.refund()
                raise
        return ticket

    def reset(self) -> None:
        """Refill the bucket completely and invalidate every outstanding ticket."""
        self._tokens = float(self.burst)
        self._updated = self._clock()
        self._epoch += 1

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.qps)

    def _give_back(self, epoch: int) -> bool:
        if epoch != self._epoch:
            return False
        self._refill()
        self._tokens = min(float(self.burst), self._tokens + 1.0)
        return True


class RateLimiter:
    """Routes :meth:`acquire` calls to per-endpoint buckets; other endpoints are unlimited."""

    def __init__(self, buckets: Mapping[str, TokenBucket]) -> None:
        self.buckets = dict(buckets)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> RateLimiter:
        """Build the ``search`` and ``fetch`` buckets from ``settings``.

        Chat completions and the balance endpoint get no bucket on purpose.
        """
        common: dict[str, Any] = {"clock": clock, "sleep": sleep, "max_wait": settings.limiter_max_wait}
        return cls(
            {
                SEARCH: TokenBucket(settings.search_qps, settings.search_burst, name=SEARCH, **common),
                FETCH: TokenBucket(settings.fetch_qps, settings.fetch_burst, name=FETCH, **common),
            }
        )

    async def acquire(self, endpoint: str) -> Ticket:
        """Take a token for ``endpoint``; endpoints without a bucket pass straight through."""
        bucket = self.buckets.get(endpoint)
        if bucket is None:
            return Ticket()
        return await bucket.acquire()


_registry: weakref.WeakSet[TokenBucket] = weakref.WeakSet()


def reset_state() -> None:
    """Refill every live bucket in the process (used between tests)."""
    for bucket in list(_registry):
        bucket.reset()
