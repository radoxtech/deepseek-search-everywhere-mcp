"""Fetch wiring: response cache, single-flight and rate limiter for ``deepseek_fetch``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.ratelimit import RateLimiter
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

URL = "https://example.com/article"
PAGE = {"title": "Article", "content": "\\# Heading\n\nBody text\\, with escapes" + "x" * 500}
NOT_FOUND = {"ok": False, "error": {"code": "http_404", "message": "failed to fetch URL"}}


class Upstream:
    """Fake ``server._request`` that counts fetch calls and serves a scripted response.

    With a ``gate`` set, each call signals ``entered`` and then blocks until the gate opens;
    a call cancelled while blocked signals ``cancelled`` before the cancellation propagates.
    """

    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.default: dict[str, Any] = {"ok": True, "data": PAGE}
        self.calls: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"path": path, **kwargs})
        if self.gate is not None:
            self.entered.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return self.responses.pop(0) if self.responses else self.default


@pytest.fixture
def upstream(fake_upstream: Any) -> Upstream:
    return fake_upstream(Upstream())


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch, fake_clock: Any) -> RateLimiter:
    """A limiter on a frozen clock, so token balances are exact."""
    instance = RateLimiter.from_settings(Settings(), clock=fake_clock.monotonic)
    monkeypatch.setattr(server, "_rate_limiter", instance)
    return instance


async def test_second_identical_fetch_is_served_from_cache(upstream: Upstream) -> None:
    first = await server.deepseek_fetch(URL)
    second = await server.deepseek_fetch(URL)
    assert len(upstream.calls) == 1
    assert upstream.calls[0]["path"] == server.FETCH_PATH
    assert upstream.calls[0]["payload"] == {"url": URL}
    assert first["cached"] is False
    assert second["cached"] is True
    assert {k: v for k, v in first.items() if k != "cached"} == {
        k: v for k, v in second.items() if k != "cached"
    }


async def test_shaping_options_share_one_upstream_call(upstream: Upstream) -> None:
    full = await server.deepseek_fetch(URL)
    short = await server.deepseek_fetch(URL, max_chars=20)
    raw = await server.deepseek_fetch(URL, clean=False)
    assert len(upstream.calls) == 1
    assert short["cached"] is True and raw["cached"] is True
    assert short["truncated"] is True and full["truncated"] is False
    assert short["content"] != full["content"]
    assert raw["content"] == PAGE["content"]
    assert full["content"].startswith("# Heading")


async def test_cached_page_is_not_truncated_by_shaping(upstream: Upstream) -> None:
    await server.deepseek_fetch(URL, max_chars=10)
    again = await server.deepseek_fetch(URL, clean=False)
    assert again["cached"] is True
    assert again["content"] == PAGE["content"]


async def test_refresh_bypasses_cache_read(upstream: Upstream) -> None:
    await server.deepseek_fetch(URL)
    refreshed = await server.deepseek_fetch(URL, refresh=True)
    assert len(upstream.calls) == 2
    assert refreshed["cached"] is False
    assert (await server.deepseek_fetch(URL))["cached"] is True
    assert len(upstream.calls) == 2


async def test_error_404_is_not_cached(upstream: Upstream) -> None:
    upstream.responses = [NOT_FOUND]
    first = await server.deepseek_fetch(URL)
    second = await server.deepseek_fetch(URL)
    assert {key: value for key, value in first.items() if key != "elapsed_ms"} == NOT_FOUND
    assert "cached" not in first
    assert second["ok"] is True and second["cached"] is False
    assert len(upstream.calls) == 2


async def test_error_timeout_is_not_cached(upstream: Upstream) -> None:
    upstream.default = {"ok": False, "error": {"code": "timeout", "message": "timed out"}}
    await server.deepseek_fetch(URL)
    await server.deepseek_fetch(URL)
    assert len(upstream.calls) == 2


async def test_empty_page_is_not_cached(upstream: Upstream) -> None:
    upstream.responses = [{"ok": True, "data": {"title": "Empty", "content": ""}}]
    first = await server.deepseek_fetch(URL)
    second = await server.deepseek_fetch(URL)
    assert first["content"] == "" and first["notes"]
    assert second["cached"] is False
    assert len(upstream.calls) == 2


async def test_cache_disabled_always_hits_upstream(upstream: Upstream) -> None:
    set_settings(Settings(cache_enabled=False))
    await server.deepseek_fetch(URL)
    second = await server.deepseek_fetch(URL)
    assert second["cached"] is False
    assert len(upstream.calls) == 2


async def test_cache_key_uses_exact_url_without_fragment() -> None:
    key = server._fetch_cache_key(URL)
    assert server._fetch_cache_key(f"  {URL}  ") == key
    assert server._fetch_cache_key(f"{URL}#section-2") == key
    assert server._fetch_cache_key(URL + "/") != key
    assert server._fetch_cache_key("https://EXAMPLE.com/article") != key
    assert server._fetch_cache_key(URL + "?utm_source=x") != key


async def test_fragment_variants_share_one_upstream_call(upstream: Upstream) -> None:
    await server.deepseek_fetch(URL)
    assert (await server.deepseek_fetch(f"{URL}#comments"))["cached"] is True
    assert len(upstream.calls) == 1


async def test_fetch_and_search_caches_are_separate(upstream: Upstream) -> None:
    await server.deepseek_fetch(URL)
    assert server._fetch_cache_key(URL) != server._search_cache_key({"url": URL})
    assert len(server._get_fetch_cache()) == 1
    assert server._get_search_cache() is not None and len(server._get_search_cache()) == 0


async def test_cache_hit_takes_no_token(upstream: Upstream, limiter: RateLimiter) -> None:
    bucket = limiter.buckets["fetch"]
    await server.deepseek_fetch(URL)
    await server.deepseek_fetch(URL, max_chars=50)
    assert bucket.tokens == bucket.burst - 1
    assert limiter.buckets["search"].tokens == limiter.buckets["search"].burst


async def test_concurrent_fetches_share_one_call_and_one_token(
    upstream: Upstream, limiter: RateLimiter
) -> None:
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(server.deepseek_fetch(URL, max_chars=n)) for n in (10, 50, 100)]
    # Every caller joins the single flight before its first await, so all three are waiting
    # by the time the shared call reaches upstream.
    await upstream.entered.wait()
    upstream.gate.set()
    results = await asyncio.gather(*tasks)
    assert len(upstream.calls) == 1
    assert [len(r["content"]) for r in results] == sorted(len(r["content"]) for r in results)
    bucket = limiter.buckets["fetch"]
    assert bucket.tokens == bucket.burst - 1


async def test_cancelled_leader_refunds_its_token(upstream: Upstream, limiter: RateLimiter) -> None:
    upstream.gate = asyncio.Event()
    task = asyncio.create_task(server.deepseek_fetch(URL))
    await upstream.entered.wait()
    assert len(upstream.calls) == 1
    bucket = limiter.buckets["fetch"]
    assert bucket.tokens == bucket.burst - 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The refund runs as the cancellation leaves the upstream call, before ``cancelled`` wakes us.
    await upstream.cancelled.wait()
    assert bucket.tokens == bucket.burst


async def test_local_rate_limit_is_reported_and_not_cached(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch, fake_clock: Any
) -> None:
    settings = Settings(fetch_qps=0.01, fetch_burst=1, limiter_max_wait=1)
    monkeypatch.setattr(
        server, "_rate_limiter", RateLimiter.from_settings(settings, clock=fake_clock.monotonic)
    )
    assert (await server.deepseek_fetch(URL))["ok"] is True
    limited = await server.deepseek_fetch(URL, refresh=True)
    assert limited["ok"] is False
    assert limited["error"]["code"] == "rate_limited_local"
    assert len(upstream.calls) == 1
    assert (await server.deepseek_fetch(URL))["cached"] is True


async def test_invalid_url_touches_neither_cache_nor_limiter(
    upstream: Upstream, limiter: RateLimiter
) -> None:
    result = await server.deepseek_fetch("ftp://example.com/file")
    assert result["error"]["code"] == "bad_url"
    assert upstream.calls == []
    assert limiter.buckets["fetch"].tokens == limiter.buckets["fetch"].burst


async def test_reset_state_drops_cached_pages(upstream: Upstream) -> None:
    await server.deepseek_fetch(URL)
    server.reset_state()
    assert (await server.deepseek_fetch(URL))["cached"] is False
    assert len(upstream.calls) == 2


@pytest.mark.parametrize(
    ("url", "advice"),
    [
        (URL, False),
        ("https://example.com/report.PDF?download=1#page=2", True),
        ("https://www.reddit.com/r/example/comments/abc123/title/", True),
        ("https://old.reddit.com/r/example/", True),
        ("https://redd.it/abc123", True),
        ("https://example.com/pdf-guide", False),
        ("https://arxiv.org/pdf/1706.03762", True),
        ("https://arxiv.org/pdf/1706.03762v7/", True),
        ("https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1.full.pdf", True),
        ("https://www.medrxiv.org/content/10.1101/2024.02.02.24302222v1/PDF/x", True),
        ("https://example.com/pdf/", False),
        ("https://example.com/pdf/guide/intro", False),
        ("https://example.com/404", False),
        ("https://notreddit.com/page", False),
    ],
)
async def test_404_suggests_search_only_for_pdf_and_reddit(
    upstream: Upstream, url: str, advice: bool
) -> None:
    upstream.responses = [
        {"ok": False, "error": {"code": "http_404", "message": "failed to fetch URL", "status": 404}}
    ]
    result = await server.deepseek_fetch(url)
    assert result["ok"] is False
    assert result["error"]["code"] == "http_404"
    assert result["error"]["status"] == 404
    assert ("deepseek_search_and_read" in result["error"]["message"]) is advice
    assert result["error"]["message"].startswith("failed to fetch URL")
