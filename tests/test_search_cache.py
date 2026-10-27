"""Search wiring: response cache, single-flight, rate limiter and rewrite cache."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server, transport
from deepseek_search_everywhere_mcp.ratelimit import RateLimiter
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

RESULTS = [
    {"title": "One", "url": "https://example.com/one", "site_name": "Example",
     "content": "a" * 500},
    {"title": "One", "url": "https://example.com/one/", "site_name": "Example",
     "content": "b" * 500},
    {"title": "Two", "url": "https://example.com/two", "site_name": "Example",
     "content": "c" * 500},
]


class Upstream:
    """Fake ``server._request`` that counts search calls and serves a fixed response."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"ok": True, "data": {"results": RESULTS}}
        self.calls: list[dict[str, Any]] = []
        self.gate: asyncio.Event | None = None

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"path": path, **kwargs})
        if self.gate is not None:
            await self.gate.wait()
        return self.response


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    fake = Upstream()
    monkeypatch.setattr(server, "_request", fake)
    monkeypatch.setattr(server, "_api_key", lambda: "test-key")
    return fake


@pytest.fixture
def limiter(monkeypatch: pytest.MonkeyPatch, fake_clock: Any) -> RateLimiter:
    """A limiter on a frozen clock, so token balances are exact."""
    instance = RateLimiter.from_settings(Settings(), clock=fake_clock.monotonic)
    monkeypatch.setattr(server, "_rate_limiter", instance)
    return instance


async def _search(query: str = "test query", *, total: int = 4000, dedupe: bool = True,
                  rewrite: bool = False, refresh: bool = False) -> dict[str, Any]:
    return await server._search(query, 5, None, True, None, 1000, total, dedupe, rewrite,
                                refresh=refresh)


async def test_second_identical_search_is_served_from_cache(upstream: Upstream) -> None:
    first = await _search()
    second = await _search()
    assert len(upstream.calls) == 1
    assert first["ok"] is True and second["ok"] is True
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["results"] == second["results"]
    assert not any("cache" in note for note in second["notes"])


async def test_shaping_options_share_one_upstream_call(upstream: Upstream) -> None:
    wide = await _search(total=4000)
    narrow = await _search(total=600)
    raw = await _search(dedupe=False)
    assert len(upstream.calls) == 1
    assert narrow["cached"] is True and raw["cached"] is True
    assert sum(len(r.get("content", "")) for r in narrow["results"]) < sum(
        len(r.get("content", "")) for r in wide["results"]
    )
    assert wide["deduped"] == 1 and wide["count"] == 2
    assert raw["deduped"] == 0 and raw["count"] == 3


async def test_cached_entry_is_not_mutated_by_shaping(upstream: Upstream) -> None:
    await _search(total=10)
    again = await _search(total=4000)
    assert again["cached"] is True
    assert again["results"][0]["content"] == "a" * 500


async def test_refresh_bypasses_cache_read(upstream: Upstream) -> None:
    await _search()
    refreshed = await _search(refresh=True)
    assert len(upstream.calls) == 2
    assert refreshed["cached"] is False
    assert (await _search())["cached"] is True
    assert len(upstream.calls) == 2


async def test_errors_are_not_cached(upstream: Upstream) -> None:
    upstream.response = {"ok": False, "error": {"code": "http_500", "message": "boom"}}
    first = await _search()
    second = await _search()
    assert first["ok"] is False and second["ok"] is False
    assert "cached" not in second
    assert len(upstream.calls) == 2


async def test_empty_results_are_not_cached(upstream: Upstream) -> None:
    upstream.response = {"ok": True, "data": {"results": []}}
    await _search()
    second = await _search()
    assert second["cached"] is False
    assert len(upstream.calls) == 2


async def test_cache_disabled_always_hits_upstream(upstream: Upstream) -> None:
    set_settings(Settings(cache_enabled=False))
    await _search()
    second = await _search()
    assert second["cached"] is False
    assert len(upstream.calls) == 2


async def test_api_key_is_not_part_of_the_cache_key(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _search()
    monkeypatch.setattr(server, "_api_key", lambda: "another-key")
    assert (await _search())["cached"] is True
    assert len(upstream.calls) == 1


async def test_cache_key_depends_only_on_endpoint_and_payload() -> None:
    payload = {"query": "q", "count": 5, "need_content": True}
    reordered = {"need_content": True, "count": 5, "query": "q"}
    assert server._search_cache_key(payload) == server._search_cache_key(reordered)
    assert server._search_cache_key(payload) != server._search_cache_key({**payload, "count": 6})


async def test_rewrite_is_cached(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    rewrites: list[str] = []

    async def fake_rewrite(query: str) -> str:
        rewrites.append(query)
        return "how sleep works"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    first = await _search("jak działa sen", rewrite=True)
    second = await _search("jak działa sen", rewrite=True)
    assert rewrites == ["jak działa sen"]
    assert len(upstream.calls) == 1
    assert second["cached"] is True
    assert second["query_original"] == "jak działa sen"
    assert "query rewritten (flash): 'how sleep works'" in second["notes"]
    assert first["notes"] == second["notes"]


async def test_failed_rewrite_is_not_cached(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    rewrites: list[str] = []

    async def fake_rewrite(query: str) -> None:
        rewrites.append(query)
        return None

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    await _search(rewrite=True)
    second = await _search(rewrite=True)
    assert len(rewrites) == 2
    assert "query rewrite failed; using original query" in second["notes"]


async def test_refresh_still_uses_rewrite_cache(upstream: Upstream, monkeypatch: pytest.MonkeyPatch) -> None:
    rewrites: list[str] = []

    async def fake_rewrite(query: str) -> str:
        rewrites.append(query)
        return "rewritten query"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    await _search(rewrite=True)
    await _search(rewrite=True, refresh=True)
    assert len(rewrites) == 1
    assert len(upstream.calls) == 2


async def test_concurrent_rewrites_share_one_flash_call(upstream: Upstream,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    gate = asyncio.Event()
    paths: list[str] = []

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        paths.append(path)
        if path == "/v1/chat/completions":
            await gate.wait()
            return {"ok": True, "data": {"choices": [{"message": {"content": "sleep science"}}]}}
        return {"ok": True, "data": {"results": RESULTS}}

    monkeypatch.setattr(server, "_request", fake_request)
    tasks = [asyncio.create_task(_search("jak działa sen", rewrite=True)) for _ in range(10)]
    await asyncio.sleep(0)
    gate.set()
    envelopes = await asyncio.gather(*tasks)
    assert paths.count("/v1/chat/completions") == 1
    assert paths.count(server.SEARCH_PATH) == 1
    assert all(envelope["ok"] is True for envelope in envelopes)
    assert server._get_rewrite_flight().in_flight() == 0


async def test_cache_hit_takes_no_token(upstream: Upstream, limiter: RateLimiter) -> None:
    bucket = limiter.buckets["search"]
    await _search()
    await _search()
    assert bucket.tokens == bucket.burst - 1


async def test_concurrent_searches_share_one_call_and_one_token(
    upstream: Upstream, limiter: RateLimiter
) -> None:
    upstream.gate = asyncio.Event()
    tasks = [asyncio.create_task(_search()) for _ in range(5)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    upstream.gate.set()
    results = await asyncio.gather(*tasks)
    assert len(upstream.calls) == 1
    assert all(r["ok"] for r in results)
    bucket = limiter.buckets["search"]
    assert bucket.tokens == bucket.burst - 1


async def test_cancelled_leader_refunds_its_token(upstream: Upstream, limiter: RateLimiter) -> None:
    upstream.gate = asyncio.Event()
    task = asyncio.create_task(_search())
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(upstream.calls) == 1
    bucket = limiter.buckets["search"]
    assert bucket.tokens == bucket.burst - 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(3):
        await asyncio.sleep(0)
    assert bucket.tokens == bucket.burst


async def test_local_rate_limit_is_reported_and_not_cached(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch, fake_clock: Any
) -> None:
    settings = Settings(search_qps=0.01, search_burst=1, limiter_max_wait=1)
    monkeypatch.setattr(server, "_rate_limiter",
                        RateLimiter.from_settings(settings, clock=fake_clock.monotonic))
    assert (await _search())["ok"] is True
    limited = await _search(refresh=True)
    assert limited["ok"] is False
    assert limited["error"]["code"] == "rate_limited_local"
    assert len(upstream.calls) == 1
    assert (await _search())["cached"] is True


async def test_transport_retries_take_a_single_token(
    monkeypatch: pytest.MonkeyPatch, limiter: RateLimiter
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    attempts: list[str] = []

    async def flaky(method: str, url: str, path: str, **kwargs: Any) -> tuple[dict[str, Any], float | None]:
        attempts.append(path)
        if len(attempts) == 1:
            return {"ok": False, "error": {"code": "http_503", "message": "busy"}}, 0.0
        return {"ok": True, "data": {"results": RESULTS}}, None

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(transport, "_attempt", flaky)
    monkeypatch.setattr(transport.asyncio, "sleep", no_sleep)
    result = await _search()
    assert result["ok"] is True
    assert len(attempts) == 2
    bucket = limiter.buckets["search"]
    assert bucket.tokens == bucket.burst - 1


async def test_reset_state_drops_cached_searches(upstream: Upstream) -> None:
    await _search()
    server.reset_state()
    assert (await _search())["cached"] is False
    assert len(upstream.calls) == 2
