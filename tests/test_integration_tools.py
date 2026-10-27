"""Cross-tool behaviour: SSRF, batching, research and the caches/limiter working together.

Per-tool cases live in ``test_fetch_many.py`` and ``test_research.py``; these tests combine
them. Everything is offline: ``server._request`` is stubbed and the limiter runs on a fake
clock with an injected ``sleep``, so nothing waits for real time.
"""

from __future__ import annotations

from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.ratelimit import RateLimiter
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

PUBLIC = "https://example.com/page"
INTERNAL = "http://127.0.0.1/admin"
OTHER = "https://example.org/other"
SHORT = "short"
LONG = "Search snippet about the topic. " * 40


class Upstream:
    """Fake ``server._request``: search results are configurable, fetches echo the URL."""

    def __init__(self, results: list[dict[str, Any]] | None = None, clock: Any = None) -> None:
        self.results = results or []
        self.clock = clock
        self.searches: list[dict[str, Any]] = []
        self.fetches: list[str] = []
        self.fetch_times: list[float] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.searches.append(payload)
            return {"ok": True, "data": {"results": self.results}}
        if path == server.FETCH_PATH:
            self.fetches.append(payload["url"])
            if self.clock is not None:
                self.fetch_times.append(self.clock.monotonic())
            body = f"Fetched {payload['url']} " + "z" * 200
            return {"ok": True, "data": {"title": "Fetched", "content": body}}
        raise AssertionError(f"unexpected upstream path {path!r}")


def _result(url: str, content: str) -> dict[str, Any]:
    return {"title": f"T {url}", "url": url, "authority": 2, "rerank_score": 0.5,
            "site_name": "Site", "content": content}


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    def _install(upstream: Upstream) -> Upstream:
        monkeypatch.setattr(server, "_request", upstream)
        monkeypatch.setattr(server, "_api_key", lambda: "test-key")
        return upstream

    return _install


async def test_mixed_batch_makes_one_upstream_call_and_reports_two_errors(install: Any) -> None:
    upstream = install(Upstream())
    urls = [PUBLIC, INTERNAL, PUBLIC, "not a url", OTHER]
    result = await server.deepseek_fetch_many(urls)
    assert result["ok"] is True
    assert [item["url"] for item in result["results"]][0] == PUBLIC
    errors = [(item["url"], item["error"]["code"]) for item in result["results"] if not item["ok"]]
    assert errors == [(INTERNAL, "ssrf_blocked"), ("not a url", "bad_url")]
    assert [item["ok"] for item in result["results"]] == [True, False, False, True]
    assert sorted(upstream.fetches) == sorted([PUBLIC, OTHER])


async def test_research_marks_internal_source_ssrf_blocked_and_still_succeeds(install: Any) -> None:
    results = [_result(PUBLIC, SHORT), _result(INTERNAL, SHORT), _result(OTHER, LONG)]
    upstream = install(Upstream(results))
    result = await server.deepseek_research("internal source question")
    assert result["ok"] is True
    by_url = {source["url"]: source for source in result["sources"]}
    assert by_url[INTERNAL]["error"]["code"] == "ssrf_blocked"
    assert by_url[INTERNAL]["origin"] == "search"
    assert by_url[PUBLIC]["origin"] == "fetch" and by_url[PUBLIC]["error"] is None
    assert by_url[OTHER]["origin"] == "search"
    assert upstream.fetches == [PUBLIC]


async def test_search_content_is_not_served_as_a_fetch_cache_hit(install: Any) -> None:
    results = [_result(PUBLIC, LONG)]
    upstream = install(Upstream(results))
    search = await server.deepseek_search_and_read("topic")
    assert search["ok"] is True and search["cached"] is False
    fetched = await server.deepseek_fetch(PUBLIC)
    assert fetched["ok"] is True
    assert fetched["cached"] is False
    assert upstream.fetches == [PUBLIC]
    assert len(server._get_search_cache()) == 1
    assert len(server._get_fetch_cache()) == 1
    assert (await server.deepseek_fetch(PUBLIC))["cached"] is True
    assert (await server.deepseek_search_and_read("topic"))["cached"] is True
    assert len(upstream.searches) == 1 and upstream.fetches == [PUBLIC]


async def test_batch_of_ten_is_paced_by_the_limiter_without_blocking(
    monkeypatch: pytest.MonkeyPatch, fake_clock: Any
) -> None:
    upstream = Upstream(clock=fake_clock)
    monkeypatch.setattr(server, "_request", upstream)
    monkeypatch.setattr(server, "_api_key", lambda: "test-key")
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        fake_clock.advance(delay)

    settings = Settings(fetch_qps=5, fetch_burst=5, limiter_max_wait=5)
    set_settings(settings)
    monkeypatch.setattr(server, "_rate_limiter",
                        RateLimiter.from_settings(settings, clock=fake_clock.monotonic, sleep=sleep))
    urls = [f"https://example.com/{i}" for i in range(10)]
    result = await server.deepseek_fetch_many(urls)
    assert result["ok"] is True and all(item["ok"] for item in result["results"])
    assert len(upstream.fetches) == 10
    # A bucket of burst 5 refilled at 5/s never lets the n-th call start before (n - 5) / 5 s.
    for n, started in enumerate(sorted(upstream.fetch_times), 1):
        assert started >= (n - 5) / 5 - 1e-9
    assert upstream.fetch_times.count(0.0) == 5
    assert sum(delays) == pytest.approx(max(upstream.fetch_times))
    assert max(upstream.fetch_times) == pytest.approx(1.0)
