"""Search wiring: cache keys, shaping, dedupe, refresh and the rewrite cache."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import runtime, server

RESULTS = [
    {"title": "One", "url": "https://example.com/one", "site_name": "Example", "content": "a" * 500},
    {"title": "One", "url": "https://example.com/one/", "site_name": "Example", "content": "b" * 500},
    {"title": "Two", "url": "https://example.com/two", "site_name": "Example", "content": "c" * 500},
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
def upstream(fake_upstream: Any) -> Upstream:
    return fake_upstream(Upstream())


async def _search(
    query: str = "test query",
    *,
    total: int = 4000,
    dedupe: bool = True,
    rewrite: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    return await server._search(query, 5, None, True, None, 1000, total, dedupe, rewrite, refresh=refresh)


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
    assert len(upstream.calls) == 1
    assert narrow["cached"] is True
    assert sum(len(r.get("content", "")) for r in narrow["results"]) < sum(
        len(r.get("content", "")) for r in wide["results"]
    )
    assert wide["deduped"] == 1 and wide["count"] == 2


async def test_dedupe_off_uses_its_own_upstream_request(upstream: Upstream) -> None:
    await _search()
    raw = await _search(dedupe=False)
    assert len(upstream.calls) == 2  # dedupe=False does not over-fetch, so the payload differs
    assert raw["cached"] is False
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


async def test_api_key_is_not_part_of_the_cache_key(
    upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _search()
    monkeypatch.setattr(runtime, "_api_key", lambda: "another-key")
    assert (await _search())["cached"] is True
    assert len(upstream.calls) == 1


async def test_cache_key_depends_only_on_endpoint_and_payload() -> None:
    payload = {"query": "q", "count": 5, "need_content": True}
    reordered = {"need_content": True, "count": 5, "query": "q"}
    assert server._search_cache_key(payload) == server._search_cache_key(reordered)
    assert server._search_cache_key(payload) != server._search_cache_key({**payload, "count": 6})


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


async def test_concurrent_rewrites_share_one_flash_call(upstream: Upstream, fake_upstream: Any) -> None:
    gate = asyncio.Event()
    paths: list[str] = []

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        paths.append(path)
        if path == "/v1/chat/completions":
            await gate.wait()
            return {"ok": True, "data": {"choices": [{"message": {"content": "sleep science"}}]}}
        return {"ok": True, "data": {"results": RESULTS}}

    fake_upstream(fake_request)
    tasks = [asyncio.create_task(_search("jak działa sen", rewrite=True)) for _ in range(10)]
    await asyncio.sleep(0)
    gate.set()
    envelopes = await asyncio.gather(*tasks)
    assert paths.count("/v1/chat/completions") == 1
    assert paths.count(server.SEARCH_PATH) == 1
    assert all(envelope["ok"] is True for envelope in envelopes)
    assert server._get_rewrite_flight().in_flight() == 0
