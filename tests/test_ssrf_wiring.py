"""SSRF wiring: blocked URLs are rejected before the cache, limiter or upstream is touched."""

from __future__ import annotations

from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server


class Upstream:
    """Fake ``server._request`` that counts calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"path": path, **kwargs})
        return {"ok": True, "data": {"title": "T", "content": "body"}}


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    fake = Upstream()
    monkeypatch.setattr(server, "_request", fake)
    return fake


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "https://printer.local/",
        "https://user:pass@example.com/",
    ],
)
async def test_blocked_urls_never_reach_upstream(upstream: Upstream, url: str) -> None:
    result = await server.deepseek_fetch(url)
    assert result["ok"] is False
    assert result["error"]["code"] == "ssrf_blocked"
    assert upstream.calls == []


@pytest.mark.parametrize("refresh", [False, True])
async def test_blocked_url_is_not_cached_or_rate_limited(upstream: Upstream, refresh: bool) -> None:
    for _ in range(3):
        result = await server.deepseek_fetch("http://10.0.0.1/", refresh=refresh)
        assert result["error"]["code"] == "ssrf_blocked"
    assert upstream.calls == []


async def test_wrong_scheme_keeps_bad_url_code(upstream: Upstream) -> None:
    result = await server.deepseek_fetch("ftp://example.com/file")
    assert result["error"]["code"] == "bad_url"
    assert upstream.calls == []


async def test_malformed_url_maps_to_bad_url(upstream: Upstream) -> None:
    result = await server.deepseek_fetch("http://")
    assert result["error"]["code"] == "bad_url"
    assert upstream.calls == []


async def test_public_url_proceeds_to_upstream(upstream: Upstream) -> None:
    result = await server.deepseek_fetch("https://example.com/a")
    assert result["ok"] is True
    assert len(upstream.calls) == 1
    assert upstream.calls[0]["payload"]["url"] == "https://example.com/a"


def test_reject_url_helper_is_reusable() -> None:
    assert server._reject_url("https://example.com/") is None
    rejected = server._reject_url("http://127.0.0.1/")
    assert rejected is not None and rejected["error"]["code"] == "ssrf_blocked"
    rejected = server._reject_url("http://")
    assert rejected is not None and rejected["error"]["code"] == "bad_url"
