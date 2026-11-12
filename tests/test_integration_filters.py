"""Integration tests for explicit search filters and budget guarding."""

from __future__ import annotations

import asyncio
from typing import Any

from deepseek_search_everywhere_mcp import server


def _install_search(fake_upstream, items: list[dict[str, Any]] | None = None):
    return fake_upstream(responses={"/v1/tools/search": {"ok": True, "data": {"results": items or []}}})


def test_filters_applied_to_query_used(fake_upstream):
    _install_search(fake_upstream)
    result = asyncio.run(
        server.deepseek_search(
            "quantum",
            include_domains=["arxiv.org", "nature.com"],
            exclude_domains=["wikipedia.org"],
            after="2024-01-01",
            before="2025-01-01",
        )
    )
    assert result["ok"] is True
    used = result["query_used"]
    assert "site:arxiv.org" in used
    assert "site:nature.com" in used
    assert "-site:wikipedia.org" in used
    assert "after:2024-01-01" in used
    assert "before:2025-01-01" in used
    assert any("Filters applied:" in note for note in result["notes"])


def test_filters_reject_bad_domain(fake_upstream):
    _install_search(fake_upstream)
    result = asyncio.run(server.deepseek_search("test", include_domains=["bad domain with spaces"]))
    assert result["ok"] is False
    assert result["error"]["code"] == "bad_domain"


def test_filters_accept_idn_domains_and_send_unicode_to_search(fake_upstream):
    upstream = _install_search(fake_upstream)
    result = asyncio.run(
        server.deepseek_search(
            "klimat",
            include_domains=["źródło.pl", "münchen.de"],
            exclude_domains=["xn--rdo-gna01bwk.pl"],
        )
    )
    assert result["ok"] is True
    used = result["query_used"]
    assert "site:źródło.pl" in used
    assert "site:münchen.de" in used
    assert "-site:źródło.pl" in used
    call_query = upstream.calls[0]["payload"]["query"]
    assert "site:źródło.pl" in call_query
    assert "site:münchen.de" in call_query
    assert "-site:źródło.pl" in call_query


def test_filters_reject_bad_date(fake_upstream):
    _install_search(fake_upstream)
    result = asyncio.run(server.deepseek_search("test", after="2024-02-31"))
    assert result["ok"] is False
    assert result["error"]["code"] == "bad_date"


def test_filters_budget_guard_rejects_overflow(fake_upstream):
    _install_search(fake_upstream)
    # 80 chars query + 60 chars filter = > 128 chars
    long_query = "a" * 80
    result = asyncio.run(
        server.deepseek_search(
            long_query,
            include_domains=["longdomainname1.org", "longdomainname2.org"],
            after="2024-01-01",
        )
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "query_too_long"
    assert "when filters" in result["error"]["message"]


def test_search_full_supports_filters(fake_upstream):
    _install_search(
        fake_upstream,
        [
            {
                "title": "Quantum Paper",
                "url": "https://arxiv.org/abs/123",
                "content": "Full text of quantum paper content over 200 chars " * 5,
            }
        ],
    )
    result = asyncio.run(
        server.deepseek_search(
            "quantum",
            content="full",
            include_domains=["arxiv.org"],
            after="2024-01-01",
        )
    )
    assert result["ok"] is True
    assert "site:arxiv.org" in result["query_used"]
    assert "after:2024-01-01" in result["query_used"]


MUENCHEN_RESULTS = [
    {"title": "Zugspitze", "url": "https://www.münchen.de/wetter", "content": "Zugspitze Wetter heute"},
    {"title": "ORAS", "url": "https://www.orasonline.lt/zugspitze", "content": "Zugspitze oras"},
    {"title": "Yr", "url": "https://www.yr.no/en/forecast/zugspitze", "content": "Zugspitze forecast"},
]


def test_include_domains_enforced_on_returned_results_idn(fake_upstream):
    _install_search(fake_upstream, MUENCHEN_RESULTS)
    result = asyncio.run(server.deepseek_search("Zugspitze Wetter", include_domains=["münchen.de"]))
    assert result["ok"] is True
    assert [r["url"] for r in result["results"]] == ["https://www.münchen.de/wetter"]
    assert result["count"] == 1
    assert any("Dropped 2 result(s)" in n and "domain filter" in n for n in result["notes"])


def test_exclude_domains_enforced_on_returned_results(fake_upstream):
    _install_search(fake_upstream, MUENCHEN_RESULTS)
    result = asyncio.run(server.deepseek_search("Zugspitze", exclude_domains=["yr.no", "orasonline.lt"]))
    assert [r["url"] for r in result["results"]] == ["https://www.münchen.de/wetter"]
    assert any("Dropped 2 result(s)" in n for n in result["notes"])


def test_no_domain_filter_note_when_nothing_dropped(fake_upstream):
    _install_search(fake_upstream, MUENCHEN_RESULTS)
    result = asyncio.run(server.deepseek_search("Zugspitze"))
    assert result["count"] == 3
    assert not any("domain filter" in n for n in result["notes"])


def test_search_many_enforces_domain_filters(fake_upstream):
    _install_search(fake_upstream, MUENCHEN_RESULTS)
    result = asyncio.run(
        server.deepseek_search_many(["Zugspitze Wetter", "Zugspitze Schnee"], include_domains=["münchen.de"])
    )
    assert result["ok"] is True
    for item in result["results"]:
        urls = [r["url"] for r in item["results"]]
        assert urls == ["https://www.münchen.de/wetter"]
