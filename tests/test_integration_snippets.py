"""Integration tests for lexical snippets and sectional pagination in fetch tools."""

from __future__ import annotations

import asyncio

import pytest

from deepseek_search_everywhere_mcp import server

SAMPLE_PAGE = """# First Chapter: Classical Bits

Classical computers use bits representing 0 or 1.

## Second Chapter: Quantum Mechanics

Quantum computers use qubits with superposition and entanglement.

## Third Chapter: Fault Tolerance

Error correction codes are essential to scale physical qubits.
"""


def _install_fetch(fake_upstream, content: str = SAMPLE_PAGE):
    return fake_upstream(
        responses={"/v1/tools/fetch": {"ok": True, "data": {"title": "Quantum Guide", "content": content}}}
    )


def test_fetch_with_query_snippets(fake_upstream):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", query="superposition"))
    assert result["ok"] is True
    assert result["query"] == "superposition"
    assert result["snippets_count"] >= 1
    assert "Quantum Mechanics" in result["content"]
    assert "superposition" in result["content"]


def test_fetch_with_section_pagination(fake_upstream):
    _install_fetch(fake_upstream)
    # Section 1
    s1 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", section=1))
    assert s1["ok"] is True
    assert s1["section"] == 1
    assert s1["total_sections"] == 3
    assert s1["has_more"] is True
    assert "Classical Bits" in s1["content"]

    # Section 3 (final)
    s3 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", section=3))
    assert s3["ok"] is True
    assert s3["section"] == 3
    assert s3["has_more"] is False
    assert "Fault Tolerance" in s3["content"]


def test_fetch_section_out_of_bounds(fake_upstream):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", section=10))
    assert result["ok"] is False
    assert result["error"]["code"] == "bad_request"
    assert "out of bounds" in result["error"]["message"]


def test_fetch_many_with_query(fake_upstream):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch_many(["https://example.com/quantum"], query="qubits"))
    assert result["ok"] is True
    assert len(result["results"]) == 1
    item = result["results"][0]
    assert item["ok"] is True
    assert "Excerpt" in item["content"]


def test_fetch_offset_limit_pagination(fake_upstream):
    _install_fetch(fake_upstream)
    # Fetch first 30 chars
    p1 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", offset=0, limit=30))
    assert p1["ok"] is True
    assert p1["offset"] == 0
    assert p1["chars"] == 30
    assert p1["total_chars"] == len(SAMPLE_PAGE)
    assert p1["has_more"] is True
    assert p1["content"] == SAMPLE_PAGE[:30]

    # Fetch next 30 chars
    p2 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", offset=30, limit=30))
    assert p2["ok"] is True
    assert p2["offset"] == 30
    assert p2["chars"] == 30
    assert p2["has_more"] is True
    assert p2["content"] == SAMPLE_PAGE[30:60]


def test_fetch_offset_out_of_bounds(fake_upstream):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", offset=99999))
    assert result["ok"] is True
    assert result["offset"] == 99999
    assert result["chars"] == 0
    assert result["content"] == ""
    assert result["has_more"] is False


def test_fetch_offset_limit_rejections(fake_upstream):
    _install_fetch(fake_upstream)
    res1 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", offset=-1))
    assert res1["ok"] is False
    assert res1["error"]["code"] == "bad_request"
    assert "offset must be at least 0" in res1["error"]["message"]

    res2 = asyncio.run(server.deepseek_fetch("https://example.com/quantum", limit=0))
    assert res2["ok"] is False
    assert res2["error"]["code"] == "bad_request"
    assert "limit must be greater than 0" in res2["error"]["message"]


def test_fetch_find_text_success(fake_upstream):
    _install_fetch(fake_upstream)
    # Search for "quantum mechanics" (case-insensitive)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", find_text="QUANTUM MECHANICS"))
    assert result["ok"] is True
    assert result["find_text"] == "QUANTUM MECHANICS"
    assert "Quantum Mechanics" in result["content"]
    assert result["offset"] == SAMPLE_PAGE.lower().find("quantum mechanics")


def test_fetch_find_text_not_found(fake_upstream):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", find_text="nonexistent text"))
    assert result["ok"] is True
    assert result["content"] == ""
    assert result["chars"] == 0
    assert any("not found in page content" in note for note in result["notes"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"section": 1, "query": "test"},
        {"section": 1, "offset": 10},
        {"section": 1, "limit": 50},
        {"section": 1, "find_text": "test"},
        {"query": "test", "offset": 10},
        {"query": "test", "limit": 50},
        {"query": "test", "find_text": "test"},
        {"find_text": "test", "offset": 10},
        {"find_text": "test", "limit": 50},
        {"section": 1, "query": "test", "find_text": "test"},
    ],
)
def test_fetch_conflicting_selectors_rejected(fake_upstream, kwargs):
    _install_fetch(fake_upstream)
    result = asyncio.run(server.deepseek_fetch("https://example.com/quantum", **kwargs))
    assert result["ok"] is False
    assert result["error"]["code"] == "bad_request"
    assert (
        "Selectors section, query, offset/limit, and find_text are mutually exclusive."
        in result["error"]["message"]
    )


def _long_page() -> str:
    filler = "Routine maintenance paragraph with unrelated operational details only. " * 5
    block = "\n\n".join([filler] * 30)
    return (
        block
        + "\n\nAn incremental backup is created with pg_basebackup --incremental.\n\n"
        + block
        + "\n\nCombining an incremental backup requires pg_combinebackup.\n\n"
        + block
    )


def test_fetch_query_returns_short_focused_excerpts(fake_upstream):
    page = _long_page()
    assert len(page) > 30_000
    _install_fetch(fake_upstream, page)
    result = asyncio.run(
        server.deepseek_fetch(
            "https://example.com/pg", query="incremental backup", max_snippets=2, max_chars=20_000
        )
    )
    assert result["ok"] is True
    assert result["snippets_count"] == 2
    parts = [p for p in result["content"].split("\n\n---\n\n") if p.strip()]
    assert len(parts) == 2
    assert all("incremental backup" in p.lower() for p in parts)
    assert all(len(p) <= 1700 for p in parts)
    assert result["chars"] < 3500
    assert result["truncated"] is False


def test_fetch_query_without_matches_has_clear_note_and_empty_content(fake_upstream):
    _install_fetch(fake_upstream, _long_page())
    result = asyncio.run(server.deepseek_fetch("https://example.com/pg", query="kubernetes ingress"))
    assert result["ok"] is True
    assert result["content"] == ""
    assert result["snippets_count"] == 0
    assert any("No excerpt" in n and "kubernetes ingress" in n for n in result["notes"])
    assert not any("PDF" in n for n in result["notes"])


def test_fetch_many_query_returns_focused_excerpts(fake_upstream):
    _install_fetch(fake_upstream, _long_page())
    result = asyncio.run(
        server.deepseek_fetch_many(
            ["https://example.com/pg"], query="incremental backup", max_chars_per_url=20_000
        )
    )
    item = result["results"][0]
    assert item["ok"] is True
    assert len(item["content"]) < 3500
    assert "incremental backup" in item["content"].lower()
