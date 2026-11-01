"""Integration tests for lexical snippets and sectional pagination in fetch tools."""

from __future__ import annotations

import asyncio

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
