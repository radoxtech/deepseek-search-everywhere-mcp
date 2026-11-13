"""Unit tests for deep research decomposition, budgeting, and citation parsing."""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp.deep_research import (
    DEPTH_CONFIG,
    build_decomposition_messages,
    build_synthesis_messages,
    build_synthesis_system_prompt,
    parse_subqueries,
    report_body_words,
    sanitize_citations,
    select_synthesis_context,
)


def test_build_decomposition_messages():
    messages = build_decomposition_messages("quantum computing", max_queries=3)
    assert len(messages) == 2
    assert "quantum computing" in messages[1]["content"]


def test_parse_subqueries_success():
    raw_response = '{"queries": ["quantum error correction", "qubits superposition"]}'
    queries = parse_subqueries(raw_response, "quantum", max_queries=3)
    assert len(queries) == 2
    assert queries[0] == "quantum error correction"
    assert queries[1] == "qubits superposition"


def test_parse_subqueries_fallback_on_invalid():
    queries = parse_subqueries("invalid json", "quantum computing", max_queries=3)
    assert len(queries) == 1
    assert queries[0] == "quantum computing"


def test_select_synthesis_context_caps_at_sources_and_chars():
    sources = [
        {"id": i, "title": f"T{i}", "url": f"https://a.test/{i}", "content": "x" * 5000, "rerank_score": 2.0}
        for i in range(1, 30)
    ]
    selected_low = select_synthesis_context(sources, depth="low")
    assert len(selected_low) == 6
    total_chars_low = sum(len(s["content"]) for s in selected_low)
    assert total_chars_low <= 60000
    assert [s["id"] for s in selected_low] == list(range(1, len(selected_low) + 1))

    huge_sources = [
        {"id": i, "title": f"T{i}", "url": f"https://a.test/{i}", "content": "x" * 30000, "rerank_score": 2.0}
        for i in range(1, 30)
    ]
    selected_huge = select_synthesis_context(huge_sources, depth="low")
    assert sum(len(s["content"]) for s in selected_huge) <= 60000


def test_build_synthesis_messages():
    sources = [{"id": 1, "title": "Paper", "url": "https://a.test", "content": "Quantum decoherence details"}]
    messages = build_synthesis_messages("quantum", sources)
    assert len(messages) == 2
    assert "[1] Paper (https://a.test)" in messages[1]["content"]


def test_sanitize_citations_removes_hallucinated():
    text = "According to [1] and [2], quantum computing scales well. However, [99] claimed otherwise."
    cleaned = sanitize_citations(text, max_id=2)
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[99]" not in cleaned


def test_synthesis_prompt_carries_tier_targets_and_ceilings():
    for depth in ("low", "medium", "high", "max"):
        cfg = DEPTH_CONFIG[depth]
        prompt = build_synthesis_system_prompt(depth)
        assert str(cfg.target_words) in prompt
        assert str(cfg.max_words) in prompt
        assert cfg.max_words == int(cfg.target_words * 1.2)


def test_synthesis_prompt_has_section_budgets_and_ceiling_for_every_tier():
    from deepseek_search_everywhere_mcp.deep_research import SECTION_COUNTS

    assert SECTION_COUNTS["low"] < SECTION_COUNTS["max"]
    assert SECTION_COUNTS["low"] <= SECTION_COUNTS["medium"] <= SECTION_COUNTS["high"]
    for depth in ("low", "medium", "high", "max"):
        cfg = DEPTH_CONFIG[depth]
        prompt = build_synthesis_system_prompt(depth)
        n = SECTION_COUNTS[depth]
        per_section = cfg.target_words // n
        assert f"{n} sections" in prompt
        assert f"about {per_section} words" in prompt
        assert f"hard ceiling of {cfg.max_words} words" in prompt
        assert "References" in prompt


def test_report_body_words_ignores_references_section():
    body = "word " * 10
    report = (
        f"## Findings\n\n{body}\n\n## References\n\n[1] A very long title with many words - https://a.com"
    )
    assert report_body_words(report) == len(f"## Findings\n\n{body}".split())
    assert report_body_words(body) == 10
    assert report_body_words("") == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://www.cl.cam.ac.uk/teaching/2021/ConcDisSys/DistributedSystems-1B-L8a.pdf#1#1#1",
            "https://www.cl.cam.ac.uk/teaching/2021/ConcDisSys/DistributedSystems-1B-L8a.pdf",
        ),
        (
            "https://browse-export.arxiv.org/pdf/2509.07157#4#11#4",
            "https://browse-export.arxiv.org/pdf/2509.07157",
        ),
        (
            "https://dl.acm.org/doi/pdf/10.1145/3600006.3613150?download=true&__cf_chl_tk=cQafXY.-1&x=1#5#6#5",
            "https://dl.acm.org/doi/pdf/10.1145/3600006.3613150?download=true&x=1",
        ),
        (
            "https://lists.debian.org/debian-wnpp/2026/07/msg00302.html?pow_referer=",
            "https://lists.debian.org/debian-wnpp/2026/07/msg00302.html",
        ),
        (
            "https://a.example/p?id=7&utm_source=x&fbclid=1&gclid=2&_x_tr_sl=en&__cf_chl_rt_tk=z&UTM_medium=m",
            "https://a.example/p?id=7",
        ),
        ("https://a.example/p?a=1&b=%20x", "https://a.example/p?a=1&b=%20x"),
    ],
)
def test_clean_source_url(raw: str, expected: str) -> None:
    from deepseek_search_everywhere_mcp.deep_research import clean_source_url

    assert clean_source_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "url", "expected"),
    [
        (
            "<table><tr><td>Target</td><td>Properties</td></tr></table>",
            "https://www.host.example/x",
            "host.example",
        ),
        (
            "> ⚠️ **Historischer Auditbericht** – Befunde ohne Bezug",
            "https://audit.example/r",
            "audit.example",
        ),
        ("**Refinement based safety proofs", "https://a.example/p", "Refinement based safety proofs"),
        ("  Plain   title \n with\tspaces ", "https://a.example/p", "Plain title with spaces"),
        (
            "A [link text](https://x.test) and `code` and <b>bold</b>",
            "https://a.example/p",
            "A link text and code and bold",
        ),
        ("", "https://www.empty.example/p", "empty.example"),
        (None, "https://www.none.example/p", "none.example"),
        ("   ", "https://blank.example/p", "blank.example"),
    ],
)
def test_clean_source_title(raw: str | None, url: str, expected: str) -> None:
    from deepseek_search_everywhere_mcp.deep_research import clean_source_title

    assert clean_source_title(raw, url) == expected


def test_clean_source_title_truncates_long_snippet_like_titles() -> None:
    from deepseek_search_everywhere_mcp.deep_research import clean_source_title

    raw = "This is a very long snippet-like title that keeps going " * 6
    out = clean_source_title(raw, "https://a.example/p")
    assert len(out) <= 100
    assert out.endswith("…")
    assert out.startswith("This is a very long snippet-like title")


def test_select_synthesis_context_propagates_date() -> None:
    from deepseek_search_everywhere_mcp.deep_research import select_synthesis_context

    sources = [
        {"title": "A", "url": "https://a.example/1", "content": "text", "date": "2026-03-04"},
        {"title": "B", "url": "https://b.example/1", "content": "text"},
    ]
    ctx = select_synthesis_context(sources, "low")
    dates = {s["title"]: s["date"] for s in ctx}
    assert dates == {"A": "2026-03-04", "B": None}


def test_rank_dedupes_urls_that_differ_only_by_fragment_or_challenge_params() -> None:
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    results = [
        [{"url": "https://dl.example/x.pdf#1#1"}],
        [{"url": "https://dl.example/x.pdf?__cf_chl_tk=abc#2"}, {"url": "https://other.example/y"}],
    ]
    selected, _ = rank_and_diversify_sources(results, k_sources=5)
    assert [i["url"] for i in selected] == ["https://dl.example/x.pdf#1#1", "https://other.example/y"]


def test_rank_prefers_higher_authority_within_each_round_then_rerank_score():
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    results = [
        [
            {"url": "https://a1.example/1", "title": "A1", "authority": 0.2, "rerank_score": 0.9},
            {"url": "https://a2.example/1", "title": "A2", "authority": 0.9, "rerank_score": 0.1},
        ],
        [
            {"url": "https://b1.example/1", "title": "B1", "authority": 0.8, "rerank_score": 0.1},
            {"url": "https://b2.example/1", "title": "B2", "authority": 0.9, "rerank_score": 0.5},
        ],
        [
            {"url": "https://c1.example/1", "title": "C1", "authority": 0.8, "rerank_score": 0.7},
            {"url": "https://c2.example/1", "title": "C2"},
        ],
    ]
    selected, _ = rank_and_diversify_sources(results, k_sources=6)
    # Round 1: B1 and C1 tie on authority 0.8 (C1 wins on rerank), A1 trails; round 2: B2/A2 tie 0.9.
    assert [s["title"] for s in selected] == ["C1", "B1", "A1", "B2", "A2", "C2"]


def test_rank_without_authority_keeps_round_robin_order():
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    results = [
        [{"url": "https://a.example/1", "title": "A1"}, {"url": "https://a.example/2", "title": "A2"}],
        [{"url": "https://b.example/1", "title": "B1"}],
    ]
    selected, _ = rank_and_diversify_sources(results, k_sources=3)
    assert [s["title"] for s in selected] == ["A1", "B1", "A2"]


def test_rank_drops_denylisted_hosts_with_note():
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources
    from deepseek_search_everywhere_mcp.settings import DEFAULT_SOURCE_DENYLIST

    results = [
        [
            {"url": "https://lib.ezproxy.uni.edu/login?url=x", "title": "ezproxy"},
            {"url": "https://feedly.com/i/entry/1", "title": "keep1"},
            {"url": "https://good.example/p", "title": "keep2"},
        ]
    ]
    selected, notes = rank_and_diversify_sources(results, k_sources=10, denylist=DEFAULT_SOURCE_DENYLIST)
    assert [s["title"] for s in selected] == ["keep1", "keep2"]
    assert any("1" in n and "denylist" in n.lower() for n in notes)


def test_rank_denylist_configurable_to_empty():
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    results = [[{"url": "https://lib.ezproxy.uni.edu/i/1", "title": "ezproxy"}]]
    selected, notes = rank_and_diversify_sources(results, k_sources=3, denylist=())
    assert [s["title"] for s in selected] == ["ezproxy"]
    assert notes == []


def test_clean_source_title_keeps_comparison_operators() -> None:
    from deepseek_search_everywhere_mcp.deep_research import clean_source_title

    assert clean_source_title("p<0.05 vs I²>50%", "https://a.example/p") == "p<0.05 vs I²>50%"


def test_synthesis_prompt_requires_second_hand_attribution() -> None:
    prompt = build_synthesis_system_prompt("medium")
    assert "(Author year, as cited in [n])" in prompt
    assert "never attribute" in prompt
    assert "prefer the primary source" in prompt
    assert "it only cites" in prompt


@pytest.mark.parametrize(
    ("url", "content", "expected"),
    [
        (
            "https://pubmed.ncbi.nlm.nih.gov/38123456/",
            "",
            {"doi": None, "pmid": "38123456", "pmcid": None},
        ),
        (
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10123456/",
            "",
            {"doi": None, "pmid": None, "pmcid": "PMC10123456"},
        ),
        (
            "https://doi.org/10.1136/bmj-2023-076242",
            "",
            {"doi": "10.1136/bmj-2023-076242", "pmid": None, "pmcid": None},
        ),
        (
            "https://www.sciencedirect.com/science/article/pii/S0148296323001234",
            "",
            {"doi": None, "pmid": None, "pmcid": None},
        ),
        (
            "https://example.org/page?download=true",
            "No identifiers here.",
            {"doi": None, "pmid": None, "pmcid": None},
        ),
        (
            "https://example.org/page",
            "See doi:10.1000/xyz.123.",
            {"doi": "10.1000/xyz.123", "pmid": None, "pmcid": None},
        ),
    ],
)
def test_source_identifiers(url: str, content: str, expected: dict[str, str | None]) -> None:
    from deepseek_search_everywhere_mcp.deep_research import source_identifiers

    assert source_identifiers(url, content) == expected


@pytest.mark.parametrize(
    ("origin", "content", "expected"),
    [
        ("fetch", "", "full_text"),
        ("search", "x" * 200, "abstract"),
        ("search", "x" * 199, "snippet_only"),
    ],
)
def test_source_access_rule(origin: str, content: str, expected: str) -> None:
    from deepseek_search_everywhere_mcp.deep_research import source_access

    assert source_access(origin, content) == expected
