"""Unit tests for deep research decomposition, budgeting, and citation parsing."""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp import deep_research
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
        (
            "https://pdf.sciencedirectassets.com/123?X-Amz-Security-Token=abc&X-Amz-Expires=300"
            "&X-Amz-Signature=sig&X-Amz-Credential=cred",
            "https://pdf.sciencedirectassets.com/123",
        ),
        (
            "https://blob.example.com/doc.pdf?sv=2022&se=2026&sp=r&sr=b&sig=abc",
            "https://blob.example.com/doc.pdf",
        ),
        ("https://example.com/p?sv=1&sp=r", "https://example.com/p?sv=1&sp=r"),
        ("https://example.com/p?token=abc&id=7", "https://example.com/p?id=7"),
        (
            "https://journals-lww-com.libproxy.ucl.ac.uk/article/123",
            "https://journals.lww.com/article/123",
        ),
        (
            "https://arxiv-org.ezproxy.obspm.fr/pdf/2509.07157",
            "https://arxiv.org/pdf/2509.07157",
        ),
        (
            "https://journals-lww-com.libproxy.ucl.ac.uk/doi/10.1097/XYZ.0000000000001234",
            "https://doi.org/10.1097/XYZ.0000000000001234",
        ),
        ("https://lib.ezproxy.uni.edu/login?url=x", "https://lib.ezproxy.uni.edu/login?url=x"),
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

    results = [
        [
            {"url": "https://lib.ezproxy.uni.edu/login?url=x", "title": "ezproxy"},
            {"url": "https://feedly.com/i/entry/1", "title": "keep1"},
            {"url": "https://good.example/p", "title": "keep2"},
        ]
    ]
    selected, notes = rank_and_diversify_sources(results, k_sources=10, denylist=("*.ezproxy.*",))
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
            {"doi": None, "pmid": None, "pmcid": None},
        ),
        (
            "https://blog.example.com/post",
            "We review the exciting paper doi: 10.1000/xyz.123 here.",
            {"doi": None, "pmid": None, "pmcid": None},
        ),
        (
            "https://www.nature.com/articles/s41586-023-0001",
            "Header without doi.\n## References\n1. Author et al. doi: 10.1000/ref.1",
            {"doi": None, "pmid": None, "pmcid": None},
        ),
        (
            "https://www.nature.com/articles/s41586-023-0001",
            "Nature Article\ndoi: 10.1038/s41586-023-0001\nAbstract text...",
            {"doi": "10.1038/s41586-023-0001", "pmid": None, "pmcid": None},
        ),
        (
            "https://academic.oup.com/nar/article/1",
            "Nucleic Acids Research\nPMID: 12345678\nAbstract text...",
            {"doi": None, "pmid": "12345678", "pmcid": None},
        ),
        (
            "https://blog.example.org/p",
            "Header text mentioning PMID: 12345678 in blog.",
            {"doi": None, "pmid": None, "pmcid": None},
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


def test_dead_research_constants_are_removed() -> None:
    from deepseek_search_everywhere_mcp import deep_research, server

    for name in ("MAX_SYNTHESIS_SOURCES", "MAX_SYNTHESIS_CHARS", "SYNTHESIS_SYSTEM_PROMPT"):
        assert not hasattr(deep_research, name)
    for name in ("SYNTHESIS_TIMEOUT", "RESEARCH_DEADLINE", "RESEARCH_PAGE_CHARS", "RESEARCH_TOTAL_CHARS"):
        assert not hasattr(server, name)


def test_synthesis_prompt_requires_verbatim_numbers() -> None:
    from deepseek_search_everywhere_mcp.deep_research import build_synthesis_system_prompt

    prompt = build_synthesis_system_prompt("medium")
    assert "use only numbers that appear verbatim in the cited source" in prompt.lower()


def test_number_grounding_flags_number_from_other_source() -> None:
    from deepseek_search_everywhere_mcp.deep_research import ground_report_numbers

    sources = [
        {
            "id": 1,
            "title": "Bremer 2025 Study",
            "url": "https://example.com/1",
            "content": "Bremer evaluated 4 RCTs with 200 participants.",
        },
        {
            "id": 2,
            "title": "Garcia 2024 Study",
            "url": "https://example.com/2",
            "content": "Garcia included 11 studies with 934 participants and r = -0.19.",
        },
    ]
    report = (
        "Bremer reported 11 studies with 934 participants [1]. "
        "Garcia observed a negative correlation with r = -0.19 [2]. "
        "An uncited sentence with 42 items."
    )

    cleaned_report, unverified_numbers, unverified_count, notes = ground_report_numbers(report, sources)

    assert unverified_count == 2
    assert len(unverified_numbers) == 2
    assert unverified_numbers[0]["number"] == "11"
    assert unverified_numbers[0]["citation"] == "[1]"
    assert unverified_numbers[0]["sentence"] == "Bremer reported 11 studies with 934 participants [1]."
    assert unverified_numbers[1]["number"] == "934"
    assert unverified_numbers[1]["citation"] == "[1]"
    assert (
        cleaned_report == "Bremer reported 11 studies with 934 participants [1] [unverified]. "
        "Garcia observed a negative correlation with r = -0.19 [2]. "
        "An uncited sentence with 42 items."
    )
    assert len(notes) == 1
    assert "2 unverified number(s)" in notes[0]


def test_classify_source_type() -> None:
    from deepseek_search_everywhere_mcp.deep_research import classify_source_type

    assert (
        classify_source_type(
            "https://pubmed.ncbi.nlm.nih.gov/123/",
            "Creatine: A randomized controlled trial in resistance training",
        )
        == "rct"
    )
    assert (
        classify_source_type(
            "https://pubmed.ncbi.nlm.nih.gov/456/",
            "Systematic review and meta-analysis of cartilage repair",
        )
        == "meta-analysis"
    )
    assert (
        classify_source_type(
            "https://pubmed.ncbi.nlm.nih.gov/789/",
            "Systematic review of exercise interventions",
        )
        == "systematic_review"
    )
    assert (
        classify_source_type(
            "https://cochranelibrary.com/cdsr/doi/10.1002/14651858/full",
            "Clinical practice guideline on knee osteoarthritis",
        )
        == "guideline"
    )
    assert (
        classify_source_type(
            "https://arxiv.org/abs/2301.00001",
            "Quantum error mitigation methods",
        )
        == "preprint"
    )
    assert (
        classify_source_type(
            "https://docs.python.org/3/library/asyncio.html",
            "asyncio — Asynchronous I/O",
        )
        == "documentation"
    )
    assert (
        classify_source_type(
            "https://www.rfc-editor.org/rfc/rfc9110",
            "RFC 9110: HTTP Semantics",
        )
        == "specification"
    )
    assert (
        classify_source_type(
            "https://github.com/astral-sh/ruff",
            "astral-sh/ruff: An extremely fast Python linter",
        )
        == "repository"
    )
    assert (
        classify_source_type(
            "https://stackoverflow.com/questions/123456",
            "How to parse JSON in Python?",
        )
        == "qa"
    )
    assert (
        classify_source_type(
            "https://shop.example.com/products/creatine-500g",
            "Buy Pure Creatine Monohydrate 500g - Best Price",
        )
        == "vendor"
    )
    assert (
        classify_source_type(
            "https://blog.example.com/post/my-thoughts",
            "My personal thoughts on fitness apps",
        )
        == "blog"
    )
    assert (
        classify_source_type(
            "https://www.youtube.com/watch?v=abcdef12345",
            "Lecture on Quantum Computing",
        )
        == "video"
    )


def test_ranking_prefers_academic_and_technical_sources() -> None:
    from deepseek_search_everywhere_mcp.deep_research import rank_and_diversify_sources

    academic_results = [
        [
            {
                "url": "https://shop.supplements.example/buy",
                "title": "Buy Creatine Fast Shipping",
                "authority": 5.0,
                "rerank_score": 5.0,
            },
        ],
        [
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/38123456/",
                "title": "Randomized controlled trial of creatine supplementation",
                "authority": 2.0,
                "rerank_score": 2.0,
                "doi": "10.1000/123",
            },
        ],
    ]
    ranked_acad, _ = rank_and_diversify_sources(academic_results, k_sources=2, sources="academic")
    assert "pubmed" in ranked_acad[0]["url"]

    technical_results = [
        [
            {
                "url": "https://geeksforgeeks.org/python-asyncio/",
                "title": "Python Asyncio Tutorial",
                "authority": 5.0,
                "rerank_score": 5.0,
            },
        ],
        [
            {
                "url": "https://docs.python.org/3/library/asyncio.html",
                "title": "asyncio Documentation",
                "authority": 2.0,
                "rerank_score": 2.0,
            },
        ],
    ]
    ranked_tech, _ = rank_and_diversify_sources(technical_results, k_sources=2, sources="technical")
    assert "docs.python.org" in ranked_tech[0]["url"]


def test_decomposition_prompt_source_preferences() -> None:
    from deepseek_search_everywhere_mcp.deep_research import build_decomposition_messages

    msgs_acad = build_decomposition_messages("knee cartilage", sources="academic")
    text_acad = " ".join(m["content"] for m in msgs_acad).lower()
    assert "randomized controlled trial" in text_acad or "meta-analysis" in text_acad

    msgs_tech = build_decomposition_messages("fastmcp", sources="technical")
    text_tech = " ".join(m["content"] for m in msgs_tech).lower()
    assert "documentation" in text_tech or "release notes" in text_tech


def test_unverified_marker_goes_before_final_punctuation() -> None:
    from deepseek_search_everywhere_mcp.deep_research import _mark_unverified

    assert (
        _mark_unverified("A review pooled 11 studies [1].") == "A review pooled 11 studies [1] [unverified]."
    )
    assert _mark_unverified("Effect was 0,45 [2]!") == "Effect was 0,45 [2] [unverified]!"
    assert _mark_unverified("No punctuation [1]") == "No punctuation [1] [unverified]"
    assert _mark_unverified('He said "18% [1]."') == 'He said "18% [1] [unverified]."'


_GROUNDING_SOURCE = [
    {
        "title": "Note",
        "date": "2026-01-22",
        "content": "In total 90.50% of episodes failed. We ran 35 surveys. Temperature fell by −5 °C. "
        "Tasks took 5 to 61 minutes.",
    }
]


@pytest.mark.parametrize(
    "sentence",
    [
        "Results — 90.50% of episodes failed [1].",
        "Results—90.50% of episodes failed [1].",
        "They ran – 35 surveys [1].",
        "The note is dated 22 January 2026 [1].",
        "The note is dated January 22, 2026 [1].",
        "Notatka z 22 stycznia 2026 [1].",
        "The note of 2026-01-22 lists 35 surveys [1].",
        "COVID-19 data covered 35 surveys [1].",
        "Temperature fell −5 °C [1].",
        "Tasks took 5–61 minutes [1].",
    ],
)
def test_grounding_ignores_dashes_dates_and_names(sentence: str) -> None:
    report, unverified, count, _ = deep_research.ground_report_numbers(sentence, _GROUNDING_SOURCE)
    assert (unverified, count) == ([], 0)
    assert "[unverified]" not in report


def test_grounding_still_flags_a_missing_number() -> None:
    _, unverified, count, _ = deep_research.ground_report_numbers(
        "Results — 77% of episodes failed [1].", _GROUNDING_SOURCE
    )
    assert count == 1
    assert unverified[0]["number"] == "77%"


@pytest.mark.parametrize(
    ("origin", "length", "expected"),
    [
        ("fetch", 10, "full_text"),
        ("search", deep_research.SOURCE_FULL_TEXT_CHARS, "full_text"),
        ("search", deep_research.SOURCE_FULL_TEXT_CHARS - 1, "abstract"),
        ("search", deep_research.SOURCE_SNIPPET_ONLY_CHARS, "abstract"),
        ("search", deep_research.SOURCE_SNIPPET_ONLY_CHARS - 1, "snippet_only"),
    ],
)
def test_source_access(origin: str, length: int, expected: str) -> None:
    assert deep_research.source_access(origin, "x" * length) == expected


@pytest.mark.parametrize(
    ("title", "url"),
    [
        ("raw.githubusercontent.com", "https://raw.githubusercontent.com/a/b/c.md"),
        ("Patch size", "https://arxiv.org/abs/2608.15579"),
        ("The human baseline is included as an undated reference", "https://example.org/p.pdf"),
        ("4.6 RQ4: Effectiveness of TestLoc in practice", "https://example.org/p"),
        ("Item Response Theory (IRT) has become…", "https://example.org/p"),
        ("", "https://example.org/p"),
    ],
)
def test_fragment_titles_are_detected(title: str, url: str) -> None:
    assert deep_research.is_fragment_title(title, url)


@pytest.mark.parametrize(
    "title",
    [
        "Clarifying limitations of time horizon",
        "Attention Is All You Need",
        "GPT-5 System Card",
        "Kubernetes",
        "Claude Code",
        "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        "The State of AI Agents in 2026 Has Changed",
        "iOS 26",
    ],
)
def test_real_titles_are_kept(title: str) -> None:
    assert not deep_research.is_fragment_title(title, "https://example.org/p")
    assert deep_research.best_source_title(title, "https://example.org/p", "# Other heading\n") == title


def test_fragment_title_is_replaced_by_a_content_heading() -> None:
    content = "Title: SWE-Patch: Measuring Patch Size in Agents\n\nAbstract. We study ..."
    assert deep_research.best_source_title("Patch size", "https://arxiv.org/abs/1", content) == (
        "SWE-Patch: Measuring Patch Size in Agents"
    )
    md = "Intro text\n\n# Measuring AI Ability to Complete Long Tasks\n\n## 2.1 Setup"
    assert deep_research.best_source_title("2.1 Setup", "https://metr.org/x", md) == (
        "Measuring AI Ability to Complete Long Tasks"
    )


def test_fragment_title_stays_without_a_better_one() -> None:
    assert (
        deep_research.best_source_title("Patch size", "https://arxiv.org/abs/1", "no heading") == "Patch size"
    )


def test_grounding_drop_removes_sentences_and_empty_paragraphs() -> None:
    report = "Intro 35 surveys ran [1]. Then 77% failed [1].\n\nOnly 88 left [1].\n\nNo numbers here."
    cleaned, unverified, count, notes = deep_research.ground_report_numbers(
        report, _GROUNDING_SOURCE, drop=True
    )
    assert cleaned == "Intro 35 surveys ran [1].\n\nNo numbers here."
    assert count == 2 and [u["number"] for u in unverified] == ["77%", "88"]
    assert any("Removed 2 sentence(s)" in note for note in notes)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "Title: Attention Is All You Need\nAuthors:Ashish Vaswani, Noam Shazeer and Łukasz Kaiser\n",
            ["Ashish Vaswani", "Noam Shazeer", "Łukasz Kaiser"],
        ),
        ("# Post\n\nBy Simon Willison\n\nToday I tried ...", ["Simon Willison"]),
        ("Authors: Jane Doe1,2, John Smith* (MIT), et al.", ["Jane Doe", "John Smith"]),
        ("# Raport\nAutorzy: Anna Kowalska i Jan Nowak", ["Anna Kowalska", "Jan Nowak"]),
        ("Authors: Ludwig van Beethoven, Ana de la Cruz", ["Ludwig van Beethoven", "Ana de la Cruz"]),
        ("Authors: The METR team", None),
        ("We did it by default\nby running 5 jobs", None),
        ("no author line here", None),
        ("x" * 1600 + "\nAuthors: Jane Doe", None),
    ],
)
def test_source_authors(content: str, expected: list[str] | None) -> None:
    assert deep_research.source_authors("https://example.org/paper", content) == expected
