"""Unit tests for deep research decomposition, budgeting, and citation parsing."""

from __future__ import annotations

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
