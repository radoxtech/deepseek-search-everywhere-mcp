"""Tests for multilingual prompt guidance across planning, decomposition, and synthesis."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.agent_research import _gap_queries
from deepseek_search_everywhere_mcp.agent_roles import REPORT_PLAN_PROMPT
from deepseek_search_everywhere_mcp.deep_research import (
    DECOMPOSE_SYSTEM_PROMPT,
    ResearchAspect,
    build_decomposition_messages,
    build_synthesis_system_prompt,
)


def test_decomposition_messages_adaptive_language_when_lang_none() -> None:
    topic = "Wdrożenie KSeF w małej firmie budowlanej"
    messages = build_decomposition_messages(topic, max_queries=3, lang=None)
    user_content = messages[1]["content"]
    assert "Search language: Topic's natural language" in user_content
    assert "Search language: English" not in user_content


def test_decomposition_messages_explicit_language() -> None:
    messages = build_decomposition_messages("Wdrożenie KSeF", max_queries=3, lang="pl")
    user_content = messages[1]["content"]
    assert "Search language: Polish" in user_content


def test_decompose_system_prompt_language_rules() -> None:
    assert "match language to context" in DECOMPOSE_SYSTEM_PROMPT
    assert "local language for domestic laws" in DECOMPOSE_SYSTEM_PROMPT
    assert "English for global tech" in DECOMPOSE_SYSTEM_PROMPT
    assert "never translate standardized proper nouns" in DECOMPOSE_SYSTEM_PROMPT


def test_synthesis_prompt_preserves_technical_industry_terms() -> None:
    prompt = build_synthesis_system_prompt("medium")
    assert "retain standard industry technical terms" in prompt
    assert "legal identifiers in their accepted form" in prompt


def test_report_plan_prompt_enforces_question_language() -> None:
    assert "Write section titles in the language of the original question" in REPORT_PLAN_PROMPT


def test_gap_queries_clean_without_hardcoded_english_suffixes() -> None:
    aspects = [
        ResearchAspect(
            aspect="terminy wdrożenia KSeF",
            kind="primary",
            looking_for="terminy obowiązkowego KSeF 2026",
            query="KSeF terminy obowiązkowe",
        )
    ]
    blocks = [{"sentences": []}]  # empty sentences -> gap
    queries = _gap_queries(aspects, blocks, set(), topic="KSeF wdrożenie w Polsce")
    assert len(queries) > 0
    # Must not contain clumsy English words 'data' or 'figures'
    for q in queries:
        assert " data " not in q
        assert " figures " not in q
        assert " report " not in q
