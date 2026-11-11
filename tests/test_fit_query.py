"""Unit tests for fit_query and query trimming on word/operator boundaries."""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp.server import MAX_QUERY, fit_query


def test_fit_query_short_query_untouched() -> None:
    used, trimmed = fit_query("python asyncio")
    assert used == "python asyncio"
    assert trimmed is False


def test_fit_query_with_lang_prefix() -> None:
    used, trimmed = fit_query("asyncio tasks", lang="pl")
    assert used == "lang:pl asyncio tasks"
    assert trimmed is False


def test_fit_query_with_existing_lang_does_not_duplicate() -> None:
    used, trimmed = fit_query("lang:es asyncio tasks", lang="es")
    assert used == "lang:es asyncio tasks"
    assert trimmed is False


def test_fit_query_with_filters_suffix() -> None:
    used, trimmed = fit_query("asyncio tasks", filter_ops="site:python.org")
    assert used == "asyncio tasks site:python.org"
    assert trimmed is False


def test_fit_query_cuts_at_word_boundary() -> None:
    # 20 chars query, limit to 15
    used, trimmed = fit_query("alpha beta gamma delta", max_query=15)
    assert trimmed is True
    # "alpha beta gamma delta"[:15] is "alpha beta gamm" -> cut at last space -> "alpha beta"
    assert used == "alpha beta"
    assert len(used) <= 15


def test_fit_query_drops_half_cut_operator() -> None:
    # Query ends with site:example.com, cut hits mid-operator
    query = "python tutorials site:example.com"
    # Available room = 25 -> "python tutorials site:ex"
    used, trimmed = fit_query(query, max_query=25)
    assert trimmed is True
    assert "site:" not in used
    assert used == "python tutorials"
    assert len(used) <= 25


def test_fit_query_drops_unbalanced_quotes() -> None:
    query = 'python guides "modern async patterns that are very long"'
    # Available room = 35 -> 'python guides "modern async pattern'
    used, trimmed = fit_query(query, max_query=35)
    assert trimmed is True
    assert used.count('"') % 2 == 0
    assert '"' not in used
    assert used == "python guides"
    assert len(used) <= 35


def test_fit_query_preserves_unicode_characters() -> None:
    # Multibyte Polish text: "zażółć gęślą jaźń"
    query = "zażółć gęślą jaźń oraz inne polskie znaki diakrytyczne"
    used, trimmed = fit_query(query, max_query=20)
    assert trimmed is True
    assert used == "zażółć gęślą jaźń"
    assert len(used) <= 20


def test_fit_query_operators_alone_exceed_limit_raises() -> None:
    filter_ops = "site:" + "x" * 125  # 130 chars
    with pytest.raises(ValueError, match="Operators and language filter exceed"):
        fit_query("test", filter_ops=filter_ops, max_query=MAX_QUERY)


def test_fit_query_single_word_no_spaces_trimmed_to_limit() -> None:
    word = "a" * 150
    used, trimmed = fit_query(word, max_query=128)
    assert trimmed is True
    assert len(used) == 128
