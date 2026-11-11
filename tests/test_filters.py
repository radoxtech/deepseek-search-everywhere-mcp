"""Unit tests for search filters compilation and validation."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.filters import (
    compile_filters,
    normalize_date,
    normalize_domain,
)


def test_normalize_domain_success():
    assert normalize_domain("arxiv.org") == ("arxiv.org", None)
    assert normalize_domain("https://arxiv.org/abs/123") == ("arxiv.org", None)
    assert normalize_domain("WWW.NATURE.COM") == ("nature.com", None)
    assert normalize_domain("http://sub.domain.co.uk/path") == ("sub.domain.co.uk", None)


def test_normalize_domain_rejections():
    domain, err = normalize_domain("")
    assert err is not None
    assert domain is None

    domain, err = normalize_domain("not a domain")
    assert err is not None
    assert domain is None

    domain, err = normalize_domain("https://")
    assert err is not None


def test_normalize_date_success():
    assert normalize_date("2024-01-15") == ("2024-01-15", None)
    assert normalize_date("2024") == ("2024", None)


def test_normalize_date_rejections():
    _, err = normalize_date("2024-02-31")  # invalid day
    assert "Invalid calendar date" in str(err)

    _, err = normalize_date("1980")  # too old
    assert "Year must be between" in str(err)

    _, err = normalize_date("invalid-date")
    assert "Date must be in YYYY-MM-DD" in str(err)


def test_compile_filters_all_operators():
    ops, code, err = compile_filters(
        include_domains=["arxiv.org", "nature.com"],
        exclude_domains=["wiki.org"],
        exclude_terms=["ads"],
        exact_phrases=["ai"],
        after="2024",
        before="2025",
    )
    assert code is None
    assert err is None
    assert ops == 'site:arxiv.org OR site:nature.com -site:wiki.org -ads "ai" after:2024 before:2025'


def test_compile_filters_include_domains_single():
    ops, code, err = compile_filters(include_domains=["arxiv.org"])
    assert code is None
    assert ops == "site:arxiv.org"


def test_compile_filters_exclude_terms():
    ops, code, err = compile_filters(exclude_terms=["ads", "-promo", "ads"])
    assert code is None
    assert ops == "-ads -promo"


def test_compile_filters_exclude_terms_rejections():
    _, code, err = compile_filters(exclude_terms=[""])
    assert code == "bad_term"
    assert "cannot be empty" in str(err)

    _, code, err = compile_filters(exclude_terms=["has space"])
    assert code == "bad_term"
    assert "single word" in str(err)


def test_compile_filters_exact_phrases():
    ops, code, err = compile_filters(exact_phrases=['"quantum computing"', "deep research"])
    assert code is None
    assert ops == '"quantum computing" "deep research"'


def test_compile_filters_exact_phrases_rejections():
    _, code, err = compile_filters(exact_phrases=[""])
    assert code == "bad_phrase"
    assert "cannot be empty" in str(err)

    _, code, err = compile_filters(exact_phrases=['foo "bar" baz'])
    assert code == "bad_phrase"
    assert "internal quotes" in str(err)

    _, code, err = compile_filters(exact_phrases=["line1\nline2"])
    assert code == "bad_phrase"
    assert "newlines" in str(err)


def test_compile_filters_deduplicates():
    ops, code, _ = compile_filters(include_domains=["arxiv.org", "https://www.arxiv.org/"])
    assert code is None
    assert ops == "site:arxiv.org"


def test_compile_filters_too_long():
    long_domains = [f"domain{i}.example.com" for i in range(12)]
    ops, code, err = compile_filters(include_domains=long_domains)
    assert code == "filters_too_long"
    assert "exceed 96 characters" in str(err)
