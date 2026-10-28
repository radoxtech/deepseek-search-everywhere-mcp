"""A missing upstream date falls back to a date (ISO or month name) found at the start of the text."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

TODAY = date(2026, 10, 28)
# Shaped like a real index entry for a Polish survey report without an upstream date.
REPORT = (
    "# PUBLIKACJE\n\n### Korzystanie z internetu w 2025 roku\n\n2025-06-26\n\n"
    "Autorzy: Anna Kowalska, Jan Nowak\n\nNiemal ośmiu na dziesięciu dorosłych (79%) regularnie korzysta "
    "z internetu. Badanie objęło 98 299 osób."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Opublikowano 2025-06-26.", "2025-06-26"),
        ("Updated: 2024-02-29T10:00:00Z by staff", "2024-02-29"),
        ("(2019-12-31)", "2019-12-31"),
        ("Wersja 2.2025-01-01 i v2025-01-01 oraz build-2025-01-01", None),
        ("ID 12025-06-26 and 2025-06-2611 and 2025-06-26-1", None),
        ("Release 1.2.3, 98 299 uczestników, tel. 2025 06 26", None),
        ("2025-02-30 then 2025-13-01", None),
        ("1989-12-31 and 2028-01-01", None),
        ("1990-01-01", "1990-01-01"),
        ("2027-12-31", "2027-12-31"),
        ("2025-02-30 then 2025-03-01", "2025-03-01"),
    ],
)
def test_date_in(text: str, expected: str | None) -> None:
    assert server._date_in(text, TODAY) == expected


def test_content_date_only_looks_at_the_start_of_the_text() -> None:
    late = "x" * (server.CONTENT_DATE_WINDOW + 10) + " 2025-06-26"
    assert server._content_date(None, late, TODAY) is None
    assert server._content_date("Raport 2025-06-26", late, TODAY) == "2025-06-26"
    assert server._content_date(None, REPORT, TODAY) == "2025-06-26"


def _result(fake_upstream, tool: str, **item: Any) -> dict[str, Any]:
    base = {"title": "PUBLIKACJE", "url": "https://survey.example.pl/raport?id=1", "site_name": "Example"}
    fake_upstream(default={"ok": True, "data": {"results": [{**base, **item}]}})
    return asyncio.run(getattr(server, tool)("internet 2025", count=1))["results"][0]


def test_search_uses_content_date_when_upstream_has_none(fake_upstream) -> None:
    result = _result(fake_upstream, "deepseek_search_and_read", date=None, content=REPORT)
    assert (result["date"], result["date_source"]) == ("2025-06-26", "content")


def test_web_search_uses_summary_without_content(fake_upstream) -> None:
    result = _result(fake_upstream, "deepseek_web_search", summary=REPORT)
    assert (result["date"], result["date_source"]) == ("2025-06-26", "content")


def test_upstream_date_wins(fake_upstream) -> None:
    upstream = "2025-11-23T16:00:00+00:00"
    result = _result(fake_upstream, "deepseek_search_and_read", date=upstream, content=REPORT)
    assert (result["date"], result["date_source"]) == (upstream, "upstream")


def test_no_date_anywhere_stays_null(fake_upstream) -> None:
    result = _result(
        fake_upstream, "deepseek_search_and_read", content="Badanie objęło 98 299 osób w 2025 roku."
    )
    assert (result["date"], result["date_source"]) == (None, None)


def test_research_sources_carry_date_source(fake_upstream) -> None:
    item = {"title": "PUBLIKACJE", "url": "https://survey.example.pl/raport?id=1", "content": REPORT * 2}
    fake_upstream(default={"ok": True, "data": {"results": [item]}})
    source = asyncio.run(server.deepseek_research("internet 2025", max_pages=1))["sources"][0]
    assert (source["date"], source["date_source"]) == ("2025-06-26", "content")


def test_research_takes_a_date_from_a_fetched_gap(fake_upstream) -> None:
    item = {"title": "Raport", "url": "https://survey.example.pl/raport?id=2", "content": "krótko"}
    fake_upstream(
        responses={
            server.SEARCH_PATH: {"ok": True, "data": {"results": [item]}},
            server.FETCH_PATH: {"ok": True, "data": {"title": "Raport", "content": REPORT}},
        }
    )
    source = asyncio.run(server.deepseek_research("internet 2025", max_pages=1))["sources"][0]
    assert source["origin"] == "fetch"
    assert (source["date"], source["date_source"]) == ("2025-06-26", "content")


# Shaped like a journal table-of-contents entry from the search index (no ISO date, no upstream date).
JOURNAL_TOC = (
    "Pediatrics in Review\n\nSeptember 2025: 151(9):1125-1146\n\n"
    "Original article. Short-form video use and attention in 98 299 participants."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("September 2025: 151(9):1125-1146", "2025-09-01"),
        ("published in SEPTEMBER 2025", "2025-09-01"),
        ("wrzesień 2025", "2025-09-01"),
        ("Raport, Wrzesień 2025 r.", "2025-09-01"),
        ("15 września 2025", "2025-09-15"),
        ("Warszawa, 3 stycznia 2024.", "2024-01-03"),
        ("September 15, 2025", "2025-09-15"),
        ("12 May 2025 and June 2024", "2025-05-12"),
        ("31 lutego 2025, potem marzec 2025", "2025-03-01"),
        ("98 299 uczestników w 2025 roku", None),
        ("lato 2025", None),
        ("Sept 2025, Sep 2025, wrz. 2025", None),
        ("majątek 2025, Septembers 2025", None),
        ("June 1989 and December 2028", None),
        ("June 2025-2026 season", None),
        ("June 20251", None),
        ("Psychological Bulletin 2025 September\n\nThe resurgence", "2025-09-01"),
        ("Pediatrics. 2024 październik; 12(3)", "2024-10-01"),
        ("2025 Sep; 151(9) and 2025 Septembers", None),
        ("in 1850 June and in 2025-06 June", None),
        ("First June 2024, then 2025 May", "2024-06-01"),
        ("By Hilary DarrahSeptember 2025", None),
        ("Hilary Darrah2025 September", None),
    ],
)
def test_month_date_in(text: str, expected: str | None) -> None:
    assert server._month_date_in(text, TODAY) == expected


def test_content_date_reads_a_month_and_year() -> None:
    assert server._content_date(None, JOURNAL_TOC, TODAY) == "2025-09-01"
    assert server._content_date("Wyniki badania, wrzesień 2025", "Bez daty.", TODAY) == "2025-09-01"


def test_content_date_prefers_an_iso_date() -> None:
    text = "September 2025 issue\n\nPublished 2025-06-26."
    assert server._content_date(None, text, TODAY) == "2025-06-26"
    assert server._content_date("CBOS: czerwiec 2025", REPORT, TODAY) == "2025-06-26"


def test_content_date_ignores_a_month_outside_the_window() -> None:
    late = "x" * (server.CONTENT_DATE_WINDOW + 10) + " September 2025"
    assert server._content_date(None, late, TODAY) is None


def test_content_date_window_reaches_past_a_long_page_header() -> None:
    header = "Journal navigation, authors and affiliations. " * 7  # about 330 characters
    assert 320 < len(header) < 340
    assert server._content_date(None, header + "2025 September; 151(9)", TODAY) == "2025-09-01"
    assert server._content_date(None, header + "Published 2025-06-26.", TODAY) == "2025-06-26"


def test_content_date_window_is_600_characters() -> None:
    assert server.CONTENT_DATE_WINDOW == 600
    fits = "x" * (600 - len(" September 2025")) + " September 2025"
    assert server._content_date(None, fits, TODAY) == "2025-09-01"
    assert server._content_date(None, "x" * 601 + " September 2025", TODAY) is None
    assert server._content_date(None, "x" * 601 + " 2025-06-26", TODAY) is None


def test_search_reads_a_month_date_from_content(fake_upstream) -> None:
    result = _result(fake_upstream, "deepseek_search_and_read", date=None, content=JOURNAL_TOC)
    assert (result["date"], result["date_source"]) == ("2025-09-01", "content")


def test_upstream_date_wins_over_a_month_date(fake_upstream) -> None:
    upstream = "2025-11-23T16:00:00+00:00"
    result = _result(fake_upstream, "deepseek_search_and_read", date=upstream, content=JOURNAL_TOC)
    assert (result["date"], result["date_source"]) == (upstream, "upstream")
