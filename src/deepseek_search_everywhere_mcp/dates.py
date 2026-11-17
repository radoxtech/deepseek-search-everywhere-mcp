"""Publication-date detection for results: upstream dates, ISO and month-name dates in content, URL dates."""

from __future__ import annotations

import re
import urllib.parse
from datetime import UTC, date, datetime, timedelta, timezone
from datetime import time as dt_time
from typing import Any

from .text import (
    _decode_json_escapes,
    _unescape_markdown,
)

# A calendar date written as YYYY-MM-DD on its own (optionally followed by an ISO time), not part
# of a longer number, identifier or version string.
_ISO_DATE = re.compile(r"(?<![\w./-])(\d{4})-(\d{2})-(\d{2})(?![0-9]|[-./][0-9]|[A-SU-Za-z])")


# How far into a page's text a publication date (ISO or month name) is looked for, and how much raw
# text is cleaned first. Journal pages often open with a few hundred characters of site navigation,
# authors and affiliations before the issue line, so the window reaches past them.
CONTENT_DATE_WINDOW = 600


_CONTENT_DATE_RAW = 2000


MIN_CONTENT_YEAR = 1990


def _date_in(text: str, today: date | None = None) -> str | None:
    """Return the first valid ``YYYY-MM-DD`` date in ``text`` (year 1990 to next year), or None."""
    max_year = (today or datetime.now(UTC).date()).year + 1
    for match in _ISO_DATE.finditer(text):
        year, month, day = (int(part) for part in match.groups())
        if not MIN_CONTENT_YEAR <= year <= max_year:
            continue
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


# Full English month names mapped to the month number. Abbreviations are not accepted.
_MONTH_NAMES: dict[str, int] = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


# "September 2025", with an optional day: "15 September 2025", "September 15, 2025".
_MONTH_DATE = re.compile(
    r"(?<![\w.,/-])(?:(?P<day>\d{1,2})\s+)?"
    r"(?P<month>" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")"
    r"(?:\s+(?P<day_after>\d{1,2}),)?\s+(?P<year>\d{4})(?![\w-]|[.,/]\d)",
    re.IGNORECASE,
)


# The year-first order used in PubMed-style citations: "2025 September".
_YEAR_MONTH_DATE = re.compile(
    r"(?<![\w.,/-])(?P<year>\d{4})\s+"
    r"(?P<month>" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")(?![\w-])",
    re.IGNORECASE,
)


def _month_date_in(text: str, today: date | None = None) -> str | None:
    """Return the first valid month-name date in ``text`` as ``YYYY-MM-DD``, or None.

    A full English month name next to a year (1990 to next year), as "September 2025" or
    "2025 September", gives the first of that month; a day written with the month ("15 September
    2025", "September 15, 2025") gives that day. The earliest match wins.
    """
    max_year = (today or datetime.now(UTC).date()).year + 1
    matches = [*_MONTH_DATE.finditer(text), *_YEAR_MONTH_DATE.finditer(text)]
    for match in sorted(matches, key=lambda found: found.start()):
        year = int(match["year"])
        if not MIN_CONTENT_YEAR <= year <= max_year:
            continue
        groups = match.groupdict()
        day = groups.get("day") or groups.get("day_after")
        try:
            return date(year, _MONTH_NAMES[match["month"].lower()], int(day) if day else 1).isoformat()
        except ValueError:
            continue
    return None


def _content_date(title: Any, text: Any, today: date | None = None) -> str | None:
    """Find a publication date in a result's title or the first characters of its text.

    Used only when the upstream gave no date. The text is cleaned the way content is before the
    first ``CONTENT_DATE_WINDOW`` characters are searched. An ISO ``YYYY-MM-DD`` date anywhere in
    the title or that window is preferred; otherwise the first month-name date is used. A bare
    year is never taken as a date.
    """
    candidates: list[str] = []
    if isinstance(title, str):
        candidates.append(title)
    if isinstance(text, str) and text:
        head = _decode_json_escapes(_unescape_markdown(text[:_CONTENT_DATE_RAW]))
        candidates.append(head[:CONTENT_DATE_WINDOW])
    for find in (_date_in, _month_date_in):
        if found := next((hit for hit in (find(c, today) for c in candidates) if hit), None):
            return found
    return None


# The upstream index stores a date-only publication date as midnight in UTC+8 and returns it converted
# to UTC, so 2026-01-22 arrives as "2026-01-21T16:00:00+00:00". That exact time is read back in UTC+8.
_UPSTREAM_DATE_ZONE = timezone(timedelta(hours=8))


_UPSTREAM_MIDNIGHT_UTC = dt_time(16, 0)


# A full Y-M-D date in a URL path ("/2026-06-26T05...", "/2026/03/20/"), not part of a longer number.
_URL_DATE = re.compile(r"(?<!\d)(\d{4})([-/])(\d{2})\2(\d{2})(?!\d)")


def _upstream_date(upstream: Any) -> Any:
    """Return the upstream date as ``YYYY-MM-DD``; a value that is not an ISO date is returned unchanged.

    A timestamp at exactly 16:00:00 UTC is a UTC+8 midnight (a date-only value) and gives the next
    calendar day; any other timezone-aware timestamp gives its UTC date.
    """
    if not isinstance(upstream, str):
        return upstream
    try:
        parsed = datetime.fromisoformat(upstream.strip().replace("Z", "+00:00"))
    except ValueError:
        return upstream
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    utc = parsed.astimezone(UTC)
    if utc.timetz().replace(tzinfo=None) == _UPSTREAM_MIDNIGHT_UTC:
        return utc.astimezone(_UPSTREAM_DATE_ZONE).date().isoformat()
    return utc.date().isoformat()


def _url_date(url: Any, today: date | None = None) -> str | None:
    """Return the first valid ``YYYY-MM-DD`` date in a URL's path (year 1990 to next year), or None."""
    if not isinstance(url, str) or not url:
        return None
    try:
        path = urllib.parse.urlsplit(url).path
    except ValueError:
        return None
    max_year = (today or datetime.now(UTC).date()).year + 1
    for match in _URL_DATE.finditer(path):
        year, month, day = int(match[1]), int(match[3]), int(match[4])
        if not MIN_CONTENT_YEAR <= year <= max_year:
            continue
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


def _dated(upstream: Any, title: Any, text: Any, url: Any = None) -> tuple[Any, str | None]:
    """Return ``(date, date_source)``: a date in the URL path, else the upstream date, else one found
    in the content, else None.

    A URL that carries a full date (blog posts, notes, dated repository paths) names the publication
    day more reliably than the upstream metadata, which can hold the date of the site or repository.
    """
    if found := _url_date(url):
        return found, "url"
    if upstream:
        return _upstream_date(upstream), "upstream"
    found = _content_date(title, text)
    return (found, "content") if found else (None, None)
