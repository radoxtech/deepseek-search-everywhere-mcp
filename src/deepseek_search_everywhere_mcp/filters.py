"""Domain and date filter compilation and budget guarding for search queries.

Validates and serializes ``include_domains``, ``exclude_domains``, ``after``, and ``before``
into upstream search query operators without exceeding DeepSeek's 128-character limit.
"""

from __future__ import annotations

import datetime
import re
from urllib.parse import urlparse

MAX_FILTER_OPS_CHARS = 96
_DATE_RE = re.compile(r"^\d{4}(?:-\d{2}-\d{2})?$")
_DOMAIN_CHAR_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$|^localhost$")


def normalize_domain(raw: str) -> tuple[str | None, str | None]:
    """Strip protocol, path and whitespace from ``raw``; validate domain characters.

    Returns ``(domain, None)`` on success, or ``(None, error_message)`` on failure.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return None, "Domain cannot be empty."
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        cleaned = parsed.netloc or parsed.path
    cleaned = cleaned.split("/", 1)[0].split(":", 1)[0].strip().lower()
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    if not cleaned or not _DOMAIN_CHAR_RE.match(cleaned):
        return None, f"Invalid domain: {raw!r}."
    return cleaned, None


def normalize_date(raw: str) -> tuple[str | None, str | None]:
    """Validate ISO date ``YYYY-MM-DD`` or year ``YYYY`` between 1990 and next year.

    Returns ``(date_str, None)`` on success, or ``(None, error_message)`` on failure.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return None, "Date cannot be empty."
    if not _DATE_RE.match(cleaned):
        return None, f"Date must be in YYYY-MM-DD or YYYY format: {raw!r}."
    max_year = datetime.datetime.now(datetime.UTC).year + 1
    if len(cleaned) == 4:
        year = int(cleaned)
        if not 1990 <= year <= max_year:
            return None, f"Year must be between 1990 and {max_year}: {raw!r}."
        return cleaned, None
    try:
        parsed = datetime.datetime.strptime(cleaned, "%Y-%m-%d").date()
    except ValueError:
        return None, f"Invalid calendar date: {raw!r}."
    if not 1990 <= parsed.year <= max_year:
        return None, f"Year must be between 1990 and {max_year}: {raw!r}."
    return cleaned, None


def compile_filters(
    *,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Compile explicit filter parameters into a space-separated operator string.

    Returns ``(ops_string, error_code, error_message)``.
    """
    ops: list[str] = []

    if include_domains:
        seen_inc: set[str] = set()
        for raw in include_domains:
            domain, err = normalize_domain(raw)
            if err:
                return "", "bad_domain", err
            assert domain is not None
            if domain not in seen_inc:
                seen_inc.add(domain)
                ops.append(f"site:{domain}")

    if exclude_domains:
        seen_exc: set[str] = set()
        for raw in exclude_domains:
            domain, err = normalize_domain(raw)
            if err:
                return "", "bad_domain", err
            assert domain is not None
            if domain not in seen_exc:
                seen_exc.add(domain)
                ops.append(f"-site:{domain}")

    if after:
        date_str, err = normalize_date(after)
        if err:
            return "", "bad_date", err
        assert date_str is not None
        ops.append(f"after:{date_str}")

    if before:
        date_str, err = normalize_date(before)
        if err:
            return "", "bad_date", err
        assert date_str is not None
        ops.append(f"before:{date_str}")

    ops_string = " ".join(ops)
    if len(ops_string) > MAX_FILTER_OPS_CHARS:
        return (
            "",
            "filters_too_long",
            f"Combined filters exceed {MAX_FILTER_OPS_CHARS} characters ({len(ops_string)} chars).",
        )

    return ops_string, None, None
