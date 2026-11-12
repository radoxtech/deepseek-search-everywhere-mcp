"""Domain, keyword and date filter compilation and budget guarding for search queries.

Validates and serializes ``include_domains``, ``exclude_domains``, ``exclude_terms``,
``exact_phrases``, ``after``, and ``before`` into upstream search query operators
without exceeding DeepSeek's 128-character limit.
"""

from __future__ import annotations

import datetime
import re
from urllib.parse import urlparse

MAX_FILTER_OPS_CHARS = 96
_DATE_RE = re.compile(r"^\d{4}(?:-\d{2}-\d{2})?$")
_DOMAIN_CHAR_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$|^localhost$")


def normalize_domain(raw: str) -> tuple[str | None, str | None]:
    """Strip protocol, path and whitespace from ``raw``; validate domain characters via IDNA.

    Returns ``(unicode_domain, None)`` on success, or ``(None, error_message)`` on failure.
    Always returns the domain in Unicode form for upstream search.
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
    if not cleaned:
        return None, f"Invalid domain: {raw!r}."

    if cleaned == "localhost":
        return "localhost", None

    try:
        ascii_form = cleaned.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        return None, f"Invalid domain: {raw!r}."

    if not _DOMAIN_CHAR_RE.match(ascii_form):
        return None, f"Invalid domain: {raw!r}."

    try:
        unicode_form = ascii_form.encode("ascii").decode("idna")
    except (UnicodeError, ValueError):
        unicode_form = cleaned

    return unicode_form, None


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
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Compile explicit filter parameters into a space-separated operator string.

    Returns ``(ops_string, error_code, error_message)``.
    """
    ops: list[str] = []

    if include_domains:
        seen_inc: set[str] = set()
        site_clauses: list[str] = []
        for raw in include_domains:
            domain, err = normalize_domain(raw)
            if err:
                return "", "bad_domain", err
            assert domain is not None
            if domain not in seen_inc:
                seen_inc.add(domain)
                site_clauses.append(f"site:{domain}")
        if site_clauses:
            ops.append(" OR ".join(site_clauses))

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

    if exclude_terms:
        seen_terms: set[str] = set()
        for raw in exclude_terms:
            cleaned = (raw or "").strip()
            if not cleaned:
                return "", "bad_term", "Excluded term cannot be empty."
            cleaned = cleaned.lstrip("-").strip()
            if not cleaned or any(c.isspace() or c in "\"'" for c in cleaned):
                return (
                    "",
                    "bad_term",
                    f"Excluded term must be a single word without whitespace or quotes: {raw!r}.",
                )
            cleaned_lower = cleaned.lower()
            if cleaned_lower not in seen_terms:
                seen_terms.add(cleaned_lower)
                ops.append(f"-{cleaned}")

    if exact_phrases:
        seen_phrases: set[str] = set()
        for raw in exact_phrases:
            cleaned = (raw or "").strip().strip('"').strip("'")
            if not cleaned:
                return "", "bad_phrase", "Exact phrase cannot be empty."
            if "\n" in cleaned or "\r" in cleaned:
                return "", "bad_phrase", f"Exact phrase cannot contain newlines: {raw!r}."
            if '"' in cleaned:
                return "", "bad_phrase", f"Exact phrase cannot contain internal quotes: {raw!r}."
            cleaned_lower = cleaned.lower()
            if cleaned_lower not in seen_phrases:
                seen_phrases.add(cleaned_lower)
                ops.append(f'"{cleaned}"')

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


MAX_QUERY = 128


def fit_query(
    query: str,
    *,
    lang: str | None = None,
    filter_ops: str | None = None,
    max_query: int = MAX_QUERY,
) -> tuple[str, bool]:
    """Fit query, language operator and filter operators into ``max_query`` characters.

    Operators and language are prioritized. If they alone exceed ``max_query``, raises
    ValueError("operators_exceed_limit").
    When the query must be trimmed:
    - Cuts at the last word boundary (space).
    - Drops any half-cut operator or token.
    - Drops unbalanced double quotes.
    - Preserves Unicode characters intact.

    Returns ``(fitted_query, trimmed)``.
    """
    clean_q = re.sub(r"\s+", " ", query).strip()

    lang_prefix = ""
    if lang:
        if not re.search(r"(?:^|\s)lang:\S+", clean_q):
            lang_prefix = f"lang:{lang} "

    filter_suffix = ""
    if filter_ops and filter_ops.strip():
        filter_suffix = f" {filter_ops.strip()}"

    fixed_len = len(lang_prefix) + len(filter_suffix)
    if fixed_len > max_query:
        raise ValueError("Operators and language filter exceed maximum query length.")

    available = max_query - fixed_len

    if not clean_q:
        fitted = f"{lang_prefix.strip()}{filter_suffix}".strip()
        return fitted, False

    if len(clean_q) <= available:
        fitted = f"{lang_prefix}{clean_q}{filter_suffix}".strip()
        return re.sub(r"\s+", " ", fitted), False

    if available <= 0:
        fitted = f"{lang_prefix.strip()}{filter_suffix}".strip()
        return fitted, True

    cut = clean_q[:available]

    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space].rstrip()

    tokens = cut.split()
    if tokens:
        last_tok = tokens[-1]
        tok_pos = clean_q.find(last_tok, max(0, len(cut) - len(last_tok) - 5))
        if tok_pos >= 0:
            tok_end = tok_pos + len(last_tok)
            if tok_end < len(clean_q) and not clean_q[tok_end].isspace():
                if len(tokens) > 1 or ":" in last_tok or last_tok.startswith(("-", "+")):
                    tokens.pop()
                    cut = " ".join(tokens).rstrip()

    tokens = cut.split()
    while tokens and (tokens[-1].endswith(":") or tokens[-1] in ("-", "+")):
        tokens.pop()
    cut = " ".join(tokens).rstrip()

    if cut.count('"') % 2 != 0:
        last_q = cut.rfind('"')
        cut = cut[:last_q].rstrip()

    tokens = cut.split()
    while tokens and (tokens[-1].endswith(":") or tokens[-1] in ("-", "+")):
        tokens.pop()
    cut = " ".join(tokens).rstrip()

    fitted = f"{lang_prefix}{cut}{filter_suffix}".strip()
    return re.sub(r"\s+", " ", fitted), True
