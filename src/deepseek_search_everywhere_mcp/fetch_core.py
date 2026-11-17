"""Fetch core shared by the fetch, extract and research tools.

Validates URLs statically, calls and caches the upstream fetch, cleans page content, selects
excerpts and falls back to a search when a page cannot be read. Other modules call
``fetch_core._fetch`` so one patch covers them all.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from . import runtime, search_core
from .boilerplate import remove_boilerplate
from .egress import check_fetch_egress
from .envelope import (
    COST_PER_REQUEST,
    _envelope_meta,
)
from .errors import (
    _fail,  # re-exported for callers and tests
)
from .filters import (
    fit_query,
)
from .ratelimit import FETCH
from .redact import redact
from .runtime import (
    _cached_call,
    _canonical_json,
    _get_budget_tracker,
    _get_fetch_cache,
    _get_fetch_flight,
    _limited_request,
    _sha256,
)
from .sanitize import (
    sanitize_untrusted_text,
    strip_hidden_unicode,
)
from .snippets import extract_snippets, split_sections
from .text import (
    _cap_raw_content,
    _collapse_repeated_title,
    _truncate,
    _unescape_markdown,
)
from .transport import _base_url
from .urls import (
    CODE_INVALID,
    UrlError,
    validate_url,
)

MAX_URL = 8192


FETCH_TIMEOUT = 60.0


FETCH_PATH = "/v1/tools/fetch"


def _fetch_cache_key(url: str) -> str:
    """Key a fetch by endpoint and the exact URL, minus its fragment (never sent upstream).

    The URL is deliberately not canonicalised; shaping options (``clean``, ``max_chars``)
    are applied per call and stay out of the key, as does the API key.
    """
    target = url.strip().split("#", 1)[0]
    return _sha256(_canonical_json({"path": FETCH_PATH, "base_url": _base_url(), "url": target}))


async def _fetch_upstream(url: str, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``fetch`` token, call upstream, cache a non-empty page.

    Returns ``{"data": ...}`` with the raw upstream page on success or an error envelope.
    """
    budget = _get_budget_tracker()
    reservation = None
    if budget is not None:
        reservation = await budget.reserve(COST_PER_REQUEST)
        if reservation is None:
            return _fail(
                "budget_exceeded",
                f"Daily spend limit of ${budget.limit_usd:.2f} exceeded.",
                402,
            )
    try:
        result = await _limited_request(FETCH, FETCH_PATH, {"url": url}, FETCH_TIMEOUT)
    except BaseException:
        if reservation is not None:
            await reservation.release()
        raise

    if not result.get("ok"):
        if reservation is not None:
            await reservation.release()
        return result

    if reservation is not None:
        await reservation.commit(COST_PER_REQUEST)

    record = {"data": result["data"]}
    cache = _get_fetch_cache()
    if cache is not None and result["data"].get("content"):
        cache.set(key, record)
    return record


def _reject_url(url: str) -> dict[str, Any] | None:
    """Statically validate ``url``; return an error envelope if it is rejected, else ``None``.

    Malformed URLs map to ``bad_url`` and blocked targets (internal hosts, literal private
    addresses, credentials in the URL) to ``ssrf_blocked``. Every tool that forwards a URL
    to DeepSeek calls this before touching the cache, the limiter or the network.
    """
    try:
        validate_url(url)
    except UrlError as exc:
        return _fail("bad_url" if exc.code == CODE_INVALID else exc.code, exc.message, param="url")
    return None


def _check_fetch_url(url: str) -> dict[str, Any] | None:
    """Validate a stripped fetch URL; return an error envelope if it is unusable, else ``None``."""
    if not url:
        return _fail("empty_url", "url must not be empty.", param="url")
    if len(url) > MAX_URL:
        return _fail("url_too_long", f"url must be at most {MAX_URL} characters.", param="url")
    if not re.match(r"^https?://", url, re.I):
        return _fail("bad_url", "url must start with http:// or https://.", param="url")
    rejection = _reject_url(url)
    if rejection is not None:
        return rejection
    egress_err = check_fetch_egress(url, runtime.get_settings())
    if egress_err is not None:
        return _fail("egress_blocked", egress_err)
    return None


_SEARCH_ADVICE = (
    "Direct fetch is unavailable for this document: use deepseek_search instead, "
    "which returns page content via the search index."
)


# A PDF served without an extension under a /pdf/<id> path, as arXiv, bioRxiv and medRxiv do.
_PDF_ID_PATH = re.compile(r"/pdf/[^/]+/?$")


def _needs_search_advice(url: str) -> bool:
    """True for a PDF path (``.pdf`` or ``/pdf/<id>``; query and fragment ignored) or a Reddit host."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    return (
        parts.path.lower().endswith(".pdf")
        or _PDF_ID_PATH.search(parts.path.lower()) is not None
        or host in ("reddit.com", "redd.it")
        or host.endswith((".reddit.com", ".redd.it"))
    )


def _with_search_advice(error: dict[str, Any], url: str) -> dict[str, Any]:
    """Append the search-and-read advice to a 404 for a PDF or Reddit URL; code and status stay."""
    detail = error.get("error")
    if not isinstance(detail, dict) or detail.get("status") != 404 or not _needs_search_advice(url):
        return error
    message = redact(f"{detail.get('message', '')} — {_SEARCH_ADVICE}")
    return {**error, "error": {**detail, "message": message}}


def _is_canonical_match(target_url: str, candidate_url: str) -> bool:
    """Check if candidate_url matches target_url canonically.

    Must share the same normalized host (no parent/subdomain match).
    For PDFs: same host and same filename (ignoring optional .pdf extension).
    For non-PDFs: same host and same normalized path.
    """
    try:
        t_parts = urllib.parse.urlsplit(target_url)
        c_parts = urllib.parse.urlsplit(candidate_url)
    except ValueError:
        return False

    t_host = (t_parts.hostname or "").lower()
    c_host = (c_parts.hostname or "").lower()
    if t_host.startswith("www."):
        t_host = t_host[4:]
    if c_host.startswith("www."):
        c_host = c_host[4:]
    if not t_host or t_host != c_host:
        return False

    is_pdf = t_parts.path.lower().endswith(".pdf") or _PDF_ID_PATH.search(t_parts.path.lower()) is not None

    t_path = t_parts.path.rstrip("/")
    c_path = c_parts.path.rstrip("/")

    if is_pdf:
        t_filename = t_path.split("/")[-1].removesuffix(".pdf") if t_path else ""
        c_filename = c_path.split("/")[-1].removesuffix(".pdf") if c_path else ""
        return bool(t_filename and t_filename == c_filename)

    return t_path == c_path


async def _search_fallback_fetch(
    url: str,
    *,
    refresh: bool = False,
    strip_boilerplate: bool = True,
) -> dict[str, Any] | None:
    """Attempt to retrieve page content via DeepSeek search when fetch returns 404 for PDF/Reddit.

    Returns ``{"data": {"title": ..., "content": ...}, "fallback_from_search": True}`` on success,
    or ``None`` if search yielded no matching results for the target domain or had empty content.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        target_host = (parts.hostname or "").lower()
    except ValueError:
        return None

    host_for_site = target_host[4:] if target_host.startswith("www.") else target_host
    path_segments = [s for s in parts.path.split("/") if s and s not in ("pdf", "comments", "r")]
    query_terms = " ".join(path_segments[-3:]) if path_segments else ""
    try:
        search_query, _ = fit_query(query_terms, filter_ops=f"site:{host_for_site}")
    except ValueError:
        return None

    search_res = await search_core._search(
        search_query,
        count=2,
        ori_question=None,
        content="full",
        lang=None,
        max_chars_per_result=40000,
        max_total_chars=40000,
        dedupe=True,
        rewrite=False,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
    )
    if not search_res.get("ok"):
        return None

    for res in search_res.get("results") or []:
        res_url = res.get("url") or ""
        if not _is_canonical_match(url, res_url):
            continue

        raw_content = res.get("content") or ""
        if raw_content.strip():
            return {
                "data": {
                    "title": res.get("title") or "",
                    "content": raw_content,
                },
                "fallback_from_search": True,
            }
    return None


async def _fetch(
    url: str,
    clean: bool,
    max_chars: int,
    *,
    offset: int | None = None,
    limit: int | None = None,
    find_text: str | None = None,
    query: str | None = None,
    section: int | None = None,
    max_snippets: int = 3,
    refresh: bool,
    strip_boilerplate: bool = True,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Fetch an already validated ``url`` through the cache, single-flight and limiter.

    Cleaning, snippets, sectional reading, pagination and ``strip_boilerplate`` shape the cached page and
    never change the cache key.

    Returns the ``deepseek_fetch`` envelope, or an error envelope from upstream or the limiter.
    """
    selectors = [
        section is not None,
        query is not None,
        offset is not None or limit is not None,
        find_text is not None,
    ]
    if sum(bool(s) for s in selectors) > 1:
        return _fail(
            "bad_request", "Selectors section, query, offset/limit, and find_text are mutually exclusive."
        )
    if offset is not None and offset < 0:
        return _fail("bad_request", "offset must be at least 0.", param="offset")
    if limit is not None and limit <= 0:
        return _fail("bad_request", "limit must be greater than 0.", param="limit")

    key = _fetch_cache_key(url)
    record, cached = await _cached_call(
        _get_fetch_cache(),
        _get_fetch_flight(),
        key,
        lambda: _fetch_upstream(url, key),
        refresh=refresh,
    )
    fallback_used = False
    if record.get("ok") is False:
        if fallback_to_search and record.get("error", {}).get("status") == 404 and _needs_search_advice(url):
            fallback_record = await _search_fallback_fetch(
                url,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
            )
            if fallback_record is not None:
                record = fallback_record
                cached = False
                fallback_used = True
            else:
                return _with_search_advice(record, url)
        else:
            return _with_search_advice(record, url)

    data = record["data"]
    raw_content = data.get("content") or ""
    raw_len = len(raw_content)
    content = _cap_raw_content(raw_content, max_chars)
    content_capped = raw_len > len(content)
    capped_limit = len(content)
    if clean:
        content = _unescape_markdown(content)
    only_boilerplate = False
    if strip_boilerplate:
        stripped = remove_boilerplate(content, url)
        only_boilerplate = bool(content.strip()) and not stripped
        content = stripped
    content, sanitize_notes = sanitize_untrusted_text(content)

    total_sections: int | None = None
    section_has_more: bool | None = None
    section_index: int | None = None
    snippets_count: int | None = None
    no_match_note: str | None = None

    if section is not None:
        if section < 1:
            return _fail("bad_request", "section must be at least 1.", param="section")
        sections = split_sections(content)
        total_sections = len(sections)
        if total_sections == 0:
            content = ""
        elif section > total_sections:
            return _fail(
                "bad_request",
                f"section {section} out of bounds; document has {total_sections} section(s).",
                param="section",
            )
        else:
            section_index = section
            content = sections[section - 1]["content"]
            section_has_more = section < total_sections
    elif query:
        had_content = bool(content.strip())
        content, snippets_count = extract_snippets(
            content, query, max_snippets=max_snippets, max_chars=max_chars
        )
        if had_content and snippets_count == 0:
            no_match_note = (
                f"No excerpt contains the query terms: '{query}'. The page was fetched but nothing matched."
            )

    total_chars = raw_len if content_capped else len(content)
    notes = [no_match_note] if no_match_note else _fetch_notes(content, only_boilerplate)
    notes.extend(sanitize_notes)
    if content_capped:
        notes.append(
            f"Content was capped at {capped_limit} characters before processing "
            f"(original document length: {raw_len} chars)."
        )

    if find_text:
        target = find_text.strip()
        if target:
            search_start = offset if offset is not None else 0
            idx = content.lower().find(target.lower(), search_start)
            if idx >= 0:
                offset = idx
            else:
                notes.append(f"find_text '{target}' not found in page content.")
                content = ""

    is_windowed = offset is not None or limit is not None or find_text is not None
    if is_windowed:
        start = offset if offset is not None else 0
        if offset is not None and offset >= capped_limit:
            if content_capped:
                notes.append(
                    f"offset {offset} is beyond the capped content limit ({capped_limit} chars); "
                    f"original length is {raw_len} chars."
                )
            else:
                notes.append(f"offset {offset} is beyond the total document length ({capped_limit} chars).")
        effective_limit = min(limit if limit is not None else max_chars, max_chars)
        content = content[start : start + effective_limit]
        window_has_more = (start + len(content)) < total_chars
        truncated = window_has_more
    else:
        start = 0
        content, truncated = _truncate(content, max_chars)
        window_has_more = None

    if fallback_used:
        cost = 2 * COST_PER_REQUEST
    else:
        cost = 0.0 if cached else COST_PER_REQUEST
    if fallback_used or record.get("fallback_from_search"):
        notes.append("Content retrieved via DeepSeek search index (direct fetch unavailable).")
    source = "search_index" if (fallback_used or record.get("fallback_from_search")) else "fetch"
    envelope: dict[str, Any] = {
        "ok": True,
        "url": url,
        "source": source,
        "title": (
            strip_hidden_unicode(_collapse_repeated_title(data.get("title")))[0]
            if isinstance(_collapse_repeated_title(data.get("title")), str)
            else _collapse_repeated_title(data.get("title"))
        ),
        "content": content,
        "chars": len(content),
        "total_chars": total_chars,
        "truncated": truncated,
        "cached": cached,
        "cost_usd": cost,
        "untrusted_content": True,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if content_capped:
        envelope["content_capped"] = True
    if section is not None:
        envelope["section"] = section_index
        envelope["total_sections"] = total_sections
        envelope["has_more"] = section_has_more
    elif is_windowed:
        envelope["offset"] = start
        envelope["has_more"] = window_has_more
        if find_text:
            envelope["find_text"] = find_text
    if query:
        envelope["query"] = query
        envelope["snippets_count"] = snippets_count
    return envelope


def _fetch_notes(content: str, only_boilerplate: bool) -> list[str]:
    """Explain an empty fetch: a page that was nothing but boilerplate, or one with no content at all."""
    if content:
        return []
    if only_boilerplate:
        return [
            "Empty content: the page rendered only a cookie-consent banner or site interface, which was "
            "removed. Try deepseek_search instead, or strip_boilerplate=False to see it."
        ]
    return ["Empty content: the page may be a PDF, JS-only or blocked. Try deepseek_search instead."]
