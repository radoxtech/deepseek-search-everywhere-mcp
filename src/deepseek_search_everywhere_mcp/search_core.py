"""Search core shared by the search, fetch and research tools.

Builds and caches upstream search calls, rewrites non-English queries, dedupes results, detects
dates and fits result text into the caller's character budgets. ``_search`` returns a full
search envelope; other modules call it as ``search_core._search`` so one patch covers them all.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from . import runtime
from .boilerplate import remove_boilerplate
from .dates import (
    _dated,
)
from .egress import check_search_egress
from .envelope import (
    COST_PER_REQUEST,
    TRUNCATED_NOTE,
    _envelope_meta,
)
from .errors import (
    _fail,  # re-exported for callers and tests
)
from .filters import MAX_QUERY, compile_filters, filter_results_by_domain, fit_query
from .ratelimit import SEARCH
from .redact import redact
from .runtime import (
    CHAT_TIMEOUT,
    REWRITE_MODEL,
    _cached_call,
    _canonical_json,
    _get_budget_tracker,
    _get_rewrite_cache,
    _get_rewrite_flight,
    _get_search_cache,
    _get_search_flight,
    _limited_request,
    _sha256,
    logger,
)
from .sanitize import (
    sanitize_untrusted_text,
)
from .text import (
    _cap_raw_content,
    _clean_index_text,
    _decode_json_escapes,
    _strip_html_tags,
    _title_or_host,
    _truncate,
    _unescape_markdown,
)
from .transport import _base_url
from .urls import (
    _canonical_url,
)

SEARCH_TIMEOUT = 45.0


SEARCH_PATH = "/v1/tools/search"


SYSTEM_PROMPT = (
    "You rewrite search queries. Output ONLY the rewritten query, no quotes, no explanation. "
    "Translate the query to English. Keep search operators verbatim: site:, -site:, lang:, after:, "
    "before:, and quoted phrases. Keep proper nouns, product names and numbers unchanged. "
    "Maximum 120 characters."
)


MIN_TITLE_SITE_CHARS = 20


MIN_TITLE_SITE_WORDS = 3


MAX_COUNT = 50  # largest ``count`` accepted from callers and by the upstream search API


# With dedupe on, the upstream is asked for up to max(OVERFETCH_MIN, ceil(count * OVERFETCH_RATIO))
# extra results (capped at MAX_COUNT) so that removing duplicates still leaves ``count`` results.
OVERFETCH_MIN = 2


OVERFETCH_RATIO = 0.5


# Default summary cap per result when the search returns no page content (need_content=False).
SEARCH_SUMMARY_CHARS = 400


def _search_cache_key(payload: dict[str, Any]) -> str:
    """Key a search by endpoint and upstream payload only (no key, no shaping options)."""
    return _sha256(_canonical_json({"path": SEARCH_PATH, "base_url": _base_url(), "payload": payload}))


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _title_site_key(item: dict) -> tuple[str, str] | None:
    title = _norm_text(item.get("title"))
    site = _norm_text(item.get("site_name"))
    if not title or not site:
        return None
    if len(title) < MIN_TITLE_SITE_CHARS or len(title.split()) < MIN_TITLE_SITE_WORDS:
        return None
    return (title, site)


def _dedupe_key(item: dict) -> tuple | None:
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        canonical = _canonical_url(url)
        if canonical is not None:
            return ("url", canonical)
    title_site = _title_site_key(item)
    if title_site is None:
        return None
    return ("title", *title_site)


def _is_https_url(item: dict[str, Any]) -> bool:
    url = item.get("url")
    return isinstance(url, str) and url.lower().startswith("https://")


def _dedupe_results(items: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    seen_keys: dict[tuple, int] = {}
    seen_title_sites: dict[tuple[str, str], int] = {}

    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue

        key = _dedupe_key(item)
        title_site = _title_site_key(item)

        target_idx: int | None = None
        if key is not None and key in seen_keys:
            target_idx = seen_keys[key]
        elif title_site is not None and title_site in seen_title_sites:
            target_idx = seen_title_sites[title_site]

        if target_idx is not None:
            removed += 1
            if _is_https_url(item) and not _is_https_url(kept[target_idx]):
                kept[target_idx] = item
            continue

        idx = len(kept)
        if key is not None:
            seen_keys[key] = idx
        if title_site is not None:
            seen_title_sites[title_site] = idx
        kept.append(item)

    return kept, removed


def _upstream_count(count: int, dedupe: bool) -> int:
    """Number of results to request upstream: ``count`` plus a dedupe margin, capped at the API max."""
    if not dedupe:
        return count
    extra = max(OVERFETCH_MIN, math.ceil(count * OVERFETCH_RATIO))
    return min(count + extra, MAX_COUNT)


async def _rewrite_query(query: str) -> str | None:
    try:
        result = await runtime._chat_request(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            thinking={"type": "disabled"},
            model=REWRITE_MODEL,
            max_tokens=64,
            temperature=0,
            timeout=CHAT_TIMEOUT,
            attempts=1,
        )
        if not result.get("ok"):
            return None
        content = result.get("content")
        if not isinstance(content, str):
            return None
        first = next((line.strip() for line in content.splitlines() if line.strip()), "")
        if len(first) >= 2 and first[0] in "\"'" and first[-1] in "\"'":
            first = first[1:-1]
        cleaned, _ = fit_query(first)
        if not cleaned:
            return None
        return cleaned
    except Exception as exc:  # broad on purpose: rewriting is best-effort and must never fail a search
        logger.debug("Query rewrite failed: %s", redact(f"{type(exc).__name__}: {exc}"))
        return None


async def _cached_rewrite(query: str) -> str | None:
    """Rewrite ``query`` through the rewrite cache and single-flight group.

    Identical concurrent rewrites share one model call; failed rewrites are not remembered.
    """
    cache = _get_rewrite_cache()
    key = (_base_url(), REWRITE_MODEL, _sha256(SYSTEM_PROMPT), query)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    async def leader() -> str | None:
        rewritten = await _rewrite_query(query)
        if cache is not None and rewritten is not None:
            cache.set(key, rewritten)
        return rewritten

    return (await _get_rewrite_flight().run(key, leader)).value


async def _search_upstream(payload: dict[str, Any], query_original: str | None, key: str) -> dict[str, Any]:
    """Leader-only path: take a ``search`` token, call upstream, cache a usable answer.

    Returns ``{"data": ..., "query_original": ...}`` on success or an error envelope.
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
        result = await _limited_request(SEARCH, SEARCH_PATH, payload, SEARCH_TIMEOUT)
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

    record = {"data": result["data"], "query_original": query_original}
    cache = _get_search_cache()
    if cache is not None and result["data"].get("results"):
        cache.set(key, record)
    return record


def _filter_param(
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    exclude_terms: list[str] | None,
    exact_phrases: list[str] | None,
    after: str | None,
    before: str | None,
) -> str | None:
    """Name the first filter parameter that fails on its own, or None when only their combination does."""
    single: dict[str, dict[str, Any]] = {
        "include_domains": {"include_domains": include_domains},
        "exclude_domains": {"exclude_domains": exclude_domains},
        "exclude_terms": {"exclude_terms": exclude_terms},
        "exact_phrases": {"exact_phrases": exact_phrases},
        "after": {"after": after},
        "before": {"before": before},
    }
    for name, kwargs in single.items():
        if next(iter(kwargs.values())) and compile_filters(**kwargs)[1]:
            return name
    return None


def _check_search_args(
    query: str, count: int, lang: str | None, filter_ops: str | None = None
) -> dict[str, Any] | None:
    """Validate query, count, lang and filter budget; return an error envelope or None."""
    if not query:
        return _fail("empty_query", "query must not be empty.", param="query")
    egress_err = check_search_egress(query, runtime.get_settings())
    if egress_err is not None:
        return _fail("egress_blocked", egress_err)
    ops_len = len(filter_ops) if filter_ops else 0
    budget = MAX_QUERY - (ops_len + 1 if ops_len else 0)
    if len(query) > budget:
        if ops_len:
            return _fail(
                "query_too_long",
                f"query is {len(query)} characters; it must be at most {budget} when filters "
                f"({ops_len} chars) are applied ({MAX_QUERY} in total).",
                param="query",
            )
        return _fail(
            "query_too_long",
            f"query is {len(query)} characters; it must be at most {MAX_QUERY}.",
            param="query",
        )
    if not 1 <= count <= MAX_COUNT:
        return _fail("bad_count", f"count must be between 1 and {MAX_COUNT}.", param="count")
    if lang and not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", lang):
        return _fail("bad_lang", "lang must look like 'es' or 'pt-BR'.", param="lang")
    return None


async def _maybe_rewrite(
    query: str, ori_question: str | None, rewrite: bool, lang: str | None
) -> tuple[str, str | None, str | None, bool]:
    """Apply the optional query rewrite.

    Returns ``(query, ori_question, query_original, rewrite_failed)``; ``query_original`` is
    set only when the rewrite changed the query, and then also seeds a missing ``ori_question``.
    """
    if not (rewrite and lang is None and runtime._api_key()):
        return query, ori_question, None, False
    rewritten = await _cached_rewrite(query)
    if rewritten is None:
        return query, ori_question, None, True
    if rewritten == query:
        return query, ori_question, None, False
    return rewritten, ori_question or query[:MAX_QUERY], query, False


def _build_search_payload(
    query: str,
    count: int,
    dedupe: bool,
    need_content: bool,
    ori_question: str | None,
    *,
    localization: str | None = None,
) -> dict[str, Any]:
    """Build the upstream search payload (``query`` already carries any ``lang:`` operator)."""
    payload: dict[str, Any] = {
        "query": query,
        "count": _upstream_count(count, dedupe),
        "need_content": need_content,
    }
    if ori_question:
        payload["ori_question"] = ori_question[:MAX_QUERY]
    if localization and localization.strip():
        payload["localization"] = localization.strip()
    return payload


def _result_entry(item: dict[str, Any]) -> dict[str, Any]:
    """Map one upstream result to its metadata fields, with the host as a blank-title fallback."""
    when, date_source = _dated(
        item.get("date"), item.get("title"), item.get("content") or item.get("summary"), item.get("url")
    )
    entry: dict[str, Any] = {
        "title": _title_or_host(item),
        "url": item.get("url"),
        "date": when,
        "date_source": date_source,
        "authority": item.get("authority"),
        "rerank_score": item.get("rerank_score"),
        "site": item.get("site_name"),
    }
    snippet = _clean_index_text(item.get("snippet") or "")
    if snippet:
        entry["snippet"] = snippet
    return entry


def _summary_cap(summary_chars: int | None, need_content: bool, per_result: int) -> int:
    """Return the per-result summary cap: ``summary_chars`` (or its default) bounded by ``per_result``.

    The default is ``SEARCH_SUMMARY_CHARS`` without page content, and ``per_result`` with it (the
    summary is then only a fallback for results that came back without content). 0 omits summaries.
    """
    if summary_chars is None:
        summary_chars = per_result if need_content else SEARCH_SUMMARY_CHARS
    return min(summary_chars, per_result)


def _apply_char_budget(
    entry: dict[str, Any],
    item: dict[str, Any],
    need_content: bool,
    per_result: int,
    remaining: int,
    summary_cap: int,
    strip_boilerplate: bool = True,
) -> tuple[int, bool]:
    """Add the result's content (or summary) to ``entry``, cut to the per-result and total budgets.

    Content is unescaped (Markdown escapes, then any JSON escapes left by the search index), rid of
    leftover simple HTML tags outside code and, with ``strip_boilerplate``, of known site-UI lines before it
    is capped at ``per_result``; a summary, used when there is no content, is capped at
    ``summary_cap`` and is left out entirely when that cap is 0. Without content (snippets mode), a
    summary is only a fallback for results that have no snippet. With content, ``chars`` is the
    length returned.
    Returns ``(remaining, truncated)``: the budget left afterwards and whether any text was cut.
    """
    body = item.get("content") if need_content else None
    if body:
        budget_target = max(0, min(per_result, remaining))
        capped_body = _cap_raw_content(body, budget_target)
        cleaned = _strip_html_tags(_decode_json_escapes(_unescape_markdown(capped_body)))
        if strip_boilerplate:
            cleaned = remove_boilerplate(cleaned, item.get("url"))
        cleaned, _ = sanitize_untrusted_text(cleaned)
        entry["content"], cut = _truncate(cleaned, budget_target)
        entry["chars"] = len(entry["content"])
        return remaining - len(entry["content"]), cut
    if not need_content and entry.get("snippet"):
        # Snippets mode: the snippet is the result text, so the upstream summary is left out.
        return remaining, False
    summary = _clean_index_text(item.get("summary") or "")
    if summary and summary_cap > 0:
        entry["summary"], cut = _truncate(summary, max(0, min(summary_cap, remaining)))
        return remaining - len(entry["summary"]), cut
    return remaining, False


def _shape_results(
    raw: list[dict[str, Any]],
    need_content: bool,
    per_result: int,
    total: int,
    summary_cap: int,
    strip_boilerplate: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """Shape upstream results under the character budgets; return ``(results, truncated)``.

    With ``need_content`` set, results stop once the total budget is spent.
    """
    remaining = total
    shaped: list[dict[str, Any]] = []
    truncated = False
    for item in raw:
        entry = _result_entry(item)
        remaining, cut = _apply_char_budget(
            entry, item, need_content, per_result, remaining, summary_cap, strip_boilerplate
        )
        truncated = truncated or cut
        shaped.append(entry)
        if need_content and remaining <= 0:
            break
    return shaped, truncated


def _search_notes(
    used: str,
    query_original: str | None,
    rewrite_failed: bool,
    removed: int,
    trimmed: bool,
    lang: str | None,
    truncated: bool,
    filter_ops: str | None = None,
    domain_dropped: int = 0,
) -> list[str]:
    """Assemble the (redacted) notes of a search envelope, in their fixed order."""
    notes: list[str] = []
    if query_original is not None:
        notes.append(f"query rewritten (flash): '{used}'")
    elif rewrite_failed:
        notes.append("query rewrite failed; using original query")
    if removed:
        notes.append(f"deduped {removed} duplicate result(s)")
    if trimmed:
        notes.append(f"Query trimmed to fit the {MAX_QUERY}-character limit.")
    if lang:
        notes.append(f"Language operator lang:{lang} applied.")
    if filter_ops:
        notes.append(f"Filters applied: {filter_ops}")
    if domain_dropped:
        notes.append(
            f"Dropped {domain_dropped} result(s) not matching the include_domains/exclude_domains "
            "domain filter (the search ignored it)."
        )
    if truncated:
        notes.append(TRUNCATED_NOTE)
    return [redact(note) for note in notes]


async def _search(
    query: str,
    count: int,
    ori_question: str | None,
    content: Literal["snippets", "full"] | bool,
    lang: str | None,
    max_chars_per_result: int,
    max_total_chars: int,
    dedupe: bool = True,
    rewrite: bool = False,
    *,
    localization: str | None = None,
    exclude_terms: list[str] | None = None,
    exact_phrases: list[str] | None = None,
    after: str | None = None,
    before: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    refresh: bool = False,
    summary_chars: int | None = None,
    strip_boilerplate: bool = True,
) -> dict[str, Any]:
    """Run one validated, cached search and shape the envelope shared by the search tools.

    ``summary_chars`` caps the summary of results without content (see ``_summary_cap``).
    ``strip_boilerplate`` removes known site-UI lines from page content; it only shapes the
    response, so the cache key and the cached upstream data are the same either way.
    """
    if localization is not None:
        if not isinstance(localization, str):
            return _fail("bad_request", "localization must be a string.")
        if len(localization.strip()) > 64:
            return _fail("bad_request", "localization exceeds maximum length of 64 characters.")

    if isinstance(content, bool):
        content = "full" if content else "snippets"
    if content not in ("snippets", "full"):
        return _fail("bad_request", "content must be 'snippets' or 'full'.", param="content")
    need_content = content == "full"

    query = (query or "").strip()
    ops_string, filter_code, filter_err = compile_filters(
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        exclude_terms=exclude_terms,
        exact_phrases=exact_phrases,
        after=after,
        before=before,
    )
    if filter_code:
        return _fail(
            filter_code,
            filter_err or "Invalid search filters.",
            param=_filter_param(
                include_domains, exclude_domains, exclude_terms, exact_phrases, after, before
            ),
        )

    rejection = _check_search_args(query, count, lang, filter_ops=ops_string)
    if rejection is not None:
        return rejection
    if summary_chars is not None and summary_chars < 0:
        return _fail("bad_request", "summary_chars must be at least 0.")

    query, ori_question, query_original, rewrite_failed = await _maybe_rewrite(
        query, ori_question, rewrite, lang
    )
    try:
        used, trimmed = fit_query(query, lang=lang, filter_ops=ops_string)
    except ValueError:
        return _fail(
            "query_too_long", f"Filters and language exceed the {MAX_QUERY}-character limit.", param="query"
        )

    payload = _build_search_payload(
        used, count, dedupe, need_content, ori_question, localization=localization
    )
    key = _search_cache_key(payload)
    record, cached = await _cached_call(
        _get_search_cache(),
        _get_search_flight(),
        key,
        lambda: _search_upstream(payload, query_original, key),
        refresh=refresh,
    )
    if record.get("ok") is False:
        return record

    raw = record["data"].get("results") or []
    raw, domain_dropped = filter_results_by_domain(raw, include_domains, exclude_domains)
    removed = 0
    if dedupe:
        raw, removed = _dedupe_results(raw)
        raw = raw[:count]
    summary_cap = _summary_cap(summary_chars, need_content, max_chars_per_result)
    shaped, truncated = _shape_results(
        raw, need_content, max_chars_per_result, max_total_chars, summary_cap, strip_boilerplate
    )

    cost = 0.0 if cached else COST_PER_REQUEST
    envelope: dict[str, Any] = {
        "ok": True,
        "query_used": used,
        "count": len(shaped),
        "content": content,
        "deduped": removed,
        "cached": cached,
        "requests_used": 0 if cached else 1,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": shaped,
        "notes": _search_notes(
            used,
            query_original,
            rewrite_failed,
            removed,
            trimmed,
            lang,
            truncated and need_content,
            filter_ops=ops_string,
            domain_dropped=domain_dropped,
        ),
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope
