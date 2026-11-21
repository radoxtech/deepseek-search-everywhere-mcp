"""The deepseek_search tool: one query or a list of queries run in parallel."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Literal

from pydantic import Field

from .. import search_core
from ..batch import (
    _wait_or_cancel,
)
from ..envelope import (
    COST_PER_REQUEST,
    TRUNCATED_NOTE,
    _all_cached,
    _envelope_meta,
)
from ..errors import (
    _fail,  # re-exported for callers and tests
)
from ..gap_fill import (
    _fill_gaps,
)
from ..redact import redact
from ..runtime import (
    _timed,
)
from ..search_core import (
    MAX_COUNT,
    SEARCH_TIMEOUT,
)

MAX_QUERIES = 5  # most queries one deepseek_search call runs
DEFAULT_MAX_TOTAL_CHARS = 20000


async def _search_one(
    query: str,
    *,
    count: int = 5,
    content: Literal["snippets", "full"] = "snippets",
    lang: str | None = None,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    refresh: bool = False,
) -> dict[str, Any]:
    """Run one query: the single-query branch of ``deepseek_search``."""
    if max_total_chars < 1:
        return _fail("bad_request", "max_total_chars must be at least 1.", param="max_total_chars")
    per_result = max(500, max_total_chars // max(1, count))
    res = await search_core._search(
        query,
        count,
        None,
        content,
        lang,
        per_result,
        max_total_chars,
        dedupe=True,
        rewrite=False,
        exclude_terms=None,
        exact_phrases=None,
        after=None,
        before=None,
        include_domains=None,
        exclude_domains=None,
        refresh=refresh,
        summary_chars=None,
        strip_boilerplate=True,
    )
    if res.get("ok") and (content == "full" or content is True):
        fill = await _fill_gaps(
            res.get("results") or [], per_result, max_total_chars, refresh=refresh, strip_boilerplate=True
        )
        if fill.requests_used:
            res["requests_used"] = (0 if res.get("cached") else 1) + fill.requests_used
            res["cost_usd"] = round(
                (0.0 if res.get("cached") else COST_PER_REQUEST) + fill.requests_used * COST_PER_REQUEST, 6
            )
            if fill.fetched:
                res["notes"].append(f"fetched {fill.fetched} page(s) with missing or short search content")
            if fill.failed:
                res["notes"].append(f"{fill.failed} fetch(es) failed; see the source's error")
    return res


def _envelope_chars(envelope: dict[str, Any]) -> int:
    return len(json.dumps(envelope))


def _fit_payload_budget(envelope: dict[str, Any], budget: int) -> bool:
    """Trim result text from the end until the serialized envelope fits ``budget``; return True if trimmed."""
    if _envelope_chars(envelope) <= budget:
        return False
    slots: list[tuple[dict[str, Any], str]] = []
    for entry in reversed(envelope.get("results") or []):
        for item in reversed(entry.get("results") or []):
            for key in ("content", "snippet", "summary"):
                if isinstance(item.get(key), str) and item[key]:
                    slots.append((item, key))
    while _envelope_chars(envelope) > budget:
        progressed = False
        for item, key in slots:
            if _envelope_chars(envelope) <= budget:
                break
            text = item.get(key) or ""
            if not text:
                continue
            overflow = _envelope_chars(envelope) - budget
            cut = min(len(text), max(overflow, 32))
            item[key] = text[: len(text) - cut].rstrip()
            progressed = True
        if not progressed:
            break
    return True


@_timed
async def deepseek_search(
    query: Annotated[
        str | list[str],
        Field(
            description="One query (<= 128 chars; operators like site: allowed) or a list of up to "
            f"{MAX_QUERIES} run in parallel"
        ),
    ],
    *,
    count: Annotated[
        int, Field(description=f"Number of ranked results to return (1-{MAX_COUNT}, default 5)")
    ] = 5,
    content: Annotated[
        Literal["snippets", "full"],
        Field(
            description="Result format: 'snippets' (default: snippet, else summary capped at 400 chars) "
            "or 'full' (gap-filled text)"
        ),
    ] = "snippets",
    lang: Annotated[
        str | None,
        Field(
            description=(
                "Language code for sources (e.g. 'en', 'de', 'pl'). Prefer 'en' for tech/science docs; "
                "set code for local laws/markets."
            )
        ),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh results (default False)"),
    ] = False,
) -> dict[str, Any]:
    """Search the live web: ranked results with snippets or full page text, for one query or a list.

    Primary search and retrieval tool for current facts, news, docs and sources. query is one string
    (<= 128 chars, operators and filters included) or a list; a list runs at most 5 unique queries and
    lists the rest in skipped_queries. count is 1-50 (default 5). $0.00075 per query. Snippets by
    default; content="full" returns page text up to 20,000 characters, shared across a list, and adds
    $0.00075 for each page whose text it has to fetch. Results are deduplicated, dated (date,
    date_source) and sanitized as untrusted content. Search is capped at 10 req/s (burst 20).

    When NOT to use: a known URL (deepseek_fetch), fields as JSON (deepseek_extract), or an in-depth
    investigation (deepseek_quick_research or deepseek_deep_agent_research).

    On errors, read error.code: invalid_input (fix the call), blocked_by_policy (do not retry),
    temporary_failure (retry after a short wait), limit_reached or account_error (stop and tell the
    user).
    """
    if isinstance(query, str):
        return await _search_one(
            query,
            count=count,
            content=content,
            lang=lang,
            refresh=refresh,
        )
    return await _search_list(
        query,
        count=count,
        content=content,
        lang=lang,
        refresh=refresh,
    )


async def _search_list(
    queries: list[str],
    *,
    count: int = 5,
    content: Literal["snippets", "full"] = "snippets",
    lang: str | None = None,
    refresh: bool = False,
    deadline: float | None = None,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> dict[str, Any]:
    """Run multiple queries in parallel: the list branch of ``deepseek_search``."""
    if isinstance(content, bool):
        content = "full" if content else "snippets"
    if content not in ("snippets", "full"):
        return _fail("bad_request", "content must be 'snippets' or 'full'.", param="content")
    if not isinstance(queries, list) or not queries:
        return _fail("bad_request", "query must be a string or a non-empty list of strings.", param="query")
    if not all(isinstance(q, str) for q in queries):
        return _fail("bad_request", "every item of a query list must be a string.", param="query")
    if count < 1 or count > MAX_COUNT:
        return _fail("bad_count", f"count must be between 1 and {MAX_COUNT}.", param="count")
    if max_total_chars < 1:
        return _fail("bad_request", "max_total_chars must be at least 1.", param="max_total_chars")

    unique_queries: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for q in queries:
        trimmed = q.strip()
        if not trimmed:
            continue
        if trimmed in seen:
            duplicates += 1
            continue
        seen.add(trimmed)
        unique_queries.append(trimmed)

    if not unique_queries:
        return _fail("bad_request", "a query list must contain at least one non-empty query.", param="query")
    skipped_queries = unique_queries[MAX_QUERIES:]
    unique_queries = unique_queries[:MAX_QUERIES]

    per_query_chars = max(1000, max_total_chars // len(unique_queries))

    async def _run_one(q: str) -> dict[str, Any]:
        return await _search_one(
            q,
            count=count,
            content=content,
            lang=lang,
            max_total_chars=per_query_chars,
            refresh=refresh,
        )

    tasks = [asyncio.create_task(_run_one(q)) for q in unique_queries]
    timeout = deadline if deadline is not None else SEARCH_TIMEOUT
    await _wait_or_cancel(tasks, timeout)

    deadline_error = _fail("deadline_exceeded", f"Search query timed out after {timeout:g} s.")
    results: list[dict[str, Any]] = []
    requests_used = 0
    cached_count = 0
    failed_count = 0
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate query/queries")
    if skipped_queries:
        notes.append(
            f"ran the first {MAX_QUERIES} queries; {len(skipped_queries)} skipped (listed in "
            "skipped_queries): send them in another call"
        )

    for q, task in zip(unique_queries, tasks, strict=True):
        if task.cancelled():
            res = deadline_error
        elif task.exception() is not None:
            res = _fail("network_error", redact(str(task.exception())))
        else:
            res = task.result()

        if isinstance(res, dict) and res.get("ok"):
            is_cached = res.get("cached", False)
            if is_cached:
                cached_count += 1
            else:
                requests_used += 1
            results.append(
                {
                    "query": q,
                    "ok": True,
                    "count": len(res.get("results") or []),
                    "results": res.get("results") or [],
                    "cached": is_cached,
                    "notes": res.get("notes") or [],
                }
            )
            for n in res.get("notes") or []:
                if n not in notes:
                    notes.append(n)
        else:
            failed_count += 1
            err = res.get("error") if isinstance(res, dict) else None
            results.append(
                {
                    "query": q,
                    "ok": False,
                    "count": 0,
                    "results": [],
                    "cached": False,
                    "error": err,
                    "notes": res.get("notes", []) if isinstance(res, dict) else [],
                }
            )

    cost = round(requests_used * COST_PER_REQUEST, 5)
    envelope: dict[str, Any] = {
        "ok": True,
        "count": len(results),
        "total_results": sum(r["count"] for r in results),
        "content": content,
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": cost,
        "untrusted_content": True,
        "results": results,
        "notes": notes,
        "_meta": _envelope_meta(cost, untrusted_content=True),
    }
    if skipped_queries:
        envelope["skipped_queries"] = skipped_queries
    if _fit_payload_budget(envelope, max_total_chars):
        envelope["notes"].append(TRUNCATED_NOTE)
        _fit_payload_budget(envelope, max_total_chars)
    return envelope
