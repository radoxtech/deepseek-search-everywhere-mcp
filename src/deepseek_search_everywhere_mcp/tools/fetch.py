"""The deepseek_fetch tool: one URL, or a list of URLs read in parallel."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from .. import fetch_core, runtime
from ..batch import (
    _assemble_batch,
    _dedupe_urls,
    _run_batch,
)
from ..errors import _fail
from ..fetch_core import (
    _check_fetch_url,
)
from ..runtime import (
    _timed,
)
from ..snippets import MAX_SNIPPETS

SINGLE_MAX_CHARS = 20000  # default max_chars for one URL
LIST_MAX_CHARS = 4000  # default max_chars per URL for a list
DEFAULT_MAX_TOTAL_CHARS = 20000


@_timed
async def deepseek_fetch(
    url: Annotated[
        str | list[str],
        Field(description="Public HTTP(S) URL to read, or a list of up to 10 URLs read in parallel"),
    ],
    *,
    query: Annotated[
        str | None,
        Field(description="Return only the excerpts that best match these terms (default None)"),
    ] = None,
    max_chars: Annotated[
        int | None,
        Field(description="Max characters of page text per URL (default 20000; 4000 per URL for a list)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages (default False)"),
    ] = False,
) -> dict[str, Any]:
    """Read known web pages as clean Markdown: one URL, or a list of 1-10 URLs read in parallel.

    Use it when you already know the URL. $0.00075 per URL; cached pages are free. A single URL returns
    up to 20,000 characters; pass query= to get only the most relevant excerpts (fast lexical match, no
    LLM). A list reads up to 5 URLs concurrently, 4,000 characters per URL and 20,000 in total, skips
    blanks, fetches duplicates once, keeps input order and reports a failed URL in place (partial: true).
    URLs must be public HTTP(S), at most 8,192 characters. A 404 on a PDF or Reddit URL falls back to the
    search index. Fetches are capped at 5 requests/second (burst 10). Content is sanitized: images
    neutralized, hidden text stripped.

    When NOT to use: finding pages by topic (deepseek_search), fields as JSON (deepseek_extract), or
    an in-depth investigation (deepseek_quick_research or deepseek_deep_agent_research).

    Errors: page_unavailable (change approach), blocked_by_policy (do not retry), temporary_failure
    (retry), invalid_input (fix).

    Returns for one URL: {"ok": True, "url": str, "source": "fetch" | "search_index", "title": str,
    "content": str, "chars": int, "total_chars": int, "truncated": bool, "cached": bool,
    "cost_usd": float, "notes": [...]}. For a list: {"ok": True, "count": int, "results": [...],
    "partial": bool, "failed_count": int, "cost_usd": float}.
    """
    if isinstance(url, str):
        return await _fetch_one(
            url,
            max_chars=SINGLE_MAX_CHARS if max_chars is None else max_chars,
            query=query,
            refresh=refresh,
        )
    return await _fetch_list(
        url,
        max_chars_per_url=LIST_MAX_CHARS if max_chars is None else max_chars,
        max_total_chars=(
            DEFAULT_MAX_TOTAL_CHARS
            if max_chars is None
            else max(DEFAULT_MAX_TOTAL_CHARS, max_chars * len(url))
        ),
        query=query,
        refresh=refresh,
    )


async def _fetch_one(
    url: str,
    *,
    max_chars: int,
    query: str | None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Read one page: the single-URL branch of ``deepseek_fetch``."""
    if max_chars < 1:
        return _fail("bad_request", "max_chars must be at least 1.", param="max_chars")
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
    return await fetch_core._fetch(
        url,
        True,
        max_chars,
        query=query,
        max_snippets=MAX_SNIPPETS,
        refresh=refresh,
    )


async def _fetch_list(
    urls: list[str],
    *,
    max_chars_per_url: int,
    max_total_chars: int,
    query: str | None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Read several pages in parallel: the list branch of ``deepseek_fetch``."""
    settings = runtime.get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail(
            "bad_request", f"a URL list must contain 1-{settings.batch_max_urls} entries.", param="url"
        )
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars and max_total_chars must be at least 1.", param="max_chars")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "a URL list must contain at least one non-empty URL.", param="url")
    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    tasks = await _run_batch(
        items,
        True,
        per_url,
        refresh=refresh,
        query=query,
        deadline=None,
    )
    return _assemble_batch(items, tasks, duplicates, max_total_chars, deadline=None)
