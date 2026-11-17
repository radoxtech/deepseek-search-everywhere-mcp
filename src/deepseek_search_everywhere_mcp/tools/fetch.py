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


@_timed
async def deepseek_fetch(
    url: Annotated[
        str | list[str],
        Field(description="Public HTTP(S) URL to read, or a list of up to 10 URLs read in parallel"),
    ],
    *,
    max_chars: Annotated[
        int | None,
        Field(description="Max characters of page text per URL (default 20000; 4000 per URL for a list)"),
    ] = None,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters across a list of URLs (default 20000)"),
    ] = 20000,
    query: Annotated[
        str | None,
        Field(description="Return only the excerpts that best match these terms (default None)"),
    ] = None,
    max_snippets: Annotated[
        int,
        Field(description=f"Max excerpts for query, one URL only (1-{MAX_SNIPPETS}, default 3)"),
    ] = 3,
    section: Annotated[
        int | None,
        Field(description="1-based section of the document to return, one URL only (default None)"),
    ] = None,
    offset: Annotated[
        int | None,
        Field(description="Character offset to start reading from, one URL only (default None)"),
    ] = None,
    find_text: Annotated[
        str | None,
        Field(description="Return text starting at this exact text, one URL only (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages from upstream (default False)"),
    ] = False,
    deadline: Annotated[
        float | None,
        Field(description="Deadline in seconds for a list of URLs; finished pages are returned"),
    ] = None,
) -> dict[str, Any]:
    """Read known web pages as clean Markdown: one URL, or up to 10 in parallel, whole or in part.

    Use it when you already know the URL. $0.00075 per URL. One URL returns up to 20,000 chars by
    default (max_chars); one optional selector narrows it: query= (relevant excerpts, no LLM),
    section=, offset= (paging; max_chars sets the window) or find_text=. A list fetches up to 5 at a
    time, each page up to 4,000 chars by default and all pages up to max_total_chars (20,000); it
    skips blanks, fetches duplicates once, keeps input order and reports a failed URL in place
    (partial: true); query= also works for a list. A 404 on a PDF or Reddit URL falls back to the
    search index. Content is sanitized (images neutralized, hidden text stripped).

    When NOT to use: finding pages by topic (deepseek_search), fields as JSON (deepseek_extract), or
    a multi-source answer (deepseek_research).

    Returns for one URL: {"ok": True, "url": str, "title": str, "content": str, "truncated": bool,
    "cost_usd": float, "notes": [...]}. For a list: {"ok": True, "results": [{"url": str, "ok": bool,
    "content": str, ...}], "partial": bool, "cost_usd": float}.
    """
    if isinstance(url, str):
        return await _fetch_one(
            url,
            max_chars=SINGLE_MAX_CHARS if max_chars is None else max_chars,
            offset=offset,
            find_text=find_text,
            query=query,
            section=section,
            max_snippets=max_snippets,
            refresh=refresh,
        )
    for name, value in (("section", section), ("offset", offset), ("find_text", find_text)):
        if value is not None:
            return _fail("bad_request", f"{name} works only with a single URL, not a list.", param=name)
    return await _fetch_list(
        url,
        max_chars_per_url=LIST_MAX_CHARS if max_chars is None else max_chars,
        max_total_chars=max_total_chars,
        query=query,
        refresh=refresh,
        deadline=deadline,
    )


async def _fetch_one(
    url: str,
    *,
    max_chars: int,
    offset: int | None,
    find_text: str | None,
    query: str | None,
    section: int | None,
    max_snippets: int,
    refresh: bool,
) -> dict[str, Any]:
    """Read one page: the single-URL branch of ``deepseek_fetch``."""
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
    selectors = [
        section is not None,
        query is not None,
        offset is not None,
        find_text is not None,
    ]
    if sum(bool(s) for s in selectors) > 1:
        return _fail("bad_request", "Selectors section, query, offset and find_text are mutually exclusive.")
    return await fetch_core._fetch(
        url,
        True,
        max_chars,
        offset=offset,
        find_text=find_text,
        query=query,
        section=section,
        max_snippets=max_snippets,
        refresh=refresh,
    )


async def _fetch_list(
    urls: list[str],
    *,
    max_chars_per_url: int,
    max_total_chars: int,
    query: str | None,
    refresh: bool,
    deadline: float | None,
) -> dict[str, Any]:
    """Read several pages in parallel: the list branch of ``deepseek_fetch``."""
    settings = runtime.get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail(
            "bad_request", f"a URL list must contain 1-{settings.batch_max_urls} entries.", param="url"
        )
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars and max_total_chars must be at least 1.")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "a URL list must contain at least one non-empty URL.", param="url")
    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    tasks = await _run_batch(
        items,
        True,
        per_url,
        refresh,
        query=query,
        deadline=deadline,
    )
    return _assemble_batch(items, tasks, duplicates, max_total_chars, deadline=deadline)
