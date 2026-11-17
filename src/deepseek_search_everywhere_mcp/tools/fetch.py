"""The deepseek_fetch and deepseek_fetch_many tools."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from .. import fetch_core, runtime
from ..batch import (
    _assemble_batch,
    _dedupe_urls,
    _run_batch,
)
from ..errors import (
    _fail,  # re-exported for callers and tests
)
from ..fetch_core import (
    _check_fetch_url,
)
from ..runtime import (
    _timed,
)
from ..snippets import MAX_SNIPPETS


@_timed
async def deepseek_fetch(
    url: Annotated[
        str,
        Field(description="HTTP or HTTPS URL to fetch (must be public, no private IP/intranet)"),
    ],
    *,
    clean: Annotated[
        bool,
        Field(description="Remove Markdown backslash escapes from the page text (default True)"),
    ] = True,
    max_chars: Annotated[
        int,
        Field(description="Max characters of page text to return (default 20000)"),
    ] = 20000,
    offset: Annotated[
        int | None,
        Field(description="Character offset to start reading from (mutually exclusive with selectors)"),
    ] = None,
    limit: Annotated[
        int | None,
        Field(description="Max characters to return from offset (mutually exclusive with selectors)"),
    ] = None,
    find_text: Annotated[
        str | None,
        Field(description="Locate text snippet and return surrounding context (mutually exclusive)"),
    ] = None,
    query: Annotated[
        str | None,
        Field(description="Search page content for query terms and extract best matching snippets"),
    ] = None,
    section: Annotated[
        int | None,
        Field(description="1-based section number to extract from document (mutually exclusive)"),
    ] = None,
    max_snippets: Annotated[
        int,
        Field(description=f"Max snippets to return when using query selector (1-{MAX_SNIPPETS}, default 3)"),
    ] = 3,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh page from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners, navigation and site UI boilerplate (default True)"),
    ] = True,
    fallback_to_search: Annotated[
        bool,
        Field(description="On 404, fall back to searching exact URL in index (default True)"),
    ] = True,
) -> dict[str, Any]:
    """Read one web page as clean Markdown, or only the part of it you need.

    Use it when you already know the URL. Flat $0.00075 per call; up to 20,000 chars by default
    (max_chars). One optional selector narrows the text: section=, query= (relevant excerpts, no
    LLM), offset=/limit= (paging) or find_text=. A 404 on a PDF or Reddit URL falls back to the
    search index. Content is sanitized (images neutralized, hidden text stripped).

    When NOT to use: finding pages by topic (deepseek_search), several URLs (deepseek_fetch_many),
    fields as JSON (deepseek_extract), or a multi-source answer (deepseek_research).

    Returns: {"ok": True, "url": str, "title": str, "content": str, "truncated": bool,
    "cost_usd": float, "notes": [...]}.
    """
    url = (url or "").strip()
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return rejection
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
    return await fetch_core._fetch(
        url,
        clean,
        max_chars,
        offset=offset,
        limit=limit,
        find_text=find_text,
        query=query,
        section=section,
        max_snippets=max_snippets,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
        fallback_to_search=fallback_to_search,
    )


@_timed
async def deepseek_fetch_many(
    urls: Annotated[
        list[str],
        Field(description="List of HTTP/HTTPS URLs to fetch in parallel (max 10 URLs)"),
    ],
    *,
    max_chars_per_url: Annotated[
        int,
        Field(description="Max characters of page text returned per URL (default 4000)"),
    ] = 4000,
    max_total_chars: Annotated[
        int,
        Field(description="Max total characters across all fetched URLs (default 20000)"),
    ] = 20000,
    clean: Annotated[
        bool,
        Field(description="Remove Markdown backslash escapes from the page text (default True)"),
    ] = True,
    query: Annotated[
        str | None,
        Field(description="Optional query to extract relevant snippets from each page (default None)"),
    ] = None,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages from upstream (default False)"),
    ] = False,
    strip_boilerplate: Annotated[
        bool,
        Field(description="Remove cookie banners and site UI boilerplate from content (default True)"),
    ] = True,
    deadline: Annotated[
        float | None,
        Field(description="Overall timeout deadline in seconds; partial results returned on expiry"),
    ] = None,
    fallback_to_search: Annotated[
        bool,
        Field(description="On 404, fall back to searching exact URL in index (default True)"),
    ] = True,
) -> dict[str, Any]:
    """Read up to 10 known web pages in parallel as clean Markdown in one call.

    Use it when the URLs are known and you need their text or matching excerpts (query=); fetches
    up to 5 at a time. $0.00075 per URL; each page gets up to max_chars_per_url (default 4,000) and all pages
    together up to max_total_chars (default 20,000). Skips blanks, fetches duplicates once, keeps
    input order and reports a failed URL in place (partial: true). Content is sanitized.

    When NOT to use: discovering pages (deepseek_search_many), one page with section or offset
    selectors (deepseek_fetch), fields as JSON (deepseek_extract_many), or a synthesized answer
    (deepseek_research).

    Returns: {"ok": True, "results": [{"url": str, "ok": bool, "content": str, ...}],
    "partial": bool, "cost_usd": float}.
    """
    settings = runtime.get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail("bad_request", f"urls must contain 1-{settings.batch_max_urls} entries.")
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars_per_url and max_total_chars must be at least 1.")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "urls must contain at least one non-empty URL.")
    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    tasks = await _run_batch(
        items,
        clean,
        per_url,
        refresh,
        strip_boilerplate,
        query=query,
        deadline=deadline,
        fallback_to_search=fallback_to_search,
    )
    return _assemble_batch(items, tasks, duplicates, max_total_chars, deadline=deadline)
