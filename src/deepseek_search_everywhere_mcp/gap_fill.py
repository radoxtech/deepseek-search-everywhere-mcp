"""Gap filling: fetch full pages for search results whose content came back too thin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import runtime
from .batch import (
    _assemble_batch,
    _BatchItem,
    _run_batch,
)
from .dates import (
    _content_date,
)
from .deep_research import (
    best_source_title,
    classify_source_type,
    is_fragment_title,
    source_access,
    source_authors,
    source_identifiers,
)
from .envelope import (
    TRUNCATED_NOTE,
)
from .fetch_core import (
    _check_fetch_url,
)
from .text import (
    _url_host,
)
from .urls import (
    _canonical_url,
)

# Search results whose content is shorter than this are treated as gaps and fetched directly.
RESEARCH_GAP_CHARS = 200


def _gap_check(source: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Classify a source: ``(url, None)`` to fetch, ``(None, error)`` if unfetchable, else no gap."""
    content = source.get("content") or ""
    if len(content) >= RESEARCH_GAP_CHARS:
        return None, None
    url = source["url"].strip() if isinstance(source.get("url"), str) else ""
    rejection = _check_fetch_url(url)
    if rejection is not None:
        return None, rejection["error"]
    return url, None


@dataclass
class _GapFill:
    """Outcome of fetching the research gaps (counts only cover the fetch phase)."""

    requests_used: int = 0
    cached_count: int = 0
    fetched: int = 0
    failed: int = 0
    trimmed: bool = False


def _select_gaps(
    sources: list[dict[str, Any]], fill: _GapFill, max_pages: int | None = None
) -> dict[str, dict[str, Any]]:
    """Pick the sources to fetch, keyed by canonical URL; mark rejected ones in place.

    At most ``max_pages`` distinct URLs are selected; rejections count as failures.
    """
    limit = max_pages if max_pages is not None else runtime.get_settings().research_max_pages
    gaps: dict[str, dict[str, Any]] = {}
    for source in sources:
        url, rejection = _gap_check(source)
        if rejection is not None:
            source["error"] = rejection
            fill.failed += 1
        key = None if url is None else (_canonical_url(url) or url)
        if key is not None and key not in gaps and len(gaps) < limit:
            gaps[key] = source
    return gaps


async def _fill_gaps(
    sources: list[dict[str, Any]],
    per_page_chars: int,
    total_chars: int,
    refresh: bool,
    strip_boilerplate: bool = True,
    deadline: float | None = None,
    max_pages: int | None = None,
    share_total: bool = True,
) -> _GapFill:
    """Fetch sources with missing or short content and merge the fetched pages in place.

    Each page gets up to ``per_page_chars``; with ``share_total`` that cap is also limited to an equal
    share of ``total_chars`` (search results), otherwise only the batch total is enforced (research).
    """
    fill = _GapFill()
    gaps = _select_gaps(sources, fill, max_pages=max_pages)
    if not gaps:
        return fill
    items: list[_BatchItem] = [(source["url"].strip(), None) for source in gaps.values()]
    per_url = max(1, min(per_page_chars, total_chars // len(items))) if share_total else per_page_chars
    batch_deadline = deadline if deadline is not None else runtime.get_settings().research_gap_deadline
    tasks = await _run_batch(
        items,
        clean=True,
        per_url=per_url,
        refresh=refresh,
        strip_boilerplate=strip_boilerplate,
        deadline=batch_deadline,
        fallback_to_search=True,
    )
    batch = _assemble_batch(items, tasks, duplicates=0, max_total_chars=total_chars, deadline=batch_deadline)
    fill.requests_used = batch["requests_used"]
    fill.cached_count = batch["cached_count"]
    fill.trimmed = TRUNCATED_NOTE in batch["notes"]
    for source, item in zip(gaps.values(), batch["results"], strict=True):
        if not item["ok"]:
            source["error"] = item["error"]
            fill.failed += 1
        elif item["content"]:
            source["content"] = item["content"]
            page_url = source["url"] or ""
            if is_fragment_title(source["title"], page_url):
                # A usable fetched title (or the page's first heading) beats a host or fragment title;
                # any fetched title still beats a blank or host-fallback one.
                fetched = best_source_title(item["title"], page_url, item["content"])
                blank = not source["title"] or source["title"] == _url_host(source["url"])
                if not is_fragment_title(fetched, page_url) or (blank and item["title"]):
                    source["title"] = fetched
            source["origin"] = "fetch"
            source["cached"] = item["cached"]
            source.update(source_identifiers(source["url"] or "", source["content"]))
            source["access"] = source_access("fetch", source["content"])
            source["authors"] = source_authors(page_url, source["content"]) or source.get("authors")
            source["source_type"] = classify_source_type(
                source["url"] or "", source["title"], source["content"]
            )
            if source.get("date") is None and (found := _content_date(None, item["content"])):
                source["date"], source["date_source"] = found, "content"
            fill.fetched += 1
    return fill
