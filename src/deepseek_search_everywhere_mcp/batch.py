"""Batch machinery for multi-URL tools: URL dedupe, concurrent isolated fetches and envelope assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from . import fetch_core, runtime
from .envelope import (
    COST_PER_REQUEST,
    TRUNCATED_NOTE,
    _all_cached,
    _envelope_meta,
)
from .errors import (
    _fail,  # re-exported for callers and tests
)
from .fetch_core import (
    _check_fetch_url,
)
from .redact import redact
from .runtime import (
    logger,
)
from .urls import (
    _canonical_url,
)

# Error codes that mean no upstream request was made for a batch item.
_LOCAL_ERROR_REASONS = frozenset({"rate_limited_local", "deadline_exceeded", "internal_error"})


def _batch_item(url: str, envelope: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``deepseek_fetch`` envelope to one result of a ``deepseek_fetch`` URL list."""
    if envelope.get("ok") is False:
        return {"url": url, "ok": False, "error": envelope["error"], "chars": 0, "cached": False}
    return {
        "url": url,
        "ok": True,
        "title": envelope["title"],
        "content": envelope["content"],
        "chars": len(envelope["content"]),
        "cached": envelope["cached"],
        "untrusted_content": True,
    }


async def _fetch_isolated(
    url: str,
    clean: bool,
    max_chars: int,
    refresh: bool,
    semaphore: asyncio.Semaphore,
    strip_boilerplate: bool = True,
    query: str | None = None,
    fallback_to_search: bool = True,
) -> dict[str, Any]:
    """Run one batch fetch under ``semaphore``; unexpected exceptions become error envelopes."""
    async with semaphore:
        try:
            return await fetch_core._fetch(
                url,
                clean,
                max_chars,
                query=query,
                refresh=refresh,
                strip_boilerplate=strip_boilerplate,
                fallback_to_search=fallback_to_search,
            )
        except Exception as exc:  # broad on purpose: one failing URL must not fail the whole batch
            message = redact(f"{type(exc).__name__}: {exc}")
            logger.debug("Batch fetch of a URL failed: %s", message)
            return _fail("internal_error", message)


# A batch entry: the stripped URL and its rejection envelope (``None`` when it is fetchable).
_BatchItem = tuple[str, dict[str, Any] | None]


def _dedupe_urls(urls: list[str]) -> tuple[list[_BatchItem], int]:
    """Strip and validate ``urls``, skip empty entries and collapse canonical duplicates.

    Returns ``(items, duplicates)``. Rejected URLs are kept (keyed verbatim) so they can be
    reported in place; fetchable ones are keyed by their canonical form.
    """
    items: list[_BatchItem] = []
    seen: set[str] = set()
    duplicates = 0
    for raw in urls:
        url = raw.strip() if isinstance(raw, str) else ""
        if not url:
            continue
        rejection = _check_fetch_url(url)
        key = url if rejection is not None else (_canonical_url(url) or url)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        items.append((url, rejection))
    return items, duplicates


async def _wait_or_cancel(tasks: Iterable[asyncio.Task[Any]], timeout: float) -> None:
    """Wait up to ``timeout`` seconds for ``tasks``, then cancel and reap those still running.

    A cancelled task is how the batch tools report an item that missed the deadline.
    """
    tasks = list(tasks)
    try:
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
    finally:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _batch_deadline_error(deadline: float | None) -> dict[str, Any]:
    """Error envelope for a batch item that was not fetched before the batch deadline."""
    seconds = deadline if deadline is not None else runtime.get_settings().batch_deadline
    return _fail("deadline_exceeded", f"Not fetched within the {seconds:g} s batch deadline.")


def _batch_envelope(
    index: int,
    rejection: dict[str, Any] | None,
    tasks: dict[int, asyncio.Task[dict[str, Any]]],
    deadline_error: dict[str, Any],
) -> dict[str, Any]:
    """Return a batch item's fetch envelope: its rejection, the deadline error or the task result."""
    if rejection is not None:
        return rejection
    task = tasks[index]  # every accepted item has a task
    return deadline_error if task.cancelled() else task.result()


async def _run_batch(
    items: list[_BatchItem],
    clean: bool,
    per_url: int,
    refresh: bool,
    strip_boilerplate: bool = True,
    query: str | None = None,
    deadline: float | None = None,
    fallback_to_search: bool = True,
) -> dict[int, asyncio.Task[dict[str, Any]]]:
    """Fetch every non-rejected item concurrently under the batch concurrency and deadline.

    Returns the tasks by item index; those still pending at the deadline are cancelled.
    """
    settings = runtime.get_settings()
    timeout = deadline if deadline is not None else settings.batch_deadline
    semaphore = asyncio.Semaphore(settings.batch_concurrency)
    tasks = {
        index: asyncio.create_task(
            _fetch_isolated(
                url,
                clean,
                per_url,
                refresh,
                semaphore,
                strip_boilerplate,
                query=query,
                fallback_to_search=fallback_to_search,
            )
        )
        for index, (url, rejection) in enumerate(items)
        if rejection is None
    }
    await _wait_or_cancel(tasks.values(), timeout)
    return tasks


def _batch_results(
    items: list[_BatchItem],
    tasks: dict[int, asyncio.Task[dict[str, Any]]],
    max_total_chars: int,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], bool, int]:
    """Collect batch results in input order, trimming content to ``max_total_chars`` overall.

    Returns ``(results, trimmed, requests_used)``.
    """
    deadline_error = _batch_deadline_error(deadline)
    results: list[dict[str, Any]] = []
    remaining = max_total_chars
    trimmed = False
    requests_used = 0
    for index, (url, rejection) in enumerate(items):
        envelope = _batch_envelope(index, rejection, tasks, deadline_error)
        item = _batch_item(url, envelope)
        if item["ok"]:
            trimmed = trimmed or envelope["truncated"]
            if item["chars"] > remaining:
                item["content"] = item["content"][:remaining]
                item["chars"] = remaining
                trimmed = True
            remaining -= item["chars"]
            requests_used += not item["cached"]
        elif rejection is None and item["error"].get("reason") not in _LOCAL_ERROR_REASONS:
            requests_used += 1
        results.append(item)
    return results, trimmed, requests_used


def _assemble_batch(
    items: list[_BatchItem],
    tasks: dict[int, asyncio.Task[dict[str, Any]]],
    duplicates: int,
    max_total_chars: int,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Build the ``deepseek_fetch`` URL-list envelope from the finished batch."""
    results, trimmed, requests_used = _batch_results(items, tasks, max_total_chars, deadline=deadline)
    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")
    if trimmed:
        notes.append(TRUNCATED_NOTE)
    failed_count = sum(not item["ok"] for item in results)
    cached_count = sum(item["cached"] for item in results)
    cost = round(requests_used * COST_PER_REQUEST, 5)
    return {
        "ok": True,
        "count": len(results),
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
