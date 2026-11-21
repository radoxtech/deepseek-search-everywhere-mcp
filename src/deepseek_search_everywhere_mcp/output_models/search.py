"""Output model for ``deepseek_search``: one query, a list of queries, or an error envelope."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope


class SearchOutput(Envelope):
    """Published shape of a ``deepseek_search`` result.

    The single-query branch fills ``query_used``, ``deduped`` and the result list; the list branch
    fills the aggregate counters, ``skipped_queries`` and a per-query ``results`` list instead. The
    error branch carries only ``ok`` and ``error``; every other field is optional.
    """

    cost_usd: float | None = Field(default=None, description="Estimated cost of the call in US dollars.")
    untrusted_content: bool | None = Field(
        default=None, description="True when the response carries untrusted content from the web."
    )
    query_used: str | None = Field(
        default=None, description="The query actually sent upstream after filtering (single-query branch)."
    )
    query_original: str | None = Field(
        default=None, description="The caller's original query when it differed from the one used."
    )
    count: int | None = Field(
        default=None, description="Number of results in this response, or number of queries in a list run."
    )
    total_results: int | None = Field(
        default=None, description="Total number of results across every query in a list run."
    )
    content: str | None = Field(
        default=None, description="Result format that was requested: 'snippets' or 'full'."
    )
    deduped: int | None = Field(
        default=None, description="Number of duplicate results removed by the dedupe heuristic."
    )
    cached: bool | None = Field(
        default=None, description="True when the response came from the local cache and cost nothing."
    )
    cached_count: int | None = Field(
        default=None, description="Number of queries in a list run served from the cache."
    )
    requests_used: int | None = Field(
        default=None, description="Number of billable upstream requests this call made."
    )
    failed_count: int | None = Field(default=None, description="Number of queries in a list run that failed.")
    partial: bool | None = Field(
        default=None, description="True when at least one query in a list run failed."
    )
    skipped_queries: list[str] | None = Field(
        default=None, description="Queries beyond the five-per-call limit that were not run."
    )
    results: list[dict[str, Any]] | None = Field(
        default=None, description="Result entries for a single query, or per-query blocks for a list run."
    )
    notes: list[str] | None = Field(
        default=None, description="Human-readable notes about filtering, caching, dedupe or truncation."
    )
    elapsed_ms: int | None = Field(default=None, description="Wall-clock time of the call in milliseconds.")
