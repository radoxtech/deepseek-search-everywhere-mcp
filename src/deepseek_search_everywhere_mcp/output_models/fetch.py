"""The published output model of ``deepseek_fetch``: one URL, a URL list or an error envelope."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope


class FetchOutput(Envelope):
    """Shape of every ``deepseek_fetch`` response.

    A single-URL call publishes the page fields; a list call publishes the batch fields
    (``count``, ``results`` and the request accounting) instead. An error publishes only
    the inherited ``ok`` and ``error`` plus ``elapsed_ms``.
    """

    cost_usd: float | None = Field(
        default=None,
        description="Cost of the call, USD; cached content is free.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted web content.",
    )
    url: str | None = Field(
        default=None,
        description="Fetched URL, in a single-URL response.",
    )
    source: str | None = Field(
        default=None,
        description='Page text origin: "fetch" or "search_index" (index fallback).',
    )
    title: str | None = Field(
        default=None,
        description="Page title, cleaned of hidden Unicode.",
    )
    content: str | None = Field(
        default=None,
        description="Page text as Markdown, trimmed to the character budget.",
    )
    chars: int | None = Field(
        default=None,
        description="Characters in ``content``.",
    )
    total_chars: int | None = Field(
        default=None,
        description="Characters in the page before truncation.",
    )
    truncated: bool | None = Field(
        default=None,
        description="True when the page text was cut to the char budget.",
    )
    cached: bool | None = Field(
        default=None,
        description="True when from cache (single) or every list result was.",
    )
    cached_count: int | None = Field(
        default=None,
        description="List results served from cache.",
    )
    count: int | None = Field(
        default=None,
        description="Results returned for a URL list.",
    )
    requests_used: int | None = Field(
        default=None,
        description="Upstream requests billed for a list; cache excluded.",
    )
    failed_count: int | None = Field(
        default=None,
        description="List results that failed.",
    )
    partial: bool | None = Field(
        default=None,
        description="True when at least one URL in the list failed.",
    )
    results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-URL results in input order, each with its own fields.",
    )
    notes: list[str] | None = Field(
        default=None,
        description="Notes such as dedupe and truncation warnings.",
    )
    content_capped: bool | None = Field(
        default=None,
        description="True when the page was capped before excerpt selection.",
    )
    query: str | None = Field(
        default=None,
        description="Query used to select excerpts, when given.",
    )
    snippets_count: int | None = Field(
        default=None,
        description="Excerpt snippets returned for ``query``.",
    )
    elapsed_ms: int | None = Field(
        default=None,
        description="Wall time of the call in milliseconds.",
    )


__all__ = ["FetchOutput"]
