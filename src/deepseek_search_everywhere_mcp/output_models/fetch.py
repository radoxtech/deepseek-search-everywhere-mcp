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
        description="Estimated cost of the call in US dollars; cached content is free.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted content retrieved from the web.",
    )
    url: str | None = Field(
        default=None,
        description="The fetched URL, in a single-URL response.",
    )
    source: str | None = Field(
        default=None,
        description='Where the page text came from: "fetch" for a direct fetch or "search_index" '
        "for the search-index fallback.",
    )
    title: str | None = Field(
        default=None,
        description="Page title, cleaned of hidden Unicode and repeated text.",
    )
    content: str | None = Field(
        default=None,
        description="Page text as clean Markdown, trimmed to the character budget.",
    )
    chars: int | None = Field(
        default=None,
        description="Number of characters in ``content``.",
    )
    total_chars: int | None = Field(
        default=None,
        description="Number of characters in the page before truncation.",
    )
    truncated: bool | None = Field(
        default=None,
        description="True when the page text was cut to respect the character budget.",
    )
    cached: bool | None = Field(
        default=None,
        description="True when the page came from the local cache (single) or every result did (list).",
    )
    cached_count: int | None = Field(
        default=None,
        description="Number of list results served from the local cache.",
    )
    count: int | None = Field(
        default=None,
        description="Number of results returned for a URL list.",
    )
    requests_used: int | None = Field(
        default=None,
        description="Upstream requests billed for a URL list; cached results are excluded.",
    )
    failed_count: int | None = Field(
        default=None,
        description="Number of list results that failed.",
    )
    partial: bool | None = Field(
        default=None,
        description="True when at least one URL in the list failed.",
    )
    results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-URL results in input order; each entry keeps its own ok, error, title, "
        "content and cache fields.",
    )
    notes: list[str] | None = Field(
        default=None,
        description="Human-readable notes such as dedupe and truncation warnings.",
    )
    content_capped: bool | None = Field(
        default=None,
        description="True when the page was capped before excerpt selection.",
    )
    query: str | None = Field(
        default=None,
        description="The query used to select excerpts, when one was given.",
    )
    snippets_count: int | None = Field(
        default=None,
        description="Number of excerpt snippets returned for ``query``.",
    )
    elapsed_ms: int | None = Field(
        default=None,
        description="Wall-clock duration of the call in milliseconds.",
    )


__all__ = ["FetchOutput"]
