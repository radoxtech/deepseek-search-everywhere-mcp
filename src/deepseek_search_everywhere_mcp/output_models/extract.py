"""Output model for ``deepseek_extract``: the single-URL, list and error envelopes.

The tool returns three shapes through one annotation: a single-page result (``url``, ``title``, ``data``,
``evidence``, ``extraction_truncated``), a parallel list result (``count``, ``requests_used``,
``cached_count``, ``failed_count``, ``partial``, ``results``), and an error envelope (``ok: false`` plus
``error``). Every non-inherited field is optional so all three validate against this one model.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope


class ExtractOutput(Envelope):
    """Public envelope of ``deepseek_extract`` for a single URL, a URL list and an error."""

    cost_usd: float | None = Field(
        default=None,
        description="Estimated cost of the request in US dollars.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted content retrieved from the web.",
    )
    url: str | None = Field(
        default=None,
        description="URL the single-page extraction read, echoed from the request.",
    )
    title: str | None = Field(
        default=None,
        description="Title of the fetched page, when the upstream provided one.",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Extracted fields or schema-shaped object; values without a verified quote are null.",
    )
    evidence: dict[str, Any] | None = Field(
        default=None,
        description="Verbatim evidence quote that backs each extracted value, keyed by field name.",
    )
    extraction_truncated: bool | None = Field(
        default=None,
        description="True when the model output hit the token limit and was cut off.",
    )
    cached: bool | None = Field(
        default=None,
        description="True when the page came from the local cache (for a list: every page did).",
    )
    cached_count: int | None = Field(
        default=None,
        description="Number of list entries served from the local cache.",
    )
    count: int | None = Field(
        default=None,
        description="Number of entries in ``results`` for a URL-list extraction.",
    )
    requests_used: int | None = Field(
        default=None,
        description="Number of billable upstream requests the list extraction used.",
    )
    failed_count: int | None = Field(
        default=None,
        description="Number of list entries that failed extraction.",
    )
    partial: bool | None = Field(
        default=None,
        description="True when at least one entry of a list extraction failed.",
    )
    cost_breakdown: dict[str, Any] | None = Field(
        default=None,
        description="Cost split into search, fetch and LLM parts, in US dollars.",
    )
    results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-URL extraction envelopes for a list request, in input order.",
    )
    notes: list[str] | None = Field(
        default=None,
        description="Human-readable notes about nullified values, truncation, cache or deduplication.",
    )
    elapsed_ms: int | None = Field(
        default=None,
        description="Wall-clock time of the tool call in milliseconds.",
    )


__all__ = ["ExtractOutput"]
