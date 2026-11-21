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
        description="Cost of the call, USD.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted web content.",
    )
    url: str | None = Field(
        default=None,
        description="URL read by the single-page extraction.",
    )
    title: str | None = Field(
        default=None,
        description="Fetched page title, when upstream provided one.",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Extracted fields; values lacking a verified quote are null.",
    )
    evidence: dict[str, Any] | None = Field(
        default=None,
        description="Verbatim quote backing each value, keyed by field.",
    )
    extraction_truncated: bool | None = Field(
        default=None,
        description="True when the model output was cut at the token limit.",
    )
    cached: bool | None = Field(
        default=None,
        description="True when from cache (list: every page was).",
    )
    cached_count: int | None = Field(
        default=None,
        description="List entries served from cache.",
    )
    count: int | None = Field(
        default=None,
        description="Entries in ``results`` for a URL-list extraction.",
    )
    requests_used: int | None = Field(
        default=None,
        description="Billable upstream requests the list extraction used.",
    )
    failed_count: int | None = Field(
        default=None,
        description="List entries that failed extraction.",
    )
    partial: bool | None = Field(
        default=None,
        description="True when at least one list entry failed.",
    )
    cost_breakdown: dict[str, Any] | None = Field(
        default=None,
        description="Cost split into search, fetch and LLM, USD.",
    )
    results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-URL extraction results in input order.",
    )
    notes: list[str] | None = Field(
        default=None,
        description="Notes on nullified values, truncation, cache or dedupe.",
    )
    elapsed_ms: int | None = Field(
        default=None,
        description="Wall time of the call in milliseconds.",
    )


__all__ = ["ExtractOutput"]
