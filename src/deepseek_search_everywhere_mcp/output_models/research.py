"""Output model for ``deepseek_quick_research`` and ``deepseek_check_research``.

Every field is optional: one envelope shape has to describe four different call outcomes
(``done``, ``running``, ``needs_input`` and an error), and ``deepseek_check_research`` is not
wrapped in ``_timed``, so some paths omit ``elapsed_ms`` and ``job_id``. ``status`` and
``delivery`` stay plain strings on purpose: the published vocabulary is documented in the
description, and an unseen value must never turn into a call-time validation failure.

Both tools publish the same field set, so one :class:`ResearchOutput` models both; the
``QuickResearchOutput`` and ``CheckResearchOutput`` names stay as aliases. The model extends
:class:`~.Envelope` directly, so each published output schema is self-contained.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope

__all__ = ["CheckResearchOutput", "QuickResearchOutput", "ResearchOutput"]


class ResearchOutput(Envelope):
    """Shared envelope of the two research tools: a finished report, a running job or a paged job."""

    cost_usd: float | None = Field(
        default=None,
        description="Cost of the call, USD; 0 when a finished result was reused.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted web content.",
    )
    derived_from_untrusted: bool | None = Field(
        default=None,
        description="True when a model wrote the report from untrusted web content.",
    )
    injection_suspected: int | None = Field(
        default=None,
        description="Prompt-injection attempts detected and neutralized in sources.",
    )
    status: str | None = Field(
        default=None,
        description="Job state: done, running (poll) or needs_input (client synthesis).",
    )
    delivery: str | None = Field(
        default=None,
        description="Delivery: 'inline', 'polled' or 'client'.",
    )
    job_id: str | None = Field(
        default=None,
        description="Job id to pass to deepseek_check_research while not done.",
    )
    depth: str | None = Field(default=None, description="Depth tier run: low, medium or high.")
    elapsed_ms: int | None = Field(default=None, description="Wall time of the call in milliseconds.")
    progress: dict[str, Any] | None = Field(
        default=None,
        description="Running-job progress: phase, step, of, timings, deadline.",
    )
    next_action: str | None = Field(
        default=None, description="Suggested next step: poll, page or submit a report."
    )
    topic: str | None = Field(default=None, description="Research topic as submitted.")
    subqueries: list[str] | None = Field(
        default=None, description="Queries the job decomposed the topic into."
    )
    report: str | None = Field(
        default=None,
        description="Cited Markdown report; null before it exists or on failure.",
    )
    synthesized: bool | None = Field(
        default=None,
        description="True when a model wrote the report; false when only sources.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Ranked sources: id, title, url, date, access, type, authors."
    )
    unverified_numbers: list[dict[str, Any]] | None = Field(
        default=None, description="Report numbers not found in their cited source."
    )
    unverified_number_count: int | None = Field(
        default=None, description="Report numbers lacking support in their cited source."
    )
    partial: bool | None = Field(
        default=None, description="True when some pages failed to fetch; evidence incomplete."
    )
    requests_used: int | None = Field(default=None, description="Upstream search and fetch requests spent.")
    cached: bool | None = Field(
        default=None, description="True when the result came from cache at zero cost."
    )
    cached_count: int | None = Field(default=None, description="Upstream requests that were cache hits.")
    cost_breakdown: dict[str, Any] | None = Field(
        default=None, description="Cost split into search, fetch and LLM, USD."
    )
    notes: list[str] | None = Field(
        default=None, description="Notes on truncation, reuse, filtering or verification."
    )
    step: str | None = Field(
        default=None, description="Step awaited when status is needs_input; 'report' today."
    )
    context_page: int | None = Field(default=None, description="Page of sources returned, 1-based.")
    context_pages: int | None = Field(default=None, description="Total source pages for the job.")
    context: list[dict[str, Any]] | None = Field(
        default=None, description="One page of sources to synthesize from, with id and content."
    )
    synthesis_instructions: str | None = Field(
        default=None, description="Instructions for the client report; page 1 only."
    )


QuickResearchOutput = ResearchOutput
CheckResearchOutput = ResearchOutput
