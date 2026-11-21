"""Output models for ``deepseek_quick_research`` and ``deepseek_check_research``.

Every field is optional: one envelope shape has to describe four different call outcomes
(``done``, ``running``, ``needs_input`` and an error), and ``deepseek_check_research`` is not
wrapped in ``_timed``, so some paths omit ``elapsed_ms`` and ``job_id``. ``status`` and
``delivery`` stay plain strings on purpose: the published vocabulary is documented in the
description, and an unseen value must never turn into a call-time validation failure.

Both models extend :class:`~.Envelope` directly and repeat the shared optional fields, so each
published output schema is self-contained.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope

__all__ = ["CheckResearchOutput", "QuickResearchOutput"]


class QuickResearchOutput(Envelope):
    """Envelope of ``deepseek_quick_research``: a finished report, a running job or a paged job."""

    cost_usd: float | None = Field(
        default=None,
        description="Total cost of the call in US dollars; 0 when a finished result was reused.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted content retrieved from the web.",
    )
    derived_from_untrusted: bool | None = Field(
        default=None,
        description="True when the report was written by a model from untrusted web content.",
    )
    injection_suspected: int | None = Field(
        default=None,
        description="Number of prompt-injection attempts detected and neutralized in the sources.",
    )
    status: str | None = Field(
        default=None,
        description=(
            "Job state: 'done' (report ready), 'running' (poll with deepseek_check_research) or "
            "'needs_input' (client synthesis; read the sources and submit the report)."
        ),
    )
    delivery: str | None = Field(
        default=None,
        description="How the result was delivered: 'inline', 'polled' or 'client'.",
    )
    job_id: str | None = Field(
        default=None,
        description="Job identifier to pass to deepseek_check_research while the job is not done.",
    )
    depth: str | None = Field(default=None, description="Depth tier the job ran at: low, medium or high.")
    elapsed_ms: int | None = Field(
        default=None, description="Wall time of the call in milliseconds, measured on a monotonic clock."
    )
    progress: dict[str, Any] | None = Field(
        default=None,
        description="Progress of a running job: phase, step, of, time in phase and the depth deadline.",
    )
    next_action: str | None = Field(
        default=None, description="Suggested next step, such as polling the job or submitting a report."
    )
    topic: str | None = Field(default=None, description="The research topic as submitted.")
    subqueries: list[str] | None = Field(
        default=None, description="Queries the job decomposed the topic into before searching."
    )
    report: str | None = Field(
        default=None,
        description="Cited report in Markdown; null before the report exists or when synthesis failed.",
    )
    synthesized: bool | None = Field(
        default=None,
        description="True when the report was written by a model; false when only sources returned.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Ranked sources with id, title, url, date, access, type and authors."
    )
    unverified_numbers: list[dict[str, Any]] | None = Field(
        default=None, description="Numbers in the report that were not found in their cited source."
    )
    unverified_number_count: int | None = Field(
        default=None, description="How many report numbers lacked support in their cited source."
    )
    partial: bool | None = Field(
        default=None, description="True when some pages failed to fetch, so the evidence is incomplete."
    )
    requests_used: int | None = Field(
        default=None, description="Upstream search and fetch requests spent on this job."
    )
    cached: bool | None = Field(
        default=None, description="True when the result came from the cache or the job store at zero cost."
    )
    cached_count: int | None = Field(default=None, description="How many upstream requests were cache hits.")
    cost_breakdown: dict[str, Any] | None = Field(
        default=None, description="Cost split into search, fetch and llm parts, in US dollars."
    )
    notes: list[str] | None = Field(
        default=None, description="Diagnostic notes about truncation, reuse, filtering or verification."
    )
    step: str | None = Field(
        default=None,
        description="Step the job waits for when status is 'needs_input'; 'report' today.",
    )
    context_page: int | None = Field(
        default=None, description="Page number of the sources returned, 1-based."
    )
    context_pages: int | None = Field(default=None, description="Total number of source pages for the job.")
    context: list[dict[str, Any]] | None = Field(
        default=None, description="One page of sources to synthesize from, with id, title, url and content."
    )
    synthesis_instructions: str | None = Field(
        default=None, description="Instructions for writing the client report; sent with page 1 only."
    )


class CheckResearchOutput(Envelope):
    """Envelope of ``deepseek_check_research``: the shared job fields plus client-synthesis paging."""

    cost_usd: float | None = Field(
        default=None,
        description="Total cost of the call in US dollars; 0 when a finished result was reused.",
    )
    untrusted_content: bool | None = Field(
        default=None,
        description="True when the response carries untrusted content retrieved from the web.",
    )
    derived_from_untrusted: bool | None = Field(
        default=None,
        description="True when the report was written by a model from untrusted web content.",
    )
    injection_suspected: int | None = Field(
        default=None,
        description="Number of prompt-injection attempts detected and neutralized in the sources.",
    )
    status: str | None = Field(
        default=None,
        description=(
            "Job state: 'done' (report ready), 'running' (poll again with the same job_id) or "
            "'needs_input' (client synthesis; read the sources and submit the report)."
        ),
    )
    delivery: str | None = Field(
        default=None,
        description="How the result was delivered: 'inline', 'polled' or 'client'.",
    )
    job_id: str | None = Field(
        default=None,
        description="Job identifier being polled; absent on the verified client-report envelope.",
    )
    depth: str | None = Field(default=None, description="Depth tier the job ran at: low, medium or high.")
    elapsed_ms: int | None = Field(
        default=None,
        description="Job wall time in milliseconds when known; this tool is not time-wrapped.",
    )
    progress: dict[str, Any] | None = Field(
        default=None,
        description="Progress of a running job: phase, step, of, time in phase and the depth deadline.",
    )
    next_action: str | None = Field(
        default=None, description="Suggested next step, such as paging, submitting a report or polling."
    )
    topic: str | None = Field(default=None, description="The research topic as submitted.")
    subqueries: list[str] | None = Field(
        default=None, description="Queries the job decomposed the topic into before searching."
    )
    report: str | None = Field(
        default=None,
        description="Cited report in Markdown; null before the report exists or when synthesis failed.",
    )
    synthesized: bool | None = Field(
        default=None,
        description="True when the report was written by a model; false when only sources returned.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Ranked sources with id, title, url, date, access, type and authors."
    )
    unverified_numbers: list[dict[str, Any]] | None = Field(
        default=None, description="Numbers in the report that were not found in their cited source."
    )
    unverified_number_count: int | None = Field(
        default=None, description="How many report numbers lacked support in their cited source."
    )
    partial: bool | None = Field(
        default=None, description="True when some pages failed to fetch, so the evidence is incomplete."
    )
    requests_used: int | None = Field(
        default=None, description="Upstream search and fetch requests spent on this job."
    )
    cached: bool | None = Field(
        default=None, description="True when the result came from the cache or the job store at zero cost."
    )
    cached_count: int | None = Field(default=None, description="How many upstream requests were cache hits.")
    cost_breakdown: dict[str, Any] | None = Field(
        default=None, description="Cost split into search, fetch and llm parts, in US dollars."
    )
    notes: list[str] | None = Field(
        default=None, description="Diagnostic notes about truncation, reuse, filtering or verification."
    )
    step: str | None = Field(
        default=None, description="Step the job waits for when status is 'needs_input'; 'report' today."
    )
    context_page: int | None = Field(
        default=None, description="Page number of the sources returned, 1-based."
    )
    context_pages: int | None = Field(default=None, description="Total number of source pages for the job.")
    context: list[dict[str, Any]] | None = Field(
        default=None, description="One page of sources to synthesize from, with id, title, url and content."
    )
    synthesis_instructions: str | None = Field(
        default=None, description="Instructions for writing the client report; sent with page 1 only."
    )
