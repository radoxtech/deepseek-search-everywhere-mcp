"""Output model for ``deepseek_deep_agent_research``: the autonomous multi-round research envelope."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope


class AgentResearchOutput(Envelope):
    """Published shape of a ``deepseek_deep_agent_research`` result.

    The model covers every path the tool can take: a finished run with a synthesized report, a
    finished run that returned the gathered sources instead of a report (``synthesized`` false and
    ``partial`` true), a job still ``running`` after the inline wait, and the budget-guard error.
    Each path publishes a subset of the fields, so every field beyond ``ok``, ``_meta`` and
    ``error`` is optional. ``model_config`` is inherited from :class:`Envelope` (``extra="ignore"``
    and ``populate_by_name=True``).
    """

    cost_usd: float | None = Field(default=None, description="Cost of the call, USD.")
    untrusted_content: bool | None = Field(
        default=None, description="True when the response carries untrusted web content."
    )
    derived_from_untrusted: bool | None = Field(
        default=None, description="True when the report was written from untrusted web content."
    )
    injection_suspected: int | None = Field(
        default=None, description="Prompt-injection attempts detected in the report."
    )
    status: str | None = Field(
        default=None,
        description="Job state: 'done' or 'running' (poll it); absent on errors.",
    )
    delivery: str | None = Field(
        default=None,
        description="Delivery of a finished result: 'inline' or 'polled'.",
    )
    job_id: str | None = Field(default=None, description="Research job identifier (running results only).")
    depth: str | None = Field(default=None, description="Depth tier run: 'medium', 'high' or 'max'.")
    elapsed_ms: int | None = Field(default=None, description="Wall time of the job in milliseconds.")
    progress: dict[str, Any] | None = Field(
        default=None,
        description="Running-job progress: phase, step, of, detail, timings, deadline.",
    )
    next_action: str | None = Field(
        default=None, description="Next step for a running job, e.g. poll its job_id."
    )
    topic: str | None = Field(default=None, description="Cleaned research topic that was run.")
    subqueries: list[str] | None = Field(
        default=None, description="Search queries the agent planned and ran."
    )
    report: str | None = Field(
        default=None,
        description="Cited Markdown report; null when none was written.",
    )
    synthesized: bool | None = Field(
        default=None,
        description="True when a model wrote the report; false when only sources.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Source entries backing the report or returned instead."
    )
    unverified_numbers: list[dict[str, Any]] | None = Field(
        default=None, description="Report numbers not found in their cited source."
    )
    unverified_number_count: int | None = Field(
        default=None, description="Report numbers that could not be verified."
    )
    partial: bool | None = Field(
        default=None,
        description="True when some pages failed or no report was synthesized.",
    )
    plan: list[dict[str, Any]] | None = Field(
        default=None,
        description="Aspect plan: one entry per aspect with name and query.",
    )
    rounds: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-round gains: new facts, gaps filled, new sources, saturated.",
    )
    facts_per_round: list[int] | None = Field(default=None, description="New facts per round.")
    estimate_usd: float | None = Field(default=None, description="Upfront cost estimate for the run, USD.")
    stopped_reason: str | None = Field(
        default=None,
        description="Stop: max_rounds, saturated, budget, no_gaps, retrieval_failed, empty_pages",
    )
    requests_used: int | None = Field(default=None, description="Billable upstream requests this run made.")
    cached: bool | None = Field(
        default=None, description="True when the whole result came from cache at zero cost."
    )
    cached_count: int | None = Field(default=None, description="Upstream requests served from cache.")
    cost_breakdown: dict[str, Any] | None = Field(
        default=None, description="Cost split into search, fetch and LLM, USD."
    )
    notes: list[str] | None = Field(default=None, description="Notes about filtering, caching or truncation.")


__all__ = ["AgentResearchOutput"]
