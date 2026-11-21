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

    cost_usd: float | None = Field(default=None, description="Estimated cost of the call in US dollars.")
    untrusted_content: bool | None = Field(
        default=None, description="True when the response carries untrusted content from the web."
    )
    derived_from_untrusted: bool | None = Field(
        default=None, description="True when the report was written from untrusted web content."
    )
    injection_suspected: int | None = Field(
        default=None, description="Number of prompt-injection attempts detected in the report."
    )
    status: str | None = Field(
        default=None,
        description=(
            "Job state: 'done' for a finished run, 'running' while the job needs polling. Absent on errors."
        ),
    )
    delivery: str | None = Field(
        default=None,
        description="How a finished result was delivered: 'inline' in this call or 'polled' later.",
    )
    job_id: str | None = Field(
        default=None, description="Stable identifier of the research job (running results only)."
    )
    depth: str | None = Field(
        default=None, description="Depth tier that was actually run: 'medium', 'high' or 'max'."
    )
    elapsed_ms: int | None = Field(default=None, description="Wall-clock time of the job in milliseconds.")
    progress: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Progress of a running job: phase, step, of, optional detail, phase_elapsed_ms and deadline_ms."
        ),
    )
    next_action: str | None = Field(
        default=None, description="What to do next with a running job, e.g. poll its job_id."
    )
    topic: str | None = Field(default=None, description="The cleaned research topic that was run.")
    subqueries: list[str] | None = Field(
        default=None, description="Search queries the agent planned and ran."
    )
    report: str | None = Field(
        default=None,
        description="Finished cited Markdown report; None when no report was written.",
    )
    synthesized: bool | None = Field(
        default=None,
        description="True when the model wrote a report; false when only the gathered sources were returned.",
    )
    sources: list[dict[str, Any]] | None = Field(
        default=None, description="Public source entries backing the report or returned instead of it."
    )
    unverified_numbers: list[dict[str, Any]] | None = Field(
        default=None, description="Numbers in the report that were not found in their cited source."
    )
    unverified_number_count: int | None = Field(
        default=None, description="How many numbers in the report could not be verified."
    )
    partial: bool | None = Field(
        default=None,
        description="True when some pages could not be fetched or no report was synthesized.",
    )
    plan: list[dict[str, Any]] | None = Field(
        default=None,
        description="The aspect plan: one entry per aspect with its 'aspect' name and 'query'.",
    )
    rounds: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Per-round evidence gains: new_facts, new_numeric_facts, gaps_filled, new_sources and saturated."
        ),
    )
    facts_per_round: list[int] | None = Field(
        default=None, description="New facts gained in each round, one integer per round."
    )
    estimate_usd: float | None = Field(
        default=None, description="Upfront cost estimate for the run in US dollars."
    )
    stopped_reason: str | None = Field(
        default=None,
        description=(
            "Why rounds stopped: 'max_rounds', 'saturated', 'budget', 'no_gaps', 'retrieval_failed' "
            "or 'empty_pages'."
        ),
    )
    requests_used: int | None = Field(
        default=None, description="Number of billable upstream requests this run made."
    )
    cached: bool | None = Field(
        default=None, description="True when the whole result came from the cache and cost nothing."
    )
    cached_count: int | None = Field(
        default=None, description="Number of upstream requests served from the cache."
    )
    cost_breakdown: dict[str, Any] | None = Field(
        default=None, description="Cost split into 'search', 'fetch' and 'llm' in US dollars."
    )
    notes: list[str] | None = Field(
        default=None, description="Human-readable notes about filtering, caching or truncation."
    )


__all__ = ["AgentResearchOutput"]
