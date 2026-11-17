"""Shared envelope pieces: the per-request price, the ``_meta`` block and cache-hit accounting."""

from __future__ import annotations

from typing import Any

# Note added by every tool whose content was cut to fit a character budget.
TRUNCATED_NOTE = "Some page content was truncated to respect the character budget."


# Upstream pricing per billable Search/Fetch request: $0.75 per 1,000 requests.
COST_PER_REQUEST = 0.00075


def _envelope_meta(cost_usd: float, untrusted_content: bool = True) -> dict[str, Any]:
    """Build the standard ``_meta`` payload reporting estimated costs and security classification."""
    return {
        "cost_usd": cost_usd,
        "untrusted_content": untrusted_content,
    }


def _all_cached(requests_used: int, cached_count: int) -> bool:
    """Top-level ``cached`` of a multi-response tool: something came from the cache and nothing upstream."""
    return cached_count > 0 and requests_used == 0
