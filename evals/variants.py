"""Variants the harness can run. Each one patches the server in the run's own fresh process.

A variant is a function that applies its patches before the job starts. Patches stay inside the harness
process; the package itself never ships a variant. Later phases register their candidates here (for
example the multi-stage synthesis variants of phase 6).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _baseline() -> None:
    """The server as it is."""


def _degraded_two_sources() -> None:
    """Deliberately worse: synthesize from the first two selected sources only.

    Used once in phase 5 to show the judge and the metrics notice a worse report.
    """
    from deepseek_search_everywhere_mcp import research_jobs

    original = research_jobs.select_synthesis_context

    def two_sources(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return original(*args, **kwargs)[:2]

    research_jobs.select_synthesis_context = two_sources


VARIANTS: dict[str, Callable[[], None]] = {
    "baseline": _baseline,
    "degraded-two-sources": _degraded_two_sources,
}


def apply(name: str) -> None:
    """Apply the patches of variant ``name``; raise ``KeyError`` for an unknown one."""
    VARIANTS[name]()
