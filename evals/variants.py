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


_OLD_DECOMPOSE_SYSTEM = (
    "You decompose complex research topics into 2 to 4 distinct, targeted web search queries. "
    'Output ONLY a valid JSON object with key "queries" containing an array of strings. '
    "Each search query must be at most 100 characters and investigate a different angle of the topic. "
    "Do not include code fences or extra text."
)


def _old_decompose() -> None:
    """Ablation: the decomposition prompt of 3.3 (plain queries, no aspects), the rest as it is now.

    Separates the effect of the aspect prompt from the verifier, source selection and synthesis rules.
    """
    from deepseek_search_everywhere_mcp import research_jobs

    def old_messages(
        topic: str, max_queries: int = 3, sources: str = "any", **_: Any
    ) -> list[dict[str, str]]:
        content = f"Decompose this research topic into {max_queries} search queries: {topic}"
        return [
            {"role": "system", "content": _OLD_DECOMPOSE_SYSTEM},
            {"role": "user", "content": content},
        ]

    research_jobs.build_decomposition_messages = old_messages


VARIANTS: dict[str, Callable[[], None]] = {
    "baseline": _baseline,
    "degraded-two-sources": _degraded_two_sources,
    "old-decompose": _old_decompose,
}


def apply(name: str) -> None:
    """Apply the patches of variant ``name``; raise ``KeyError`` for an unknown one."""
    VARIANTS[name]()
