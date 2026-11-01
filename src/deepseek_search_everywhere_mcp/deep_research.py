"""Decomposition, budgeting, and cited synthesis for deep research.

Decomposes complex topics into distinct subqueries via deepseek-flash,
budgets merged retrieval contexts, and generates factual briefings with [n] citations.
"""

from __future__ import annotations

import re
from typing import Any

from .extract import sanitize_json

MAX_SYNTHESIS_SOURCES = 6
MAX_SYNTHESIS_CHARS = 24000
DECOMPOSE_SYSTEM_PROMPT = (
    "You decompose complex research topics into 2 to 4 distinct, targeted web search queries. "
    'Output ONLY a valid JSON object with key "queries" containing an array of strings. '
    "Each search query must be at most 100 characters and investigate a different angle of the topic. "
    "Do not include code fences or extra text."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a research synthesis analyst. Write a concise, high-density factual briefing report "
    "(400-600 words) addressing the research topic based strictly on the provided sources. "
    "You MUST cite facts using numbered bracket citations [1], [2], etc., exactly matching the "
    "corresponding source number. Do not cite sources that do not support the claim. "
    "At the end of your report, provide a brief 'References' section listing each cited [n] with "
    "its source title and URL. Do not include conversational filler."
)


def build_decomposition_messages(topic: str, max_queries: int = 3) -> list[dict[str, str]]:
    """Build messages for deepseek-flash to decompose ``topic`` into subqueries."""
    return [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Decompose this research topic into {max_queries} search queries: {topic}",
        },
    ]


def parse_subqueries(response_text: str, fallback_topic: str, max_queries: int = 3) -> list[str]:
    """Parse JSON subqueries from deepseek-flash response; fall back to ``fallback_topic`` on failure."""
    parsed, err = sanitize_json(response_text)
    if err or not isinstance(parsed, dict) or "queries" not in parsed:
        return [fallback_topic[:128].strip()]

    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        return [fallback_topic[:128].strip()]

    queries: list[str] = []
    for item in raw_queries:
        if isinstance(item, str) and item.strip():
            # Guarantee each subquery respects the 128-char limit
            queries.append(item.strip()[:128].strip())

    if not queries:
        return [fallback_topic[:128].strip()]

    return queries[:max_queries]


def select_synthesis_context(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select up to 6 non-empty sources and cap total content at 24,000 characters."""
    valid_sources = [s for s in sources if s.get("content")]
    # Prioritize by rerank score / authority if present
    valid_sources.sort(
        key=lambda s: (s.get("rerank_score") or 0.0, s.get("authority") or 0.0),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    total_chars = 0
    for idx, s in enumerate(valid_sources[:MAX_SYNTHESIS_SOURCES], 1):
        content = s.get("content") or ""
        remaining = MAX_SYNTHESIS_CHARS - total_chars
        if remaining <= 0:
            break
        cut_content = content[:remaining]
        total_chars += len(cut_content)
        selected.append(
            {
                "id": idx,
                "title": s.get("title") or "Untitled",
                "url": s.get("url"),
                "content": cut_content,
            }
        )
    return selected


def build_synthesis_messages(topic: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build messages for deepseek-flash to generate cited synthesis report."""
    context_parts: list[str] = []
    for s in sources:
        header = f"[{s['id']}] {s['title']} ({s['url']})"
        context_parts.append(f"{header}\n{s['content']}")

    sources_text = "\n\n---\n\n".join(context_parts)
    return [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research Topic: {topic}\n\nAvailable Sources:\n\n{sources_text}"},
    ]


def sanitize_citations(text: str, max_id: int) -> str:
    """Normalize bracket citations, ensuring hallucinated IDs (> max_id or <= 0) are removed."""

    def _replace_citation(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= max_id:
            return match.group(0)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", _replace_citation, text)
    return re.sub(r"[ \t]{2,}", " ", cleaned)
