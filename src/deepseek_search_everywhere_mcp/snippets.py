"""Block-level lexical snippet extraction and sectional reading for web pages.

Segments Markdown content into logical sections, computes BM25-lite relevance scores
for user queries without LLM invocation, and provides sectional pagination.
"""

from __future__ import annotations

import math
import re
from typing import Any

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric terms from ``text``, excluding stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\u0080-\uffff]{2,}\b", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def split_sections(text: str, target_size: int = 1500) -> list[dict[str, Any]]:
    """Segment Markdown ``text`` into logical sections by headers or paragraph blocks.

    Each returned item is ``{"index": 1-based, "title": str, "content": str}``.
    """
    raw_text = text.strip()
    if not raw_text:
        return []

    matches = list(_HEADER_RE.finditer(raw_text))
    if matches:
        sections: list[dict[str, Any]] = []
        if matches[0].start() > 0:
            preamble = raw_text[: matches[0].start()].strip()
            if preamble:
                sections.append({"index": 1, "title": "Preamble", "content": preamble})

        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            body = raw_text[start:end].strip()
            sections.append({"index": len(sections) + 1, "title": title, "content": body})
        return sections

    # Fallback for plain text without Markdown headers: split by double newlines
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [{"index": 1, "title": "Content", "content": raw_text}]

    sections = []
    current_chunk: list[str] = []
    current_size = 0

    for p in paragraphs:
        current_chunk.append(p)
        current_size += len(p)
        if current_size >= target_size:
            sections.append(
                {
                    "index": len(sections) + 1,
                    "title": f"Section {len(sections) + 1}",
                    "content": "\n\n".join(current_chunk),
                }
            )
            current_chunk = []
            current_size = 0

    if current_chunk:
        sections.append(
            {
                "index": len(sections) + 1,
                "title": f"Section {len(sections) + 1}",
                "content": "\n\n".join(current_chunk),
            }
        )
    return sections


def extract_snippets(text: str, query: str, max_snippets: int = 3) -> tuple[str, int]:
    """Score sections against ``query`` using BM25-lite and return top formatted excerpts.

    Returns ``(formatted_snippets, snippet_count)``.
    """
    sections = split_sections(text)
    if not sections:
        return "", 0

    max_snippets = max(1, min(max_snippets, 5))
    q_terms = _tokenize(query)
    if not q_terms:
        # No searchable terms; return the first section(s)
        selected = sections[:max_snippets]
        return _format_snippets(selected), len(selected)

    doc_count = len(sections)
    # Document frequency per query term
    df: dict[str, int] = {}
    section_terms: list[list[str]] = []
    for s in sections:
        terms = _tokenize(s["content"])
        section_terms.append(terms)
        unique_terms = set(terms)
        for t in q_terms:
            if t in unique_terms:
                df[t] = df.get(t, 0) + 1

    # BM25-lite scoring
    k1 = 1.2
    b = 0.75
    avg_dl = sum(len(terms) for terms in section_terms) / max(1, doc_count)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, (s, terms) in enumerate(zip(sections, section_terms, strict=True)):
        dl = len(terms)
        title_terms = set(_tokenize(s["title"]))
        score = 0.0
        term_freqs: dict[str, int] = {}
        for t in terms:
            term_freqs[t] = term_freqs.get(t, 0) + 1

        for t in q_terms:
            tf = term_freqs.get(t, 0)
            if tf == 0:
                continue
            n_qi = df.get(t, 0)
            idf = math.log((doc_count - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
            term_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl))))
            if t in title_terms:
                term_score *= 1.5
            score += term_score

        scored.append((score, idx, s))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top = [s for _, _, s in scored[:max_snippets]]
    # Restore original document order for narrative flow
    top.sort(key=lambda s: s["index"])
    return _format_snippets(top), len(top)


def _format_snippets(sections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    total = len(sections)
    for i, s in enumerate(sections, 1):
        header = f"### [Excerpt {i} of {total}: {s['title']}]"
        parts.append(f"{header}\n{s['content']}")
    return "\n\n---\n\n".join(parts)
