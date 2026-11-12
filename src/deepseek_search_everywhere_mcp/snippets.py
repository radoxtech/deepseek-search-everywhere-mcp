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


EXCERPT_WINDOW = 600
EXCERPT_CAP = 1500
MAX_SNIPPETS = 5  # most excerpts extract_snippets returns, whatever max_snippets asks for
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_\u0080-\uffff]{2,}\b")
_MIN_PARTIAL_EXCERPT = 200


def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Move a cut window to word boundaries so it does not begin or end inside a word."""
    if start > 0:
        gap = text.find(" ", start, min(end, start + 40))
        if gap != -1 and not text[start - 1].isspace():
            start = gap + 1
    if end < len(text):
        gap = text.rfind(" ", max(start, end - 40), end)
        if gap != -1 and not text[end].isspace():
            end = gap
    return start, end


def _match_windows(content: str, q_terms: set[str], window: int, cap: int) -> list[dict[str, Any]]:
    """Return merged ``+-window`` char windows around query-term hits in one section.

    Overlapping windows merge as long as the merged span stays within ``cap``.
    """
    hits = [(m.start(), m.end(), m.group().lower()) for m in _TOKEN_RE.finditer(content)]
    hits = [h for h in hits if h[2] in q_terms]
    windows: list[dict[str, Any]] = []
    for h_start, h_end, term in hits:
        w_start, w_end = max(0, h_start - window), min(len(content), h_end + window)
        last = windows[-1] if windows else None
        if last is not None and w_start <= last["end"] and w_end - last["start"] <= cap:
            last["end"] = w_end
            last["hits"] += 1
            last["terms"].add(term)
        else:
            windows.append({"start": w_start, "end": w_end, "hits": 1, "terms": {term}})
    return windows


def extract_snippets(
    text: str,
    query: str,
    max_snippets: int = 3,
    *,
    window: int = EXCERPT_WINDOW,
    excerpt_cap: int = EXCERPT_CAP,
    max_chars: int | None = None,
) -> tuple[str, int]:
    """Return focused excerpts (``+-window`` chars around query-term matches) ranked by relevance.

    Only excerpts that contain a query term are returned; windows within a section merge when they
    overlap, each excerpt is capped at ``excerpt_cap`` characters and the total at ``max_chars``.
    Windows never cross a section boundary. Without searchable query terms the leading sections
    are returned instead (each capped). Returns ``(formatted_snippets, snippet_count)``; no match
    gives ``("", 0)``.
    """
    sections = split_sections(text)
    if not sections:
        return "", 0

    max_snippets = max(1, min(max_snippets, MAX_SNIPPETS))
    q_terms = set(_tokenize(query))
    if not q_terms:
        capped = [{**s, "content": _cut(s["content"], 0, excerpt_cap)} for s in sections[:max_snippets]]
        return _format_snippets(capped, max_chars), len(capped)

    doc_count = len(sections)
    section_terms = [set(_tokenize(s["content"])) for s in sections]
    df = {t: sum(1 for terms in section_terms if t in terms) for t in q_terms}

    candidates: list[dict[str, Any]] = []
    for pos, (s, terms) in enumerate(zip(sections, section_terms, strict=True)):
        if not (q_terms & terms):
            continue
        title_terms = set(_tokenize(s["title"]))
        for w in _match_windows(s["content"], q_terms, window, excerpt_cap):
            score = 0.1 * w["hits"]
            for t in w["terms"]:
                idf = math.log((doc_count - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                score += idf * (1.5 if t in title_terms else 1.0)
            start, end = _snap(s["content"], w["start"], w["end"])
            candidates.append(
                {
                    "score": score,
                    "order": (pos, start),
                    "index": pos + 1,
                    "title": s["title"],
                    "content": _cut(s["content"], start, end),
                }
            )

    candidates.sort(key=lambda c: (-c["score"], c["order"]))
    top = candidates[:max_snippets]
    top = _fit_total(top, max_chars)
    top.sort(key=lambda c: c["order"])
    return _format_snippets(top, None), len(top)


def _cut(content: str, start: int, end: int) -> str:
    """Slice ``content[start:end]`` and mark a cut start or end with an ellipsis."""
    piece = content[start:end].strip()
    prefix = "\u2026 " if start > 0 else ""
    suffix = " \u2026" if 0 < end < len(content) else ""
    return f"{prefix}{piece}{suffix}"


def _fit_total(excerpts: list[dict[str, Any]], max_chars: int | None) -> list[dict[str, Any]]:
    """Keep the best-ranked excerpts that fit ``max_chars`` in total (the last one may be shortened)."""
    if max_chars is None:
        return excerpts
    kept: list[dict[str, Any]] = []
    used = 0
    for ex in excerpts:
        overhead = len(_excerpt_header(ex["title"], 1, 1)) + 1 + len("\n\n---\n\n")
        room = max_chars - used - overhead
        if len(ex["content"]) <= room:
            kept.append(ex)
            used += len(ex["content"]) + overhead
        elif room >= _MIN_PARTIAL_EXCERPT:
            kept.append({**ex, "content": ex["content"][: room - 2].rstrip() + " \u2026"})
            break
        else:
            break
    return kept


def _excerpt_header(title: str, i: int, total: int) -> str:
    return f"### [Excerpt {i} of {total}: {title}]"


def _format_snippets(sections: list[dict[str, Any]], max_chars: int | None = None) -> str:
    parts: list[str] = []
    total = len(sections)
    for i, s in enumerate(sections, 1):
        parts.append(f"{_excerpt_header(s['title'], i, total)}\n{s['content']}")
    out = "\n\n---\n\n".join(parts)
    return out[:max_chars] if max_chars is not None else out
