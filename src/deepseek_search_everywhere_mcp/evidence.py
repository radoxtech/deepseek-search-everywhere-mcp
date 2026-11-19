"""Deterministic evidence bookkeeping: which source sentences belong to which aspect, what a report
misses, and what one search round added.

Nothing here calls a model, so the results cannot drift between runs. The agent tool uses this to decide
whether another round is worth its cost; the evaluation harness uses it to attribute losses to stages.
"""

from __future__ import annotations

import re
from typing import Any

STOP = frozenset(
    """the a an and or of to in on for with without from by is are was were be been being that this these
    those it its as at into over under than then can could should would may might must not no more less
    most some any all each other such when where which who whom whose what how why does do did done have
    has had per versus vs about between within also according report reports source sources data
    """.split()
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
NUMBER = re.compile(r"\d[\d.,]*%?")


def terms(text: str) -> set[str]:
    """Lowercase content words of a text, used to match an aspect or a fact against a sentence."""
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.%-]{2,}", (text or "").lower()) if w not in STOP}


def numbers(text: str) -> set[str]:
    """The figures in a text, normalised so that 1,200 and 1200 compare equal."""
    return {m.replace(",", "").rstrip(".") for m in NUMBER.findall(text or "")}


def sentences(text: str, *, shortest: int = 40, longest: int = 320) -> list[str]:
    """Split a page into sentence-sized units that are worth citing."""
    out = []
    for part in SENTENCE_SPLIT.split((text or "").strip()):
        part = " ".join(part.split())
        if shortest <= len(part) <= longest:
            out.append(part)
    return out


def evidence_matrix(
    aspects: list[Any],
    context: list[dict[str, Any]],
    *,
    per_aspect: int = 8,
    max_chars: int = 2500,
    thin_below: int = 3,
) -> list[dict[str, Any]]:
    """Pick, for each aspect, the source sentences that carry its terms and figures.

    Returns one block per aspect with ``aspect``, ``looking_for``, ``sentences`` (each a
    ``{"id": source_id, "text": sentence}``), ``chars``, ``thin`` (fewer than ``thin_below`` sentences)
    and ``empty``. An empty block is a fact of the measurement, not a reason to invent text.
    """
    blocks: list[dict[str, Any]] = []
    for aspect in aspects:
        wanted = terms(f"{getattr(aspect, 'aspect', '')} {getattr(aspect, 'looking_for', '')}")
        prefer = numbers(getattr(aspect, "looking_for", "") or "")
        scored: list[tuple[float, Any, str]] = []
        for source in context:
            for sentence in sentences(source.get("content") or ""):
                hits = len(wanted & terms(sentence))
                if hits == 0:
                    continue
                bonus = 3 * len(numbers(sentence) & prefer)
                scored.append((hits + bonus, source.get("id"), sentence))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        chars = 0
        for _, source_id, sentence in scored:
            key = sentence[:80].lower()
            if key in seen:
                continue
            if chars + len(sentence) > max_chars:
                break
            seen.add(key)
            picked.append({"id": source_id, "text": sentence})
            chars += len(sentence)
            if len(picked) >= per_aspect:
                break
        blocks.append(
            {
                "aspect": getattr(aspect, "aspect", ""),
                "looking_for": getattr(aspect, "looking_for", ""),
                "sentences": picked,
                "chars": chars,
                "thin": len(picked) < thin_below,
                "empty": not picked,
            }
        )
    return blocks


def block_text(block: dict[str, Any]) -> str:
    """Render one evidence block as the lines a section writer may cite."""
    return "\n".join(f"- [{item['id']}] {item['text']}" for item in block["sentences"])


def checklist_text(blocks: list[dict[str, Any]]) -> str:
    """Render the evidence blocks as a must-cover checklist that sits on top of the full context."""
    lines = []
    for block in blocks:
        if not block["sentences"]:
            continue
        lines.append(f"### {block['aspect']}")
        lines.extend(f"- [{item['id']}] {item['text']}" for item in block["sentences"])
    if not lines:
        return ""
    return (
        "\n\nCHECKLIST — every fact below is stated by the sources above and must appear in the report, "
        "with its source id in brackets. Cover them all; do not drop one, and do not add facts the "
        "sources do not state:\n" + "\n".join(lines)
    )


def block_ids(block: dict[str, Any]) -> list[int]:
    """The distinct source ids a section writer is allowed to cite."""
    return sorted({item["id"] for item in block["sentences"]}, key=str)


def round_gain(before: dict[str, Any], after: dict[str, Any], *, new_sources: int) -> dict[str, Any]:
    """What one search round added: new facts carrying figures, aspects that got their first evidence,
    and new sources. ``before`` and ``after`` are the evidence-block lists of the two rounds."""
    seen_before = {item["text"][:80].lower() for block in before for item in block["sentences"]}
    fresh = [
        item for block in after for item in block["sentences"] if item["text"][:80].lower() not in seen_before
    ]
    numeric = sum(1 for item in fresh if numbers(item["text"]))
    before_counts = {block["aspect"]: len(block["sentences"]) for block in before}
    gaps_filled = sum(
        1 for block in after if block["sentences"] and before_counts.get(block["aspect"], 0) == 0
    )
    return {
        "new_facts": len(fresh),
        "new_numeric_facts": numeric,
        "gaps_filled": gaps_filled,
        "new_sources": new_sources,
        "saturated": numeric == 0 and gaps_filled == 0 and new_sources == 0,
    }


def is_covered(sentence: str, report: str, *, term_share: float = 0.5) -> bool:
    """Whether a report states a source sentence: its figures, and most of its terms, appear in it."""
    want_numbers = numbers(sentence)
    have_numbers = numbers(report)
    if want_numbers and not want_numbers <= have_numbers:
        return False
    want_terms = terms(sentence)
    if not want_terms:
        return True
    hits = len(want_terms & terms(report))
    return hits / len(want_terms) >= term_share


def coverage_gaps(blocks: list[dict[str, Any]], report: str) -> list[dict[str, Any]]:
    """Per aspect, the evidence sentences the report does not state, and how many are missing."""
    gaps = []
    for block in blocks:
        missing = [item for item in block["sentences"] if not is_covered(item["text"], report)]
        gaps.append({"aspect": block["aspect"], "missing": missing, "count": len(missing)})
    return gaps


def guard(report: str, context: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Drop sentences whose citation is unknown or whose figures do not occur in the cited source.

    Works sentence by sentence, so one bad sentence does not take its paragraph with it. Figures are
    compared against the sources that sentence cites, which is what the server does for its own check.
    """
    sources = {str(source.get("id")): source.get("content") or "" for source in context}
    kept_lines: list[str] = []
    dropped: list[str] = []
    for line in report.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            kept_lines.append(line)
            continue
        survivors: list[str] = []
        for sentence in SENTENCE_SPLIT.split(line):
            cited = re.findall(r"\[(\d+)\]", sentence)
            if not cited:
                survivors.append(sentence)
                continue
            if any(cid not in sources for cid in cited):
                dropped.append(f"unknown citation: {sentence.strip()[:80]}")
                continue
            allowed = numbers(" ".join(sources[cid] for cid in cited))
            body = re.sub(r"\[\d+\]", "", sentence)
            if not numbers(body) <= allowed:
                dropped.append(f"figure not in cited source: {sentence.strip()[:80]}")
                continue
            survivors.append(sentence)
        if survivors:
            kept_lines.append(" ".join(part.strip() for part in survivors))
    return "\n".join(kept_lines), dropped


def assemble(sections: list[tuple[str, str]], lead: str = "") -> str:
    """Join written sections into one report: the server does the stitching, so nothing is lost."""
    parts = []
    if lead.strip():
        parts.append(lead.strip())
    for name, text in sections:
        body = (text or "").strip()
        if not body:
            continue
        parts.append(f"## {name}\n\n{body}")
    return "\n\n".join(parts)
