"""Deterministic building blocks for multi-stage synthesis (evals only).

The pieces here decide, without any model call, which sentences of the context belong to which aspect,
which of them made it into a written section, and which sentences may stay in the final report. They are
the part of the `matrix-repair` design that cannot drift between runs, so they are kept apart from the
prompted steps in ``evals/variants.py``.

Run the self-check with ``python evals/synthesis_lab.py``.
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


def _selftest() -> None:
    """Check the deterministic behaviour on a small synthetic context."""
    context = [
        {"id": 1, "content": "Vanadium flow batteries reached 320 USD per kWh in 2024. The cost fell."},
        {"id": 2, "content": "Electrolyte crossover limits cycle life to about 12000 cycles. Nothing else."},
    ]
    aspects = [
        type("A", (), {"aspect": "installed cost per kWh", "looking_for": "USD per kWh, 2024"})(),
        type("A", (), {"aspect": "cycle life", "looking_for": "cycle counts"})(),
        type("A", (), {"aspect": "manufacturing", "looking_for": "factories"})(),
    ]
    blocks = evidence_matrix(aspects, context)
    assert blocks[0]["sentences"] and blocks[0]["sentences"][0]["id"] == 1, blocks[0]
    assert blocks[1]["sentences"] and blocks[1]["sentences"][0]["id"] == 2, blocks[1]
    assert blocks[2]["empty"] and blocks[2]["thin"], blocks[2]

    report = "Vanadium flow batteries reached 320 USD per kWh in 2024 [1]."
    gaps = coverage_gaps(blocks, report)
    assert gaps[0]["count"] == 0, gaps[0]
    assert gaps[1]["count"] >= 1, gaps[1]

    cleaned, dropped = guard("Cost was 320 USD per kWh [1]. Cost was 999 USD [1]. Mirage [9].", context)
    assert "999" not in cleaned and "Mirage" not in cleaned, cleaned
    assert "320 USD per kWh" in cleaned and len(dropped) == 2, (cleaned, dropped)

    text = assemble([("Cost", "Body one [1]."), ("Life", "Body two [2].")], lead="Lead.")
    assert text.startswith("Lead.") and "## Cost" in text and "## Life" in text, text
    print("synthesis_lab self-check: ok")


if __name__ == "__main__":
    _selftest()
