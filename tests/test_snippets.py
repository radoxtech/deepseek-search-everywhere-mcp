"""Unit tests for block-level lexical snippet extraction and sectional pagination."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.snippets import extract_snippets, prune_source_content, split_sections

SAMPLE_DOC = """# Introduction to Quantum Computing

Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.

## Qubits and Superposition

Unlike classical bits which can be 0 or 1, qubits can exist in a superposition of states.
This allows quantum computers to process complex problems exponentially faster.

## Quantum Error Correction

Surface codes and topological qubits are modern approaches to mitigating environmental decoherence.
Error correction remains the primary engineering hurdle for fault-tolerant architectures.

## Conclusion and Future Outlook

Commercial applications in cryptography and chemistry are expected by 2030.
"""


def test_split_sections_markdown_headers():
    sections = split_sections(SAMPLE_DOC)
    assert len(sections) == 4
    assert sections[0]["title"] == "Introduction to Quantum Computing"
    assert sections[1]["title"] == "Qubits and Superposition"
    assert sections[2]["title"] == "Quantum Error Correction"
    assert sections[3]["title"] == "Conclusion and Future Outlook"
    assert sections[0]["index"] == 1
    assert sections[3]["index"] == 4


def test_split_sections_plain_text():
    plain = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    sections = split_sections(plain, target_size=10)
    assert len(sections) >= 2


def test_extract_snippets_relevant_match():
    content, count = extract_snippets(SAMPLE_DOC, "error correction decoherence", max_snippets=1)
    assert count == 1
    assert "Quantum Error Correction" in content
    assert "Surface codes" in content
    assert "Qubits and Superposition" not in content


def test_extract_snippets_multi_match():
    content, count = extract_snippets(SAMPLE_DOC, "quantum", max_snippets=2)
    assert count == 2
    assert "### [Excerpt 1 of 2:" in content
    assert "### [Excerpt 2 of 2:" in content


def test_extract_snippets_empty_query_fallback():
    content, count = extract_snippets(SAMPLE_DOC, "the a and", max_snippets=2)
    # Stopwords only -> falls back to first sections
    assert count == 2
    assert "Introduction to Quantum Computing" in content


def _filler(n_chars: int, seed: str = "filler") -> str:
    para = f"This paragraph is {seed} text about unrelated operational matters and nothing else. " * 4
    out: list[str] = []
    total = 0
    while total < n_chars:
        out.append(para)
        total += len(para) + 2
    return "\n\n".join(out)


def _excerpts(content: str) -> list[str]:
    return [p for p in content.split("\n\n---\n\n") if p.strip()]


def test_extract_snippets_long_page_two_matches_gives_two_short_excerpts():
    page = (
        _filler(12_000, "alpha")
        + "\n\nTo take an incremental backup, run pg_basebackup with the incremental option set.\n\n"
        + _filler(12_000, "beta")
        + "\n\nAn incremental backup needs the manifest of the previous backup to be present.\n\n"
        + _filler(6_000, "gamma")
    )
    assert len(page) > 30_000
    content, count = extract_snippets(page, "incremental backup", max_snippets=2)
    assert count == 2
    parts = _excerpts(content)
    assert len(parts) == 2
    for part in parts:
        assert "incremental backup" in part.lower()
        assert len(part) <= 1700  # 1500 body cap + header and ellipsis markers
    assert len(content) < 3500


def test_extract_snippets_huge_single_section_is_windowed_not_returned_whole():
    body = (
        "# Notes\n\n"
        + _filler(8_000, "x")
        + "\n\nIncremental backup starts here.\n\n"
        + _filler(8_000, "y")
        + "\n\nThe incremental backup ends here.\n\n"
        + _filler(3_000, "z")
    )
    content, count = extract_snippets(body, "incremental backup", max_snippets=3)
    assert count == 2
    assert len(content) < 3500
    assert "Incremental backup starts here." in content
    assert "The incremental backup ends here." in content


def test_extract_snippets_merges_overlapping_windows():
    page = "# Doc\n\nIncremental backup one. " + "word " * 20 + "Another incremental backup mention. Done."
    content, count = extract_snippets(page, "incremental backup", max_snippets=3)
    assert count == 1
    assert content.count("incremental backup") + content.count("Incremental backup") == 2


def test_extract_snippets_only_returns_excerpts_containing_query_terms():
    content, count = extract_snippets(SAMPLE_DOC, "decoherence", max_snippets=3)
    assert count == 1
    assert "decoherence" in content
    assert "Conclusion and Future Outlook" not in content
    assert "Qubits and Superposition" not in content


def test_extract_snippets_no_match_returns_empty():
    content, count = extract_snippets(SAMPLE_DOC, "photosynthesis chlorophyll", max_snippets=3)
    assert (content, count) == ("", 0)


def test_extract_snippets_respects_max_chars():
    page = "\n\n".join(f"## S{i}\n\nIncremental backup detail number {i}. " + "pad " * 150 for i in range(5))
    content, count = extract_snippets(page, "incremental backup", max_snippets=5, max_chars=900)
    assert len(content) <= 900
    assert count >= 1


def test_extract_snippets_stopword_fallback_is_capped():
    big = "# Intro\n\n" + _filler(6_000, "intro")
    content, count = extract_snippets(big, "the a and", max_snippets=1)
    assert count == 1
    assert len(content) <= 1700


def test_prune_source_content_short_text_unchanged():
    short = "Short document content with fewer than max characters."
    assert prune_source_content(short, query_terms=["short"], max_chars=1000) == short


def test_prune_source_content_prioritizes_query_and_metrics():
    filler = (
        "This is an ordinary generic paragraph without any specific figures or numbers. "
        "It contains standard filler descriptions of normal activities that occur regularly."
    )
    salient1 = (
        "In benchmark tests with PostgreSQL version 17, incremental backup reduced backup duration "
        "by 85% from 450 seconds down to 68 seconds with pg_basebackup."
    )
    salient2 = (
        "Warning: a common error encountered during backup replication is WAL stream interruption, "
        "resulting in a limit reached table state."
    )
    doc = f"{filler}\n\n{salient1}\n\n{filler}\n\n{salient2}\n\n{filler}"
    pruned = prune_source_content(doc, query_terms=["postgresql", "backup", "duration"], max_chars=400)
    assert salient1 in pruned
    assert "85%" in pruned


def test_prune_source_content_preserves_document_order():
    p1 = "First critical finding on version 1.0 with 500 items tested in benchmark."
    filler = "Generic filler text that takes up space and does not contain numbers or relevant terms."
    p2 = "Second critical finding on version 2.0 with 900 items tested in benchmark."
    doc = f"{p1}\n\n{filler}\n\n{filler}\n\n{p2}"
    pruned = prune_source_content(doc, query_terms=["critical", "version"], max_chars=300)
    pos1 = pruned.find("First critical finding")
    pos2 = pruned.find("Second critical finding")
    assert pos1 != -1 and pos2 != -1
    assert pos1 < pos2


def test_prune_source_content_bounds_chars():
    long_doc = "\n\n".join(
        f"Paragraph {i} has version {i} with score {i * 10}% in benchmark." for i in range(100)
    )
    pruned = prune_source_content(long_doc, query_terms=["version"], max_chars=500)
    assert len(pruned) <= 500
