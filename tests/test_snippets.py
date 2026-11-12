"""Unit tests for block-level lexical snippet extraction and sectional pagination."""

from __future__ import annotations

from deepseek_search_everywhere_mcp.snippets import extract_snippets, split_sections

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
