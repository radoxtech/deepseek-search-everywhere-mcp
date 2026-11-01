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
