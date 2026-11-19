from __future__ import annotations

from types import SimpleNamespace

from deepseek_search_everywhere_mcp import evidence as lab

CONTEXT = [
    {"id": 1, "content": "Vanadium flow batteries reached 320 USD per kWh in 2024. The cost fell."},
    {"id": 2, "content": "Electrolyte crossover limits cycle life to about 12000 cycles. Nothing else."},
]
ASPECTS = [
    SimpleNamespace(aspect="installed cost per kWh", looking_for="USD per kWh, 2024"),
    SimpleNamespace(aspect="cycle life", looking_for="cycle counts"),
    SimpleNamespace(aspect="manufacturing", looking_for="factories"),
]


def test_evidence_matrix_assigns_sentences_to_aspects() -> None:
    blocks = lab.evidence_matrix(ASPECTS, CONTEXT)

    assert [item["id"] for item in blocks[0]["sentences"]] == [1]
    assert [item["id"] for item in blocks[1]["sentences"]] == [2]
    assert blocks[2]["empty"] and blocks[2]["thin"]
    assert not blocks[0]["empty"]


def test_coverage_gaps_finds_the_missing_aspect() -> None:
    blocks = lab.evidence_matrix(ASPECTS, CONTEXT)
    report = "Vanadium flow batteries reached 320 USD per kWh in 2024 [1]."

    gaps = lab.coverage_gaps(blocks, report)

    assert gaps[0]["count"] == 0
    assert gaps[1]["count"] >= 1


def test_round_gain_reports_saturation() -> None:
    first = lab.evidence_matrix(ASPECTS, CONTEXT)

    assert lab.round_gain(first, first, new_sources=0)["saturated"]


def test_round_gain_counts_a_numeric_fact_that_fills_a_gap() -> None:
    first = lab.evidence_matrix(ASPECTS, CONTEXT)
    extra = {"id": 3, "content": "Manufacturing capacity reached 12 GWh in 2025 across three plants."}
    second = lab.evidence_matrix(ASPECTS, CONTEXT + [extra])

    gain = lab.round_gain(first, second, new_sources=1)

    assert gain["new_numeric_facts"] == 1
    assert gain["gaps_filled"] == 1
    assert not gain["saturated"]


def test_guard_drops_unknown_citations_and_foreign_figures() -> None:
    cleaned, dropped = lab.guard("Cost was 320 USD per kWh [1]. Cost was 999 USD [1]. Mirage [9].", CONTEXT)

    assert "999" not in cleaned and "Mirage" not in cleaned
    assert "320 USD per kWh" in cleaned
    assert len(dropped) == 2


def test_guard_keeps_headings_and_uncited_prose() -> None:
    cleaned, dropped = lab.guard("## Cost\n\nNo figures here.", CONTEXT)

    assert cleaned == "## Cost\n\nNo figures here."
    assert dropped == []


def test_assemble_keeps_every_section() -> None:
    text = lab.assemble([("Cost", "Body one [1]."), ("Life", "Body two [2].")], lead="Lead.")

    assert text.startswith("Lead.")
    assert "## Cost" in text and "## Life" in text
