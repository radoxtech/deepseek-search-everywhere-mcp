"""Offline checks of the harness logic (run by hand: ``pytest evals/tests``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge  # noqa: E402
import metrics  # noqa: E402


def test_a_comparison_is_won_only_when_both_orders_agree() -> None:
    assert judge.decide(["b", "b"]) == "b"
    assert judge.decide(["a", "a"]) == "a"
    assert judge.decide(["a", "b"]) == "tie"
    assert judge.decide(["tie", "tie"]) == "tie"
    assert judge.decide(["b"]) == "tie"  # a missing order is never a win


def test_labels_map_back_to_sides_in_both_orders() -> None:
    assert judge._side_of("1", "AB") == "a" and judge._side_of("2", "AB") == "b"
    assert judge._side_of("1", "BA") == "b" and judge._side_of("2", "BA") == "a"
    assert judge._side_of("tie", "BA") == "tie"


def test_cited_sentences_and_lexical_support() -> None:
    report = "# Title\n\nSurface codes reached 97% fidelity in 2024 [1]. Another claim without data [2].\n"
    sentences = metrics.cited_sentences(report)
    assert sentences == [
        ("Surface codes reached 97% fidelity in 2024.", [1]),
        ("Another claim without data.", [2]),
    ]
    source = "In 2024 the team showed surface codes reached 97% gate fidelity."
    assert metrics.supports(sentences[0][0], source)
    assert not metrics.supports("Surface codes reached 99% fidelity in 2024", source)


def test_offpeak_cost_counts_web_requests_and_tokens() -> None:
    record = {
        "envelope": {"cost_breakdown": {"search": 0.0015, "fetch": 0.00075}},
        "chat_usage": [
            {"usage": {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 200, "completion_tokens": 500}}
        ],
    }
    expected = 0.0015 + 0.00075 + (200 * 0.003 + 800 * 0.15 + 500 * 0.60) / 1_000_000
    assert metrics.offpeak_cost(record) == round(expected, 6)


def test_record_metrics_of_a_finished_job() -> None:
    record = {
        "topic_id": "t",
        "depth": "low",
        "variant": "baseline",
        "repeat": 1,
        "wall_s": 5.0,
        "envelope": {
            "ok": True,
            "status": "done",
            "report": "Fact one holds with 12 cases [1]. Bad id here [9].",
            "sources": [
                {"id": 1, "url": "https://www.example.com/a"},
                {"id": 2, "url": "https://other.org/b"},
            ],
            "unverified_numbers": [{"number": "12"}],
        },
        "dump": {"synthesis_context": [{"id": 1, "content": "Fact one holds with 12 cases in total."}]},
        "chat_usage": [],
    }
    m = metrics.record_metrics(record)
    assert m["ok"] is True
    assert m["citation_valid_rate"] == 0.5
    assert m["citation_support_rate"] == 0.5
    assert (m["sources"], m["domains"], m["unverified_numbers"]) == (2, 2, 1)


def test_a_failed_job_is_reported_with_its_reason() -> None:
    record = {"envelope": {"ok": False, "error": {"reason": "timeout"}}, "chat_usage": []}
    m = metrics.record_metrics(record)
    assert m["ok"] is False and m["reason"] == "timeout"
