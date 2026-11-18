"""Offline checks of the harness logic (run by hand: ``pytest evals/tests``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge  # noqa: E402
import keypoints  # noqa: E402
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


def test_excerpt_picks_the_passage_with_the_cited_figure() -> None:
    filler = "Background text about the study design and general methods. " * 30
    text = filler + "Mean weight change was -14.9% with semaglutide versus -2.4% with placebo. " + filler
    excerpt = judge._excerpt("Semaglutide gave -14.9% versus -2.4% with placebo", text)
    assert "-14.9%" in excerpt and "-2.4%" in excerpt


def test_excerpt_returns_two_separate_passages() -> None:
    text = "alpha 12.5 here. " + "x " * 600 + "alpha 99.1 there. " + "y " * 600
    excerpt = judge._excerpt("alpha rose from 12.5 to 99.1", text)
    assert " … " in excerpt and "12.5" in excerpt and "99.1" in excerpt


def test_keypoint_scores_count_core_coverage_and_citation_support() -> None:
    rubric = {
        "key_points": [
            {"id": "k1", "importance": "core"},
            {"id": "k2", "importance": "core"},
            {"id": "k3", "importance": "secondary"},
        ]
    }
    verdict = {
        "points": {
            "k1": {"covered": True, "sentence": 2, "supported": True},
            "k2": {"covered": True, "sentence": None, "supported": None},
            "k3": {"covered": False},
        },
        "unsupported": [4, 7],
    }
    scores = keypoints.verdict_scores(verdict, rubric)
    assert scores == {"core": 1.0, "all": 2 / 3, "core_supported": 0.5, "unsupported": 2.0}


def test_keypoint_pack_ids_are_stable_and_blind() -> None:
    first = keypoints.pack_id("pilot:medium", "medicine-glp1", 1)
    assert first == keypoints.pack_id("pilot:medium", "medicine-glp1", 1)
    assert first != keypoints.pack_id("new:medium", "medicine-glp1", 1)
    assert "pilot" not in first and len(first) == 12


def test_old_decompose_variant_restores_plain_queries(monkeypatch) -> None:
    import variants

    from deepseek_search_everywhere_mcp import research_jobs

    monkeypatch.setattr(
        research_jobs, "build_decomposition_messages", research_jobs.build_decomposition_messages
    )
    variants.apply("old-decompose")
    system, user = research_jobs.build_decomposition_messages(
        "Topic X", max_queries=3, sources="any", lang=None
    )
    assert '"queries"' in system["content"] and "aspects" not in system["content"]
    assert user["content"] == "Decompose this research topic into 3 search queries: Topic X"


def test_freeze_replays_the_saved_context(tmp_path, monkeypatch) -> None:
    import json

    import variants

    from deepseek_search_everywhere_mcp import research_jobs

    for name in (
        "_research_retrieve",
        "_research_round_two",
        "_research_rank_and_fill",
        "select_synthesis_context",
    ):
        monkeypatch.setattr(research_jobs, name, getattr(research_jobs, name))
    context = [{"id": 1, "title": "T", "url": "https://a.test/x", "content": "saved text"}]
    record = tmp_path / "r.json"
    record.write_text(json.dumps({"dump": {"synthesis_context": context}}), encoding="utf-8")
    variants.freeze(str(record))
    assert research_jobs.select_synthesis_context([], depth="medium") == context


def test_old_synthesis_variant_restores_the_33_prompt(monkeypatch) -> None:
    import variants

    from deepseek_search_everywhere_mcp import research_jobs

    monkeypatch.setattr(research_jobs, "build_synthesis_messages", research_jobs.build_synthesis_messages)
    variants.apply("old-synthesis")
    system, user = research_jobs.build_synthesis_messages(
        "Topic X",
        [{"id": 1, "title": "T", "url": "u", "content": "c"}],
        depth="medium",
        nonce="n1",
        aspects=(),
    )
    assert "high-density, authoritative" in system["content"] and "exactly 3 sections" in system["content"]
    assert user["content"].startswith("Research Topic: Topic X") and "Source index" not in user["content"]


def test_probe_flags_reference_lists_and_letters() -> None:
    import plan_probe

    refs = "Smith J et al. doi:10.1000/x. Doe A et al. 10.1000/y. " * 20
    assert plan_probe.looks_bibliographic("Some study", "https://a.test/x", refs)
    assert plan_probe.looks_bibliographic("Correspondence: trial results", "https://a.test/x", "text")
    assert plan_probe.looks_bibliographic("T", "https://www.nejm.org/doi/full/10.1056/NEJMc2211120", "text")
    assert not plan_probe.looks_bibliographic(
        "Trial results", "https://a.test/x", "We randomized 400 adults. " * 50
    )
