"""Comprehensive test suite covering every single stage of quick research and deep agent research

under the modern scaled limits and 1M context architecture (ADR 0011, ADR 0012).
"""

from __future__ import annotations

import json
from typing import Any

from test_research import DeepResearchUpstream

from deepseek_search_everywhere_mcp import deep_research, server
from deepseek_search_everywhere_mcp.deep_research import (
    DEPTH_CONFIG,
    MAX_EST_SYNTHESIS_TOKENS,
    SECTION_COUNTS,
    apply_primary_quota,
    apply_token_guard,
    build_synthesis_messages,
    build_synthesis_system_prompt,
    filter_synthesis_output,
    select_synthesis_context,
)
from deepseek_search_everywhere_mcp.research_jobs import MAX_CLIENT_REPORT_CHARS
from deepseek_search_everywhere_mcp.tools.agent import DEPTH_ROUNDS, deepseek_deep_agent_research
from deepseek_search_everywhere_mcp.tools.research import (
    deepseek_check_research,
    deepseek_quick_research,
)

# ---------------------------------------------------------------------------
# STAGE 1: Topic Input, Decomposition & Query Planning
# ---------------------------------------------------------------------------


async def test_stage1_decomposition_accepts_rich_4000_char_brief(fake_upstream: Any) -> None:
    """Stage 1: Verify MAX_TOPIC=4000 accepts complex multi-paragraph user briefs without error."""
    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    # Rich brief of 3,800 characters
    brief = (
        "Investigate the clinical efficacy, dosage regimens, safety profiles, and molecular mechanisms "
        "of GLP-1 receptor agonists and dual GIP/GLP-1 co-agonists (tirzepatide, semaglutide, retatrutide) "
        "in adult patients with type 2 diabetes mellitus and non-alcoholic steatohepatitis (NASH). "
        "Specifically analyze: 1. Glycemic control parameters (HbA1c reductions, fasting plasma glucose). "
        "2. Weight loss percentages across phase 3 trials (SURPASS, STEP programs). "
        "3. Hepatic outcomes including resolution of steatohepatitis without worsening of fibrosis. "
        "4. Gastrointestinal adverse event profiles and treatment discontinuation rates. "
    )
    long_brief = (brief * 10)[:3800]
    assert len(long_brief) == 3800

    res = await deepseek_quick_research(long_brief, depth="low")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert len(upstream.searches) >= 1


def test_stage1_subquery_scaling_across_depth_tiers() -> None:
    """Stage 1: Verify that decomposition allocates scaled subqueries for each depth tier."""
    assert DEPTH_CONFIG["low"].subqueries == 2
    assert DEPTH_CONFIG["medium"].subqueries == 3
    assert DEPTH_CONFIG["high"].subqueries == 5
    assert DEPTH_CONFIG["max"].subqueries == 6


async def test_stage1_decomposition_fallback_on_chat_failure(fake_upstream: Any) -> None:
    """Stage 1: When decomposition LLM fails, pipeline cleanly falls back to raw topic query."""
    upstream = DeepResearchUpstream(fail_decomposition=True)
    fake_upstream(upstream)

    res = await deepseek_quick_research("quantum computing fault tolerance", depth="low")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert len(upstream.searches) >= 1
    # Fallback search used the raw topic
    assert "quantum computing" in upstream.searches[0]["query"]


# ---------------------------------------------------------------------------
# STAGE 2: Multi-Angle Web Search & Deduplication
# ---------------------------------------------------------------------------


async def test_stage2_multi_search_deduplication_and_need_content(fake_upstream: Any) -> None:
    """Stage 2: Verify searches set need_content=True and deduplicate identical URLs across queries."""
    duplicate_results = [
        {
            "title": "Universal Quantum Computing",
            "url": "https://nature.test/quantum-universal",
            "content": "Superconducting qubits reach long coherence times [1].",
        },
        {
            "title": "Quantum Benchmarks",
            "url": "https://nature.test/quantum-universal",  # Exact same URL
            "content": "Duplicate content from query 2.",
        },
        {
            "title": "Surface Codes",
            "url": "https://nature.test/surface-codes",
            "content": "Surface codes achieve fault tolerance [2].",
        },
    ]

    class DedupeUpstream(DeepResearchUpstream):
        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == server.SEARCH_PATH:
                self.searches.append(kwargs.get("payload") or {})
                return {"ok": True, "data": {"results": duplicate_results}}
            return await super().__call__(method, path, **kwargs)

    upstream = DedupeUpstream()
    fake_upstream(upstream)

    res = await deepseek_quick_research("quantum architectures", depth="low")
    assert res["ok"] is True
    # Verify search payload requested full content
    for search in upstream.searches:
        assert search.get("need_content") is True or search.get("content") == "full"
    # In returned sources, URLs must be unique
    urls = [s["url"] for s in res["sources"]]
    assert len(urls) == len(set(urls))


# ---------------------------------------------------------------------------
# STAGE 3: Source Selection, Ranking & Primary Quotas
# ---------------------------------------------------------------------------


def test_stage3_source_selection_respects_scaled_k_and_source_chars() -> None:
    """Stage 3: Verify candidate selection respects the scaled K and total character budgets."""
    candidates = [
        {
            "id": i,
            "title": f"Paper {i}",
            "url": f"https://domain{i}.test/page",
            "content": "Factual evidence text " * 1000,  # ~22,000 chars each
            "rerank_score": 10.0 - (i * 0.1),
        }
        for i in range(1, 50)
    ]

    for depth, expected_k, expected_max_chars in [
        ("low", 8, 80_000),
        ("medium", 12, 240_000),
        ("high", 20, 450_000),
        ("max", 30, 800_000),
    ]:
        selected = select_synthesis_context(candidates, depth=depth)
        assert len(selected) <= expected_k
        total_chars = sum(len(s.get("content") or "") for s in selected)
        assert total_chars <= expected_max_chars


def test_stage3_primary_quota_preserves_official_and_study_sources() -> None:
    """Stage 3: Verify apply_primary_quota guarantees primary source representation."""
    pool = [
        {"id": 1, "title": "News Blog", "url": "https://techcrunch.test/x", "source_type": "news"},
        {"id": 2, "title": "Commentary", "url": "https://medium.test/x", "source_type": "commentary"},
        {"id": 3, "title": "Review", "url": "https://forbes.test/x", "source_type": "review"},
        {"id": 4, "title": "Clinical Trial", "url": "https://nejm.test/trial", "source_type": "rct"},
        {"id": 5, "title": "Official Spec", "url": "https://w3.test/spec", "source_type": "specification"},
    ]
    # Initially selected top 3 without primary
    initial = [pool[0], pool[1], pool[2]]
    swapped_selected, swaps = apply_primary_quota(initial, pool, k_sources=3)
    primary_count = sum(1 for s in swapped_selected if s.get("source_type") in {"rct", "specification"})
    assert primary_count >= 1
    assert swaps > 0


# ---------------------------------------------------------------------------
# STAGE 4: Token Guard & Context Assembly (1M Window Modernization)
# ---------------------------------------------------------------------------


def test_stage4_token_guard_respects_1m_token_limit() -> None:
    """Stage 4: Verify apply_token_guard enforces the 1M token synthesis ceiling."""
    assert MAX_EST_SYNTHESIS_TOKENS == 1_000_000

    # 5 massive sources totaling 2,500,000 characters (~1,250,000 estimated tokens)
    huge_sources = [{"id": i, "title": f"Huge Source {i}", "content": "x" * 500_000} for i in range(1, 6)]
    guarded, notes = apply_token_guard(huge_sources, max_tokens=1_000_000)

    # Must trim down to <= 2,000,000 chars (1M tokens)
    total_chars = sum(len(s.get("content") or "") for s in guarded)
    assert total_chars <= 2_000_000
    assert len(guarded) < len(huge_sources)
    assert any("Trimmed source 'Huge Source 5'" in n for n in notes)


def test_stage4_context_assembly_structure_and_continuation_markers() -> None:
    """Stage 4: Verify build_synthesis_messages builds source index and continuation markers."""
    sources = [
        {
            "id": 1,
            "title": "Large Study",
            "url": "https://journal.test/large",
            "date": "2026-05-10",
            "source_type": "study",
            "content": "Section data " * 1500,  # exceeds single block chunk
        }
    ]
    system, user = build_synthesis_messages("Test Topic", sources, depth="medium")
    assert "[1] Large Study" in user["content"]
    assert "Source index:" in user["content"]
    assert "[source 1, continued]" in user["content"]


# ---------------------------------------------------------------------------
# STAGE 5: Synthesis, Prompt Rules, Filtering & Citation Verification
# ---------------------------------------------------------------------------


def test_stage5_system_prompt_rules_and_tier_targets() -> None:
    """Stage 5: Verify build_synthesis_system_prompt sets expected targets across all 4 tiers."""
    for depth in ("low", "medium", "high", "max"):
        prompt = build_synthesis_system_prompt(depth)
        cfg = DEPTH_CONFIG[depth]
        n = SECTION_COUNTS[depth]
        per_sec = cfg.target_words // n
        assert f"{n} sections" in prompt
        assert f"about {per_sec} words" in prompt
        assert f"target is approximately {cfg.target_words} words" in prompt
        assert f"hard ceiling of {cfg.max_words} words" in prompt
        assert "CITATIONS:" in prompt
        assert "COVERAGE BEFORE STYLE:" in prompt


def test_stage5_filter_synthesis_output_security_sanitization() -> None:
    """Stage 5: Verify filter_synthesis_output neutralizes raw HTML and unallowed media."""
    malicious_report = (
        "# Results\n\n"
        "Key trial achieved 98.5% efficacy [1].\n\n"
        "![Phishing chart](https://evil.test/badge.png)\n\n"
        "<script>window.location='https://attacker.test'</script>\n\n"
        "Details in table [2]."
    )
    cleaned, suspected, notes = filter_synthesis_output(
        malicious_report, allowed_urls={"https://journal.test"}
    )
    assert "<script>" not in cleaned
    assert "![Phishing chart]" not in cleaned
    assert "[image: Phishing chart]" in cleaned
    assert "Key trial achieved 98.5% efficacy [1]" in cleaned


def test_stage5_finalization_cleans_hallucinated_citations_and_verifies_numbers() -> None:
    """Stage 5: Verify citation normalization removes hallucinated citations and grounds numbers."""
    raw_text = (
        "Efficacy reached 94.2% in primary trial [1]. "
        "Fabricated figure of 999.9% was incorrectly attributed [1]. "
        "Unrelated note cited [99]."
    )
    sources = [{"id": 1, "url": "https://a.test", "content": "Efficacy reached 94.2% in primary trial."}]
    cleaned = deep_research.normalize_citations(raw_text, max_id=1)
    assert "[1]" in cleaned
    assert "[99]" not in cleaned

    grounded, unverified, count, notes = deep_research.ground_report_numbers(cleaned, sources, drop=False)
    assert "[1]" in grounded
    # 999.9% is cited to [1] but absent from source 1 content, so it gets flagged
    assert count >= 1
    assert any(item.get("number") == "999.9%" for item in unverified)


async def test_stage5_overlength_report_retained_in_full_without_condensation(fake_upstream: Any) -> None:
    """Stage 5: Over-length report is retained in full with a note (ADR 0011)."""
    # 1,500 words on depth low (max ceiling is 1,200)
    overlength_text = "Factual research claim with confirmed data point [1]. " * 200
    upstream = DeepResearchUpstream(report_text=overlength_text)
    fake_upstream(upstream)

    res = await deepseek_quick_research("overlength test", depth="low")
    assert res["ok"] is True
    # Whole report is preserved
    assert len(res["report"].split()) > DEPTH_CONFIG["low"].max_words
    assert any("over the 1200-word ceiling" in note for note in res["notes"])


# ---------------------------------------------------------------------------
# STAGE 6: Deep Agent Research Multi-Round Execution & Scaling
# ---------------------------------------------------------------------------


class FiveRoundUpstream(DeepResearchUpstream):
    """Upstream simulating an iterative agent loop that produces new facts across multiple rounds."""

    def __init__(self) -> None:
        super().__init__()
        self.round_idx = 0

    async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload") or {}
        if path == server.SEARCH_PATH:
            self.round_idx += 1
            self.searches.append(payload)
            # Each round provides distinct numeric facts
            content_line = f"Round {self.round_idx} parameter measured {100 + self.round_idx}% in trials. "
            return {
                "ok": True,
                "data": {
                    "results": [
                        {
                            "title": f"Round {self.round_idx} Findings",
                            "url": f"https://science.test/r{self.round_idx}",
                            "content": content_line * 5,
                        }
                    ]
                },
            }
        if path == "/v1/chat/completions":
            self.chats.append(payload)
            # Plan call or section synthesis
            if len(self.chats) == 1:
                plan = {
                    "aspects": [
                        {
                            "kind": "numbers",
                            "aspect": f"metric {i}",
                            "looking_for": "percent",
                            "query": f"query {i}",
                        }
                        for i in range(1, 6)
                    ]
                }
                choice = {"message": {"role": "assistant", "content": json.dumps(plan)}}
                return {"ok": True, "data": {"choices": [choice]}}
            msg = f"Measured parameter in round {self.round_idx} [1]."
            return {"ok": True, "data": {"choices": [{"message": {"role": "assistant", "content": msg}}]}}
        return await super().__call__(method, path, **kwargs)


async def test_stage6_agent_research_five_rounds_on_max(fake_upstream: Any) -> None:
    """Stage 6: Verify deepseek_deep_agent_research scales up to 5 rounds on depth 'max'."""
    assert DEPTH_ROUNDS["max"] == 5

    upstream = FiveRoundUpstream()
    fake_upstream(upstream)

    res = await deepseek_deep_agent_research("multi round physics investigation", depth="max")
    assert res["ok"] is True
    assert res["status"] == "done"
    assert "rounds" in res
    # Verified that agent executed multiple iterative rounds
    assert len(res["rounds"]) >= 2
    assert "facts_per_round" in res
    assert "stopped_reason" in res


async def test_stage6_agent_research_saturation_early_stopping(fake_upstream: Any) -> None:
    """Stage 6: Verify agent stops early when a round yields 0 new numeric facts."""

    class SaturatedUpstream(DeepResearchUpstream):
        def __init__(self) -> None:
            super().__init__()
            self.search_count = 0

        async def __call__(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if path == server.SEARCH_PATH:
                self.search_count += 1
                self.searches.append(kwargs.get("payload") or {})
                if self.search_count == 1:
                    return {
                        "ok": True,
                        "data": {
                            "results": [
                                {
                                    "title": "Initial Findings",
                                    "url": "https://science.test/r1",
                                    "content": "Metric was 95% in 2026. " * 5,
                                }
                            ]
                        },
                    }
                # Round 2 returns no new numbers or duplicate text
                return {
                    "ok": True,
                    "data": {
                        "results": [
                            {
                                "title": "Stale repeat",
                                "url": "https://science.test/r1",
                                "content": "Metric was 95% in 2026. " * 5,
                            }
                        ]
                    },
                }
            if path == "/v1/chat/completions":
                self.chats.append(kwargs.get("payload") or {})
                if len(self.chats) == 1:
                    plan = {
                        "aspects": [
                            {"kind": "numbers", "aspect": "m1", "looking_for": "%", "query": "q1"},
                            {"kind": "numbers", "aspect": "m2", "looking_for": "%", "query": "q2"},
                        ]
                    }
                    choice = {"message": {"role": "assistant", "content": json.dumps(plan)}}
                    return {"ok": True, "data": {"choices": [choice]}}
                synth_msg = {"message": {"role": "assistant", "content": "Synthesized report [1]."}}
                return {"ok": True, "data": {"choices": [synth_msg]}}
            return await super().__call__(method, path, **kwargs)

    fake_upstream(SaturatedUpstream())
    res = await deepseek_deep_agent_research("early saturation test", depth="high")
    assert res["ok"] is True
    assert res["stopped_reason"] in {"saturated", "no_gaps"}
    assert res["facts_per_round"][-1] == 0


# ---------------------------------------------------------------------------
# STAGE 7: Client Synthesis Context Paging & 120k Character Report Verification
# ---------------------------------------------------------------------------


async def test_stage7_client_synthesis_workflow_and_120k_limit(fake_upstream: Any) -> None:
    """Stage 7: Verify client synthesis workflow, paging, and the expanded MAX_CLIENT_REPORT_CHARS=120k."""
    assert MAX_CLIENT_REPORT_CHARS == 120_000

    upstream = DeepResearchUpstream()
    fake_upstream(upstream)

    # 1. Start job in client mode
    start_res = await deepseek_quick_research("client synthesis topic", depth="low", synthesis="client")
    assert start_res["ok"] is True
    assert start_res["status"] == "needs_input"
    assert start_res["step"] == "report"
    job_id = start_res["job_id"]
    assert "context" in start_res
    assert len(start_res["context"]) >= 1

    # 2. Submit report exceeding 120,000 characters -> rejected
    oversized = "x" * 120_001
    bad_check = await deepseek_check_research(job_id=job_id, answer=oversized)
    assert bad_check["ok"] is False
    assert bad_check["error"]["reason"] == "bad_request"
    assert "at most 120000 characters" in bad_check["error"]["message"]

    # 3. Submit rich, valid client report (e.g. 10,000 chars)
    valid_client_report = (
        "## Executive Summary\n\n"
        "Fault tolerance is established via surface codes [1]. "
        "Experimental gate fidelity achieves 99.9% [2].\n\n"
        + ("Detailed factual analysis of the protocols and results [1].\n" * 100)
    )
    verified = await deepseek_check_research(job_id=job_id, answer=valid_client_report)
    assert verified["ok"] is True
    assert verified["status"] == "done"
    assert "[1]" in verified["report"]
    assert verified["cost_usd"] >= 0.0
