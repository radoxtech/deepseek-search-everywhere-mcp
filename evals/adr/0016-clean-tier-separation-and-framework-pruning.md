# ADR 0016 — Non-Overlapping Research Tiers, Pruning of Abandoned Frameworks, and Entity-Anchored Gap Detection

- **Status:** accepted
- **Date:** 2026-11-21
- **Evidence:** Experiments `E33` and `E34` in `evals/EXPERIMENTS.md`; isolated synthesis ablation on 104 atoms; empirical benchmark analysis across 388 rubric atoms.
- **Related:** `0009-agent-framework.md`, `0010-tool-surface-pruning-and-taxonomy.md`, `0015-quick-vs-deep-agent-research-and-grounding.md`

## Context

Following the initial deployment of `deepseek_quick_research` and `deepseek_deep_agent_research`, an architectural and user-experience audit revealed severe cognitive overload and semantic contradiction in tool selection:

1. **Semantic Inversion Paradox:**
   - `deepseek_quick_research` at `depth="max"` executed an untracked Round 2 gap search taking **78.6s and \$0.0318** for **59.2% atom coverage**—adding only **+0.8 pp** over `high` (34.8s, \$0.0105, 58.4%) at +126% wall time and 3x cost.
   - Concurrently, `deepseek_deep_agent_research` at `depth="low"` executed a 1-round baseline (36.2s, 47.3% atoms) with zero iterative exploration, underperforming `quick(medium)` (24.5s, 53.1% atoms).
   - A tool named "quick" was 2.5x slower and 4x more expensive than a tool named "deep agent".
2. **Abandoned Framework Bloat:**
   - Early prototyping under ADR 0009 explored Vercel `ai` and `httpx2`. In production, all tools were implemented on native `httpx` and `runtime._chat_request`.
   - Four prototype files (`ai_client.py`, `agent_loop.py`, `structured.py`, `agent_roles.py`) and two large dependencies (`httpx2>=2.16.2`, `ai[openai]>=0.10.1`) remained in the repository, supported solely by 8 dedicated unit tests.
3. **Entity Drift in Multi-Round Agent:**
   - In contentious topics such as GLP-1 and solid-state batteries, Round 2 queries generated generic keyword variants rather than drilling into specific named clinical trial codes (`STEP`, `SURMOUNT`, `SELECT`) or standard identifiers (`ISO`, `RFC`) discovered in Round 1.

## Decisions

1. **Strictly Non-Overlapping Depth Tiers:**
   - **`deepseek_quick_research`** is strictly single-pass ($\le 35$s). Removed `_research_round_two`. Supported tiers: `low` (~10–12s, \$0.003), `medium` (default, ~20–25s, \$0.006), `high` (~30–35s, \$0.010). If legacy `"max"` is supplied, it is safely mapped to `high`.
   - **`deepseek_deep_agent_research`** is strictly multi-round ($\ge 55$s, minimum 2 rounds). Supported tiers: `medium` (default, 2 rounds, ~55–65s, \$0.011–\$0.014), `high` (up to 3 rounds, ~80–100s, \$0.018–\$0.022), `max` (up to 5 rounds, ~120–150s, \$0.028–\$0.035). If legacy `"low"` is supplied, it is safely mapped to `medium`.
2. **Entity-Anchored Gap Detection:**
   - In `agent_research.py`, `_extract_discovered_entities` extracts named trial codes, statute numbers, standard identifiers, and versioned models from Round 1 summaries.
   - `_gap_queries_model` injects these anchors into the gap prompt, ensuring Round 2 queries target specific endpoints rather than generic marketing terms.
3. **Strict Grounding in Agent Synthesis:**
   - `REPORT_ASK` in `agent_research.py` enforces Rule 3 strict grounding: state only facts and numbers explicitly stated in cited passages; never extrapolate or add outside claims.
4. **Pruning Dead Code and Dependencies:**
   - Deleted `ai_client.py`, `agent_loop.py`, `structured.py`, `agent_roles.py` (migrated `REPORT_PLAN_PROMPT` into `agent_research.py`).
   - Removed `httpx2` and `ai[openai]` from `pyproject.toml`.
   - Removed `test_ai_client.py` and `test_agent_orchestration.py`.

## Consequences

- Zero cognitive overload: Quick is always fast and cheap ($\le 35$s, $\le \$0.010$); Agent is always iterative and investigative ($\ge 55$s, $\ge \$0.011$).
- Package footprint reduced, eliminating 2 heavy external dependencies.
- 100% of 1263 test cases pass offline.
