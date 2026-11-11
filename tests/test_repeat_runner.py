"""Offline unit tests for the repeat runner helper used in live_llm evaluation suites."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure tests/live is on sys.path so repeat_runner can be imported offline
LIVE_DIR = Path(__file__).resolve().parent / "live"
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))

from repeat_runner import repeat_llm, run_repeated  # noqa: E402


async def test_run_repeated_executes_async_exact_times() -> None:
    calls = 0

    async def sample() -> int:
        nonlocal calls
        calls += 1
        return calls

    results = await run_repeated(sample, repeats=3)
    assert calls == 3
    assert results == [1, 2, 3]


async def test_run_repeated_executes_sync_callable() -> None:
    calls = 0

    def sample() -> str:
        nonlocal calls
        calls += 1
        return f"run-{calls}"

    results = await run_repeated(sample, repeats=3)
    assert calls == 3
    assert results == ["run-1", "run-2", "run-3"]


async def test_run_repeated_passes_args_and_kwargs() -> None:
    seen: list[tuple[Any, Any]] = []

    async def sample(a: int, *, key: str) -> None:
        seen.append((a, key))

    await run_repeated(sample, 42, repeats=2, key="val")
    assert seen == [(42, "val"), (42, "val")]


async def test_run_repeated_fails_on_any_error() -> None:
    calls = 0

    async def flaky() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("iteration 2 failed")

    with pytest.raises(RuntimeError, match="iteration 2 failed"):
        await run_repeated(flaky, repeats=3)

    assert calls == 2


async def test_run_repeated_rejects_zero_or_negative_repeats() -> None:
    async def noop() -> None:
        pass

    with pytest.raises(ValueError, match="repeats must be at least 1"):
        await run_repeated(noop, repeats=0)

    with pytest.raises(ValueError, match="repeats must be at least 1"):
        await run_repeated(noop, repeats=-1)


async def test_repeat_llm_decorator() -> None:
    runs = 0

    @repeat_llm(repeats=3)
    async def sample_test() -> int:
        nonlocal runs
        runs += 1
        return runs

    res = await sample_test()
    assert runs == 3
    assert res == [1, 2, 3]
