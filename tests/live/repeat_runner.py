"""Repeat runner helper for live_llm evaluations and flaky-sensitive assertions.

Executes a callable ``repeats`` times; all runs must pass without error. The default comes from
the ``LIVE_REPEATS`` environment variable (1 when unset), so CI pays for one run and a manual
quality check can ask for more (``LIVE_REPEATS=3 pytest -m live_llm``).
Provides a helper callable (:func:`run_repeated`) and a test function decorator (:func:`repeat_llm`).
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Awaitable, Callable
from typing import Any


def default_repeats() -> int:
    """Return ``LIVE_REPEATS`` as an int, or 1 when it is unset or empty."""
    return int(os.environ.get("LIVE_REPEATS") or 1)


async def run_repeated[T](
    func: Callable[..., Awaitable[T] | T],
    *args: Any,
    repeats: int | None = None,
    **kwargs: Any,
) -> list[T]:
    """Execute ``func(*args, **kwargs)`` ``repeats`` times; all iterations must succeed.

    Returns the list of return values from each run. Raises immediately on any exception.
    """
    if repeats is None:
        repeats = default_repeats()
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    results: list[T] = []
    for _ in range(repeats):
        from deepseek_search_everywhere_mcp import server

        server.reset_state()
        res = func(*args, **kwargs)
        if inspect.isawaitable(res):
            res = await res
        results.append(res)
    return results


def repeat_llm(
    repeats: int | None = None,
) -> Callable[[Callable[..., Awaitable[Any] | Any]], Callable[..., Awaitable[list[Any]]]]:
    """Test function decorator running the wrapped test ``repeats`` times (all must pass)."""

    def decorator(fn: Callable[..., Awaitable[Any] | Any]) -> Callable[..., Awaitable[list[Any]]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> list[Any]:
            return await run_repeated(fn, *args, repeats=repeats, **kwargs)

        return wrapper

    return decorator
