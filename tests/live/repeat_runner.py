"""Repeat runner helper for live_llm evaluations and flaky-sensitive assertions.

Executes a callable ``repeats`` times (default 3); all runs must pass without error.
Provides a helper callable (:func:`run_repeated`) and a test function decorator (:func:`repeat_llm`).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def run_repeated(
    func: Callable[..., Awaitable[T] | T],
    *args: Any,
    repeats: int = 3,
    **kwargs: Any,
) -> list[T]:
    """Execute ``func(*args, **kwargs)`` ``repeats`` times; all iterations must succeed.

    Returns the list of return values from each run. Raises immediately on any exception.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    results: list[T] = []
    for _ in range(repeats):
        res = func(*args, **kwargs)
        if inspect.isawaitable(res):
            res = await res
        results.append(res)
    return results


def repeat_llm(
    repeats: int = 3,
) -> Callable[[Callable[..., Awaitable[Any] | Any]], Callable[..., Awaitable[list[Any]]]]:
    """Test function decorator running the wrapped test ``repeats`` times (all must pass)."""

    def decorator(fn: Callable[..., Awaitable[Any] | Any]) -> Callable[..., Awaitable[list[Any]]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> list[Any]:
            return await run_repeated(fn, *args, repeats=repeats, **kwargs)

        return wrapper

    return decorator
