"""Shared test fixtures: import path, environment isolation and runtime-state reset."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from collections.abc import Callable, Iterator

import pytest

if importlib.util.find_spec("deepseek_search_everywhere_mcp") is None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

PACKAGE = "deepseek_search_everywhere_mcp"

# Modules that may own process-wide state (cache entries, in-flight maps, rate-limit
# buckets, the rewrite cache). Each one opts in by defining a ``reset_state()`` function;
# modules that do not exist yet are skipped.
STATEFUL_MODULES = ("cache", "ratelimit", "server")


def _reset_runtime_state_now() -> None:
    for name in STATEFUL_MODULES:
        qualified = f"{PACKAGE}.{name}"
        module = sys.modules.get(qualified)
        if module is None:
            if importlib.util.find_spec(qualified) is None:
                continue
            module = importlib.import_module(qualified)
        reset = getattr(module, "reset_state", None)
        if callable(reset):
            reset()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Strip real credentials and endpoints so no test can reach the network by accident."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(tmp_path / "absent-auth.json"))
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.delenv("MCP_BEARER_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _reset_runtime_state() -> Iterator[Callable[[], None]]:
    """Reset module-level state before and after every test.

    Yields the reset callable so a test can trigger an extra reset explicitly.
    """
    _reset_runtime_state_now()
    yield _reset_runtime_state_now
    _reset_runtime_state_now()


class FakeClock:
    """Deterministic stand-in for ``time.monotonic`` that only moves when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a monotonic clock cannot go backwards")
        self._now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    """A fresh clock starting at 0.0 for cache and rate-limiter tests (no ``time.sleep``)."""
    return FakeClock()
