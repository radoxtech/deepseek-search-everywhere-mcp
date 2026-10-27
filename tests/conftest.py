"""Shared test fixtures: import path, environment isolation and runtime-state reset."""

from __future__ import annotations

import importlib
import importlib.util
import os
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
STATEFUL_MODULES = ("cache", "ratelimit", "server", "settings")

# The real key, captured once at import time before any fixture strips the environment.
# Only tests marked ``live`` get it back, through the ``live_api_key`` fixture.
_LIVE_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()


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
    monkeypatch.delenv("DEEPSEEK_MCP_CACHE", raising=False)
    monkeypatch.delenv("DEEPSEEK_MCP_LOG_LEVEL", raising=False)


@pytest.fixture
def live_api_key(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                 _isolated_env: None) -> None:
    """Re-expose the captured ``DEEPSEEK_API_KEY`` to a ``live`` test, or skip it without one.

    Depends on ``_isolated_env`` so the key is restored after the environment is stripped.
    The key itself is never returned, so it cannot leak into assertion output.
    """
    if request.node.get_closest_marker("live") is None:
        pytest.fail("live_api_key is reserved for tests marked 'live'.", pytrace=False)
    if not _LIVE_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not set; live tests need a real DeepSeek key.")
    monkeypatch.setenv("DEEPSEEK_API_KEY", _LIVE_API_KEY)


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
