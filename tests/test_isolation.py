"""Guards for the autouse fixtures in ``conftest.py``."""

import os
import sys
import types

import pytest

from deepseek_search_everywhere_mcp import server


def test_environment_is_isolated():
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert "MCP_BEARER_TOKEN" not in os.environ
    assert not os.path.exists(os.environ["DEEPSEEK_AUTH_JSON"])
    assert server._base_url() == "https://api.deepseek.com"
    assert server._api_key() == ""


def test_runtime_state_reset_calls_module_hooks(monkeypatch, fake_clock, _reset_runtime_state):
    calls = []
    stub = types.ModuleType("deepseek_search_everywhere_mcp.cache")
    stub.reset_state = lambda: calls.append("cache")
    monkeypatch.setitem(sys.modules, "deepseek_search_everywhere_mcp.cache", stub)

    _reset_runtime_state()
    assert calls == ["cache"]

    assert fake_clock.monotonic() == 0.0
    fake_clock.advance(2.5)
    assert fake_clock.monotonic() == 2.5
    with pytest.raises(ValueError):
        fake_clock.advance(-1)
