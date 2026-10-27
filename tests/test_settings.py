"""Tests for the settings module: defaults, validation and the two environment switches."""

from __future__ import annotations

import dataclasses

import pytest

from deepseek_search_everywhere_mcp import settings as settings_module
from deepseek_search_everywhere_mcp.settings import Settings, get_settings, set_settings


def test_defaults_match_the_documented_constants() -> None:
    s = Settings()
    assert (s.search_qps, s.fetch_qps, s.cache_fetch_ttl, s.response_cap_fetch) == (10, 5, 1800, 33554432)
    assert (s.search_burst, s.fetch_burst, s.limiter_max_wait) == (20, 10, 5)
    assert (s.cache_search_ttl, s.cache_search_entries, s.cache_search_bytes) == (300, 128, 16 * 1024**2)
    assert (s.cache_rewrite_ttl, s.cache_rewrite_entries, s.cache_rewrite_bytes) == (86400, 1024, 1024**2)
    assert (s.batch_max_urls, s.batch_concurrency, s.batch_deadline) == (10, 5, 120)
    assert (s.research_max_pages, s.research_page_chars, s.research_total_chars) == (5, 6000, 30000)
    assert (s.http_host, s.http_port, s.http_path) == ("127.0.0.1", 8000, "/mcp")
    assert s.cache_enabled is True
    assert s.log_level == "INFO"


def test_constructor_overrides_and_frozen() -> None:
    s = Settings(fetch_qps=2)
    assert s.fetch_qps == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.fetch_qps = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("search_qps", 0),
        ("fetch_qps", -1),
        ("cache_fetch_ttl", -1),
        ("response_cap_search", 0),
        ("cache_search_bytes", 0),
        ("batch_concurrency", 0),
        ("batch_concurrency", 11),
        ("limiter_max_wait", 0),
        ("http_port", 70000),
        ("http_path", "mcp"),
        ("log_level", "LOUD"),
    ],
)
def test_invalid_values_name_the_field(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_zero_ttl_is_allowed() -> None:
    assert Settings(cache_search_ttl=0).cache_search_ttl == 0


def test_from_env_cache_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings.from_env().cache_enabled is True
    monkeypatch.setenv("DEEPSEEK_MCP_CACHE", "0")
    assert Settings.from_env().cache_enabled is False
    monkeypatch.setenv("DEEPSEEK_MCP_CACHE", "1")
    assert Settings.from_env().cache_enabled is True
    monkeypatch.setenv("DEEPSEEK_MCP_CACHE", "maybe")
    with pytest.raises(ValueError, match="cache_enabled"):
        Settings.from_env()


def test_from_env_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MCP_LOG_LEVEL", "debug")
    assert Settings.from_env().log_level == "DEBUG"


def test_other_env_knobs_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "DEEPSEEK_MCP_SEARCH_QPS": "7",
        "DEEPSEEK_MCP_FETCH_TTL": "1",
        "DEEPSEEK_MCP_BATCH_CONCURRENCY": "9",
        "FASTMCP_PORT": "9999",
    }.items():
        monkeypatch.setenv(name, value)
    assert Settings.from_env() == Settings()


def test_get_settings_can_be_swapped_and_reset() -> None:
    custom = Settings(search_qps=1)
    set_settings(custom)
    assert get_settings() is custom
    settings_module.reset_state()
    assert get_settings() is not custom
    assert get_settings() == Settings()
