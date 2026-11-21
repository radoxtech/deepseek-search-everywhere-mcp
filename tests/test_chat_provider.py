"""The chat host is selected from the environment alone, and search never follows it.

A server that borrows another tool's config directory behaves differently for every user and cannot
be audited from its own settings, so nothing here may read a path the user did not name.
"""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp import transport
from deepseek_search_everywhere_mcp.settings import DEEPINFRA_PRICING, active_chat_pricing


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_CHAT_BASE_URL",
        "DEEPSEEK_CHAT_API_KEY",
        "DEEPSEEK_CHAT_MODEL",
        "DEEPSEEK_CHAT_TIER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_chat_follows_search_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert transport._chat_base_url() == transport._base_url()
    assert transport._chat_api_key() == "sk-deepseek"
    assert transport._chat_model("deepseek-flash") == "deepseek-flash"


def test_chat_host_is_selected_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("DEEPSEEK_CHAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setenv("DEEPSEEK_CHAT_API_KEY", "sk-deepinfra")
    monkeypatch.setenv("DEEPSEEK_CHAT_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash-1017")

    assert transport._chat_base_url() == "https://api.deepinfra.com/v1/openai"
    assert transport._chat_api_key() == "sk-deepinfra"
    assert transport._chat_model("deepseek-flash") == "deepseek-ai/DeepSeek-V4.1-Flash-1017"

    # Search keeps its own host and key: the search API exists only on the DeepSeek endpoint.
    assert transport._base_url() == "https://api.deepseek.com"


def test_openai_sdk_style_base_does_not_double_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    assert transport._chat_path("/v1/chat/completions", "https://api.deepseek.com") == "/v1/chat/completions"
    assert (
        transport._chat_path("/v1/chat/completions", "https://api.deepinfra.com/v1/openai")
        == "/chat/completions"
    )


def test_tier_header_only_for_a_tiered_host(monkeypatch: pytest.MonkeyPatch) -> None:
    assert transport._chat_headers("https://api.deepseek.com") == {}
    monkeypatch.setenv("DEEPSEEK_CHAT_TIER", "default")
    assert transport._chat_headers("https://api.deepinfra.com/v1/openai") == {
        "x-deepinfra-service-tier": "default"
    }


def test_a_third_party_chat_host_prices_at_its_own_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert active_chat_pricing().flat is None
    monkeypatch.setenv("DEEPSEEK_CHAT_BASE_URL", "https://api.deepinfra.com/v1/openai")
    assert active_chat_pricing().flat == DEEPINFRA_PRICING.flat


def test_no_credential_is_read_from_another_tools_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert transport._api_key() == ""
