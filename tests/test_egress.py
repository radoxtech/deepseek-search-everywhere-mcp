"""Tests for Egress Guard (S2, S10): blocking secret exfiltration and allow/deny lists."""

from __future__ import annotations

import json
import pathlib
import urllib.parse

import pytest

from deepseek_search_everywhere_mcp.server import (
    deepseek_fetch,
    deepseek_fetch_many,
    deepseek_search,
    deepseek_search_many,
)
from deepseek_search_everywhere_mcp.settings import Settings, set_settings

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "injection"


@pytest.fixture
def egress_secrets() -> dict[str, str]:
    return json.loads((FIXTURES_DIR / "egress_secrets.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_field",
    ["sk_key", "bearer_token", "api_key_query", "jwt_token", "aws_key"],
)
async def test_search_egress_blocks_secrets(egress_secrets: dict[str, str], secret_field: str) -> None:
    secret = egress_secrets[secret_field]
    query = f"what is {secret} in python?"
    res = await deepseek_search(query)
    assert res["ok"] is False
    assert res["error"]["code"] == "egress_blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_field",
    ["sk_key", "bearer_token", "api_key_query", "jwt_token", "aws_key"],
)
async def test_fetch_egress_blocks_secrets(egress_secrets: dict[str, str], secret_field: str) -> None:
    secret = egress_secrets[secret_field]
    url = f"https://example.com/api?k={urllib.parse.quote(secret)}"
    res = await deepseek_fetch(url)
    assert res["ok"] is False
    assert res["error"]["code"] == "egress_blocked"


@pytest.mark.asyncio
async def test_fetch_egress_allow_domains() -> None:
    # Set allowlist to only allow example.com and docs.python.org
    custom = Settings(fetch_allow_domains=("example.com", "docs.python.org"))
    set_settings(custom)
    try:
        # Denied domain
        res_denied = await deepseek_fetch("https://attacker.evil/data")
        assert res_denied["ok"] is False
        assert res_denied["error"]["code"] == "egress_blocked"
        assert "fetch_allow_domains" in res_denied["error"]["message"]

        # Subdomain of allowed domain passes egress check
        # (It will fail later on network if not mocked, but not with egress_blocked)
        res_allowed = await deepseek_fetch("https://sub.example.com/test")
        if not res_allowed.get("ok"):
            assert res_allowed["error"]["code"] != "egress_blocked"
    finally:
        set_settings(Settings())


@pytest.mark.asyncio
async def test_fetch_egress_deny_domains() -> None:
    # Set denylist to block evil.org
    custom = Settings(fetch_deny_domains=("evil.org",))
    set_settings(custom)
    try:
        res_denied = await deepseek_fetch("https://sub.evil.org/exfil")
        assert res_denied["ok"] is False
        assert res_denied["error"]["code"] == "egress_blocked"
        assert "fetch_deny_domains" in res_denied["error"]["message"]

        res_allowed = await deepseek_fetch("https://example.com/ok")
        if not res_allowed.get("ok"):
            assert res_allowed["error"]["code"] != "egress_blocked"
    finally:
        set_settings(Settings())


@pytest.mark.asyncio
async def test_egress_entropy_check_opt_in(egress_secrets: dict[str, str]) -> None:
    high_entropy = egress_secrets["high_entropy_token"]
    url = f"https://example.com/endpoint?v={high_entropy}"

    # Default: entropy check disabled
    set_settings(Settings(egress_entropy_check=False))
    res_default = await deepseek_fetch(url)
    if not res_default.get("ok"):
        assert res_default["error"]["code"] != "egress_blocked"

    # Enabled: entropy check blocks the high-entropy token
    set_settings(Settings(egress_entropy_check=True))
    try:
        res_blocked = await deepseek_fetch(url)
        assert res_blocked["ok"] is False
        assert res_blocked["error"]["code"] == "egress_blocked"
        assert "high-entropy" in res_blocked["error"]["message"]
    finally:
        set_settings(Settings())


@pytest.mark.asyncio
async def test_batch_tools_egress_blocked(egress_secrets: dict[str, str]) -> None:
    secret = egress_secrets["sk_key"]

    # deepseek_search_many with one secret query
    res_search = await deepseek_search_many([f"find {secret}", "normal query"])
    assert res_search["ok"] is True
    assert res_search["results"][0]["ok"] is False
    assert res_search["results"][0]["error"]["code"] == "egress_blocked"

    # deepseek_fetch_many with one secret URL
    res_fetch = await deepseek_fetch_many([f"https://example.com/p?k={secret}"])
    assert res_fetch["ok"] is True
    assert res_fetch["results"][0]["ok"] is False
    assert res_fetch["results"][0]["error"]["code"] == "egress_blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_variant",
    [
        "SK-1234567890abcdef",
        "Sk-abcdef1234567890",
        "BEARER abcdef1234567890abcdef",
        "bEaReR abcdef1234567890abcdef",
        "akia1234567890abcdef",
        "Akia1234567890abcdef",
        "ASIA1234567890ABCDEF",
        "asia1234567890abcdef",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "API_KEY=supersecret123",
        "Token=supersecret123",
    ],
)
async def test_egress_blocks_case_insensitive_secrets(secret_variant: str) -> None:
    res_search = await deepseek_search(f"check {secret_variant} status")
    assert res_search["ok"] is False
    assert res_search["error"]["code"] == "egress_blocked"

    res_fetch = await deepseek_fetch(f"https://example.com/api?q={urllib.parse.quote(secret_variant)}")
    assert res_fetch["ok"] is False
    assert res_fetch["error"]["code"] == "egress_blocked"
