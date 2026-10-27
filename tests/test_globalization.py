"""Language neutrality: no hardcoded source language in prompts, notes or defaults."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import server

NON_ENGLISH_QUERIES = (
    "popołudniowy spadek energii",
    "cómo mejorar el sueño",
    "睡眠の質",
)


def _search(
    fake_upstream: Any, query: str, lang: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    upstream = fake_upstream(default={"ok": True, "data": {"results": []}})
    result = asyncio.run(server._search(query, 5, None, False, lang, 1000, 4000))
    return result, upstream.calls


def test_rewrite_prompt_is_language_agnostic() -> None:
    assert "Translate the query to English" in server.SYSTEM_PROMPT
    assert "Polish" not in server.SYSTEM_PROMPT
    assert "120 characters" in server.SYSTEM_PROMPT
    assert "site:" in server.SYSTEM_PROMPT


@pytest.mark.parametrize("query", NON_ENGLISH_QUERIES)
def test_no_language_specific_note(fake_upstream: Any, query: str) -> None:
    result, calls = _search(fake_upstream, query)
    assert result["ok"] is True
    assert result["notes"] == []
    assert result["query_used"] == query
    assert calls[0]["payload"]["query"] == query
    assert not any("lang:" in note or "Polish" in note for note in result["notes"])


def test_no_language_specific_operator_without_lang(fake_upstream: Any) -> None:
    _, calls = _search(fake_upstream, "popołudniowy spadek energii")
    assert "lang:" not in calls[0]["payload"]["query"]


def test_with_lang_still_accepts_pl_code() -> None:
    used, trimmed = server._with_lang("prokrastynacja", "pl")
    assert used == "lang:pl prokrastynacja"
    assert trimmed is False


def test_bad_lang_example_is_language_neutral(fake_upstream: Any) -> None:
    result, _ = _search(fake_upstream, "test", lang="polish")
    assert result["ok"] is False
    assert "'es' or 'pt-BR'" in result["error"]["message"]
