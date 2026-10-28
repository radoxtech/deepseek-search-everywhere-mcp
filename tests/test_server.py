import asyncio
import json
import logging

import pytest

from deepseek_search_everywhere_mcp import server


def test_lang_prefix_added():
    used, trimmed = server._with_lang("prokrastynacja", "pl")
    assert used == "lang:pl prokrastynacja"
    assert trimmed is False


def test_lang_not_duplicated():
    used, _ = server._with_lang("lang:en habit formation", "pl")
    assert used == "lang:en habit formation"


def test_lang_prefix_trims_long_query():
    query = "a" * 128
    used, trimmed = server._with_lang(query, "pl")
    assert trimmed is True
    assert len(used) == server.MAX_QUERY
    assert used.startswith("lang:pl ")


def test_unescape_markdown():
    raw = "\\# Nagłówek\nOdkładasz zadania\\, choć wiesz\\."
    assert server._unescape_markdown(raw) == "# Nagłówek\nOdkładasz zadania, choć wiesz."


def test_unescape_reduces_escaped_backslash():
    assert server._unescape_markdown("a\\\\b") == "a\\b"
    assert server._unescape_markdown("C:\\\\Users") == "C:\\Users"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "JSON \\(JavaScript Object Notation\\) is a format",
            "JSON (JavaScript Object Notation) is a format",
        ),
        ("see ECMA\\-404 and RFC\\ 7159", "see ECMA-404 and RFC\\ 7159"),
        ("as in \\[1\\] and \\[RFC 8259\\]", "as in [1] and [RFC 8259]"),
        ("a\\/b, it\\'s, \\>quoted, 5 \\* 3, snake\\_case", "a/b, it's, >quoted, 5 * 3, snake_case"),
        ("\\# of items", "# of items"),
        ("issue \\#42", "issue #42"),
        ('\\!\\"\\$\\%\\&\\+\\<\\=\\?\\@\\^\\`\\{\\|\\}\\~', '!"$%&+<=?@^`{|}~'),
        ("trailing backslash\\", "trailing backslash\\"),
        ("a \\b \\n \\é stay", "a \\b \\n \\é stay"),
    ],
)
def test_unescape_all_commonmark_punctuation(raw, expected):
    assert server._unescape_markdown(raw) == expected


def test_unescape_skips_code():
    fenced = "```python\nre.sub(r'\\(\\d+\\)', '', s)\n```"
    tilde = "~~~\nC:\\-dir \\[x\\]\n~~~"
    span = "`a\\-b`"
    raw = f"Use \\(x\\) here\n{fenced}\nthen {span} and \\[1\\]\n{tilde}\nend\\."
    expected = f"Use (x) here\n{fenced}\nthen {span} and [1]\n{tilde}\nend."
    assert server._unescape_markdown(raw) == expected


def test_unescape_unterminated_fence_runs_to_the_end():
    raw = "a \\- b\n```\nc \\- d"
    assert server._unescape_markdown(raw) == "a - b\n```\nc \\- d"


def test_unescape_unmatched_backtick_is_text():
    assert server._unescape_markdown("a ` b \\- c") == "a ` b - c"


def test_unescaped_backtick_does_not_open_a_code_span():
    assert server._unescape_markdown("\\`a\\-b\\`") == "`a-b`"


def test_truncate():
    text, cut = server._truncate("abcdef", 3)
    assert cut is True
    assert text.startswith("abc")
    assert "truncated" in text


def test_truncate_disabled():
    text, cut = server._truncate("abcdef", 0)
    assert (text, cut) == ("abcdef", False)


def test_http_error_404_hint_is_neutral():
    result = server._map_http_error(404, json.dumps({"error": {"message": "failed to fetch URL"}}))
    assert result["ok"] is False
    assert result["error"]["status"] == 404
    assert "deepseek_search_and_read" not in result["error"]["message"]
    assert "not found" in result["error"]["message"]


def test_http_error_402_mentions_balance():
    result = server._map_http_error(402, json.dumps({"error": {"message": "Insufficient Balance"}}))
    assert "balance" in result["error"]["message"].lower()


def test_http_error_non_json_body():
    result = server._map_http_error(500, "upstream broke")
    assert result["ok"] is False
    assert "upstream broke" in result["error"]["message"]


def test_search_rejects_empty_query():
    result = asyncio.run(server._search("", 5, None, False, None, 1000, 4000))
    assert result["error"]["code"] == "empty_query"


def test_search_rejects_long_query():
    result = asyncio.run(server._search("a" * 129, 5, None, False, None, 1000, 4000))
    assert result["error"]["code"] == "query_too_long"


def test_search_rejects_bad_count():
    result = asyncio.run(server._search("test", 0, None, False, None, 1000, 4000))
    assert result["error"]["code"] == "bad_count"


def test_search_rejects_bad_lang():
    result = asyncio.run(server._search("test", 5, None, False, "polish", 1000, 4000))
    assert result["error"]["code"] == "bad_lang"


def test_api_key_env_wins(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert server._api_key() == "env-key"


def test_api_key_auth_json_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"deepseek": {"key": "file-key"}}))
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(auth))
    assert server._api_key() == "file-key"


def test_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(tmp_path / "absent.json"))
    assert server._api_key() == ""


def test_search_without_key_returns_error(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(tmp_path / "absent.json"))
    result = asyncio.run(server._search("test", 1, None, False, None, 1000, 4000))
    assert result["error"]["code"] == "missing_api_key"


def test_fetch_rejects_non_http_url():
    result = asyncio.run(server.deepseek_fetch("ftp://example.com/file"))
    assert result["error"]["code"] == "bad_url"


def test_fetch_rejects_long_url():
    result = asyncio.run(server.deepseek_fetch("https://x.test/" + "a" * 9000))
    assert result["error"]["code"] == "url_too_long"


def _search_response(results):
    return {"ok": True, "data": {"results": results}}


def _run_search(fake_upstream, results, **kwargs):
    calls = fake_upstream(responses={"/v1/tools/search": _search_response(results)}).calls
    result = asyncio.run(
        server._search(
            kwargs.pop("query", "test query"),
            5,
            kwargs.pop("ori_question", None),
            kwargs.pop("need_content", False),
            kwargs.pop("lang", None),
            1000,
            4000,
            **kwargs,
        )
    )
    return result, calls


def test_dedupe_removes_tracking_and_trailing_slash(fake_upstream):
    results = [
        {"title": "A", "url": "https://example.com/post?utm_source=x&b=2&a=1", "site_name": "Example"},
        {"title": "B", "url": "https://www.example.com/post/?fbclid=y&a=1&b=2", "site_name": "Mirror"},
        {"title": "C", "url": "https://EXAMPLE.com/post?a=1&b=2#frag", "site_name": "Third"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["count"] == 1
    assert result["deduped"] == 2
    assert result["results"][0]["title"] == "A"


def test_dedupe_keeps_distinct_urls(fake_upstream):
    results = [
        {"title": "One", "url": "https://example.com/one", "site_name": "Example"},
        {"title": "Two", "url": "https://example.com/two", "site_name": "Example"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["count"] == 2
    assert result["deduped"] == 0


def test_dedupe_collapses_same_title_and_site(fake_upstream):
    results = [
        {
            "title": "Hello World From The Example Blog",
            "url": "https://example.com/a",
            "site_name": "Example",
        },
        {
            "title": "hello world from the example blog",
            "url": "https://mirror.test/amp/a",
            "site_name": "example",
        },
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["count"] == 1
    assert result["deduped"] == 1
    assert result["results"][0]["url"] == "https://example.com/a"


def test_dedupe_keeps_short_generic_titles(fake_upstream):
    results = [
        {"title": "Home", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "Home", "url": "https://example.com/b", "site_name": "Example"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["count"] == 2
    assert result["deduped"] == 0


def test_dedupe_does_not_collapse_empty_title(fake_upstream):
    results = [
        {"title": "", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "", "url": "https://mirror.test/amp/a", "site_name": "example"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["count"] == 2
    assert result["deduped"] == 0


def test_dedupe_preserves_input_and_order():
    items = [
        {"title": "A", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "A", "url": "https://example.com/a/", "site_name": "Example"},
        {"title": "B", "url": "https://example.com/b", "site_name": "Example"},
    ]
    snapshot = [dict(item) for item in items]
    kept, removed = server._dedupe_results(items)
    assert items == snapshot
    assert [item["title"] for item in kept] == ["A", "B"]
    assert removed == 1


def test_dedupe_false_preserves_duplicates(fake_upstream):
    results = [
        {"title": "Same", "url": "https://example.com/a?utm_source=x", "site_name": "Example"},
        {"title": "Same", "url": "https://example.com/a", "site_name": "Example"},
    ]
    result, _ = _run_search(fake_upstream, results, dedupe=False)
    assert result["count"] == 2
    assert result["deduped"] == 0
    assert not any("deduped" in note for note in result["notes"])


def test_dedupe_note_and_count_in_envelope(fake_upstream):
    results = [
        {"title": "Same", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "Same", "url": "https://example.com/a/", "site_name": "Example"},
        {"title": "Same", "url": "https://www.example.com/a?gclid=1", "site_name": "Example"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert result["deduped"] == 2
    assert "deduped 2 duplicate result(s)" in result["notes"]
    assert result["count"] == 1


def test_rewrite_disabled_never_calls_rewrite(monkeypatch, fake_upstream):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    result, _ = _run_search(fake_upstream, [], rewrite=False)
    assert result["ok"] is True


def test_rewrite_uses_rewritten_query_and_sets_original(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return "how sleep works"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(fake_upstream, [], query="jak działa sen", rewrite=True)
    search_calls = [c for c in calls if c["path"] == "/v1/tools/search"]
    assert search_calls[0]["payload"]["query"] == "how sleep works"
    assert result["query_used"] == "how sleep works"
    assert result["query_original"] == "jak działa sen"
    assert "query rewritten (flash): 'how sleep works'" in result["notes"]


def test_rewrite_noop_adds_no_note(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return query

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(fake_upstream, [], query="test query", rewrite=True)
    assert calls[0]["payload"]["query"] == "test query"
    assert "query_original" not in result
    assert not any("rewrite" in note for note in result["notes"])


def test_rewrite_failure_falls_back_to_original(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return None

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(fake_upstream, [], query="test query", rewrite=True)
    assert calls[0]["payload"]["query"] == "test query"
    assert result["query_used"] == "test query"
    assert "query_original" not in result
    assert "query rewrite failed; using original query" in result["notes"]


def test_rewrite_skipped_when_lang_is_set(monkeypatch, fake_upstream):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called with lang")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    result, calls = _run_search(fake_upstream, [], query="test query", lang="pl", rewrite=True)
    assert calls[0]["payload"]["query"] == "lang:pl test query"
    assert "query_original" not in result
    assert not any("rewrite" in note for note in result["notes"])


def test_rewrite_with_missing_key_makes_no_chat_call(monkeypatch, tmp_path):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called without a key")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(tmp_path / "absent.json"))
    result = asyncio.run(server._search("test", 1, None, False, None, 1000, 4000, True, True))
    assert result["error"]["code"] == "missing_api_key"


def test_rewrite_payload_is_flash_without_thinking(fake_upstream):
    response = {"ok": True, "data": {"choices": [{"message": {"content": "better query"}}]}}
    calls = fake_upstream(responses={"/v1/chat/completions": response}).calls
    rewritten = asyncio.run(server._rewrite_query("query"))
    assert rewritten == "better query"
    payload = calls[0]["payload"]
    assert calls[0]["path"] == "/v1/chat/completions"
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning" not in payload
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 64
    assert payload["stream"] is False
    assert calls[0]["timeout"] == server.CHAT_TIMEOUT
    assert calls[0]["attempts"] == 1
    assert payload["messages"][0]["content"] == server.SYSTEM_PROMPT


def test_rewrite_truncates_long_model_answer(fake_upstream):
    response = {"ok": True, "data": {"choices": [{"message": {"content": "x" * 200}}]}}
    fake_upstream(responses={"/v1/chat/completions": response})
    rewritten = asyncio.run(server._rewrite_query("query"))
    assert rewritten is not None
    assert len(rewritten) <= server.MAX_QUERY


def test_rewrite_query_returns_none_on_error_envelope(fake_upstream):
    fake_upstream(responses={"/v1/chat/completions": {"ok": False, "error": {"code": "timeout"}}})
    assert asyncio.run(server._rewrite_query("query")) is None


def test_rewrite_path_sets_ori_question_to_original(monkeypatch, fake_upstream):
    async def fake_rewrite(query):
        return "how sleep works"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    _, calls = _run_search(fake_upstream, [], query="jak działa sen", rewrite=True)
    assert calls[0]["payload"]["ori_question"] == "jak działa sen"


def _dup_results(unique: int, duplicates: int) -> list[dict]:
    items = [
        {"title": f"Result {i}", "url": f"https://example.com/{i}", "site_name": "Example"}
        for i in range(unique)
    ]
    return items + [dict(items[0]) for _ in range(duplicates)]


def test_upstream_count_overfetches_only_with_dedupe():
    assert server._upstream_count(4, False) == 4
    assert server._upstream_count(4, True) == 6
    assert server._upstream_count(1, True) == 3
    assert server._upstream_count(10, True) == 15
    assert server._upstream_count(49, True) == server.MAX_COUNT
    assert server._upstream_count(50, True) == server.MAX_COUNT


def test_topup_returns_count_despite_duplicates(fake_upstream):
    # asked for 4; upstream returns 6 of which 2 are duplicates
    result, calls = _run_search(fake_upstream, _dup_results(4, 2), dedupe=True)
    assert calls[0]["payload"]["count"] == 8  # _run_search asks for 5 -> 5 + 3
    assert result["deduped"] == 2


def test_topup_trims_to_requested_count(fake_upstream):
    calls = fake_upstream(responses={"/v1/tools/search": _search_response(_dup_results(8, 0))}).calls
    result = asyncio.run(server._search("q", 4, None, False, None, 1000, 4000))
    assert calls[0]["payload"]["count"] == 6
    assert result["count"] == 4
    assert [r["title"] for r in result["results"]] == [f"Result {i}" for i in range(4)]


def test_topup_fills_count_after_dedupe(fake_upstream):
    calls = fake_upstream(responses={"/v1/tools/search": _search_response(_dup_results(4, 2))}).calls
    result = asyncio.run(server._search("q", 4, None, False, None, 1000, 4000))
    assert calls[0]["payload"]["count"] == 6
    assert result["count"] == 4
    assert result["deduped"] == 2


def test_no_overfetch_without_dedupe(fake_upstream):
    calls = fake_upstream(responses={"/v1/tools/search": _search_response(_dup_results(2, 0))}).calls
    asyncio.run(server._search("q", 4, None, False, None, 1000, 4000, False))
    assert calls[0]["payload"]["count"] == 4


def test_overfetch_capped_at_upstream_max(fake_upstream):
    calls = fake_upstream(responses={"/v1/tools/search": _search_response([])}).calls
    asyncio.run(server._search("q", 50, None, False, None, 1000, 4000))
    assert calls[0]["payload"]["count"] == 50


def test_blank_title_falls_back_to_host(fake_upstream):
    results = [
        {"title": "", "url": "https://www.example.com/a", "site_name": "Example"},
        {"title": "   ", "url": "https://docs.example.org:8080/b", "site_name": "Docs"},
        {"title": None, "url": "not a url", "site_name": "Odd"},
        {"title": "Real", "url": "https://real.test/c", "site_name": "Real"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert [r["title"] for r in result["results"]] == ["example.com", "docs.example.org", None, "Real"]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("ADHD and social media - ADHD and social media", "ADHD and social media"),
        ("  X - X  ", "X"),
        ("Re - Re - Re - Re", "Re - Re"),
        ("A - B", "A - B"),
        ("A - A B", "A - A B"),
        ("A-A", "A-A"),
        ("A -  A", "A -  A"),
        ("X - x", "X - x"),
        (" - ", " - "),
        ("", ""),
        (None, None),
    ],
)
def test_collapse_repeated_title(title, expected):
    assert server._collapse_repeated_title(title) == expected


def test_repeated_titles_are_collapsed_in_search_results(fake_upstream):
    results = [
        {"title": "Thread - Thread", "url": "https://reddit.test/a", "site_name": "Reddit"},
        {"title": "Left - Right", "url": "https://reddit.test/b", "site_name": "Reddit"},
    ]
    result, _ = _run_search(fake_upstream, results)
    assert [r["title"] for r in result["results"]] == ["Thread", "Left - Right"]


def test_repeated_title_is_collapsed_by_fetch(fake_upstream):
    fake_upstream(
        responses={"/v1/tools/fetch": {"ok": True, "data": {"title": "Post - Post", "content": "body"}}}
    )
    result = asyncio.run(server.deepseek_fetch("https://example.com/post"))
    assert result["title"] == "Post"


def test_rewrite_exception_is_logged_redacted(fake_upstream, caplog):
    async def broken_request(method, path, **_):
        raise RuntimeError("upstream exploded with key sk-abcdefghijklmnop")

    fake_upstream(broken_request)
    with caplog.at_level(logging.DEBUG, logger=server.__name__):
        assert asyncio.run(server._rewrite_query("q")) is None
    messages = [record.getMessage() for record in caplog.records if record.name == server.__name__]
    assert any("RuntimeError" in message for message in messages)
    assert all("sk-abcdefghijklmnop" not in message for message in messages)
