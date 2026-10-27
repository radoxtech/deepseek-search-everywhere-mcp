import asyncio
import json

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


def test_unescape_keeps_double_backslash():
    assert server._unescape_markdown("a\\\\b") == "a\\\\b"


def test_truncate():
    text, cut = server._truncate("abcdef", 3)
    assert cut is True
    assert text.startswith("abc")
    assert "truncated" in text


def test_truncate_disabled():
    text, cut = server._truncate("abcdef", 0)
    assert (text, cut) == ("abcdef", False)


def test_http_error_404_mentions_pdf():
    result = server._map_http_error(404, json.dumps({"error": {"message": "failed to fetch URL"}}))
    assert result["ok"] is False
    assert result["error"]["status"] == 404
    assert "PDF" in result["error"]["message"]


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


def _install_fake_request(monkeypatch, responses):
    calls = []

    async def fake_request(method, path, *, payload=None, timeout=None, retries=2, **_):
        calls.append({"method": method, "path": path, "payload": payload, "timeout": timeout,
                      "retries": retries})
        return responses[path]

    monkeypatch.setattr(server, "_request", fake_request)
    monkeypatch.setattr(server, "_api_key", lambda: "test-key")
    return calls


def _search_response(results):
    return {"ok": True, "data": {"results": results}}


def _run_search(monkeypatch, results, **kwargs):
    calls = _install_fake_request(
        monkeypatch, {"/v1/tools/search": _search_response(results)}
    )
    result = asyncio.run(
        server._search(
            kwargs.pop("query", "test query"), 5, kwargs.pop("ori_question", None),
            kwargs.pop("need_content", False), kwargs.pop("lang", None), 1000, 4000, **kwargs
        )
    )
    return result, calls


def test_dedupe_removes_tracking_and_trailing_slash(monkeypatch):
    results = [
        {"title": "A", "url": "https://example.com/post?utm_source=x&b=2&a=1",
         "site_name": "Example"},
        {"title": "B", "url": "https://www.example.com/post/?fbclid=y&a=1&b=2",
         "site_name": "Mirror"},
        {"title": "C", "url": "https://EXAMPLE.com/post?a=1&b=2#frag", "site_name": "Third"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert result["count"] == 1
    assert result["deduped"] == 2
    assert result["results"][0]["title"] == "A"


def test_dedupe_keeps_distinct_urls(monkeypatch):
    results = [
        {"title": "One", "url": "https://example.com/one", "site_name": "Example"},
        {"title": "Two", "url": "https://example.com/two", "site_name": "Example"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert result["count"] == 2
    assert result["deduped"] == 0


def test_dedupe_collapses_same_title_and_site(monkeypatch):
    results = [
        {"title": "Hello World From The Example Blog", "url": "https://example.com/a",
         "site_name": "Example"},
        {"title": "hello world from the example blog", "url": "https://mirror.test/amp/a",
         "site_name": "example"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert result["count"] == 1
    assert result["deduped"] == 1
    assert result["results"][0]["url"] == "https://example.com/a"


def test_dedupe_keeps_short_generic_titles(monkeypatch):
    results = [
        {"title": "Home", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "Home", "url": "https://example.com/b", "site_name": "Example"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert result["count"] == 2
    assert result["deduped"] == 0


def test_dedupe_does_not_collapse_empty_title(monkeypatch):
    results = [
        {"title": "", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "", "url": "https://mirror.test/amp/a", "site_name": "example"},
    ]
    result, _ = _run_search(monkeypatch, results)
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


def test_dedupe_false_preserves_duplicates(monkeypatch):
    results = [
        {"title": "Same", "url": "https://example.com/a?utm_source=x", "site_name": "Example"},
        {"title": "Same", "url": "https://example.com/a", "site_name": "Example"},
    ]
    result, _ = _run_search(monkeypatch, results, dedupe=False)
    assert result["count"] == 2
    assert result["deduped"] == 0
    assert not any("deduped" in note for note in result["notes"])


def test_dedupe_note_and_count_in_envelope(monkeypatch):
    results = [
        {"title": "Same", "url": "https://example.com/a", "site_name": "Example"},
        {"title": "Same", "url": "https://example.com/a/", "site_name": "Example"},
        {"title": "Same", "url": "https://www.example.com/a?gclid=1", "site_name": "Example"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert result["deduped"] == 2
    assert "deduped 2 duplicate result(s)" in result["notes"]
    assert result["count"] == 1


def test_rewrite_disabled_never_calls_rewrite(monkeypatch):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    result, _ = _run_search(monkeypatch, [], rewrite=False)
    assert result["ok"] is True


def test_rewrite_uses_rewritten_query_and_sets_original(monkeypatch):
    async def fake_rewrite(query):
        return "how sleep works"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(
        monkeypatch, [], query="jak działa sen", rewrite=True
    )
    search_calls = [c for c in calls if c["path"] == "/v1/tools/search"]
    assert search_calls[0]["payload"]["query"] == "how sleep works"
    assert result["query_used"] == "how sleep works"
    assert result["query_original"] == "jak działa sen"
    assert "query rewritten (flash): 'how sleep works'" in result["notes"]


def test_rewrite_noop_adds_no_note(monkeypatch):
    async def fake_rewrite(query):
        return query

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(monkeypatch, [], query="test query", rewrite=True)
    assert calls[0]["payload"]["query"] == "test query"
    assert "query_original" not in result
    assert not any("rewrite" in note for note in result["notes"])


def test_rewrite_failure_falls_back_to_original(monkeypatch):
    async def fake_rewrite(query):
        return None

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    result, calls = _run_search(monkeypatch, [], query="test query", rewrite=True)
    assert calls[0]["payload"]["query"] == "test query"
    assert result["query_used"] == "test query"
    assert "query_original" not in result
    assert "query rewrite failed; using original query" in result["notes"]


def test_rewrite_skipped_when_lang_is_set(monkeypatch):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called with lang")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    result, calls = _run_search(monkeypatch, [], query="test query", lang="pl", rewrite=True)
    assert calls[0]["payload"]["query"] == "lang:pl test query"
    assert "query_original" not in result
    assert not any("rewrite" in note for note in result["notes"])


def test_rewrite_with_missing_key_makes_no_chat_call(monkeypatch, tmp_path):
    async def boom(query):
        raise AssertionError("_rewrite_query must not be called without a key")

    monkeypatch.setattr(server, "_rewrite_query", boom)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_AUTH_JSON", str(tmp_path / "absent.json"))
    result = asyncio.run(
        server._search("test", 1, None, False, None, 1000, 4000, True, True)
    )
    assert result["error"]["code"] == "missing_api_key"


def test_rewrite_payload_is_flash_without_thinking(monkeypatch):
    response = {"ok": True, "data": {"choices": [{"message": {"content": "better query"}}]}}
    calls = _install_fake_request(monkeypatch, {"/v1/chat/completions": response})
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
    assert calls[0]["retries"] == 1
    assert payload["messages"][0]["content"] == server.SYSTEM_PROMPT


def test_rewrite_truncates_long_model_answer(monkeypatch):
    response = {"ok": True, "data": {"choices": [{"message": {"content": "x" * 200}}]}}
    _install_fake_request(monkeypatch, {"/v1/chat/completions": response})
    rewritten = asyncio.run(server._rewrite_query("query"))
    assert rewritten is not None
    assert len(rewritten) <= server.MAX_QUERY


def test_rewrite_query_returns_none_on_error_envelope(monkeypatch):
    _install_fake_request(
        monkeypatch, {"/v1/chat/completions": {"ok": False, "error": {"code": "timeout"}}}
    )
    assert asyncio.run(server._rewrite_query("query")) is None


def test_rewrite_path_sets_ori_question_to_original(monkeypatch):
    async def fake_rewrite(query):
        return "how sleep works"

    monkeypatch.setattr(server, "_rewrite_query", fake_rewrite)
    _, calls = _run_search(monkeypatch, [], query="jak działa sen", rewrite=True)
    assert calls[0]["payload"]["ori_question"] == "jak działa sen"



def test_blank_title_falls_back_to_host(monkeypatch):
    results = [
        {"title": "", "url": "https://www.example.com/a", "site_name": "Example"},
        {"title": "   ", "url": "https://docs.example.org:8080/b", "site_name": "Docs"},
        {"title": None, "url": "not a url", "site_name": "Odd"},
        {"title": "Real", "url": "https://real.test/c", "site_name": "Real"},
    ]
    result, _ = _run_search(monkeypatch, results)
    assert [r["title"] for r in result["results"]] == [
        "example.com", "docs.example.org", None, "Real"]
