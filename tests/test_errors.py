"""Tests for closed error code set and HTTP error mapping."""

from __future__ import annotations

import json

import pytest

from deepseek_search_everywhere_mcp.errors import (
    CLOSED_ERROR_CODES,
    _fail,
    _map_http_error,
)


@pytest.mark.parametrize(
    ("status", "path", "expected_code", "hint_keyword"),
    [
        (400, "/v1/tools/search", "bad_request", "invalid"),
        (401, "/v1/chat/completions", "auth_failed", "authentication"),
        (401, "/v1/tools/search", "auth_failed", "authentication"),
        (401, "/v1/tools/fetch", "auth_failed", "authentication"),
        (401, "/user/balance", "auth_failed", "authentication"),
        (402, "/v1/tools/search", "insufficient_balance", "balance"),
        (403, "/v1/tools/fetch", "blocked_by_site", "security policy"),
        (403, "/v1/chat/completions", "auth_failed", "forbidden"),
        (403, "/user/balance", "auth_failed", "forbidden"),
        (403, "/v1/tools/search", "auth_failed", "forbidden"),
        (404, "/v1/tools/fetch", "not_found", "not found"),
        (413, "/v1/tools/search", "bad_request", "too large"),
        (429, "/v1/tools/search", "upstream_rate_limited", "rate limit"),
        (500, "/v1/tools/search", "upstream_error", ""),
        (502, "/v1/tools/search", "upstream_error", ""),
        (503, "/v1/tools/search", "upstream_error", ""),
        (504, "/v1/tools/fetch", "upstream_timeout", "timeout"),
        (418, "/v1/tools/search", "bad_request", ""),
        (599, "/v1/tools/search", "upstream_error", ""),
    ],
)
def test_map_http_error_statuses_and_paths(
    status: int, path: str, expected_code: str, hint_keyword: str
) -> None:
    body = json.dumps({"error": {"message": "upstream issue", "code": "upstream_orig_code"}})
    envelope = _map_http_error(status, body, path=path)

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == expected_code
    assert envelope["error"]["code"] in CLOSED_ERROR_CODES
    assert envelope["error"]["status"] == status
    assert envelope["error"]["upstream_code"] == "upstream_orig_code"
    assert envelope["error"]["upstream_message"] == "upstream issue"
    if hint_keyword:
        assert hint_keyword.lower() in envelope["error"]["message"].lower()


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 413, 422, 429, 500, 502, 503, 504, 520])
@pytest.mark.parametrize(
    "path",
    ["/v1/tools/fetch", "/v1/tools/search", "/v1/chat/completions", "/user/balance", ""],
)
def test_all_statuses_and_paths_belong_to_closed_code_set(status: int, path: str) -> None:
    body = json.dumps({"error": {"message": "something failed", "code": "raw_code"}})
    envelope = _map_http_error(status, body, path=path)
    assert envelope["error"]["code"] in CLOSED_ERROR_CODES


def test_map_http_error_handles_non_json_body() -> None:
    body = "Plain text gateway timeout"
    envelope = _map_http_error(504, body, path="/v1/tools/fetch")

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "upstream_timeout"
    assert envelope["error"]["code"] in CLOSED_ERROR_CODES
    assert envelope["error"]["upstream_message"] == body
    assert "upstream_code" not in envelope["error"]
    assert "timeout" in envelope["error"]["message"].lower()


def test_map_http_error_redacts_credentials_in_message() -> None:
    body = json.dumps({"error": {"message": "invalid key: sk-ant-api03-secretsecretsecret-AAAA"}})
    envelope = _map_http_error(401, body, path="/v1/chat/completions")

    assert "secretsecretsecret" not in envelope["error"]["message"]
    assert "secretsecretsecret" not in envelope["error"]["upstream_message"]
    assert envelope["error"]["code"] == "auth_failed"


def test_fail_enforces_closed_code_set(caplog: pytest.LogCaptureFixture) -> None:
    valid = _fail("auth_failed", "msg")
    assert valid["error"]["code"] == "auth_failed"

    # In production, _fail must NOT raise for an unknown code: log at WARNING and return internal_error
    with caplog.at_level("WARNING"):
        envelope = _fail("arbitrary_unknown_code", "msg")
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "internal_error"
    assert "arbitrary_unknown_code" in caplog.text


def test_src_error_codes_belong_to_closed_set() -> None:
    import ast
    import pathlib

    src_dir = pathlib.Path(__file__).parent.parent / "src" / "deepseek_search_everywhere_mcp"
    found_codes: set[str] = set()

    for p in src_dir.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "_fail":
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        found_codes.add(node.args[0].value)
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values, strict=False):
                    if isinstance(k, ast.Constant) and k.value == "code":
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            found_codes.add(v.value)

    # Filter return error codes from filters.py
    filters_file = src_dir / "filters.py"
    if filters_file.exists():
        tree = ast.parse(filters_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                for elt in node.value.elts:
                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                        and elt.value.startswith("bad_")
                    ):
                        found_codes.add(elt.value)
                    if isinstance(elt, ast.Constant) and elt.value == "filters_too_long":
                        found_codes.add(elt.value)

    missing = found_codes - CLOSED_ERROR_CODES
    assert not missing, f"Found codes in src/ not in CLOSED_ERROR_CODES: {missing}"


@pytest.mark.asyncio
async def test_filter_validation_errors_produce_valid_envelopes() -> None:
    from deepseek_search_everywhere_mcp import server

    # 1. exclude_terms with whitespace -> bad_term
    r1 = await server.deepseek_search("test", exclude_terms=["two words"])
    assert r1["ok"] is False
    assert r1["error"]["code"] == "bad_term"

    # 2. exact_phrases with internal quotes -> bad_phrase
    r2 = await server.deepseek_search("test", exact_phrases=['a"b'])
    assert r2["ok"] is False
    assert r2["error"]["code"] == "bad_phrase"

    # 3. oversized filters -> filters_too_long
    many_domains = [f"domain{i}.com" for i in range(20)]
    r3 = await server.deepseek_search("test", include_domains=many_domains)
    assert r3["ok"] is False
    assert r3["error"]["code"] == "filters_too_long"

    # 4. bad date -> bad_date
    r4 = await server.deepseek_search("test", after="invalid-date")
    assert r4["ok"] is False
    assert r4["error"]["code"] == "bad_date"


@pytest.mark.asyncio
async def test_all_tools_validation_errors_return_error_envelopes() -> None:
    from deepseek_search_everywhere_mcp import server

    # deepseek_search with invalid arguments
    res = await server.deepseek_search("")
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    res = await server.deepseek_search("hello", count=0)
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    # deepseek_search_many with invalid arguments
    res = await server.deepseek_search_many([])
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    res = await server.deepseek_search_many(["test"], count=999)
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    # deepseek_fetch with invalid arguments
    res = await server.deepseek_fetch("")
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    res = await server.deepseek_fetch("ftp://example.com")
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    res = await server.deepseek_fetch("http://example.com", offset=-1)
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    # deepseek_fetch_many with invalid arguments
    res = await server.deepseek_fetch_many([])
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    # deepseek_extract with invalid arguments
    res = await server.deepseek_extract("http://example.com")  # missing fields and schema
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    res = await server.deepseek_extract("http://example.com", fields=["a"], schema={"type": "object"})
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES

    # deepseek_extract_many with invalid arguments
    res = await server.deepseek_extract_many([], fields=["a"])
    assert res["ok"] is False
    assert res["error"]["code"] in CLOSED_ERROR_CODES
