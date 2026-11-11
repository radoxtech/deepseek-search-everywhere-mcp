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


def test_fail_enforces_closed_code_set() -> None:
    valid = _fail("auth_failed", "msg")
    assert valid["error"]["code"] == "auth_failed"

    with pytest.raises(ValueError, match="is not in the closed error code set"):
        _fail("arbitrary_unknown_code", "msg")
