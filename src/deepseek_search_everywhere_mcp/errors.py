"""Error envelopes shared by the transport and the tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from .redact import redact

logger = logging.getLogger(__name__)

# Error codes an agent acts on, each with one recommended action. The finer cause stays in
# ``error.reason`` (a closed set below) for diagnostics; agents decide by ``code`` and ``action``.
CODE_ACTIONS: dict[str, str] = {
    "invalid_input": "fix_input",
    "blocked_by_policy": "do_not_retry",
    "page_unavailable": "change_approach",
    "extraction_failed": "change_approach",
    "temporary_failure": "retry",
    "job_not_found": "start_again",
    "limit_reached": "ask_user",
    "account_error": "ask_user",
    "internal_error": "retry",
}

CLOSED_ERROR_CODES: frozenset[str] = frozenset(CODE_ACTIONS)

# Every reason maps to exactly one code.
REASON_CODES: dict[str, str] = {
    # Input validation
    "bad_request": "invalid_input",
    "empty_query": "invalid_input",
    "query_too_long": "invalid_input",
    "bad_count": "invalid_input",
    "bad_lang": "invalid_input",
    "bad_url": "invalid_input",
    "empty_url": "invalid_input",
    "url_too_long": "invalid_input",
    "topic_too_long": "invalid_input",
    "bad_domain": "invalid_input",
    "bad_date": "invalid_input",
    "bad_term": "invalid_input",
    "bad_phrase": "invalid_input",
    "filters_too_long": "invalid_input",
    # Security and URL policy
    "ssrf_blocked": "blocked_by_policy",
    "disallowed_url": "blocked_by_policy",
    "private_url": "blocked_by_policy",
    "disallowed_protocol": "blocked_by_policy",
    "egress_blocked": "blocked_by_policy",
    # The page cannot be read
    "not_found": "page_unavailable",
    "blocked_by_site": "page_unavailable",
    "empty_page": "page_unavailable",
    "response_too_large": "page_unavailable",
    # Extraction
    "extraction_failed": "extraction_failed",
    "extraction_truncated": "extraction_failed",
    # Transient failures
    "rate_limited_local": "temporary_failure",
    "upstream_rate_limited": "temporary_failure",
    "timeout": "temporary_failure",
    "upstream_timeout": "temporary_failure",
    "deadline_exceeded": "temporary_failure",
    "network_error": "temporary_failure",
    "upstream_error": "temporary_failure",
    "request_failed": "temporary_failure",
    "invalid_json": "temporary_failure",
    # Jobs
    "job_not_found": "job_not_found",
    # Spending limits
    "budget_exceeded": "limit_reached",
    "max_cost_reached": "limit_reached",
    # Account
    "missing_api_key": "account_error",
    "auth_failed": "account_error",
    "insufficient_balance": "account_error",
    # Bugs
    "internal_error": "internal_error",
}

CLOSED_ERROR_REASONS: frozenset[str] = frozenset(REASON_CODES)

HTTP_HINTS = {
    400: "Invalid request parameters.",
    401: "Authentication failed. Check your DEEPSEEK_API_KEY.",
    402: "Insufficient DeepSeek balance. Top up or switch provider.",
    403: "Access forbidden. Check your account permissions or API key.",
    404: "Page not found, or the site could not be extracted.",
    413: "Request body too large.",
    429: "Rate limit exceeded (upstream); retry later.",
    504: "Upstream timeout; the page took too long to fetch.",
}


def _fail(
    reason: str,
    message: str,
    status: int | None = None,
    *,
    param: str | None = None,
    upstream_code: str | None = None,
    upstream_message: str | None = None,
) -> dict[str, Any]:
    """Build the ``{"ok": False, "error": {...}}`` envelope returned by every tool.

    ``reason`` is the fine cause from ``CLOSED_ERROR_REASONS``; it decides the public ``code`` and its
    ``action``. ``param`` names the offending parameter of an input error.
    """
    if reason not in CLOSED_ERROR_REASONS:
        logger.warning("Unknown error reason %r encountered; returning internal_error", reason)
        reason = "internal_error"
    code = REASON_CODES[reason]
    err: dict[str, Any] = {"code": code, "message": message, "action": CODE_ACTIONS[code]}
    if param is not None:
        err["param"] = param
    err["reason"] = reason
    if status is not None:
        err["status"] = status
    if upstream_code is not None:
        err["upstream_code"] = upstream_code
    if upstream_message is not None:
        err["upstream_message"] = redact(upstream_message)
    return {"ok": False, "error": err}


def _map_http_status_to_code(status: int, path: str = "") -> str:
    """Map an HTTP status code and request path to a closed error reason."""
    if status == 400:
        return "bad_request"
    if status == 401:
        return "auth_failed"
    if status == 402:
        return "insufficient_balance"
    if status == 403:
        if path == "/v1/tools/fetch" or path.endswith("/fetch"):
            return "blocked_by_site"
        return "auth_failed"
    if status == 404:
        return "not_found"
    if status == 413:
        return "bad_request"
    if status == 429:
        return "upstream_rate_limited"
    if status == 504:
        return "upstream_timeout"
    if 500 <= status < 600:
        return "upstream_error"
    if 400 <= status < 500:
        return "bad_request"
    return "upstream_error"


def _http_hint(status: int, path: str = "") -> str | None:
    """Return a path-aware hint for human readers."""
    if status == 403 and (path == "/v1/tools/fetch" or path.endswith("/fetch")):
        return "Rejected by the target site's security policy."
    return HTTP_HINTS.get(status)


def _map_http_error(status: int, body: str, path: str = "") -> dict[str, Any]:
    """Turn an upstream HTTP error body into an error envelope with a closed code and human hint."""
    upstream_code: str | None = None
    upstream_msg: str | None = None
    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            err_obj = payload.get("error")
            if isinstance(err_obj, dict):
                upstream_msg = err_obj.get("message")
                raw_code = err_obj.get("code")
                if raw_code is not None:
                    upstream_code = str(raw_code)
    except (json.JSONDecodeError, ValueError):
        pass

    raw_msg = upstream_msg or body[:400]
    code = _map_http_status_to_code(status, path)
    hint = _http_hint(status, path)
    message = f"{raw_msg} — {hint}" if hint else raw_msg

    return _fail(
        code,
        redact(message),
        status,
        upstream_code=upstream_code,
        upstream_message=raw_msg,
    )
