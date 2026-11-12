"""Error envelopes shared by the transport and the tools."""

from __future__ import annotations

import json
import logging
from typing import Any

from .redact import redact

logger = logging.getLogger(__name__)

CLOSED_ERROR_CODES: frozenset[str] = frozenset(
    {
        # Upstream mapped error codes
        "auth_failed",
        "insufficient_balance",
        "upstream_rate_limited",
        "upstream_timeout",
        "upstream_error",
        "not_found",
        "blocked_by_site",
        "bad_request",
        # Local transport and system error codes
        "missing_api_key",
        "rate_limited_local",
        "timeout",
        "deadline_exceeded",
        "response_too_large",
        "invalid_json",
        "network_error",
        "budget_exceeded",
        "internal_error",
        "request_failed",
        # Input validation codes
        "empty_query",
        "query_too_long",
        "bad_count",
        "bad_lang",
        "bad_url",
        "empty_url",
        "url_too_long",
        "topic_too_long",
        "bad_domain",
        "bad_date",
        "bad_term",
        "bad_phrase",
        "filters_too_long",
        # Security and URL validation codes
        "ssrf_blocked",
        "disallowed_url",
        "private_url",
        "disallowed_protocol",
        "egress_blocked",
        # Tool specific codes
        "empty_page",
        "extraction_failed",
        "extraction_truncated",
    }
)

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
    code: str,
    message: str,
    status: int | None = None,
    *,
    upstream_code: str | None = None,
    upstream_message: str | None = None,
) -> dict[str, Any]:
    """Build the ``{"ok": False, "error": {...}}`` envelope returned by every tool."""
    if code not in CLOSED_ERROR_CODES:
        logger.warning("Unknown error code %r encountered; returning internal_error", code)
        code = "internal_error"

    err: dict[str, Any] = {"code": code, "message": message}
    if status is not None:
        err["status"] = status
    if upstream_code is not None:
        err["upstream_code"] = upstream_code
    if upstream_message is not None:
        err["upstream_message"] = redact(upstream_message)
    return {"ok": False, "error": err}


def _map_http_status_to_code(status: int, path: str = "") -> str:
    """Map an HTTP status code and request path to a closed error code."""
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
