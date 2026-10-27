"""Error envelopes shared by the transport and the tools."""

from __future__ import annotations

import json
from typing import Any

from .redact import redact

HTTP_HINTS = {
    400: "Invalid request parameters.",
    402: "Insufficient DeepSeek balance. Top up or switch provider.",
    403: "Rejected by the target site's security policy.",
    404: "Fetch failed: PDFs, Reddit and some sites are not extractable via /tools/fetch. "
         "Use deepseek_search_and_read instead, which returns content via the search index.",
    413: "Request body too large.",
    429: "Rate limit exceeded (search 100 QPS, fetch 25 QPS per account).",
    504: "Upstream timeout; the page took too long to fetch.",
}


def _fail(code: str, message: str, status: int | None = None) -> dict[str, Any]:
    """Build the ``{"ok": False, "error": {...}}`` envelope returned by every tool."""
    err: dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if status is not None:
        err["error"]["status"] = status
    return err


def _map_http_error(status: int, body: str) -> dict[str, Any]:
    """Turn an upstream HTTP error body into an error envelope with a human hint."""
    try:
        payload = json.loads(body)
        message = (payload.get("error") or {}).get("message") or body[:400]
        code = (payload.get("error") or {}).get("code") or "http_error"
    except json.JSONDecodeError:
        message, code = body[:400], "http_error"

    if status in HTTP_HINTS:
        message = f"{message} — {HTTP_HINTS[status]}"
    return _fail(code, redact(message), status)
