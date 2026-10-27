"""HTTP transport to the DeepSeek API: credentials, retries and bounded response reads.

Bodies are streamed and counted so an oversized upstream response is rejected with
``response_too_large`` before it is buffered or parsed; error bodies are capped too.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import pathlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .errors import _fail, _map_http_error
from .settings import get_settings

BASE_DEFAULT = "https://api.deepseek.com"
FETCH_PATH = "/v1/tools/fetch"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
ERROR_BODY_CAP = 64 * 1024
DEFAULT_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 10.0


def _base_url() -> str:
    return os.environ.get("DEEPSEEK_BASE_URL", BASE_DEFAULT).rstrip("/")


def _api_key() -> str:
    """Return the DeepSeek key from ``DEEPSEEK_API_KEY`` or an auth JSON file, else ``""``.

    Without ``DEEPSEEK_AUTH_JSON`` the legacy opencode location is tried.
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    configured = os.environ.get("DEEPSEEK_AUTH_JSON")
    candidates = [pathlib.Path(configured).expanduser()] if configured else [
        pathlib.Path.home() / ".local/share/opencode/auth.json",
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
            found = (data.get("deepseek") or {}).get("key") or ""
        except (OSError, ValueError):
            continue
        if found.strip():
            return found.strip()
    return ""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _retry_after(value: str | None) -> float:
    """Parse ``Retry-After`` (seconds or HTTP-date) into a delay clamped to 0-10 s.

    Missing or unparseable values fall back to the default delay instead of raising.
    """
    value = (value or "").strip()
    if not value:
        return DEFAULT_RETRY_DELAY
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return DEFAULT_RETRY_DELAY
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - _utcnow()).total_seconds()
    if not math.isfinite(seconds):
        return DEFAULT_RETRY_DELAY
    return min(max(seconds, 0.0), MAX_RETRY_DELAY)


def _response_cap(path: str) -> int:
    settings = get_settings()
    return settings.response_cap_fetch if path == FETCH_PATH else settings.response_cap_search


async def _read_capped(response: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes; report whether the body was longer (rest is left unread)."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        room = limit - size
        if len(chunk) > room:
            chunks.append(chunk[:room])
            return b"".join(chunks), True
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks), False


async def _attempt(method: str, url: str, path: str, *, payload: dict[str, Any] | None,
                   headers: dict[str, str], timeout: float) -> tuple[dict[str, Any], float | None]:
    """Run one HTTP exchange; return the envelope and a retry delay when it is retryable."""
    cap = _response_cap(path)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(method, url, json=payload, headers=headers) as response:
            status = response.status_code
            if status >= 400:
                raw, _ = await _read_capped(response, ERROR_BODY_CAP)
                error = _map_http_error(status, raw.decode("utf-8", errors="replace"))
                if status in RETRY_STATUSES:
                    return error, _retry_after(response.headers.get("Retry-After"))
                return error, None
            body, too_large = await _read_capped(response, cap)
    if too_large:
        return _fail("response_too_large",
                     f"Response from {path} exceeded the {cap}-byte limit.", status), None
    try:
        return {"ok": True, "data": json.loads(body)}, None
    except ValueError:
        return _fail("invalid_json", "DeepSeek returned a non-JSON response.", status), None


async def request(method: str, path: str, *, payload: dict[str, Any] | None = None,
                  timeout: float, retries: int = 2) -> dict[str, Any]:
    """Call the DeepSeek API and return ``{"ok": True, "data": ...}`` or an error envelope.

    Each of the ``retries`` attempts gets its own ``timeout``; 429 and 5xx gateway errors,
    timeouts and network errors are retried, other 4xx responses are returned at once.
    """
    key = _api_key()
    if not key:
        return _fail("missing_api_key",
                     "No DeepSeek key: set DEEPSEEK_API_KEY or a readable DEEPSEEK_AUTH_JSON.")

    url = f"{_base_url()}{path}"
    headers = {"Authorization": f"Bearer {key}"}
    last: dict[str, Any] = _fail("request_failed", "No attempt was made.")

    for attempt in range(retries):
        delay = DEFAULT_RETRY_DELAY
        try:
            result, retry_delay = await _attempt(method, url, path, payload=payload,
                                                 headers=headers, timeout=timeout)
            if retry_delay is None:
                return result
            last, delay = result, retry_delay
        except httpx.TimeoutException:
            last = _fail("timeout", f"Request to {path} timed out after {timeout:.0f}s.")
        except httpx.RequestError as exc:
            last = _fail("network_error", f"{type(exc).__name__}: {exc}")
        if attempt + 1 < retries:
            await asyncio.sleep(delay)
    return last
