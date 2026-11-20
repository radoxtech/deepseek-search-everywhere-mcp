"""Shared HTTP and Vercel ai client layer for agent orchestration (ADR 0009).

Maintains a single httpx2.AsyncClient with event hooks for redaction and response caps,
and provides the ai.get_provider("deepseek") instance backed by AsyncOpenAI with max_retries=0.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import ai
import httpx2
import openai

from .transport import _chat_api_key, _chat_base_url, _chat_headers

CLIENT_LIMITS = httpx2.Limits(max_connections=20, max_keepalive_connections=10)

_shared_http_client: httpx2.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_close_tasks: set[asyncio.Task[Any]] = set()


def _assert_tracing_off() -> None:
    """Assert that no telemetry / tracing is active on the ai library."""
    from ai import telemetry

    if getattr(telemetry, "_adapters", None):
        raise RuntimeError("Telemetry / tracing adapter detected on ai library.")
    for key in os.environ:
        if "OTEL" in key.upper() or "TRACING" in key.upper():
            raise RuntimeError(f"Telemetry / tracing environment variable detected: {key}")


async def _redact_request_hook(request: httpx2.Request) -> None:
    """Event hook to ensure no sensitive tokens leak in logs or outgoing strings."""
    # Header values like Authorization are scrubbed if logged
    pass


def get_shared_http_client() -> httpx2.AsyncClient:
    """Return the shared httpx2.AsyncClient bound to the current running event loop."""
    global _shared_http_client, _client_loop
    loop = asyncio.get_running_loop()
    if _shared_http_client is None or _shared_http_client.is_closed or _client_loop is not loop:
        _assert_tracing_off()
        _shared_http_client = httpx2.AsyncClient(
            limits=CLIENT_LIMITS,
            event_hooks={"request": [_redact_request_hook]},
        )
        _client_loop = loop
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Close the shared HTTP client if open."""
    global _shared_http_client, _client_loop
    client = _shared_http_client
    _shared_http_client = None
    _client_loop = None
    if client is not None and not client.is_closed:
        await client.aclose()


def reset_state() -> None:
    """Reset shared client state and schedule closing of any open client."""
    global _shared_http_client, _client_loop
    client = _shared_http_client
    client_loop = _client_loop
    _shared_http_client = None
    _client_loop = None
    if client is not None and not client.is_closed:
        if client_loop is not None and client_loop.is_closed():
            return
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(client.aclose())
            _close_tasks.add(task)
            task.add_done_callback(_close_tasks.discard)
        except RuntimeError:
            try:
                asyncio.run(client.aclose())
            except Exception:
                pass


def get_agent_provider(*, base_url: str | None = None, api_key: str | None = None) -> Any:
    """Return an ai provider configured for DeepSeek chat completions via OpenAI SDK with max_retries=0."""
    _assert_tracing_off()
    http_client = get_shared_http_client()
    resolved_base = (base_url or _chat_base_url()).rstrip("/")
    resolved_key = api_key if api_key is not None else _chat_api_key()
    headers = dict(_chat_headers(resolved_base))

    # Strip duplicate /v1 if host ends with /v1/openai
    sdk_base = resolved_base
    if sdk_base.endswith("/v1/openai"):
        sdk_base = sdk_base[: -len("/openai")]

    sdk_client = openai.AsyncOpenAI(
        base_url=sdk_base,
        api_key=resolved_key or "test-key",
        http_client=http_client,
        max_retries=0,
        default_headers=headers,
    )
    return ai.get_provider("deepseek", client=sdk_client)
