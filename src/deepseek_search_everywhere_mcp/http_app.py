"""Streamable HTTP application with a mandatory bearer token.

``build_http_app()`` returns a Starlette application serving the MCP tools over streamable
HTTP (stateless, JSON responses). Every request to the MCP path must carry
``Authorization: Bearer <token>``; the token comes from ``MCP_BEARER_TOKEN`` unless passed in.
Serving the application (for example with uvicorn) is left to the caller.
"""

from __future__ import annotations

import json
import os
import secrets

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from .server import SERVER_NAME, create_server
from .settings import Settings, get_settings

ENV_BEARER_TOKEN = "MCP_BEARER_TOKEN"

_UNAUTHORIZED_BODY = json.dumps({"error": "unauthorized"}).encode()
_WWW_AUTHENTICATE = f'Bearer realm="{SERVER_NAME}"'.encode()


def _normalize_path(path: str) -> str:
    return path.rstrip("/") or "/"


class BearerAuthMiddleware:
    """Pure ASGI middleware that rejects MCP requests without the expected bearer token.

    Only HTTP requests to ``path`` (with or without a trailing slash) are checked; lifespan
    events and other paths pass through untouched. The request body is never read.
    """

    def __init__(self, app: ASGIApp, token: bytes, path: str = "/mcp") -> None:
        if not token:
            raise ValueError("bearer token must not be empty")
        self.app = app
        self._token = token
        self._path = _normalize_path(path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _normalize_path(scope["path"]) != self._path:
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                    (b"www-authenticate", _WWW_AUTHENTICATE),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})

    def _authorized(self, scope: Scope) -> bool:
        values = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        if len(values) != 1:
            return False
        scheme, _, credentials = values[0].strip().partition(b" ")
        if scheme.lower() != b"bearer":
            return False
        return secrets.compare_digest(credentials.strip(), self._token)


def _resolve_token(token: str | None) -> bytes:
    value = os.environ.get(ENV_BEARER_TOKEN, "") if token is None else token
    value = value.strip()
    if not value:
        raise ValueError(f"HTTP transport requires a bearer token: set {ENV_BEARER_TOKEN}")
    return value.encode("utf-8")


def build_http_app(settings: Settings | None = None, *, token: str | None = None) -> Starlette:
    """Build a new bearer-protected Starlette application for the streamable HTTP transport.

    Each call creates a fresh FastMCP server, because a session manager can run only once.
    DNS-rebinding protection is disabled on purpose: the bearer token is the trust boundary,
    and the Host check would answer 421 to legitimate clients when binding to ``0.0.0.0``.
    Raises ``ValueError`` when no token is given and ``MCP_BEARER_TOKEN`` is unset or empty.
    """
    settings = settings or get_settings()
    secret = _resolve_token(token)
    server = create_server(
        host=settings.http_host,
        port=settings.http_port,
        streamable_http_path=settings.http_path,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    app = server.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=secret, path=settings.http_path)
    return app
