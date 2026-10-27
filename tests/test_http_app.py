"""Bearer gate and lifespan of the streamable HTTP application (no real listener)."""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from deepseek_search_everywhere_mcp.http_app import BearerAuthMiddleware, build_http_app
from deepseek_search_everywhere_mcp.settings import Settings

TOKEN = "s3cret-token"
BASE_URL = "http://127.0.0.1:8000"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _post_initialize(client: TestClient, headers: dict[str, str] | None = None):
    return client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **(headers or {})})


def test_auth_missing_header_is_rejected() -> None:
    with TestClient(build_http_app(Settings(), token=TOKEN), base_url=BASE_URL) as client:
        response = _post_initialize(client)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert response.json() == {"error": "unauthorized"}


@pytest.mark.parametrize("value", ["Bearer wrong", f"Basic {TOKEN}", TOKEN, "Bearer", "Bearer żółć"])
def test_auth_wrong_credentials_are_rejected(value: str) -> None:
    with TestClient(build_http_app(Settings(), token=TOKEN), base_url=BASE_URL) as client:
        response = client.post(
            "/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": value.encode()}
        )
    assert response.status_code == 401


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_auth_valid_token_reaches_mcp(scheme: str) -> None:
    with TestClient(build_http_app(Settings(), token=TOKEN), base_url=BASE_URL) as client:
        response = _post_initialize(client, {"Authorization": f"{scheme} {TOKEN}"})
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "deepseek-search-everywhere-mcp"


def test_auth_non_ascii_token_is_compared_as_bytes() -> None:
    token = "zażółć-gęślą"
    app = build_http_app(Settings(), token=token)
    with TestClient(app, base_url=BASE_URL) as client:
        ok = client.post(
            "/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": f"Bearer {token}".encode()}
        )
        bad = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": "Bearer x"})
    assert ok.status_code == 200
    assert bad.status_code == 401


def test_auth_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_BEARER_TOKEN", TOKEN)
    with TestClient(build_http_app(Settings()), base_url=BASE_URL) as client:
        assert _post_initialize(client).status_code == 401
        assert _post_initialize(client, {"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_auth_token_is_always_required() -> None:
    with pytest.raises(ValueError, match="MCP_BEARER_TOKEN") as excinfo:
        build_http_app(Settings())
    with pytest.raises(ValueError):
        build_http_app(Settings(), token="  ")
    assert TOKEN not in str(excinfo.value)


def test_auth_only_guards_the_mcp_path() -> None:
    async def other(request: Request) -> PlainTextResponse:
        return PlainTextResponse("open")

    inner = Starlette(routes=[Route("/mcp", other), Route("/other", other)])
    app = BearerAuthMiddleware(inner, token=TOKEN.encode(), path="/mcp")
    client = TestClient(app, base_url=BASE_URL)
    assert client.get("/other").status_code == 200
    assert client.get("/mcp").status_code == 401
    assert client.get("/mcp/", follow_redirects=False).status_code == 401
    assert client.get("/mcp", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


async def test_lifespan_passes_without_token() -> None:
    seen: list[str] = []

    async def inner(scope, receive, send) -> None:
        seen.append(scope["type"])

    async def receive() -> dict:
        return {"type": "lifespan.startup"}

    async def send(message: dict) -> None:
        raise AssertionError("middleware must not answer lifespan events")

    await BearerAuthMiddleware(inner, token=b"t")({"type": "lifespan"}, receive, send)
    assert seen == ["lifespan"]


def test_lifespan_fresh_apps_start_and_stop_twice() -> None:
    first = build_http_app(Settings(), token=TOKEN)
    second = build_http_app(Settings(), token=TOKEN)
    assert isinstance(first, Starlette)
    assert first is not second
    for app in (first, second):
        with TestClient(app, base_url=BASE_URL) as client:
            assert _post_initialize(client, {"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_no_health_route() -> None:
    app = build_http_app(Settings(), token=TOKEN)
    paths = {getattr(route, "path", None) for route in app.routes}
    assert paths == {"/mcp"}
