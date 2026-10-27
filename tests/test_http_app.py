"""Bearer gate, lifespan and JSON-RPC behaviour of the streamable HTTP application.

Everything runs in-process through ``TestClient``: no uvicorn, no port, no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from deepseek_search_everywhere_mcp import server
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
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def http_app() -> Starlette:
    """A fresh application per test: a session manager can only run once."""
    return build_http_app(Settings(), token=TOKEN)


@pytest.fixture
def client(http_app: Starlette) -> Iterator[TestClient]:
    """A loopback client inside ``with``, so the application lifespan runs."""
    with TestClient(http_app, base_url=BASE_URL) as test_client:
        yield test_client


def _rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _post_initialize(client: TestClient, headers: dict[str, str] | None = None):
    return client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **(headers or {})})


def test_auth_missing_header_is_rejected(client: TestClient) -> None:
    response = _post_initialize(client)
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert response.json() == {"error": "unauthorized"}


@pytest.mark.parametrize("value", ["Bearer wrong", f"Basic {TOKEN}", TOKEN, "Bearer", "Bearer żółć"])
def test_auth_wrong_credentials_are_rejected(client: TestClient, value: str) -> None:
    response = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": value.encode()})
    assert response.status_code == 401


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_auth_valid_token_reaches_mcp(client: TestClient, scheme: str) -> None:
    response = _post_initialize(client, {"Authorization": f"{scheme} {TOKEN}"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert body["result"]["serverInfo"]["name"] == "deepseek-search-everywhere-mcp"


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


def test_no_health_route(http_app: Starlette) -> None:
    paths = {getattr(route, "path", None) for route in http_app.routes}
    assert paths == {"/mcp"}


def test_tools_list_returns_every_tool(client: TestClient) -> None:
    response = client.post("/mcp", json=_rpc("tools/list"), headers={**MCP_HEADERS, **AUTH})
    assert response.status_code == 200
    names = [tool["name"] for tool in response.json()["result"]["tools"]]
    assert names == [tool.__name__ for tool in server.TOOLS]
    assert len(names) == 6


def test_tools_call_uses_stubbed_upstream(client: TestClient, fake_upstream: Any) -> None:
    balance = {"ok": True, "data": {"is_available": True, "balance_infos": [{"currency": "USD"}]}}
    upstream = fake_upstream(default=balance)
    response = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "deepseek_balance", "arguments": {}}),
        headers={**MCP_HEADERS, **AUTH},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["is_available"] is True
    assert [(call["method"], call["path"]) for call in upstream.calls] == [("GET", "/user/balance")]


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"content": b"{not json", "headers": {"Content-Type": "application/json"}}, 400),
        ({"json": {"foo": 1}}, 400),
        ({"content": b"hello", "headers": {"Content-Type": "text/plain"}}, 400),
        ({"json": INITIALIZE, "headers": {"Accept": "text/html"}}, 406),
    ],
    ids=["invalid-json", "not-json-rpc", "wrong-content-type", "wrong-accept"],
)
def test_body_errors_with_valid_token_are_not_auth_errors(
    client: TestClient, kwargs: dict[str, Any], status: int
) -> None:
    request = dict(kwargs)
    headers = {**MCP_HEADERS, **AUTH, **request.pop("headers", {})}
    response = client.post("/mcp", headers=headers, **request)
    assert response.status_code == status
    assert "www-authenticate" not in response.headers


def test_body_errors_without_token_are_auth_errors(client: TestClient) -> None:
    response = client.post(
        "/mcp", content=b"{not json", headers={**MCP_HEADERS, "Content-Type": "application/json"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "mcp.example.org"])
def test_host_header_is_not_rejected(client: TestClient, host: str) -> None:
    # DNS-rebinding protection is off by design (the bearer token is the trust boundary),
    # so no Host value may produce 421 Misdirected Request.
    response = _post_initialize(client, {**AUTH, "Host": host})
    assert response.status_code == 200
