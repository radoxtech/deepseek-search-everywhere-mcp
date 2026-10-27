"""Tests for the command-line interface (no network, no real listener)."""

from __future__ import annotations

import ast
import importlib.metadata
import logging
import pathlib
import sys
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import cli, http_app, server

TOKEN_VAR = "MCP_BEARER_TOKEN"


def test_help_lists_all_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for option in ("--transport", "--host", "--port", "--path", "--check", "--version"):
        assert option in out
    assert "--token" not in out


def test_version_matches_package_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.split()[-1] == importlib.metadata.version(cli.PROG)


def test_check_reports_balance_without_key_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-test-secret-value"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    async def balance() -> dict[str, Any]:
        return {"ok": True, "balance": "1.00"}

    monkeypatch.setattr(server, "deepseek_balance", balance)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--check"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert '"balance": "1.00"' in out
    assert "DeepSeek key: set" in out
    assert "len=" not in out
    assert secret not in out


def test_check_without_key_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--check"])
    assert exc.value.code == 1
    assert "MISSING" in capsys.readouterr().out


def test_check_conflicts_with_http_transport(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--check", "--transport", "streamable-http"])
    assert exc.value.code == 2
    assert "--check" in capsys.readouterr().err


def test_http_without_token_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(TOKEN_VAR, raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--transport", "streamable-http"])
    assert exc.value.code == 2
    assert TOKEN_VAR in capsys.readouterr().err


def test_http_without_uvicorn_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(TOKEN_VAR, "t0ken")
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--transport", "streamable-http"])
    assert exc.value.code == 2
    assert f"{cli.PROG}[http]" in capsys.readouterr().err


def test_http_runs_uvicorn_with_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("uvicorn")
    import uvicorn

    monkeypatch.setenv(TOKEN_VAR, "t0ken")
    calls: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))
    built: dict[str, Any] = {}

    def fake_build(settings: Any, *, token: str) -> str:
        built.update(settings=settings, token=token)
        return "app"

    monkeypatch.setattr(cli, "build_http_app", fake_build)
    cli.main(["--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9001", "--path", "/x"])
    assert calls == {"app": "app", "host": "0.0.0.0", "port": 9001, "log_level": "warning"}
    assert built["token"] == "t0ken"
    assert built["settings"].http_path == "/x"


def test_http_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("uvicorn")
    import uvicorn

    monkeypatch.setenv(TOKEN_VAR, "t0ken")
    calls: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(kwargs))
    monkeypatch.setattr(cli, "build_http_app", lambda settings, *, token: "app")
    cli.main(["--transport", "streamable-http"])
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8000)


def test_invalid_port_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_VAR, "t0ken")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--transport", "streamable-http", "--port", "70000"])
    assert exc.value.code == 2


def test_stdio_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[bool] = []
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: ran.append(True))
    cli.main([])
    assert ran == [True]


def test_uvicorn_is_never_imported_at_module_level() -> None:
    """``uvicorn`` is an optional extra: only function-level imports are allowed.

    (Importing the module is not a usable check: the ``mcp`` dependency pulls uvicorn in.)
    """
    source = pathlib.Path(cli.__file__).parent
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            assert not any(n.split(".")[0] == "uvicorn" for n in names), path.name


def test_http_app_module_is_the_one_cli_uses() -> None:
    assert cli.build_http_app is http_app.build_http_app


@pytest.fixture
def logger_levels() -> Any:
    """Restore the levels the CLI sets on its loggers."""
    saved = {name: logging.getLogger(name).level for name in cli.LOGGERS}
    yield
    for name, level in saved.items():
        logging.getLogger(name).setLevel(level)


def test_default_log_level_silences_mcp_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, logger_levels: None
) -> None:
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: None)
    cli.main([])
    sdk = logging.getLogger("mcp.server.lowlevel.server")
    sdk.info("Processing request of type %s", "CallToolRequest")
    sdk.warning("still visible")
    assert [record.getMessage() for record in caplog.records if record.name == sdk.name] == ["still visible"]
    assert not logging.getLogger(cli.LOGGERS[0]).isEnabledFor(logging.INFO)


def test_log_level_env_enables_debug(monkeypatch: pytest.MonkeyPatch, logger_levels: None) -> None:
    monkeypatch.setenv("DEEPSEEK_MCP_LOG_LEVEL", "debug")
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: None)
    cli.main([])
    assert logging.getLogger("mcp.server.lowlevel.server").isEnabledFor(logging.INFO)
    assert logging.getLogger(cli.LOGGERS[0]).isEnabledFor(logging.DEBUG)


def test_invalid_log_level_exits_2(monkeypatch: pytest.MonkeyPatch, logger_levels: None) -> None:
    monkeypatch.setenv("DEEPSEEK_MCP_LOG_LEVEL", "loud")
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_servers_default_to_a_quiet_log_level() -> None:
    assert server.create_server().settings.log_level == "WARNING"
    assert server.create_server(log_level="DEBUG").settings.log_level == "DEBUG"
