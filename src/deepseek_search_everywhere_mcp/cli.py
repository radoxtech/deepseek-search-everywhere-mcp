"""Command-line entry point for the DeepSeek Search Everywhere MCP server."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version

from . import server
from .http_app import ENV_BEARER_TOKEN, build_http_app
from .settings import Settings, get_settings

PROG = "deepseek-search-everywhere-mcp"
STDIO = "stdio"
STREAMABLE_HTTP = "streamable-http"

EXIT_USAGE = 2

# Loggers whose level follows DEEPSEEK_MCP_LOG_LEVEL: this package and the MCP SDK, which
# otherwise logs "Processing request of type ..." at INFO for every request.
LOGGERS = (__package__ or "deepseek_search_everywhere_mcp", "mcp")


def _build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="MCP server for the DeepSeek Search & Fetch API.",
        epilog=f"The streamable-http transport always requires the {ENV_BEARER_TOKEN} "
        "environment variable (the token is never accepted on the command line).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version(PROG)}")
    parser.add_argument(
        "--transport",
        choices=(STDIO, STREAMABLE_HTTP),
        default=STDIO,
        help="transport to serve (default: %(default)s)",
    )
    parser.add_argument("--host", help=f"HTTP bind address (default: {defaults.http_host})")
    parser.add_argument("--port", type=int, help=f"HTTP port (default: {defaults.http_port})")
    parser.add_argument("--path", help=f"HTTP endpoint path (default: {defaults.http_path})")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify credentials and connectivity, then exit without starting MCP",
    )
    return parser


def _configure_logging(level: str) -> None:
    """Send log records to stderr and apply ``level`` to this package's and the SDK's loggers.

    The levels are set on the loggers themselves, so they hold even when the root logger was
    already configured (FastMCP configures it once when the first server is built).
    """
    level = level.upper()
    logging.basicConfig(level=level)
    for name in LOGGERS:
        logging.getLogger(name).setLevel(level)


def _http_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Settings:
    """Overlay the CLI options on the environment-derived settings (CLI wins)."""
    overrides = {
        field: value
        for field, value in (
            ("http_host", args.host),
            ("http_port", args.port),
            ("http_path", args.path),
        )
        if value is not None
    }
    try:
        return dataclasses.replace(get_settings(), **overrides)
    except ValueError as exc:
        parser.error(str(exc))


def _serve_http(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    token = os.environ.get(ENV_BEARER_TOKEN, "").strip()
    if not token:
        parser.exit(EXIT_USAGE, f"{PROG}: error: set {ENV_BEARER_TOKEN} to serve over HTTP\n")
    settings = _http_settings(parser, args)
    try:
        import uvicorn
    except ImportError:
        parser.exit(
            EXIT_USAGE,
            f"{PROG}: error: install {PROG}[http] to use the streamable-http transport\n",
        )
    uvicorn.run(
        build_http_app(settings, token=token),
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the server or the connectivity check."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _configure_logging(get_settings().log_level)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check:
        if args.transport != STDIO:
            parser.error("--check cannot be combined with --transport streamable-http")
        sys.exit(asyncio.run(server._check()))
    if args.transport == STREAMABLE_HTTP:
        _serve_http(parser, args)
    else:
        server.mcp.run()
