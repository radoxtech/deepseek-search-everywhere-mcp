"""Command-line entry point for the DeepSeek Search Everywhere MCP server."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from . import server

PROG = "deepseek-search-everywhere-mcp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="MCP server for the DeepSeek Search & Fetch API (stdio transport).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify credentials and connectivity, then exit without starting MCP",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the server or the connectivity check."""
    args = _build_parser().parse_args(argv)
    if args.check:
        raise SystemExit(asyncio.run(server._check()))
    server.mcp.run()
