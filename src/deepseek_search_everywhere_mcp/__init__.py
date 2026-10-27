"""MCP server for the DeepSeek Search & Fetch API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .server import mcp

try:
    __version__ = version("deepseek-search-everywhere-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "mcp"]
