"""MCP server for the DeepSeek Search & Fetch API."""

from __future__ import annotations

from ._version import __version__
from .server import mcp

__all__ = ["__version__", "mcp"]
