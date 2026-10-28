"""Single source of the installed package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepseek-search-everywhere-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"
