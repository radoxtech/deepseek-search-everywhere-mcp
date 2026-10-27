"""Runtime settings: documented defaults as constants, two environment switches.

Every number below is a default baked into the :class:`Settings` fields; tests (and
embedders) override them through the constructor. Only ``DEEPSEEK_MCP_CACHE`` and
``DEEPSEEK_MCP_LOG_LEVEL`` are read from the environment. HTTP host, port and path
come from the command line, not from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

MIB = 1024 * 1024

ENV_CACHE = "DEEPSEEK_MCP_CACHE"
ENV_LOG_LEVEL = "DEEPSEEK_MCP_LOG_LEVEL"

# Default level for this package's, the MCP SDK's and uvicorn's loggers: quiet unless asked.
DEFAULT_LOG_LEVEL = "WARNING"

# deepseek_fetch_many never runs more fetches than this at once.
MAX_BATCH_CONCURRENCY = 5


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable server configuration; invalid values raise ``ValueError`` naming the field."""

    cache_enabled: bool = True
    log_level: str = DEFAULT_LOG_LEVEL

    # Response caches: TTL in seconds, entry limit, total size limit in bytes.
    cache_search_ttl: float = 300
    cache_search_entries: int = 128
    cache_search_bytes: int = 16 * MIB
    cache_fetch_ttl: float = 1800
    cache_fetch_entries: int = 128
    cache_fetch_bytes: int = 32 * MIB
    cache_rewrite_ttl: float = 86400
    cache_rewrite_entries: int = 1024
    cache_rewrite_bytes: int = 1 * MIB

    # Upstream response size caps in bytes.
    response_cap_search: int = 8 * MIB
    response_cap_fetch: int = 32 * MIB

    # Per-endpoint token buckets: sustained rate, burst size, longest wait for a token.
    search_qps: float = 10
    search_burst: int = 20
    fetch_qps: float = 5
    fetch_burst: int = 10
    limiter_max_wait: float = 5

    # Batch fetching.
    batch_max_urls: int = 10
    batch_concurrency: int = 5
    batch_deadline: float = 120

    # Research tool budget.
    research_max_pages: int = 5

    # HTTP transport defaults (the CLI may override them).
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    http_path: str = "/mcp"

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            name = field.name
            if name.endswith("_qps") and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            if name.endswith("_ttl") and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")
            if name.endswith(("_bytes", "_entries", "_burst", "_pages", "_urls")) and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            if name.startswith("response_cap_") and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            if name in {"limiter_max_wait", "batch_deadline"} and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if not 1 <= self.batch_concurrency <= MAX_BATCH_CONCURRENCY:
            raise ValueError(
                f"batch_concurrency must be between 1 and {MAX_BATCH_CONCURRENCY}, "
                f"got {self.batch_concurrency!r}"
            )
        if not 0 < self.http_port < 65536:
            raise ValueError(f"http_port must be between 1 and 65535, got {self.http_port!r}")
        if not self.http_path.startswith("/"):
            raise ValueError(f"http_path must start with '/', got {self.http_path!r}")
        if self.log_level.upper() not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(_LOG_LEVELS)}, got {self.log_level!r}")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from defaults plus ``DEEPSEEK_MCP_CACHE`` and ``DEEPSEEK_MCP_LOG_LEVEL``."""
        overrides: dict[str, Any] = {}
        cache = os.environ.get(ENV_CACHE, "").strip()
        if cache:
            if cache not in {"0", "1"}:
                raise ValueError(f"cache_enabled: {ENV_CACHE} must be 0 or 1, got {cache!r}")
            overrides["cache_enabled"] = cache == "1"
        level = os.environ.get(ENV_LOG_LEVEL, "").strip()
        if level:
            overrides["log_level"] = level.upper()
        return cls(**overrides)


_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})

_current: Settings | None = None


def get_settings() -> Settings:
    """Return the active settings, building them from the environment on first use."""
    global _current
    if _current is None:
        _current = Settings.from_env()
    return _current


def set_settings(settings: Settings | None) -> None:
    """Swap the active settings (``None`` re-reads the environment on the next access)."""
    global _current
    _current = settings


def reset_state() -> None:
    """Forget the active settings so the next :func:`get_settings` rebuilds them."""
    set_settings(None)
