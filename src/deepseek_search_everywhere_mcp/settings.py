"""Runtime settings: documented defaults as constants, two environment switches.

Every number below is a default baked into the :class:`Settings` fields; tests (and
embedders) override them through the constructor. Only ``DEEPSEEK_MCP_CACHE`` and
``DEEPSEEK_MCP_LOG_LEVEL`` are read from the environment. HTTP host, port and path
come from the command line, not from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any

MIB = 1024 * 1024

ENV_CACHE = "DEEPSEEK_MCP_CACHE"
ENV_LOG_LEVEL = "DEEPSEEK_MCP_LOG_LEVEL"
ENV_DAILY_BUDGET = "DEEPSEEK_DAILY_BUDGET_USD"
ENV_FETCH_ALLOW_DOMAINS = "DEEPSEEK_FETCH_ALLOW_DOMAINS"
ENV_FETCH_DENY_DOMAINS = "DEEPSEEK_FETCH_DENY_DOMAINS"
ENV_EGRESS_ENTROPY_CHECK = "DEEPSEEK_EGRESS_ENTROPY_CHECK"

# Default level for this package's, the MCP SDK's and uvicorn's loggers: quiet unless asked.
DEFAULT_LOG_LEVEL = "WARNING"

# deepseek_fetch_many never runs more fetches than this at once.
MAX_BATCH_CONCURRENCY = 5


@dataclass(frozen=True, slots=True)
class LLMRates:
    cache_hit_per_1m: float
    cache_miss_per_1m: float
    output_per_1m: float


@dataclass(frozen=True, slots=True)
class LLMPricing:
    peak: LLMRates = LLMRates(cache_hit_per_1m=0.006, cache_miss_per_1m=0.30, output_per_1m=1.20)
    off_peak: LLMRates = LLMRates(cache_hit_per_1m=0.003, cache_miss_per_1m=0.15, output_per_1m=0.60)


DEFAULT_LLM_PRICING = LLMPricing()


def get_pricing_period(when: datetime | None = None) -> str:
    """Return 'peak' or 'off_peak' based on the UTC timestamp.

    Peak: Mon-Fri 01:00-04:00 and 06:00-10:00 UTC.
    All other times (weekends, other UTC hours) are off-peak.
    """
    if when is None:
        when = datetime.now(UTC)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    else:
        when = when.astimezone(UTC)

    if when.weekday() in (5, 6):
        return "off_peak"

    hour = when.hour
    if (1 <= hour < 4) or (6 <= hour < 10):
        return "peak"
    return "off_peak"


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
    chat_qps: float = 10
    chat_burst: int = 20
    limiter_max_wait: float = 5

    # Batch fetching.
    batch_max_urls: int = 10
    batch_concurrency: int = 5
    batch_deadline: float = 120

    # Research tool budget.
    research_max_pages: int = 5
    research_gap_deadline: float = 15.0

    # Daily spend budget limit in USD (None means disabled).
    daily_budget_usd: float | None = None

    # LLM pricing rates for peak and off-peak periods.
    llm_pricing: LLMPricing = DEFAULT_LLM_PRICING

    # HTTP transport defaults (the CLI may override them).
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    http_path: str = "/mcp"

    # Egress guard settings (S2, S10).
    fetch_allow_domains: tuple[str, ...] | None = None
    fetch_deny_domains: tuple[str, ...] | None = None
    egress_entropy_check: bool = False

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
            if name in {"limiter_max_wait", "batch_deadline", "research_gap_deadline"} and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if self.daily_budget_usd is not None and not self.daily_budget_usd > 0:
            raise ValueError(f"daily_budget_usd must be > 0, got {self.daily_budget_usd!r}")
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
        if self.fetch_allow_domains is not None:
            if not isinstance(self.fetch_allow_domains, tuple):
                object.__setattr__(self, "fetch_allow_domains", tuple(self.fetch_allow_domains))
            for d in self.fetch_allow_domains:
                if not isinstance(d, str) or not d.strip() or "/" in d or ":" in d:
                    raise ValueError(f"fetch_allow_domains must contain valid domain names, got {d!r}")
        if self.fetch_deny_domains is not None:
            if not isinstance(self.fetch_deny_domains, tuple):
                object.__setattr__(self, "fetch_deny_domains", tuple(self.fetch_deny_domains))
            for d in self.fetch_deny_domains:
                if not isinstance(d, str) or not d.strip() or "/" in d or ":" in d:
                    raise ValueError(f"fetch_deny_domains must contain valid domain names, got {d!r}")

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
        budget = os.environ.get(ENV_DAILY_BUDGET, "").strip()
        if budget:
            try:
                val = float(budget)
                if val <= 0:
                    raise ValueError
                overrides["daily_budget_usd"] = val
            except ValueError:
                raise ValueError(
                    f"daily_budget_usd: {ENV_DAILY_BUDGET} must be a positive number, got {budget!r}"
                ) from None
        allow = os.environ.get(ENV_FETCH_ALLOW_DOMAINS, "").strip()
        if allow:
            overrides["fetch_allow_domains"] = tuple(d.strip().lower() for d in allow.split(",") if d.strip())
        deny = os.environ.get(ENV_FETCH_DENY_DOMAINS, "").strip()
        if deny:
            overrides["fetch_deny_domains"] = tuple(d.strip().lower() for d in deny.split(",") if d.strip())
        entropy = os.environ.get(ENV_EGRESS_ENTROPY_CHECK, "").strip().lower()
        if entropy:
            overrides["egress_entropy_check"] = entropy in {"1", "true", "yes"}
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
