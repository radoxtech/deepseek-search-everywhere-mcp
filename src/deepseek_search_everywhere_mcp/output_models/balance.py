"""The ``deepseek_check_balance`` output model: account balance plus this session's usage."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from . import Envelope


class BalanceOutput(Envelope):
    """Top-level shape of ``deepseek_check_balance`` in ``structuredContent``.

    Inherits ``ok``, ``_meta`` and ``error`` from :class:`Envelope`. Every field the tool emits is
    declared here; the upstream returns ``balances`` either as a list of currency objects or as the
    raw mapping, so the field accepts both.
    """

    is_available: bool | None = Field(
        default=None,
        description="True when the account can spend, per the upstream flag.",
    )
    balances: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="Per-currency balances: a list or the raw upstream mapping.",
    )
    session: dict[str, Any] | None = Field(
        default=None,
        description="Session metrics: requests, cache hits, hit rate, spend.",
    )
    elapsed_ms: int | None = Field(
        default=None,
        description="Wall time of the call in milliseconds.",
    )


__all__ = ["BalanceOutput"]
