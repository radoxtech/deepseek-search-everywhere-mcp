"""Pydantic models shared by every tool envelope: ``Envelope``, ``Meta`` and ``ErrorInfo``.

The package is the single source of truth for the public JSON shape. Per-tool result models live in
sibling modules (``output_models.search``, ``output_models.fetch`` and so on); this module holds only
the fields every tool publishes.

``Envelope.meta`` is aliased to ``_meta`` because pydantic treats leading-underscore *field names* as
private attributes: the Python attribute stays ``meta`` while the published JSON property is ``_meta``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Meta(BaseModel):
    """The ``_meta`` block published by every tool: estimated cost and security classification."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    cost_usd: float = Field(description="Estimated cost of the request in US dollars.")
    untrusted_content: bool = Field(
        description="True when the response carries untrusted content retrieved from the web."
    )


class ErrorInfo(BaseModel):
    """The ``error`` object of a failed tool response."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    code: str = Field(description="Stable error code from the closed set in errors.py.")
    message: str = Field(description="Human-readable explanation of the failure.")
    action: str | None = Field(
        default=None,
        description="Suggested next step for the caller: retry, fix the call or change the approach.",
    )
    reason: str | None = Field(
        default=None,
        description="Fine-grained failure reason from the closed reason set in errors.py.",
    )
    param: str | None = Field(
        default=None,
        description="Name of the offending input parameter, for input errors.",
    )
    status: int | None = Field(
        default=None,
        description="Upstream HTTP status code, when one was received.",
    )
    upstream_code: str | None = Field(
        default=None,
        description="Upstream provider error code, when one was received.",
    )
    upstream_message: str | None = Field(
        default=None,
        description="Redacted upstream error message, when one was received.",
    )


class Envelope(BaseModel):
    """Shared top-level shape of every tool response envelope."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ok: bool = Field(description="True when the tool call succeeded, false on any error.")
    meta: Meta | None = Field(
        default=None,
        alias="_meta",
        serialization_alias="_meta",
        description="Cost and security metadata, published in JSON as ``_meta``.",
    )
    error: ErrorInfo | None = Field(
        default=None,
        description="Error details, present when ``ok`` is false.",
    )


__all__ = ["Envelope", "ErrorInfo", "Meta"]
