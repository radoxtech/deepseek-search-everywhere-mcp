"""Structured JSON output handling for DeepSeek (ADR 0009).

DeepSeek rejects native json_schema with HTTP 400. This module enforces
response_format={"type": "json_object"}, validates against Pydantic models in Python,
and retries once if validation fails.
"""

from __future__ import annotations

from typing import Any

import openai
from pydantic import BaseModel, ValidationError


class StructuredOutputError(Exception):
    """Raised when structured JSON output validation fails after retries."""

    def __init__(self, message: str, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


async def generate_structured[T: BaseModel](
    sdk_client: openai.AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    max_tokens: int | None = 2048,
    temperature: float | None = 0.0,
    thinking: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    attempts: int = 2,
) -> tuple[T, dict[str, Any]]:
    """Request JSON from DeepSeek, validate with Pydantic, and retry once on schema mismatch.

    Returns (validated_model_instance, usage_dict).
    """
    current_messages = list(messages)
    last_error: str | None = None
    last_content: str | None = None

    extra_body: dict[str, Any] = {"response_format": {"type": "json_object"}}
    if thinking is not None:
        extra_body["thinking"] = thinking

    for attempt in range(attempts):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": current_messages,
            "extra_body": extra_body,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        response = await sdk_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        last_content = content

        usage_dict = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
        }

        try:
            parsed = schema.model_validate_json(content)
            return parsed, usage_dict
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            if attempt + 1 < attempts:
                current_messages.append({"role": "assistant", "content": content})
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The response did not match schema: {last_error}. "
                            "Return ONLY valid JSON correcting these errors."
                        ),
                    }
                )

    raise StructuredOutputError(
        f"Failed to produce structured output matching {schema.__name__}: {last_error}",
        raw_content=last_content,
    )
