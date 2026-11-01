"""Structured data extraction engine for web pages.

Constructs extraction prompts for deepseek-flash, sanitizes JSON outputs,
and validates target fields or schemas without external dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def build_extract_prompt(
    fields: list[str] | None, schema: dict[str, Any] | None, instruction: str | None
) -> str:
    """Construct system prompt instructing deepseek-flash to extract target data."""
    parts = [
        "You are a structured data extractor.",
        (
            "Your task is to extract information from the provided web page text and "
            "return ONLY a valid JSON object."
        ),
        "Do not include markdown code fences (like ```json), no explanations, and no preamble.",
    ]
    if fields:
        field_list = ", ".join(f'"{f}"' for f in fields)
        parts.append(
            f"Extract the following target fields into a JSON object with these exact keys: [{field_list}]."
        )
        parts.append("If a field cannot be found or is not mentioned in the text, set its value to null.")
    elif schema:
        formatted_schema = json.dumps(schema, indent=2)
        parts.append(
            f"Extract data into a JSON object strictly conforming to this schema:\n{formatted_schema}"
        )

    if instruction:
        parts.append(f"Additional guidance: {instruction.strip()}")

    return "\n\n".join(parts)


def sanitize_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Strip markdown code fences and whitespace, extract JSON object between braces.

    Returns ``(parsed_dict, None)`` on success, or ``(None, error_message)`` on failure.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "Empty response from extraction model."

    # Strip code block fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    # Find the outermost braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, f"No JSON object found in extraction response: {raw[:150]!r}."

    candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"Failed to parse extraction JSON: {exc}."

    if not isinstance(parsed, dict):
        return None, f"Expected a JSON object, got {type(parsed).__name__}."

    return parsed, None


def validate_fields(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Ensure all requested ``fields`` are present in ``data``, populating missing ones with None."""
    result = dict(data)
    for field in fields:
        if field not in result:
            result[field] = None
    return result


def validate_schema(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Perform lightweight validation of ``data`` against ``schema`` (required and top-level types)."""
    errors: list[str] = []
    required = schema.get("required") or []
    if isinstance(required, list):
        for req in required:
            if req not in data or data[req] is None:
                errors.append(f"Missing required property: {req!r}")

    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for prop, prop_schema in properties.items():
            if prop in data and data[prop] is not None and isinstance(prop_schema, dict):
                expected_type = prop_schema.get("type")
                if expected_type in _TYPE_MAP:
                    py_type = _TYPE_MAP[expected_type]
                    val = data[prop]
                    if expected_type in ("integer", "number") and isinstance(val, bool):
                        errors.append(f"Property {prop!r} expected {expected_type}, got boolean.")
                    elif not isinstance(val, py_type):
                        errors.append(
                            f"Property {prop!r} expected {expected_type}, got {type(val).__name__}."
                        )

    return errors
