"""Structured data extraction engine for web pages.

Constructs extraction prompts for deepseek-flash, sanitizes JSON outputs,
and validates target fields or schemas without external dependencies.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
            f"Extract the following target fields: [{field_list}].\n"
            "Return a JSON object with two top-level keys:\n"
            '1. "data": an object with the extracted values for each field (or null if not found in text).\n'
            '2. "evidence": an object mapping each field name to a short verbatim quote from the page '
            "that directly supports the extracted value (or null if not found)."
        )
    elif schema:
        formatted_schema = json.dumps(schema, indent=2)
        parts.append(
            f"Extract data into a JSON object strictly conforming to this schema:\n{formatted_schema}\n"
            "Return a JSON object with two top-level keys:\n"
            '1. "data": the extracted data conforming to the schema.\n'
            '2. "evidence": an object mapping string fields/leaves to a short verbatim quote from the page '
            "supporting the value (or null if not found)."
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


_QUOTES_RE = re.compile(r'["“”„«»]')
_APOSTROPHES_RE = re.compile(r"['’‘ʻ]")
_DASHES_RE = re.compile(r"[-–—−‐‑]")
_MARKDOWN_MARKERS_RE = re.compile(r"[*_`~]+")
_SUPERSCRIPT_DIGITS_RE = re.compile(r"[\u2070-\u2079\u00b2\u00b3\u00b9]+")
_FOOTNOTE_MARKERS_RE = re.compile(
    r"\[(?:\d+(?:[,\-–—]\s*\d+)*|[a-zA-Z]|citation\s+needed|edit|note\s+\d+|nb\s+\d+)\]",
    re.IGNORECASE,
)
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")


def normalize_for_evidence(text: str) -> str:
    """Normalize text for evidence matching.

    Strips superscript digits and citation/footnote markers ([4], [a], [citation needed], [edit]),
    applies NFKC Unicode normalization, strips markdown emphasis and code markers (*, _, `, ~),
    unifies quotation marks, apostrophes and dashes, collapses whitespace, fixes space before
    punctuation, and lowercases.
    """
    if not isinstance(text, str):
        return ""
    text = _SUPERSCRIPT_DIGITS_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _FOOTNOTE_MARKERS_RE.sub("", text)
    text = _MARKDOWN_MARKERS_RE.sub("", text)
    text = _QUOTES_RE.sub('"', text)
    text = _APOSTROPHES_RE.sub("'", text)
    text = _DASHES_RE.sub("-", text)
    text = " ".join(text.lower().split())
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    return text


def _value_appears_in_quote(val: Any, quote: str, norm_quote: str) -> bool:
    """Check whether a scalar value appears in the evidence quote."""
    if isinstance(val, str):
        norm_val = normalize_for_evidence(val)
        if norm_val and norm_val in norm_quote:
            return True
        # If val is a comma/semicolon-delimited list, check if all items appear in quote
        if "," in val or ";" in val:
            parts = [normalize_for_evidence(p) for p in re.split(r"[,;]", val) if p.strip()]
            if parts and all(p in norm_quote for p in parts):
                return True
        # For string values that look numeric, compare by digits (ignoring
        # thousands separators and spaces) like int values.
        clean_val = re.sub(r"[\s,._']", "", val.strip())
        digits = clean_val.lstrip("+-")
        if digits and digits.isdigit():
            sep_pattern = r"(?<!\d)" + r"[\s,._']?".join(re.escape(d) for d in digits) + r"(?!\d)"
            norm_q = normalize_for_evidence(quote)
            if re.search(sep_pattern, quote) or re.search(sep_pattern, norm_q):
                return True
        if "." in val or "," in val:
            try:
                std_num = val.strip().replace(" ", "").replace("'", "")
                if "," in std_num and "." not in std_num and len(std_num.split(",")[1]) <= 2:
                    std_num = std_num.replace(",", ".")
                else:
                    std_num = std_num.replace(",", "")
                float_val = float(std_num)
                return _value_appears_in_quote(float_val, quote, norm_quote)
            except ValueError:
                pass
        return False
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float)):
        norm_q = normalize_for_evidence(quote)
        if isinstance(val, int):
            digits = str(abs(val))
            sep_pattern = r"(?<!\d)" + r"[\s,._']?".join(re.escape(d) for d in digits) + r"(?!\d)"
            return bool(re.search(sep_pattern, quote) or re.search(sep_pattern, norm_q))
        elif isinstance(val, float):
            if val.is_integer():
                digits = str(abs(int(val)))
                sep_pattern = r"(?<!\d)" + r"[\s,._']?".join(re.escape(d) for d in digits) + r"(?!\d)"
                if re.search(sep_pattern, quote) or re.search(sep_pattern, norm_q):
                    return True
            parts = f"{abs(val):.6f}".rstrip("0").rstrip(".").split(".")
            int_part = parts[0]
            frac_part = parts[1] if len(parts) > 1 else ""
            if frac_part:
                int_pat = r"[\s,._']?".join(re.escape(d) for d in int_part)
                frac_pat = r"[\s,._']?".join(re.escape(d) for d in frac_part)
                sep_pattern = r"(?<!\d)" + int_pat + r"[\.,]" + frac_pat + r"(?!\d)"
                return bool(re.search(sep_pattern, quote) or re.search(sep_pattern, norm_q))
            else:
                int_pat = r"[\s,._']?".join(re.escape(d) for d in int_part)
                sep_pattern = r"(?<!\d)" + int_pat + r"(?!\d)"
                return bool(re.search(sep_pattern, quote) or re.search(sep_pattern, norm_q))
    return False


def _verify_field_evidence(
    val: Any,
    field_name: str,
    evidence: dict[str, Any] | None,
    norm_page: str,
    verified_data: dict[str, Any],
    verified_evidence: dict[str, Any],
    notes: list[str],
) -> None:
    if val is None:
        verified_data[field_name] = None
        return

    if isinstance(val, dict):
        sub_ev = evidence.get(field_name) if isinstance(evidence, dict) else None
        sub_ev_dict = sub_ev if isinstance(sub_ev, dict) else evidence
        sub_data: dict[str, Any] = {}
        sub_evidence: dict[str, Any] = {}
        for sub_k, sub_v in val.items():
            _verify_field_evidence(
                sub_v,
                sub_k,
                sub_ev_dict if isinstance(sub_ev_dict, dict) else None,
                norm_page,
                sub_data,
                sub_evidence,
                notes,
            )
        verified_data[field_name] = sub_data
        if sub_evidence:
            verified_evidence[field_name] = sub_evidence
        return

    quote = evidence.get(field_name) if isinstance(evidence, dict) else None
    if quote is None or not isinstance(quote, str) or not quote.strip():
        verified_data[field_name] = None
        notes.append(f"Field '{field_name}' nullified: missing evidence quote.")
    else:
        norm_quote = normalize_for_evidence(quote)
        if not norm_quote or norm_quote not in norm_page:
            verified_data[field_name] = None
            notes.append(f"Field '{field_name}' nullified: evidence quote not found in page content.")
        elif (
            isinstance(val, (str, int, float))
            and not isinstance(val, bool)
            and not _value_appears_in_quote(val, quote, norm_quote)
        ):
            verified_data[field_name] = None
            notes.append(f"Field '{field_name}' nullified: extracted value not found in evidence quote.")
        else:
            verified_data[field_name] = val
            verified_evidence[field_name] = quote


def verify_evidence(
    data: dict[str, Any],
    evidence: dict[str, Any] | None,
    page_content: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Verify that evidence quotes for extracted fields exist in the page content.

    If an evidence quote is not found in normalized page content, or if evidence is missing
    for an extracted non-null field, the field value is set to None and a note is added.

    Returns ``(verified_data, verified_evidence, notes)``.
    """
    norm_page = normalize_for_evidence(page_content)
    verified_data: dict[str, Any] = {}
    verified_evidence: dict[str, Any] = {}
    notes: list[str] = []

    for k, v in data.items():
        _verify_field_evidence(
            v,
            k,
            evidence,
            norm_page,
            verified_data,
            verified_evidence,
            notes,
        )

    return verified_data, verified_evidence, notes
