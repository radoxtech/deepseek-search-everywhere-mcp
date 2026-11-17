"""The deepseek_extract tool (one URL or a list) and the LLM extraction step behind it.

Callers patch ``tools.extract._extract_page``; both branches look it up in this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Annotated, Any

from pydantic import Field

from .. import fetch_core, runtime
from ..batch import (
    _LOCAL_ERROR_REASONS,
    _batch_deadline_error,
    _batch_envelope,
    _dedupe_urls,
    _run_batch,
)
from ..envelope import (
    COST_PER_REQUEST,
    _all_cached,
    _envelope_meta,
)
from ..errors import (
    _fail,  # re-exported for callers and tests
)
from ..extract import (
    build_extract_prompt,
    sanitize_json,
    validate_fields,
    validate_schema,
    verify_evidence,
)
from ..fetch_core import (
    _check_fetch_url,
)
from ..runtime import (
    REWRITE_MODEL,
    _timed,
)
from ..sanitize import (
    sanitize_structure,
)

EXTRACT_CONCURRENCY = 4  # LLM extraction calls a deepseek_extract URL list runs at once


def _validate_extract_spec(fields: list[str] | None, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate that either 'fields' or 'schema' is provided and structurally sound."""
    if not fields and not schema:
        return _fail("bad_request", "Either 'fields' or 'schema' must be provided.", param="fields")
    if fields is not None and schema is not None:
        return _fail("bad_request", "Provide either 'fields' or 'schema', not both.", param="schema")
    if fields is not None:
        valid_fields = (
            isinstance(fields, list)
            and bool(fields)
            and all(isinstance(f, str) and f.strip() for f in fields)
        )
        if not valid_fields:
            return _fail(
                "bad_request", "'fields' must be a non-empty list of non-empty strings.", param="fields"
            )
    if schema is not None and not isinstance(schema, dict):
        return _fail("bad_request", "'schema' must be a dictionary.", param="schema")
    return None


def _calculate_extract_budget(
    fields: list[str] | None,
    schema: dict[str, Any] | None,
    content_len: int,
) -> tuple[int, float]:
    """Calculate adaptive max_tokens and timeout for extraction.

    Tokens: 4096 base, scales up to 16384 with large field sets or complex schemas.
    Timeout: 20s to 90s based on page content length.
    """
    field_count = len(fields) if fields else 0
    if schema:
        props = schema.get("properties")
        if isinstance(props, dict):
            field_count = max(field_count, len(props))
        else:
            field_count = max(field_count, 10)

    tokens = min(16384, max(4096, 4096 + max(0, field_count - 5) * 256))
    timeout = min(90.0, max(20.0, 20.0 + (content_len / 5000.0) * 1.0))
    return tokens, timeout


async def _extract_page(
    clean_url: str,
    fetch_res: dict[str, Any],
    system_prompt: str,
    fields: list[str] | None,
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    """Perform LLM extraction for one successfully fetched page.

    Returns ``(item_result, upstream_llm_requests)``.
    """
    content = fetch_res.get("content") or ""
    if not content.strip():
        notes = fetch_res.get("notes") or []
        msg = notes[0] if notes else "The fetched page has no readable content to extract from."
        return {
            "ok": False,
            "url": clean_url,
            "error": _fail("empty_page", msg)["error"],
        }, 0

    page_title = fetch_res.get("title") or "Untitled"
    user_content = f"URL: {clean_url}\nTitle: {page_title}\n\nPage Content:\n{content}"

    max_tokens, timeout = _calculate_extract_budget(fields, schema, len(content))
    chat_res = await runtime._chat_request(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        thinking={"type": "disabled"},
        model=REWRITE_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=timeout,
        attempts=2,
    )
    if not chat_res.get("ok"):
        return {
            "ok": False,
            "url": clean_url,
            "error": chat_res.get("error")
            or _fail("extraction_failed", "Extraction model request failed.")["error"],
            "llm_cost_usd": 0.0,
        }, 1

    finish_reason = chat_res.get("finish_reason")
    extraction_truncated = finish_reason == "length"

    raw_text = chat_res.get("content")
    llm_cost = chat_res.get("cost_usd", 0.0)
    if not isinstance(raw_text, str) or not raw_text.strip():
        err_code = "extraction_truncated" if extraction_truncated else "extraction_failed"
        err_msg = (
            "Extraction response was truncated before valid JSON could be completed."
            if extraction_truncated
            else "Invalid response payload from extraction model."
        )
        return {
            "ok": False,
            "url": clean_url,
            "error": _fail(err_code, err_msg)["error"],
            "llm_cost_usd": llm_cost,
        }, 1

    parsed, err = sanitize_json(raw_text)
    if err or parsed is None:
        err_code = "extraction_truncated" if extraction_truncated else "extraction_failed"
        err_msg = (
            "Extraction response was truncated before valid JSON could be completed."
            if extraction_truncated
            else (err or "Failed to parse extraction response as JSON.")
        )
        return {
            "ok": False,
            "url": clean_url,
            "error": _fail(err_code, err_msg)["error"],
            "llm_cost_usd": llm_cost,
        }, 1

    notes: list[str] = []
    if extraction_truncated:
        notes.append("Extraction output was truncated by token limit.")
    if fetch_res.get("cached"):
        notes.append("Page fetched from local cache.")

    if isinstance(parsed, dict) and "data" in parsed:
        raw_data = parsed["data"]
        raw_evidence = parsed.get("evidence")
        if isinstance(raw_data, dict):
            verified_data, verified_evidence, ev_notes = verify_evidence(
                raw_data, raw_evidence if isinstance(raw_evidence, dict) else {}, content
            )
            notes.extend(ev_notes)
        else:
            verified_data = raw_data
            verified_evidence = {}
    elif isinstance(parsed, dict):
        verified_data = parsed
        verified_evidence = {}
    else:
        verified_data = {}
        verified_evidence = {}

    if fields is not None and isinstance(verified_data, dict):
        verified_data = validate_fields(verified_data, fields)
    elif schema is not None and isinstance(verified_data, dict):
        schema_errors = validate_schema(verified_data, schema)
        if schema_errors:
            notes.extend(schema_errors)

    if isinstance(verified_data, dict):
        verified_data, s_notes = sanitize_structure(verified_data)
        notes.extend(s_notes)
    if verified_evidence:
        verified_evidence, ev_s_notes = sanitize_structure(verified_evidence)
        notes.extend(ev_s_notes)

    return {
        "ok": True,
        "url": clean_url,
        "title": fetch_res.get("title"),
        "data": verified_data,
        "evidence": verified_evidence,
        "extraction_truncated": extraction_truncated,
        "cached": fetch_res.get("cached", False),
        "notes": notes,
        "llm_cost_usd": llm_cost,
    }, 1


async def _extract_one(
    url: str,
    *,
    fields: list[str] | None,
    schema: dict[str, Any] | None,
    instruction: str | None,
    max_chars: int,
    refresh: bool,
) -> dict[str, Any]:
    """Extract from one page: the single-URL branch of ``deepseek_extract``."""
    spec_err = _validate_extract_spec(fields, schema)
    if spec_err is not None:
        return spec_err

    clean_url = url.strip() if isinstance(url, str) else ""
    rejection = _check_fetch_url(clean_url)
    if rejection is not None:
        return rejection

    fetch_res = await fetch_core._fetch(
        clean_url,
        clean=True,
        max_chars=max_chars,
        refresh=refresh,
    )
    if fetch_res.get("ok") is False:
        return fetch_res

    system_prompt = build_extract_prompt(fields, schema, instruction)
    item, llm_reqs = await _extract_page(clean_url, fetch_res, system_prompt, fields, schema)
    if not item.get("ok"):
        return {"ok": False, "error": item["error"]}

    fetch_cost = fetch_res.get("cost_usd", 0.0)
    llm_cost = item.get("llm_cost_usd", 0.0)
    total_cost = round(fetch_cost + llm_cost, 6)
    return {
        "ok": True,
        "url": clean_url,
        "title": item.get("title"),
        "data": item.get("data"),
        "evidence": item.get("evidence", {}),
        "extraction_truncated": item.get("extraction_truncated", False),
        "cached": fetch_res.get("cached", False),
        "cost_usd": total_cost,
        "cost_breakdown": {
            "search": 0.0,
            "fetch": round(fetch_cost, 6),
            "llm": round(llm_cost, 6),
        },
        "untrusted_content": True,
        "notes": item.get("notes", []),
        "_meta": _envelope_meta(total_cost, untrusted_content=True),
    }


async def _extract_list(
    urls: list[str],
    *,
    fields: list[str] | None,
    schema: dict[str, Any] | None,
    instruction: str | None,
    max_chars_per_url: int,
    max_total_chars: int,
    refresh: bool,
    deadline: float | None,
) -> dict[str, Any]:
    """Extract from several pages in parallel: the list branch of ``deepseek_extract``."""
    spec_err = _validate_extract_spec(fields, schema)
    if spec_err is not None:
        return spec_err

    settings = runtime.get_settings()
    if not urls or len(urls) > settings.batch_max_urls:
        return _fail(
            "bad_request", f"a URL list must contain 1-{settings.batch_max_urls} entries.", param="url"
        )
    if max_chars_per_url < 1 or max_total_chars < 1:
        return _fail("bad_request", "max_chars and max_total_chars must be at least 1.", param="max_chars")

    items, duplicates = _dedupe_urls(urls)
    if not items:
        return _fail("bad_request", "a URL list must contain at least one non-empty URL.", param="url")

    per_url = max(1, min(max_chars_per_url, max_total_chars // len(items)))
    fetch_tasks = await _run_batch(items, True, per_url, refresh, deadline=deadline)

    deadline_error = _batch_deadline_error(deadline)

    system_prompt = build_extract_prompt(fields, schema, instruction)

    extract_coros: list[Awaitable[tuple[dict[str, Any], int]]] = []
    extract_indices: dict[int, int] = {}
    fetch_envelopes: dict[int, dict[str, Any]] = {}

    extract_sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def _bounded_extract_page(u: str, e: dict[str, Any]) -> tuple[dict[str, Any], int]:
        async with extract_sem:
            return await _extract_page(u, e, system_prompt, fields, schema)

    requests_used = 0
    for index, (url, rejection) in enumerate(items):
        fetch_envelopes[index] = _batch_envelope(index, rejection, fetch_tasks, deadline_error)

        env = fetch_envelopes[index]
        if env.get("ok"):
            fetch_reqs = round(env.get("cost_usd", 0.0) / COST_PER_REQUEST)
            requests_used += fetch_reqs
            extract_indices[index] = len(extract_coros)
            extract_coros.append(_bounded_extract_page(url, env))
        elif rejection is None and env.get("error", {}).get("reason") not in _LOCAL_ERROR_REASONS:
            requests_used += 1

    extracted_results: list[tuple[dict[str, Any], int]] = []
    if extract_coros:
        extracted_results = await asyncio.gather(*extract_coros)

    results: list[dict[str, Any]] = []
    for index, (url, _) in enumerate(items):
        if index in extract_indices:
            item_result, llm_reqs = extracted_results[extract_indices[index]]
            requests_used += llm_reqs
            results.append(item_result)
        else:
            env = fetch_envelopes[index]
            results.append(
                {
                    "ok": False,
                    "url": url,
                    "error": env.get("error", _fail("request_failed", "Failed to fetch page.")["error"]),
                }
            )

    notes: list[str] = []
    if duplicates:
        notes.append(f"deduped {duplicates} duplicate URL(s)")
    if any(item.get("extraction_truncated") for item in results):
        notes.append("One or more extractions were truncated by token limit.")

    failed_count = sum(not item.get("ok", False) for item in results)
    cached_count = sum(item.get("cached", False) for item in results)
    total_fetch_cost = sum(env.get("cost_usd", 0.0) for env in fetch_envelopes.values() if env.get("ok"))
    total_llm_cost = sum(item.get("llm_cost_usd", 0.0) for item in results if item.get("ok"))
    total_cost = round(total_fetch_cost + total_llm_cost, 6)

    return {
        "ok": True,
        "count": len(results),
        "requests_used": requests_used,
        "cached": _all_cached(requests_used, cached_count),
        "cached_count": cached_count,
        "failed_count": failed_count,
        "partial": failed_count > 0,
        "cost_usd": total_cost,
        "cost_breakdown": {
            "search": 0.0,
            "fetch": round(total_fetch_cost, 6),
            "llm": round(total_llm_cost, 6),
        },
        "untrusted_content": True,
        "results": results,
        "notes": notes,
        "_meta": _envelope_meta(total_cost, untrusted_content=True),
    }


SINGLE_MAX_CHARS = 40000  # default max_chars for one URL
LIST_MAX_CHARS = 20000  # default max_chars per URL for a list


@_timed
async def deepseek_extract(
    url: Annotated[
        str | list[str],
        Field(description="Public HTTP(S) URL to extract from, or a list of up to 10 URLs done in parallel"),
    ],
    *,
    fields: Annotated[
        list[str] | None,
        Field(description="List of field names to extract (mutually exclusive with schema)"),
    ] = None,
    schema: Annotated[
        dict[str, Any] | None,
        Field(description="JSON schema describing desired output structure (mutually exclusive with fields)"),
    ] = None,
    instruction: Annotated[
        str | None,
        Field(description="Optional natural-language guidance for extraction (default None)"),
    ] = None,
    max_chars: Annotated[
        int | None,
        Field(description="Max page characters for the extractor (default 40000; 20000 per URL for a list)"),
    ] = None,
    max_total_chars: Annotated[
        int,
        Field(description="Max total page characters across a list of URLs (default 60000)"),
    ] = 60000,
    refresh: Annotated[
        bool,
        Field(description="Bypass local cache and fetch fresh pages from upstream (default False)"),
    ] = False,
    deadline: Annotated[
        float | None,
        Field(description="Deadline in seconds for a list of URLs; finished results are returned"),
    ] = None,
) -> dict[str, Any]:
    """Pull specific fields or a JSON schema out of web pages, each value backed by a verbatim quote.

    Use it for structured data (prices, names, dates, specs) instead of prose, from one URL or the
    same fields from up to 10 URLs in parallel. $0.00075 per URL for the fetch plus about
    $0.001-0.003 of LLM tokens per page. One URL reads up to 40,000 chars (max_chars); a list reads
    up to 20,000 per URL and 60,000 in total (max_total_chars), keeps input order and reports a
    failed URL in place. Pass fields= (names) or schema= (JSON schema), not both. A value whose
    evidence quote is not in the page is returned as null with a note. Output is sanitized.

    When NOT to use: reading prose (deepseek_fetch), finding pages (deepseek_search), or a topic
    report (deepseek_research).

    Returns for one URL: {"ok": True, "data": {...}, "evidence": {...}, "extraction_truncated": bool,
    "cost_usd": float}. For a list: {"ok": True, "results": [{"url": str, "ok": bool, "data": {...},
    "evidence": {...}}], "partial": bool, "cost_usd": float}.
    """
    if isinstance(url, str):
        return await _extract_one(
            url,
            fields=fields,
            schema=schema,
            instruction=instruction,
            max_chars=SINGLE_MAX_CHARS if max_chars is None else max_chars,
            refresh=refresh,
        )
    return await _extract_list(
        url,
        fields=fields,
        schema=schema,
        instruction=instruction,
        max_chars_per_url=LIST_MAX_CHARS if max_chars is None else max_chars,
        max_total_chars=max_total_chars,
        refresh=refresh,
        deadline=deadline,
    )
