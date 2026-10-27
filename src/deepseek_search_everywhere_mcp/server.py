"""MCP server for the DeepSeek Search & Fetch API.

Exposes web search, page fetch, search-and-read and balance tools over stdio.
Requires DEEPSEEK_API_KEY; DEEPSEEK_BASE_URL overrides the endpoint.

The command-line entry point lives in ``deepseek_search_everywhere_mcp.cli``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import urllib.parse
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BASE_DEFAULT = "https://api.deepseek.com"
MAX_QUERY = 128
MAX_URL = 8192
SEARCH_TIMEOUT = 60.0
FETCH_TIMEOUT = 120.0
CHAT_TIMEOUT = 8.0

SYSTEM_PROMPT = (
    "You rewrite search queries. Output ONLY the rewritten query, no quotes, no explanation. "
    "Translate Polish to English. Keep search operators verbatim: site:, -site:, lang:, after:, "
    "before:, and quoted phrases. Keep proper nouns, product names and numbers unchanged. "
    "Maximum 120 characters."
)

TRACKING_PARAMS = {"fbclid", "gclid", "yclid", "mc_cid", "mc_eid"}
MIN_TITLE_SITE_CHARS = 20
MIN_TITLE_SITE_WORDS = 3

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

mcp = FastMCP("deepseek-search-everywhere-mcp")


def _fail(code: str, message: str, status: int | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"ok": False, "error": {"code": code, "message": message}}
    if status is not None:
        err["error"]["status"] = status
    return err


def _base_url() -> str:
    return os.environ.get("DEEPSEEK_BASE_URL", BASE_DEFAULT).rstrip("/")


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    configured = os.environ.get("DEEPSEEK_AUTH_JSON")
    candidates = [pathlib.Path(configured).expanduser()] if configured else [
        pathlib.Path.home() / ".local/share/opencode/auth.json",
    ]
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
            found = (data.get("deepseek") or {}).get("key") or ""
        except (OSError, ValueError):
            continue
        if found.strip():
            return found.strip()
    return ""


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n…[truncated]", True


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("utm_") or lowered in TRACKING_PARAMS


def _canonical_url(url: str) -> str | None:
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port is None else f"{host}:{port}"
    path = parts.path
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    params = [
        (name, value)
        for name, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    params.sort()
    return urllib.parse.urlunsplit(
        (parts.scheme.lower(), netloc, path, urllib.parse.urlencode(params), "")
    )


def _title_site_key(item: dict) -> tuple[str, str] | None:
    title = _norm_text(item.get("title"))
    site = _norm_text(item.get("site_name"))
    if not title or not site:
        return None
    if len(title) < MIN_TITLE_SITE_CHARS or len(title.split()) < MIN_TITLE_SITE_WORDS:
        return None
    return (title, site)


def _dedupe_key(item: dict) -> tuple | None:
    url = item.get("url")
    if isinstance(url, str) and url.strip():
        canonical = _canonical_url(url)
        if canonical is not None:
            return ("url", canonical)
    title_site = _title_site_key(item)
    if title_site is None:
        return None
    return ("title", *title_site)


def _dedupe_results(items: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    seen: set[tuple] = set()
    seen_title_sites: set[tuple[str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            kept.append(item)
            continue

        key = _dedupe_key(item)
        title_site = _title_site_key(item)

        if (key is not None and key in seen) or (
            title_site is not None and title_site in seen_title_sites
        ):
            removed += 1
            continue

        if key is not None:
            seen.add(key)
        if title_site is not None:
            seen_title_sites.add(title_site)
        kept.append(item)

    return kept, removed


def _unescape_markdown(text: str) -> str:
    text = re.sub(r"\\([,.;:!?])", r"\1", text)
    return re.sub(r"(?m)^\\#(?=\s)", "#", text)


def _with_lang(query: str, lang: str | None) -> tuple[str, bool]:
    if not lang:
        return query, False
    if re.search(r"(?:^|\s)lang:\S+", query):
        return query, False
    prefix = f"lang:{lang} "
    room = MAX_QUERY - len(prefix)
    trimmed = query[:room]
    return prefix + trimmed, len(trimmed) < len(query)


def _looks_polish(query: str) -> bool:
    return any(ch in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ" for ch in query)


def _map_http_error(status: int, body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
        message = (payload.get("error") or {}).get("message") or body[:400]
        code = (payload.get("error") or {}).get("code") or "http_error"
    except json.JSONDecodeError:
        message, code = body[:400], "http_error"

    hints = {
        400: "Invalid request parameters.",
        402: "Insufficient DeepSeek balance. Top up or switch provider.",
        403: "Rejected by the target site's security policy.",
        404: "Fetch failed: PDFs, Reddit and some sites are not extractable via /tools/fetch. "
             "Use deepseek_search_and_read instead, which returns content via the search index.",
        413: "Request body too large.",
        429: "Rate limit exceeded (search 100 QPS, fetch 25 QPS per account).",
        504: "Upstream timeout; the page took too long to fetch.",
    }
    if status in hints:
        message = f"{message} — {hints[status]}"
    return _fail(code, message, status)


async def _request(method: str, path: str, *, payload: dict[str, Any] | None = None,
                   timeout: float, retries: int = 2) -> dict[str, Any]:
    key = _api_key()
    if not key:
        return _fail("missing_api_key",
                     "No DeepSeek key: set DEEPSEEK_API_KEY or a readable DEEPSEEK_AUTH_JSON.")

    url = f"{_base_url()}{path}"
    headers = {"Authorization": f"Bearer {key}"}
    last: dict[str, Any] = _fail("request_failed", "No attempt was made.")

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, json=payload, headers=headers)
            if response.status_code >= 400:
                last = _map_http_error(response.status_code, response.text)
                if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                    delay = float(response.headers.get("Retry-After", "2") or 2)
                    await asyncio.sleep(min(delay, 10))
                    continue
                return last
            try:
                return {"ok": True, "data": response.json()}
            except ValueError:
                return _fail("invalid_json", "DeepSeek returned a non-JSON response.",
                             response.status_code)
        except httpx.TimeoutException:
            last = _fail("timeout", f"Request to {path} timed out after {timeout:.0f}s.")
        except httpx.RequestError as exc:
            last = _fail("network_error", f"{type(exc).__name__}: {exc}")
        if attempt + 1 < retries:
            await asyncio.sleep(2)
    return last


async def _rewrite_query(query: str) -> str | None:
    payload: dict[str, Any] = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    try:
        result = await _request(
            "POST", "/v1/chat/completions", payload=payload, timeout=CHAT_TIMEOUT, retries=1
        )
        if not result.get("ok"):
            return None
        content = result["data"]["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None
        first = next((line.strip() for line in content.splitlines() if line.strip()), "")
        if len(first) >= 2 and first[0] in "\"'" and first[-1] in "\"'":
            first = first[1:-1]
        cleaned = re.sub(r"\s+", " ", first).strip()[:MAX_QUERY].strip()
        if not cleaned:
            return None
        return cleaned
    except Exception:
        return None


async def _search(query: str, count: int, ori_question: str | None, need_content: bool,
                  lang: str | None, max_chars_per_result: int,
                  max_total_chars: int, dedupe: bool = True,
                  rewrite: bool = False) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return _fail("empty_query", "query must not be empty.")
    if len(query) > MAX_QUERY:
        return _fail("query_too_long", f"query must be at most {MAX_QUERY} characters.")
    if not 1 <= count <= 50:
        return _fail("bad_count", "count must be between 1 and 50.")
    if lang and not re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", lang):
        return _fail("bad_lang", "lang must look like 'pl' or 'pt-BR'.")

    query_original: str | None = None
    rewrite_failed = False
    if rewrite and lang is None and _api_key():
        rewritten = await _rewrite_query(query)
        if rewritten is None:
            rewrite_failed = True
        elif rewritten != query:
            query_original = query
            query = rewritten
            if not ori_question:
                ori_question = query_original[:MAX_QUERY]

    used, trimmed = _with_lang(query, lang)
    payload: dict[str, Any] = {"query": used, "count": count, "need_content": need_content}
    if ori_question:
        payload["ori_question"] = ori_question[:MAX_QUERY]

    result = await _request("POST", "/v1/tools/search", payload=payload, timeout=SEARCH_TIMEOUT)
    if not result.get("ok"):
        return result

    raw = result["data"].get("results") or []
    removed = 0
    if dedupe:
        raw, removed = _dedupe_results(raw)
    remaining = max_total_chars
    shaped: list[dict[str, Any]] = []
    content_truncated = False

    for item in raw:
        entry: dict[str, Any] = {
            "title": item.get("title"),
            "url": item.get("url"),
            "date": item.get("date"),
            "authority": item.get("authority"),
            "rerank_score": item.get("rerank_score"),
            "site": item.get("site_name"),
        }
        snippet = item.get("snippet") or ""
        if snippet:
            entry["snippet"] = snippet

        body = item.get("content") if need_content else None
        if body:
            cleaned = _unescape_markdown(body)
            allowed = max(0, min(max_chars_per_result, remaining))
            entry["content"], cut = _truncate(cleaned, allowed)
            remaining -= len(entry["content"])
            entry["content_chars"] = len(cleaned)
            content_truncated = content_truncated or cut
        elif item.get("summary"):
            allowed = max(0, min(max_chars_per_result, remaining))
            entry["summary"], cut = _truncate(item["summary"], allowed)
            remaining -= len(entry["summary"])
            content_truncated = content_truncated or cut
        shaped.append(entry)

        if need_content and remaining <= 0:
            break

    notes: list[str] = []
    if query_original is not None:
        notes.append(f"query rewritten (flash): '{used}'")
    elif rewrite_failed:
        notes.append("query rewrite failed; using original query")
    if removed:
        notes.append(f"deduped {removed} duplicate result(s)")
    if trimmed:
        notes.append(f"Query trimmed to fit the {MAX_QUERY}-character limit.")
    if lang:
        notes.append(f"Language operator lang:{lang} applied.")
    elif _looks_polish(query):
        notes.append("Polish query without lang: DeepSeek may return foreign-language pages "
                     "(measured 2026-10-20). Retry with lang='pl'; for named entities also try "
                     "without it, because the operator can reduce entity precision.")
    if content_truncated:
        notes.append("Some page content was truncated to respect the character budget.")

    envelope: dict[str, Any] = {
        "ok": True,
        "query_used": used,
        "count": len(shaped),
        "need_content": need_content,
        "deduped": removed,
        "results": shaped,
        "notes": notes,
    }
    if query_original is not None:
        envelope["query_original"] = query_original
    return envelope


@mcp.tool()
async def deepseek_web_search(
    query: str,
    count: int = 10,
    ori_question: str | None = None,
    need_content: bool = False,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 40000,
    dedupe: bool = True,
    rewrite: bool = False,
) -> dict[str, Any]:
    """Search the web with DeepSeek's index. Returns titles, URLs, dates, authority and rerank scores.

    Use this for discovery and ranking. Set need_content=True to get full page text inline in the
    same call (billed as one request, no per-page charge) — this is usually cheaper and more
    reliable than calling deepseek_fetch afterwards.

    For Polish queries set lang='pl'. Without it DeepSeek can return Indonesian, Spanish or
    Portuguese pages for Polish queries; with it, exact named-entity matches can get worse, so for
    entities run both variants. Each result carries date (RFC 3339) and authority (0-4, -1 unknown).

    Args:
        query: Search query, max 128 characters. Supports site:, -site:, "phrase", lang:, after:, before:.
        count: Number of results, 1-50 (default 10).
        ori_question: Original natural-language question, max 128 characters; improves ranking.
        need_content: Return full page text per result (recommended for reading).
        lang: Two-letter language operator, e.g. 'pl' or 'en'.
        max_chars_per_result: Content cap per result when need_content is set.
        max_total_chars: Total content budget across all results.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
    """
    return await _search(query, count, ori_question, need_content, lang,
                         max_chars_per_result, max_total_chars, dedupe, rewrite)


@mcp.tool()
async def deepseek_search_and_read(
    query: str,
    count: int = 5,
    ori_question: str | None = None,
    lang: str | None = None,
    max_chars_per_result: int = 4000,
    max_total_chars: int = 30000,
    dedupe: bool = True,
    rewrite: bool = False,
) -> dict[str, Any]:
    """Search and return full page content in one request — the recommended retrieval path.

    This is search with need_content=True. Measured on 14 benchmark questions it returned content
    for PDFs and Reddit threads where deepseek_fetch returned HTTP 404, in about 0.7s per call.

    Args:
        query: Search query, max 128 characters.
        count: Number of results, 1-50 (default 5).
        ori_question: Original natural-language question, max 128 characters.
        lang: Two-letter language operator, e.g. 'pl' or 'en'.
        max_chars_per_result: Content cap per result.
        max_total_chars: Total content budget across all results.
        dedupe: Collapse near-duplicate results (same canonical URL or same title+site), default on.
        rewrite: Rewrite the query via deepseek-flash before searching, default off. Skipped when
            lang is set; falls back silently to the original query on failure.
    """
    return await _search(query, count, ori_question, True, lang,
                         max_chars_per_result, max_total_chars, dedupe, rewrite)


@mcp.tool()
async def deepseek_fetch(
    url: str,
    clean: bool = True,
    max_chars: int = 20000,
) -> dict[str, Any]:
    """Fetch one known URL and return its title and Markdown content.

    Caveat measured on 2026-10-20: this endpoint returns HTTP 404 for PDFs and for Reddit, and
    times out on some sites. When the URL is a document or a blocked site, call
    deepseek_search_and_read with the page title or a site:-scoped query instead.

    The response contains only title and content — no final URL, status code or publication date,
    so keep the provenance on your side.

    Args:
        url: Absolute http(s) URL, max 8192 characters.
        clean: Remove DeepSeek's spurious Markdown escaping (\\# at line start, \\, \\, etc.).
        max_chars: Content cap, applied after cleaning.
    """
    url = (url or "").strip()
    if not url:
        return _fail("empty_url", "url must not be empty.")
    if len(url) > MAX_URL:
        return _fail("url_too_long", f"url must be at most {MAX_URL} characters.")
    if not re.match(r"^https?://", url, re.I):
        return _fail("bad_url", "url must start with http:// or https://.")

    result = await _request("POST", "/v1/tools/fetch", payload={"url": url}, timeout=FETCH_TIMEOUT)
    if not result.get("ok"):
        return result

    data = result["data"]
    content = data.get("content") or ""
    if clean:
        content = _unescape_markdown(content)
    content, truncated = _truncate(content, max_chars)

    return {
        "ok": True,
        "url": url,
        "title": data.get("title"),
        "content": content,
        "content_chars": len(content),
        "truncated": truncated,
        "notes": ["Empty content: the page may be a PDF, JS-only or blocked. "
                  "Try deepseek_search_and_read instead."] if not content else [],
    }


@mcp.tool()
async def deepseek_balance() -> dict[str, Any]:
    """Check the DeepSeek account balance (granted and topped-up, in CNY and USD)."""
    result = await _request("GET", "/user/balance", timeout=20.0)
    if not result.get("ok"):
        return result
    data = result["data"]
    return {
        "ok": True,
        "is_available": data.get("is_available"),
        "balances": data.get("balance_infos") or data.get("balances") or data,
    }


async def _check() -> int:
    key = _api_key()
    print(f"base_url: {_base_url()}")
    print(f"DeepSeek key: {'set' if key else 'MISSING'} (len={len(key)})")
    if not key:
        print("Set DEEPSEEK_API_KEY or point DEEPSEEK_AUTH_JSON at a JSON file.")
        return 1
    result = await deepseek_balance()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1

