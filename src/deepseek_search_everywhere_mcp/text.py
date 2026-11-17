"""Text cleanup helpers for upstream content: truncation, titles, markdown, JSON escapes and HTML tags."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from .sanitize import (
    sanitize_untrusted_text,
    strip_hidden_unicode,
)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit] + "\n\n…[truncated]", True


def _url_host(url: object) -> str | None:
    """Return the URL host without ``www.``, or None when the URL has none."""
    if not isinstance(url, str):
        return None
    try:
        host = urllib.parse.urlsplit(url.strip()).hostname
    except ValueError:
        return None
    return host.removeprefix("www.") if host else None


TITLE_SEPARATOR = " - "


def _collapse_repeated_title(title: Any) -> Any:
    """Collapse a title made of two identical halves joined by `` - `` (``"X - X"`` becomes ``"X"``).

    Some pages (Reddit threads, for one) repeat the title around the separator. Any other title,
    including ``"A - B"`` and non-string values, is returned unchanged.
    """
    if not isinstance(title, str):
        return title
    stripped = title.strip()
    half, remainder = divmod(len(stripped) - len(TITLE_SEPARATOR), 2)
    if remainder or half < 1 or stripped[half : half + len(TITLE_SEPARATOR)] != TITLE_SEPARATOR:
        return title
    first, second = stripped[:half], stripped[half + len(TITLE_SEPARATOR) :]
    return first if first == second else title


def _title_or_host(item: dict) -> str | None:
    """Return the cleaned result title, or the URL host without ``www.`` when the title is blank."""
    title = _collapse_repeated_title(item.get("title"))
    if isinstance(title, str) and title.strip():
        cleaned, _ = strip_hidden_unicode(title)
        return cleaned
    fallback = _url_host(item.get("url")) or title
    if isinstance(fallback, str):
        cleaned_fb, _ = strip_hidden_unicode(fallback)
        return cleaned_fb
    return fallback


MAX_PRE_PROCESSING_CHARS = 2 * 1024 * 1024  # 2 MB ceiling


def _cap_raw_content(text: str, max_chars: int) -> str:
    """Cap untrusted text before expensive regex unescaping and boilerplate passes.

    Protects against ReDoS and memory exhaustion on huge pages while
    preserving enough buffer for boilerplate removal and Markdown unescaping
    to produce the requested ``max_chars``.
    """
    if not text:
        return ""
    pre_cap = min(len(text), max(4 * max_chars, MAX_PRE_PROCESSING_CHARS))
    return text[:pre_cap]


# One alternation, scanned left to right, so a backslash escape is consumed before any backtick
# it escapes can open an inline code span. Code (fenced blocks, then inline spans) is matched
# only to be skipped; a fence without a closing line runs to the end of the text.
_MARKDOWN_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\(?P<char>[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])",
    re.DOTALL | re.MULTILINE,
)


def _unescape_markdown(text: str) -> str:
    """Remove CommonMark backslash escapes from ``text``, leaving code untouched.

    Any ASCII punctuation character may be backslash-escaped in CommonMark, and the fetch
    backend escapes many of them (``\\(``, ``\\-``, ``\\[``, ``\\/``...), which only costs tokens
    and hurts readability. Fenced code blocks and inline code spans are skipped because a
    backslash there is literal. An escaped backslash (``\\\\``) is reduced to a single one.
    A ``\\#`` at the start of a line is unescaped like any other, so an escaped heading marker
    becomes a heading again.
    """
    return _MARKDOWN_TOKEN.sub(lambda match: match["char"] or match[0], text)


# Code first, so escapes inside fenced blocks and inline spans are matched as part of the code and kept.
_JSON_ESCAPE_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\u(?P<high>[dD][89abAB][0-9a-fA-F]{2})\\u(?P<low>[dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u(?P<unit>[0-9a-fA-F]{4})"
    r"|(?P<slash>\\/)",
    re.DOTALL | re.MULTILINE,
)


_JSON_ESCAPE_HINT = re.compile(r"\\u[0-9a-fA-F]{4}|\\/")


def _decode_json_escape(match: re.Match[str]) -> str:
    if match["slash"]:
        return "/"
    if match["high"]:
        high, low = int(match["high"], 16), int(match["low"], 16)
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
    if match["unit"]:
        code = int(match["unit"], 16)
        # Control characters and lone surrogates stay escaped: they are not text worth restoring.
        if code < 0x20 or 0xD800 <= code <= 0xDFFF:
            return match[0]
        return chr(code)
    return match[0]


def _decode_json_escapes(text: str) -> str:
    """Decode JSON string escapes (``\\uXXXX``, surrogate pairs, ``\\/``) left in search-index text.

    The search index sometimes returns a page's text still JSON-escaped, so ``ó`` arrives as
    the six characters ``\\u00f3`` and ``</p>`` as ``<\\/p>``. Fenced code blocks and inline code
    spans are skipped, so a literal escape shown in code is kept; text without such sequences
    is returned unchanged.
    """
    if not _JSON_ESCAPE_HINT.search(text):
        return text
    return _JSON_ESCAPE_TOKEN.sub(_decode_json_escape, text)


# Simple attribute-less block and inline tags that leak into index content, snippets and summaries.
_HTML_TAG = re.compile(r"</?(?:p|br|h[1-6]|div|span|li|ul|ol|strong|em|b|i)\s*/?>", re.IGNORECASE)


_TAG_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|(?P<tags>(?:[ \t]*" + _HTML_TAG.pattern + r"[ \t]*)+)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _strip_html_tags(text: str) -> str:
    """Replace runs of simple HTML tags (``<p>``, ``<br>``, ``</h2>``...) with a single space.

    Only attribute-less tags from a short list are touched, outside code, so text that merely
    mentions angle brackets (``a < b``, ``<T>``) stays as it is.
    """
    if "<" not in text:
        return text
    stripped = _TAG_TOKEN.sub(lambda match: " " if match["tags"] else match[0], text)
    return stripped.strip() if stripped != text else text


def _clean_index_text(text: str) -> str:
    """Decode JSON escapes in a search snippet or summary, drop HTML, and sanitize untrusted text."""
    cleaned = _strip_html_tags(_decode_json_escapes(text))
    cleaned, _ = sanitize_untrusted_text(cleaned)
    return cleaned
