"""Sanitization helpers for untrusted web content.

Provides stripping of hidden/invisible Unicode (tag characters, bidi controls,
zero-width characters) and neutralization of markdown images and unsafe links.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from collections.abc import Callable
from typing import Any

# Unicode Tag characters: U+E0000 - U+E007F
_TAG_MIN = 0xE0000
_TAG_MAX = 0xE007F

# Bidirectional text control characters:
# U+202A-U+202E (LRE, RLE, PDF, LRO, RLO)
# U+2066-U+2069 (LRI, RLI, FSI, PDI)
# U+200E (LRM), U+200F (RLM), U+061C (ALM)
_BIDI_CONTROLS = frozenset(
    {
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\u200e",
        "\u200f",
        "\u061c",
    }
)

# Zero-width / invisible non-ZWJ characters:
# U+200B (ZWSP), U+200C (ZWNJ), U+2060 (WJ), U+FEFF (BOM/ZWNBSP)
_ZERO_WIDTH_CHARS = frozenset({"\u200b", "\u200c", "\u2060", "\ufeff"})

_ZWJ = "\u200d"

# Markdown images: ![alt](url)
_MD_IMAGE_RE = re.compile(r"(?<!\\)!\[(?P<alt>[^\]]*)\]\((?P<url>[^\)]*)\)")

# Markdown links: [label](url) where label is not preceded by !
# Supports balanced parentheses in URL up to 1 level deep.
# The URL body is written as ``[^()\s]*(?:\([^()\s]*\)[^()\s]*)*`` instead of the
# former ``(?:[^()\s]+|\([^()\s]*\))*``. The old form nested two quantifiers over the
# same character class and backtracked exponentially on an unterminated link such as
# ``[x](aaaa...`` (4 KB was enough to peg a CPU for minutes); this form is linear.
_MD_LINK_RE = re.compile(r"(?<!\!)\[(?P<label>[^\]]*)\]\((?P<url>[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)\)")


# Start finders for ``_sub_bracketed``: where a markdown image / link can begin (same lookbehinds as above).
_MD_IMAGE_START_RE = re.compile(r"(?<!\\)!\[")
_MD_LINK_START_RE = re.compile(r"(?<!\!)\[")


def _sub_bracketed(
    pattern: re.Pattern[str],
    start_re: re.Pattern[str],
    opener_len: int,
    repl: Callable[[re.Match[str]], str],
    text: str,
    *,
    needs_close_paren: bool = False,
) -> str:
    """Linear-time equivalent of ``pattern.sub(repl, text)`` for ``[label](url)``-shaped patterns.

    ``pattern.sub`` retries every opener, and each retry rescans the label up to the next ``]``
    (or to the end of the text when there is none), which is quadratic on text such as ``[1,[1,[1,...``
    or ``![a](![a](...``. The label ends at the first ``]`` after its opener, so once an attempt fails
    every opener before that ``]`` fails the same way and is skipped. With ``needs_close_paren`` (images:
    the url is ``[^)]*`` followed by ``)``) an attempt can only fail for a missing ``)`` or a missing
    ``(`` after the ``]``; when no ``)`` is left in the text no later opener can match either.
    """
    last_paren = text.rfind(")") if needs_close_paren else -1
    out: list[str] = []
    copied = 0
    pos = 0
    while True:
        start = start_re.search(text, pos)
        if start is None:
            break
        begin = start.start()
        close = text.find("]", begin + opener_len)
        if close == -1:
            break
        match = pattern.match(text, begin)
        if match is not None:
            out.append(text[copied:begin])
            out.append(repl(match))
            copied = pos = match.end()
            continue
        if needs_close_paren and last_paren < close + 2:
            break
        pos = close + 1
    out.append(text[copied:])
    return "".join(out)


def _is_emoji_component(char: str) -> bool:
    """Return True if character is an emoji or emoji-combining component."""
    cp = ord(char)
    # Variation selectors (VS15, VS16) and keycap combining char
    if cp in (0xFE0E, 0xFE0F, 0x20E3):
        return True
    # Fitzpatrick skin tone modifiers
    if 0x1F3FB <= cp <= 0x1F3FF:
        return True
    # Common emoji code point ranges
    if (
        0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2300 <= cp <= 0x23FF
        or 0x2B50 <= cp <= 0x2B55
        or 0x1F1E6 <= cp <= 0x1F1FF
    ):
        return True
    return unicodedata.category(char) == "So"


def strip_hidden_unicode(text: str) -> tuple[str, int]:
    """Strip tag characters, bidi controls, and zero-width characters from text.

    Preserves U+200D (ZWJ) if and only if it sits within an emoji sequence.
    Returns ``(cleaned_text, stripped_count)``.
    """
    if not text:
        return text, 0

    chars = list(text)
    length = len(chars)
    out: list[str] = []
    stripped = 0

    for i, c in enumerate(chars):
        cp = ord(c)
        if _TAG_MIN <= cp <= _TAG_MAX:
            stripped += 1
            continue
        if c in _BIDI_CONTROLS:
            stripped += 1
            continue
        if c in _ZERO_WIDTH_CHARS:
            stripped += 1
            continue
        if c == _ZWJ:
            prev_ok = i > 0 and _is_emoji_component(chars[i - 1])
            next_ok = i + 1 < length and _is_emoji_component(chars[i + 1])
            if prev_ok and next_ok:
                out.append(c)
            else:
                stripped += 1
            continue
        out.append(c)

    return "".join(out), stripped


def neutralize_markdown_images(text: str) -> tuple[str, int]:
    """Replace Markdown images `![alt](url)` with safe text `[image: alt](url)` or `[image](url)`.

    Escaped markdown images `\\![alt](url)` are untouched.
    Returns ``(cleaned_text, count)``.
    """
    if not text:
        return text, 0

    count = 0

    def _replace_image(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        alt = match.group("alt").strip()
        url = match.group("url")
        if alt:
            return f"[image: {alt}]({url})"
        return f"[image]({url})"

    cleaned = _sub_bracketed(
        _MD_IMAGE_RE, _MD_IMAGE_START_RE, 2, _replace_image, text, needs_close_paren=True
    )
    return cleaned, count


def sanitize_markdown_links(text: str) -> tuple[str, int]:
    """Neutralize unsafe non-http(s) link targets like `javascript:` or `data:`.

    Unsafe link destinations are emptied: `[label]()`.
    Returns ``(cleaned_text, count)``.
    """
    if not text:
        return text, 0

    count = 0

    def _replace_link(match: re.Match[str]) -> str:
        nonlocal count
        label = match.group("label")
        url = match.group("url")
        clean_u = "".join(c for c in url.strip() if c not in "\t\r\n")
        parsed = urllib.parse.urlsplit(clean_u)
        scheme = parsed.scheme.lower()
        if scheme and scheme not in ("http", "https"):
            count += 1
            return f"[{label}]()"
        return match.group(0)

    cleaned = _sub_bracketed(_MD_LINK_RE, _MD_LINK_START_RE, 1, _replace_link, text)
    return cleaned, count


def sanitize_untrusted_text(text: str) -> tuple[str, list[str]]:
    """Sanitize untrusted text: strips hidden unicode, neutralizes images and unsafe links.

    Returns ``(cleaned_text, notes)`` where notes lists sanitization actions performed.
    """
    if not text:
        return text, []

    notes: list[str] = []
    cleaned, u_count = strip_hidden_unicode(text)
    if u_count > 0:
        notes.append(f"Sanitized {u_count} hidden Unicode character(s) from untrusted content.")

    cleaned, img_count = neutralize_markdown_images(cleaned)
    if img_count > 0:
        notes.append(f"Neutralized {img_count} markdown image(s) in untrusted content.")

    cleaned, link_count = sanitize_markdown_links(cleaned)
    if link_count > 0:
        notes.append(f"Sanitized {link_count} unsafe markdown link(s) in untrusted content.")

    return cleaned, notes


def sanitize_structure(data: Any) -> tuple[Any, list[str]]:
    """Recursively sanitize strings inside nested dicts, lists or primitives.

    Returns ``(cleaned_data, notes)``.
    """
    notes: list[str] = []

    def _walk(item: Any) -> Any:
        if isinstance(item, str):
            cleaned, n = sanitize_untrusted_text(item)
            notes.extend(n)
            return cleaned
        if isinstance(item, dict):
            return {_walk(k): _walk(v) for k, v in item.items()}
        if isinstance(item, list):
            return [_walk(v) for v in item]
        return item

    cleaned_data = _walk(data)
    seen: set[str] = set()
    deduped_notes = [n for n in notes if not (n in seen or seen.add(n))]
    return cleaned_data, deduped_notes
