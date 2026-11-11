"""Tests for content size capping and regex ReDoS / fuzz safety on large adversarial inputs."""

from __future__ import annotations

import time

from deepseek_search_everywhere_mcp.boilerplate import remove_boilerplate
from deepseek_search_everywhere_mcp.server import (
    MAX_PRE_PROCESSING_CHARS,
    _cap_raw_content,
    _decode_json_escapes,
    _strip_html_tags,
    _unescape_markdown,
)


def test_cap_raw_content_limits_huge_pages() -> None:
    # 32 MB of page content
    huge_text = "x" * (32 * 1024 * 1024)
    capped = _cap_raw_content(huge_text, max_chars=20000)
    assert len(capped) == MAX_PRE_PROCESSING_CHARS
    assert len(capped) == 2 * 1024 * 1024

    # Text smaller than 2 MB stays untouched
    small_text = "hello world" * 100
    assert _cap_raw_content(small_text, max_chars=20000) == small_text

    # Empty text returns empty string
    assert _cap_raw_content("", max_chars=20000) == ""


def test_markdown_unescape_fuzz_time_limit() -> None:
    # 2 MB adversarial input with repeated unclosed backticks and backslashes
    adversarial = ("\\# Heading\n" + "\\*" * 20 + "```\n" + "test content\n") * 35000
    assert len(adversarial) >= 1_000_000

    t0 = time.monotonic()
    result = _unescape_markdown(adversarial[: 2 * 1024 * 1024])
    elapsed = time.monotonic() - t0

    # Must complete safely in under 500 ms (linear time, no catastrophic backtracking)
    assert elapsed < 0.5, f"Markdown unescape took {elapsed:.3f} s on 2 MB input"
    assert len(result) > 0


def test_json_escapes_fuzz_time_limit() -> None:
    # 2 MB adversarial input with dense unicode and slash escapes
    adversarial = ("\\u0041\\u0042\\/\\u0043\\u0044" * 10 + " normal text ") * 30000
    assert len(adversarial) >= 1_000_000

    t0 = time.monotonic()
    result = _decode_json_escapes(adversarial[: 2 * 1024 * 1024])
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"JSON escape decoding took {elapsed:.3f} s on 2 MB input"
    assert len(result) > 0


def test_html_tag_stripping_fuzz_time_limit() -> None:
    # 2 MB adversarial input with dense nested and unclosed HTML tags
    adversarial = ("<div><p><span><b><i>Text</i></b></span></p></div><br/>" * 5) * 10000
    assert len(adversarial) >= 1_000_000

    t0 = time.monotonic()
    result = _strip_html_tags(adversarial[: 2 * 1024 * 1024])
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"HTML tag stripping took {elapsed:.3f} s on 2 MB input"
    assert len(result) > 0


def test_boilerplate_removal_fuzz_time_limit() -> None:
    # 2 MB adversarial input with repeated consent banner lines
    adversarial = ("Accept all cookies to continue\nManage cookie preferences\nPrivacy Policy\n" * 5) * 10000
    assert len(adversarial) >= 1_000_000

    t0 = time.monotonic()
    result = remove_boilerplate(adversarial[: 2 * 1024 * 1024], "https://example.com/article")
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"Boilerplate removal took {elapsed:.3f} s on 2 MB input"
    assert isinstance(result, str)
