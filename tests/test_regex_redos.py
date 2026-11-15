"""Regex ReDoS regression tests: helpers that see untrusted page content must stay (near) linear.

Each case feeds ~200 KB of adversarial text and must finish inside a short wall-clock budget. The budget is
enforced with ``SIGALRM`` so a regression fails fast instead of hanging the suite (the regex engine checks
for signals while matching). Linear code finishes these inputs in milliseconds; quadratic code needs
minutes, so the budget is generous enough not to be flaky on slow CI runners.
"""

from __future__ import annotations

import random
import re
import signal
from collections.abc import Callable
from typing import Any

import pytest

from deepseek_search_everywhere_mcp import sanitize
from deepseek_search_everywhere_mcp.deep_research import (
    _extract_numbers_and_ranges,
    _mark_unverified,
    normalize_citations,
)
from deepseek_search_everywhere_mcp.extract import normalize_for_evidence
from deepseek_search_everywhere_mcp.sanitize import sanitize_markdown_links, sanitize_untrusted_text
from deepseek_search_everywhere_mcp.server import _strip_html_tags, _unescape_markdown

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="needs SIGALRM (POSIX)")

SIZE = 200_000
BUDGET_S = 1.0


class _Budget(Exception):
    pass


def _run_within_budget(fn: Callable[..., Any], *args: Any) -> Any:
    def _on_alarm(signum: int, frame: Any) -> None:
        raise _Budget

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, BUDGET_S)
    try:
        return fn(*args)
    except _Budget:
        pytest.fail(f"{getattr(fn, '__name__', fn)} exceeded {BUDGET_S}s on {SIZE:,}-char adversarial input")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _repeat(unit: str, prefix: str = "", suffix: str = "x") -> str:
    return prefix + unit * (SIZE // len(unit)) + suffix


# --- Markdown link regex (sanitize._MD_LINK_RE) ------------------------------------------------------------

# The pattern before the fix; nested quantifiers over one class backtracked exponentially.
_OLD_MD_LINK_RE = re.compile(r"(?<!\!)\[(?P<label>[^\]]*)\]\((?P<url>(?:[^\(\)\s]+|\([^\(\)\s]*\))*)\)")


def test_unterminated_markdown_link_is_linear() -> None:
    # 4 KB of this input pegged a CPU for minutes with the old pattern.
    text = "[x](" + "a" * SIZE
    cleaned, count = _run_within_budget(sanitize_markdown_links, text)
    assert cleaned == text
    assert count == 0


def test_unterminated_markdown_link_through_untrusted_text_pipeline() -> None:
    text = "intro [x](javascript:" + "a" * SIZE
    cleaned, _notes = _run_within_budget(sanitize_untrusted_text, text)
    assert cleaned == text


def test_md_link_regex_matches_like_the_old_pattern() -> None:
    # Same language as before the fix: compare both patterns on many short random strings, where the old
    # pattern is still fast.
    rng = random.Random(1234)
    alphabet = "[]()! a:\t\\"
    for _ in range(20_000):
        sample = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        old = [(m.span(), m.groupdict()) for m in _OLD_MD_LINK_RE.finditer(sample)]
        new = [(m.span(), m.groupdict()) for m in sanitize._MD_LINK_RE.finditer(sample)]
        assert new == old, repr(sample)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[a](javascript:alert(1))", "[a]()"),
        ("[a](https://en.wikipedia.org/wiki/Foo_(bar))", "[a](https://en.wikipedia.org/wiki/Foo_(bar))"),
        ("[a](data:text/html,(x)(y))", "[a]()"),
        ("![a](javascript:x)", "![a](javascript:x)"),
    ],
)
def test_md_link_regex_keeps_behaviour(text: str, expected: str) -> None:
    assert sanitize_markdown_links(text)[0] == expected


# --- Other regexes fed with untrusted or model-written text ------------------------------------------------
# Known quadratic cases found by an adversarial scan; strict xfail so a fix flips them to a hard pass.

_QUADRATIC = pytest.mark.xfail(strict=True, reason="quadratic regex backtracking, not fixed yet")

LINEAR_CASES = [
    pytest.param(_unescape_markdown, _repeat("\\*"), id="unescape-markdown-escapes"),
    pytest.param(_strip_html_tags, _repeat("<p>"), id="strip-html-dense-tags"),
    pytest.param(sanitize_untrusted_text, _repeat("[a](https://x.y) "), id="sanitize-many-links"),
    # Whitespace is collapsed before _PUNCT_SPACE_RE runs, so this stays linear.
    pytest.param(normalize_for_evidence, _repeat(" "), id="evidence-punct-space"),
]

QUADRATIC_CASES = [
    pytest.param(sanitize_untrusted_text, _repeat("![a]("), id="md-image-unclosed", marks=_QUADRATIC),
    pytest.param(sanitize_untrusted_text, _repeat("[1,"), id="md-link-many-openers", marks=_QUADRATIC),
    pytest.param(_unescape_markdown, _repeat("`"), id="markdown-token-backticks", marks=_QUADRATIC),
    pytest.param(_strip_html_tags, _repeat(" \t", suffix="<"), id="tag-token-whitespace", marks=_QUADRATIC),
    pytest.param(_extract_numbers_and_ranges, _repeat(" "), id="grounding-numbers", marks=_QUADRATIC),
    pytest.param(
        lambda t: normalize_citations(t, 10), _repeat(" \t"), id="citations-punct", marks=_QUADRATIC
    ),
    pytest.param(_mark_unverified, _repeat("!"), id="sentence-end", marks=_QUADRATIC),
]


@pytest.mark.parametrize(("fn", "text"), LINEAR_CASES + QUADRATIC_CASES)
def test_helper_is_linear_on_adversarial_input(fn: Callable[[str], Any], text: str) -> None:
    _run_within_budget(fn, text)
