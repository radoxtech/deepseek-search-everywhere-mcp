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

from deepseek_search_everywhere_mcp import deep_research, sanitize
from deepseek_search_everywhere_mcp.deep_research import (
    _extract_numbers_and_ranges,
    _mark_unverified,
    normalize_citations,
)
from deepseek_search_everywhere_mcp.extract import normalize_for_evidence
from deepseek_search_everywhere_mcp.sanitize import sanitize_markdown_links, sanitize_untrusted_text
from deepseek_search_everywhere_mcp.server import _strip_html_tags, _unescape_markdown
from deepseek_search_everywhere_mcp.text import _HTML_TAG, _decode_json_escape, _decode_json_escapes

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
# Quadratic cases found by an adversarial scan, now fixed; the equivalence tests below pin the old output.

LINEAR_CASES = [
    # A sign must touch its digits, so a run of spaces no longer backtracks.
    pytest.param(_extract_numbers_and_ranges, _repeat(" "), id="grounding-numbers"),
    pytest.param(_extract_numbers_and_ranges, _repeat("- "), id="grounding-dashes"),
    pytest.param(_extract_numbers_and_ranges, _repeat("1 January "), id="grounding-dates"),
    pytest.param(_extract_numbers_and_ranges, _repeat("1 000 "), id="grounding-space-thousands"),
    pytest.param(_extract_numbers_and_ranges, _repeat("10^"), id="grounding-powers"),
    pytest.param(_extract_numbers_and_ranges, _repeat("1 pp "), id="grounding-percentage-points"),
    pytest.param(_unescape_markdown, _repeat("\\*"), id="unescape-markdown-escapes"),
    pytest.param(_strip_html_tags, _repeat("<p>"), id="strip-html-dense-tags"),
    pytest.param(sanitize_untrusted_text, _repeat("[a](https://x.y) "), id="sanitize-many-links"),
    # Whitespace is collapsed before _PUNCT_SPACE_RE runs, so this stays linear.
    pytest.param(normalize_for_evidence, _repeat(" "), id="evidence-punct-space"),
    pytest.param(sanitize_untrusted_text, _repeat("![a]("), id="md-image-unclosed"),
    pytest.param(sanitize_untrusted_text, _repeat("[1,"), id="md-link-many-openers"),
    pytest.param(_unescape_markdown, _repeat("`"), id="markdown-token-backticks"),
    pytest.param(_strip_html_tags, _repeat(" \t", suffix="<"), id="tag-token-whitespace"),
    pytest.param(lambda t: normalize_citations(t, 10), _repeat(" \t"), id="citations-punct"),
    pytest.param(_decode_json_escapes, _repeat("`", suffix="\\u00f3"), id="json-escape-backticks"),
    pytest.param(deep_research._strip_md_links, _repeat("[1,"), id="title-md-link-openers"),
    pytest.param(deep_research._strip_md_links, _repeat("![a]("), id="title-md-image-unclosed"),
    pytest.param(_mark_unverified, _repeat("!"), id="sentence-end"),
]


@pytest.mark.parametrize(("fn", "text"), LINEAR_CASES)
def test_helper_is_linear_on_adversarial_input(fn: Callable[[str], Any], text: str) -> None:
    _run_within_budget(fn, text)


# --- Equivalence of the fixed helpers with the patterns they replaced ---------------------------------------
# Each old pattern is kept verbatim; short random strings keep it fast, and the alphabets are dense in the
# characters that matter so openers, closers, fences and blank lines collide often.

_OLD_MD_IMAGE_RE = re.compile(r"(?<!\\)!\[(?P<alt>[^\]]*)\]\((?P<url>[^\)]*)\)")

_OLD_MARKDOWN_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\(?P<char>[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])",
    re.DOTALL | re.MULTILINE,
)

_OLD_TAG_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|(?P<tags>(?:[ \t]*" + _HTML_TAG.pattern + r"[ \t]*)+)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

_OLD_PUNCT_SPACE_RE = re.compile(r"[ \t]+([.,;:!?])")
_OLD_SENTENCE_END_RE = re.compile(r"([.!?]+)([\"'”’)\]]*)\s*$")


def _old_unescape_markdown(text: str) -> str:
    return _OLD_MARKDOWN_TOKEN.sub(lambda match: match["char"] or match[0], text)


def _old_strip_html_tags(text: str) -> str:
    if "<" not in text:
        return text
    stripped = _OLD_TAG_TOKEN.sub(lambda match: " " if match["tags"] else match[0], text)
    return stripped.strip() if stripped != text else text


def _random_strings(seed: int, alphabet: str, max_len: int, count: int = 20_000) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len))) for _ in range(count)]


def test_md_image_regex_matches_like_the_old_pattern() -> None:
    def show(match: re.Match[str]) -> str:
        return f"<{match['alt']}|{match['url']}>"

    for sample in _random_strings(4321, "![]() \\a", 16):
        old = _OLD_MD_IMAGE_RE.sub(show, sample)
        new = sanitize._sub_bracketed(
            sanitize._MD_IMAGE_RE, sanitize._MD_IMAGE_START_RE, 2, show, sample, needs_close_paren=True
        )
        assert new == old, repr(sample)


def test_md_link_substitution_matches_the_old_pattern() -> None:
    def show(match: re.Match[str]) -> str:
        return f"<{match['label']}|{match['url']}>"

    for sample in _random_strings(2468, "[]()! a:\t\\", 16):
        old = _OLD_MD_LINK_RE.sub(show, sample)
        new = sanitize._sub_bracketed(sanitize._MD_LINK_RE, sanitize._MD_LINK_START_RE, 1, show, sample)
        assert new == old, repr(sample)


def test_sanitize_untrusted_text_matches_the_old_patterns() -> None:
    def old_sanitize(text: str) -> str:
        def image(match: re.Match[str]) -> str:
            alt = match.group("alt").strip()
            return f"[image: {alt}]({match.group('url')})" if alt else f"[image]({match.group('url')})"

        def link(match: re.Match[str]) -> str:
            return (
                f"[{match['label']}]()"
                if match["url"].strip().lower().startswith("javascript:")
                else match[0]
            )

        return _OLD_MD_LINK_RE.sub(link, _OLD_MD_IMAGE_RE.sub(image, text))

    def new_sanitize(text: str) -> str:
        return sanitize_untrusted_text(text)[0]

    for sample in _random_strings(97, "![]()a:javscript \\", 24, 5_000):
        # Only the javascript: scheme is covered here; the other schemes share the same code path.
        if "javascript:" in sample:
            assert new_sanitize(sample) == old_sanitize(sample), repr(sample)


@pytest.mark.parametrize("alphabet", ["`~\\\n *a!()[]_", "``` \n\\a~", "`\n\t\\*"])
def test_unescape_markdown_matches_the_old_pattern(alphabet: str) -> None:
    for sample in _random_strings(777, alphabet, 22):
        assert _unescape_markdown(sample) == _old_unescape_markdown(sample), repr(sample)


@pytest.mark.parametrize("alphabet", ["`~\n <>pbr/ \t", "<p>/br \t\nx`", "<>pB/ \t\n"])
def test_strip_html_tags_matches_the_old_pattern(alphabet: str) -> None:
    for sample in _random_strings(555, alphabet, 22):
        assert _strip_html_tags(sample) == _old_strip_html_tags(sample), repr(sample)


def test_strip_html_tags_matches_the_old_pattern_on_whole_tags() -> None:
    rng = random.Random(31)
    pieces = [
        "<p>",
        "</p>",
        "<br/>",
        "<br />",
        "< p>",
        "<P>",
        " ",
        "\t",
        "\n",
        "`",
        "```",
        "~~~",
        "x",
        "<",
        "\n\n",
    ]
    for _ in range(20_000):
        sample = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 10)))
        assert _strip_html_tags(sample) == _old_strip_html_tags(sample), repr(sample)


def test_punct_space_regex_matches_like_the_old_pattern() -> None:
    for sample in _random_strings(11, "a .,;:!?\t\n", 16):
        old = _OLD_PUNCT_SPACE_RE.sub(r"\1", sample)
        assert deep_research._PUNCT_SPACE_RE.sub(r"\1", sample) == old, repr(sample)


def test_sentence_end_regex_matches_like_the_old_pattern() -> None:
    for sample in _random_strings(22, "a .!?)]\"'”’\t\n", 16):
        old = _OLD_SENTENCE_END_RE.search(sample)
        new = deep_research._SENTENCE_END_RE.search(sample)
        assert (new and (new.span(), new.groups())) == (old and (old.span(), old.groups())), repr(sample)


# The patterns before the fix; one alternation with code spans, and ``re.sub`` over every link opener.
_OLD_JSON_ESCAPE_TOKEN = re.compile(
    r"(?P<fence>^[ \t]*(?P<mark>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=mark)[ \t]*$|\Z))"
    r"|(?P<span>(?P<ticks>`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)(?P=ticks)(?!`))"
    r"|\\u(?P<high>[dD][89abAB][0-9a-fA-F]{2})\\u(?P<low>[dD][c-fC-F][0-9a-fA-F]{2})"
    r"|\\u(?P<unit>[0-9a-fA-F]{4})"
    r"|(?P<slash>\\/)",
    re.DOTALL | re.MULTILINE,
)
_OLD_TITLE_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def _old_decode_json_escapes(text: str) -> str:
    return _OLD_JSON_ESCAPE_TOKEN.sub(
        lambda m: m[0] if m["fence"] or m["span"] else _decode_json_escape(m), text
    )


@pytest.mark.parametrize(
    "alphabet",
    ["`~\\\nu0f3/ d8c", ["\\u00f3", "\\/", "`", "```", "\n", " ", "x", "\\ud83d\\ude00", "\\u0001"]],
)
def test_decode_json_escapes_matches_the_old_pattern(alphabet: Any) -> None:
    for sample in _random_strings(4242, alphabet, 18):
        assert _decode_json_escapes(sample) == _old_decode_json_escapes(sample), repr(sample)


@pytest.mark.parametrize("alphabet", ["[]()!a \n", "[](!x", "!![[]](())a"])
def test_strip_md_links_matches_the_old_pattern(alphabet: str) -> None:
    for sample in _random_strings(99, alphabet, 22):
        assert deep_research._strip_md_links(sample) == _OLD_TITLE_MD_LINK_RE.sub(r"\1", sample), repr(sample)
