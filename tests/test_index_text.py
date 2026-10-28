"""JSON escapes and leftover HTML tags in search-index text (content, snippet, summary)."""

from __future__ import annotations

import asyncio
from typing import Any

from deepseek_search_everywhere_mcp import server

# Shaped like a real index entry for a Polish news article: the text is still JSON-escaped.
ESCAPED_CONTENT = (
    "Kr\\u00f3tkie nagrania os\\u0142abiaj\\u0105 koncentracj\\u0119 uczni\\u00f3w. "
    "Badacze z uczelni w \\u0141odzi pisz\\u0105, \\u017ce tzw. \\u201ebrainrot\\u201d \\u2013 "
    "zjawisko znane z internetu \\u2013 da si\\u0119 zmierzy\\u0107. <p>W badaniu wzi\\u0119\\u0142o "
    "udzia\\u0142 1 204 os\\u00f3b."
)
DECODED_CONTENT = (
    "Krótkie nagrania osłabiają koncentrację uczniów. "
    "Badacze z uczelni w Łodzi piszą, że tzw. „brainrot” – "
    "zjawisko znane z internetu – da się zmierzyć. <p>W badaniu wzięło "
    "udział 1 204 osób."
)
ESCAPED_SNIPPET = (
    "informacji.<\\/p><p><br><\\/p><h2>Skutki: zaburzenia snu i samotno\\u015b\\u0107<\\/h2>"
    "<p><br><\\/p><p>Wed\\u0142ug autor\\u00f3w nawyk przewijania ... utrwala schemat."
)


def _item(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Krótkie nagrania a koncentracja",
        "url": "https://news.example.pl/artykul/123",
        "date": "2025-11-23T16:00:00+00:00",
        "authority": 3,
        "rerank_score": 1.5,
        "site_name": "Example News",
    }
    base.update(fields)
    return base


def _search(fake_upstream, tool: str = "deepseek_search_and_read", **item: Any) -> dict[str, Any]:
    fake_upstream(default={"ok": True, "data": {"results": [_item(**item)]}})
    return asyncio.run(getattr(server, tool)("brainrot", count=1))


def test_decode_unicode_escapes_restores_polish_letters() -> None:
    assert server._decode_json_escapes(ESCAPED_CONTENT) == DECODED_CONTENT


def test_decode_surrogate_pair() -> None:
    assert server._decode_json_escapes("ok \\ud83d\\ude00 done") == "ok \U0001f600 done"


def test_decode_keeps_lone_surrogates_and_control_characters() -> None:
    text = "a \\ud83d b \\u0000 c \\u0041"
    assert server._decode_json_escapes(text) == "a \\ud83d b \\u0000 c A"


def test_decode_unescapes_slash() -> None:
    assert server._decode_json_escapes("<\\/p> and a\\/b") == "</p> and a/b"


def test_decode_leaves_code_untouched() -> None:
    text = (
        "Kodowanie znaku \\u00e9 w JSON:\n\n"
        '```json\n{"name": "\\u00e9", "path": "a\\/b"}\n```\n\n'
        'Inline: `"\\u00f3"` stays.'
    )
    assert server._decode_json_escapes(text) == (
        "Kodowanie znaku é w JSON:\n\n"
        '```json\n{"name": "\\u00e9", "path": "a\\/b"}\n```\n\n'
        'Inline: `"\\u00f3"` stays.'
    )


def test_decode_without_escapes_is_identity() -> None:
    text = "Plain C:\\path\\to\\file and \\n literal, no JSON escapes."
    assert server._decode_json_escapes(text) is text


def test_strip_html_tags_in_snippet() -> None:
    cleaned = server._clean_index_text(ESCAPED_SNIPPET)
    assert cleaned == (
        "informacji. Skutki: zaburzenia snu i samotność Według autorów nawyk przewijania ... utrwala schemat."
    )


def test_strip_html_tags_leaves_other_angle_brackets_and_code() -> None:
    text = "Use List<T> when a < b; `<p>` is a tag; <pre>kept</pre> and <a href='x'>link</a>."
    assert server._strip_html_tags(text) == text


def test_search_and_read_decodes_content_snippet_and_summary(fake_upstream) -> None:
    envelope = _search(fake_upstream, content=ESCAPED_CONTENT, snippet=ESCAPED_SNIPPET)
    result = envelope["results"][0]
    assert result["content"] == DECODED_CONTENT
    assert "\\u0" not in result["snippet"] and "<\\/" not in result["snippet"]
    assert "<p>" not in result["snippet"] and "samotność" in result["snippet"]
    assert result["chars"] == len(DECODED_CONTENT)


def test_web_search_decodes_summary(fake_upstream) -> None:
    envelope = _search(fake_upstream, "deepseek_web_search", summary=ESCAPED_CONTENT)
    summary = envelope["results"][0]["summary"]
    assert summary == DECODED_CONTENT.replace(" <p>", " ")


def test_research_sources_are_decoded(fake_upstream) -> None:
    long_content = ESCAPED_CONTENT * 3  # long enough that research does not fetch the page
    fake_upstream(default={"ok": True, "data": {"results": [_item(content=long_content)]}})
    envelope = asyncio.run(server.deepseek_research("brainrot", max_pages=1))
    source = envelope["sources"][0]
    assert source["origin"] == "search"
    assert "\\u0" not in source["content"] and "Łodzi" in source["content"]
