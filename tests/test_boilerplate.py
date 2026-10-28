"""Known site-UI boilerplate is removed from page content, and nothing else is."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import re

import pytest

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.boilerplate import remove_boilerplate, rules_for

# A shortened, anonymised Reddit thread as DeepSeek returns it: English page, Chinese site UI.
THREAD = (pathlib.Path(__file__).parent / "fixtures" / "reddit_thread_en.md").read_text(encoding="utf-8")
REDDIT_URL = "https://www.reddit.com/r/example/comments/abc123/heavy_phone_use/"
OTHER_URL = "https://example.com/article"
CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
USER_LINES = (
    "# Does heavy phone use make focus worse?",
    "I have noticed that scrolling a lot seems to make it harder to concentrate. Is this common?",
    "Yes, for me too. I set app timers and it helps a bit.",
    "Switching the screen to greyscale made it less tempting.",
    "Thanks everyone, some really useful tips here.",
)


def test_reddit_thread_loses_every_ui_line_and_keeps_the_text():
    cleaned = remove_boilerplate(THREAD, REDDIT_URL)
    assert not CJK.search(cleaned)
    assert cleaned.split("\n\n") == list(USER_LINES)


def test_flag_off_keeps_raw_content_and_unknown_hosts_are_untouched():
    for url in (OTHER_URL, "https://notreddit.com/r/x", "https://reddit.com.example.org/", None, "not a url"):
        assert remove_boilerplate(THREAD, url) == THREAD
    assert rules_for("https://old.reddit.com/r/x") == rules_for("https://reddit.com/r/x") != ()


@pytest.mark.parametrize(
    "line",
    [
        "最佳",  # a single-word comment outside the sort menu
        "新",
        "我觉得新版本更好，排序方式也更清楚。",  # user text in Chinese
        "排序方式：最佳",  # not a whole-line label
        "# 评论区域 of this subreddit",
    ],
)
def test_user_text_with_cjk_is_kept(line):
    text = f"# Title\n\nFirst comment.\n\n{line}\n\nLast comment."
    assert remove_boilerplate(text, REDDIT_URL) == text


def test_menu_options_end_at_the_first_other_line():
    text = "Post.\n\n排序方式：\n\n最佳\n\n新\n\nA comment.\n\n新\n\nEnd."
    assert remove_boilerplate(text, REDDIT_URL) == "Post.\n\nA comment.\n\n新\n\nEnd."


def test_code_fences_are_left_alone():
    text = "Post.\n\n```\n# 评论区域\n排序方式：\n最佳\n```\n\n# 评论区域\n\nA comment."
    assert (
        remove_boilerplate(text, REDDIT_URL)
        == "Post.\n\n```\n# 评论区域\n排序方式：\n最佳\n```\n\nA comment."
    )


def test_whitespace_variants_match():
    text = "Post.\n\n  版主公告   阅读更多内容 »  \n\n##  [已删除]  •\n\nA comment."
    assert remove_boilerplate(text, REDDIT_URL) == "Post.\n\nA comment."


def _search_item(url: str, content: str) -> dict:
    return {"title": "Thread", "url": url, "site_name": "Reddit", "content": content}


def test_search_strips_before_budgets_and_shares_the_cache(fake_upstream):
    # The upstream escapes heading marks; unescaping runs first, so "\# 评论区域" is caught too.
    escaped = THREAD.replace("# 评论区域", "\\# 评论区域")
    fake = fake_upstream(default={"ok": True, "data": {"results": [_search_item(REDDIT_URL, escaped)]}})
    clean = asyncio.run(server.deepseek_search_and_read("q", count=1))
    raw = asyncio.run(server.deepseek_search_and_read("q", count=1, strip_boilerplate=False))
    assert len(fake.calls) == 1  # one cache entry serves both
    assert raw["cached"] is True
    result = clean["results"][0]
    assert not CJK.search(result["content"])
    assert result["content_chars"] == result["chars"] == len(result["content"])
    assert raw["results"][0]["content"] == server._unescape_markdown(escaped)

    budget = len(remove_boilerplate(THREAD, REDDIT_URL))
    cut = asyncio.run(server.deepseek_search_and_read("q", count=1, max_chars_per_result=budget))
    assert cut["results"][0]["content"] == result["content"]
    assert server.TRUNCATED_NOTE not in cut["notes"]


def test_search_leaves_other_sites_alone(fake_upstream):
    fake_upstream(default={"ok": True, "data": {"results": [_search_item(OTHER_URL, THREAD)]}})
    envelope = asyncio.run(server.deepseek_web_search("q", count=1, need_content=True))
    assert envelope["results"][0]["content"] == THREAD


def test_fetch_tools_and_research_strip_reddit_ui(fake_upstream):
    fake = fake_upstream(
        responses={
            server.SEARCH_PATH: {"ok": True, "data": {"results": [_search_item(REDDIT_URL, "short")]}},
            server.FETCH_PATH: {"ok": True, "data": {"title": "Thread", "content": THREAD}},
        }
    )
    single = asyncio.run(server.deepseek_fetch(REDDIT_URL))
    many = asyncio.run(server.deepseek_fetch_many([REDDIT_URL]))
    research = asyncio.run(server.deepseek_research("q", max_pages=1))
    raw = asyncio.run(server.deepseek_fetch(REDDIT_URL, strip_boilerplate=False))
    for content in (single["content"], many["results"][0]["content"], research["sources"][0]["content"]):
        assert content == remove_boilerplate(THREAD, REDDIT_URL)
    assert raw["content"] == THREAD
    assert raw["cached"] is True
    assert sum(call["path"] == server.FETCH_PATH for call in fake.calls) == 1


@pytest.mark.parametrize(
    "name",
    [
        "deepseek_web_search",
        "deepseek_search_and_read",
        "deepseek_fetch",
        "deepseek_fetch_many",
        "deepseek_research",
    ],
)
def test_flag_is_keyword_only_and_on_by_default(name):
    parameter = inspect.signature(getattr(server, name)).parameters["strip_boilerplate"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True
