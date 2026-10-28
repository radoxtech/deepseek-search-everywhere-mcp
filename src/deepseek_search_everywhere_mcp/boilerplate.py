"""Remove known site-UI boilerplate from page content.

DeepSeek renders some sites with their interface localised to Chinese, whatever the language of
the page itself: a Reddit thread in English still carries the comment-sort menu (``排序方式：``,
``最佳``, ``新``...), the ``# 评论区域`` heading and the sign-up footer. Those lines cost tokens and
end up in quotes.

The rules here are deliberately conservative. Each one belongs to a site and only applies to
pages on that site's domains, and a line is removed only when it consists *entirely* of a
known UI label, so user-authored text (Chinese included) is never touched. Single-word labels,
which could plausibly be a whole comment, are only removed inside the menu they belong to.
Fenced code blocks are left as they are.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field

_FENCE = re.compile(r"^[ \t]*(?P<mark>`{3,}|~{3,})")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Menu:
    """A UI menu: a header line followed by option lines (blank lines in between are allowed)."""

    header: str
    option: re.Pattern[str]


@dataclass(frozen=True)
class SiteRules:
    """Boilerplate rules for one site, applied only to pages on ``domains`` or their subdomains."""

    domains: tuple[str, ...]
    lines: frozenset[str] = frozenset()
    patterns: tuple[re.Pattern[str], ...] = ()
    menus: tuple[Menu, ...] = field(default_factory=tuple)

    def applies_to(self, host: str) -> bool:
        """Return whether ``host`` is one of the rule's domains or a subdomain of one."""
        return any(host == domain or host.endswith("." + domain) for domain in self.domains)

    def is_ui_line(self, line: str) -> bool:
        """Return whether the normalised ``line`` is a known UI label on its own."""
        return line in self.lines or any(pattern.fullmatch(line) for pattern in self.patterns)

    def menu_for(self, line: str) -> Menu | None:
        """Return the menu that ``line`` is the header of, if any."""
        return next((menu for menu in self.menus if menu.header == line), None)


REDDIT = SiteRules(
    domains=("reddit.com",),
    lines=frozenset(
        {
            "# 评论区域",  # "Comments section" heading
            "帖子已归档。无法发布新评论，并且无法进行投票。",  # "Archived post..." banner
            "版主已移除评论",  # "Comment removed by moderator"
            "版主公告",  # "Moderator announcement"
            "阅读更多内容 »",  # "Read more »"
            "版主公告 阅读更多内容 »",
            "订阅互联网上最真实的社区",  # sign-up footer
            "通过 Google 继续操作",
            "通过 Google 继续操作通过 Google 继续操作。在新标签页中打开",
            "通过电话号码继续",
            "通过电子邮件地址继续",
        }
    ),
    patterns=(
        # Author line of a deleted account ("[deleted]"), with or without heading marks and bullets.
        re.compile(r"(?:#{1,6} )?\[已删除\](?: ?•)*"),
    ),
    menus=(
        Menu(
            header="排序方式：",  # "Sort by:"
            option=re.compile(r"(?:最佳|点赞最多|新|有争议|旧|问答|最受欢迎)(?:（默认）)?|打开评论排序选项"),
        ),
    ),
)

RULES: tuple[SiteRules, ...] = (REDDIT,)


def _host(url: str | None) -> str | None:
    """Return the lower-case host of ``url``, or None when it has none."""
    if not url:
        return None
    try:
        return urllib.parse.urlsplit(url.strip()).hostname
    except ValueError:
        return None


def rules_for(url: str | None) -> tuple[SiteRules, ...]:
    """Return the rule sets that apply to a page at ``url``."""
    host = _host(url)
    if host is None:
        return ()
    return tuple(rules for rules in RULES if rules.applies_to(host))


def _normalise(line: str) -> str:
    return _SPACES.sub(" ", line).strip()


def _strip_lines(text: str, rules: tuple[SiteRules, ...]) -> str:
    """Drop UI lines and menus outside code fences, collapsing the blank runs left behind."""
    out: list[str] = []
    fence: str | None = None
    menu: Menu | None = None
    dropped = False  # a line was dropped since the last kept non-blank line
    for line in text.split("\n"):
        fence_match = _FENCE.match(line)
        if fence is not None or fence_match:
            if fence is None and fence_match:
                fence = fence_match["mark"][0]
            elif fence_match and fence_match["mark"][0] == fence:
                fence = None
            out.append(line)
            menu, dropped = None, False
            continue
        normalised = _normalise(line)
        if not normalised:
            if not (dropped and (not out or not out[-1].strip())):
                out.append(line)
            continue
        if menu is not None and menu.option.fullmatch(normalised):
            dropped = True
            continue
        menu = next((m for r in rules if (m := r.menu_for(normalised)) is not None), None)
        if menu is not None or any(r.is_ui_line(normalised) for r in rules):
            dropped = True
            continue
        out.append(line)
        dropped = False
    if dropped:
        while out and not out[-1].strip():
            out.pop()
    return "\n".join(out)


def remove_boilerplate(text: str, url: str | None) -> str:
    """Return ``text`` without the known site-UI lines of the site ``url`` belongs to.

    Pages on sites without rules, and pages without a usable URL, are returned unchanged.
    """
    rules = rules_for(url)
    if not rules or not text:
        return text
    return _strip_lines(text, rules)
