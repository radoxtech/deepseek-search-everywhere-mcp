"""A cookie-consent banner at the very start of the content is removed, and nothing else is."""

from __future__ import annotations

import asyncio
import pathlib

from deepseek_search_everywhere_mcp import server
from deepseek_search_everywhere_mcp.boilerplate import remove_boilerplate, strip_consent_banner

# A shortened IAB TCF consent dialog as DeepSeek renders it for a Polish news site.
BANNER_PL = (pathlib.Path(__file__).parent / "fixtures" / "consent_banner_pl.md").read_text(encoding="utf-8")
BANNER_ONET_STYLE = (
    "Zanim klikniesz którykolwiek przycisk prosimy o przeczytanie do końca tej informacji – dotyczy ona "
    "Twoich danych osobowych.\n"
    "Klikając „Przejdź do serwisu” udzielasz zgody na przetwarzanie Twoich danych osobowych przez nas i "
    "naszych Zaufanych Partnerów w celach marketingowych.\n"
    "Zgoda jest dobrowolna. Możesz jej odmówić lub ograniczyć jej zakres klikając w "
    "„Ustawienia zaawansowane”.\n"
)
BANNER_EN = (
    "## We value your privacy\n\n"
    "We and our partners use cookies and similar technologies to store and access information on your "
    "device, for personalised advertising and content, measurement and audience research.\n\n"
    "Accept all\n\nReject all\n\nManage preferences\n"
)
BANNER_DE = (
    "Wir verwenden Cookies und ähnliche Technologien, um Inhalte und Werbung zu personalisieren und die "
    "Nutzung unserer Website zu analysieren. Ihre Einwilligung können Sie jederzeit widerrufen.\n\n"
    "[Alle akzeptieren]\n\nEinstellungen\n"
)
ARTICLE = (
    "# Krótkie nagrania a koncentracja\n\n"
    "Nowe badanie pokazuje, że częste przewijanie krótkich filmów osłabia uwagę.\n\n"
    "Autorzy przeanalizowali dane kilkudziesięciu tysięcy uczestników."
)
URL = "https://news.example.pl/artykul/123"


def test_polish_tcf_banner_alone_is_removed_entirely():
    assert strip_consent_banner(BANNER_PL) == ""


def test_banner_before_article_leaves_the_article_intact():
    for banner in (BANNER_PL, BANNER_ONET_STYLE, BANNER_EN, BANNER_DE):
        assert strip_consent_banner(banner + "\n" + ARTICLE) == ARTICLE


def test_article_about_privacy_is_kept():
    text = (
        "## We value your privacy, says the regulator\n\n"
        "The data protection authority fined an advertising network for setting cookies without consent. "
        "Partners of the network shared tracking identifiers with vendors.\n\n"
        "The company said it would appeal."
    )
    assert strip_consent_banner(text) == text  # no dialog control, so no banner


def test_privacy_phrases_in_the_middle_are_kept():
    text = ARTICLE + "\n\n" + BANNER_EN + "\nMore article text."
    assert strip_consent_banner(text) == text


def test_banner_must_open_the_text():
    text = "# Title first\n\n" + BANNER_EN + "\n" + ARTICLE
    assert strip_consent_banner(text) == text


def test_code_block_ends_the_banner_and_is_kept():
    code = "```text\nAccept all\n```"
    assert strip_consent_banner(BANNER_EN + "\n" + code) == code


def test_empty_and_plain_text_are_unchanged():
    assert strip_consent_banner("") == ""
    assert strip_consent_banner(ARTICLE) == ARTICLE


def test_remove_boilerplate_applies_the_consent_rule_on_any_site():
    assert remove_boilerplate(BANNER_EN + "\n" + ARTICLE, URL) == ARTICLE
    assert remove_boilerplate(BANNER_EN + "\n" + ARTICLE, None) == ARTICLE


def test_fetch_of_a_banner_only_page_returns_empty_content_with_a_note(fake_upstream):
    fake_upstream(default={"ok": True, "data": {"title": "Artykuł", "content": BANNER_PL}})
    envelope = asyncio.run(server.deepseek_fetch(URL))
    assert envelope["ok"] is True and envelope["content"] == "" and envelope["chars"] == 0
    assert "cookie-consent banner" in envelope["notes"][0]


def test_fetch_strip_boilerplate_off_keeps_the_banner(fake_upstream):
    fake_upstream(default={"ok": True, "data": {"title": "Artykuł", "content": BANNER_EN + "\n" + ARTICLE}})
    envelope = asyncio.run(server.deepseek_fetch(URL, strip_boilerplate=False))
    assert envelope["content"].startswith("## We value your privacy")


def test_search_content_loses_a_leading_banner(fake_upstream):
    item = {"title": "Artykuł", "url": URL, "content": BANNER_DE + "\n" + ARTICLE}
    fake_upstream(default={"ok": True, "data": {"results": [item]}})
    envelope = asyncio.run(server.deepseek_search_and_read("koncentracja", count=1))
    assert envelope["results"][0]["content"] == ARTICLE
