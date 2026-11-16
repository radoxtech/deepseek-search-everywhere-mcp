"""Static URL validation tests: accepted public URLs and the table of rejected targets (offline)."""

from __future__ import annotations

import pytest

from deepseek_search_everywhere_mcp.urls import UrlError, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a",
        "http://example.com",
        "HTTPS://Example.COM/Path?q=1#frag",
        "https://example.com./a",
        "https://example.com:8443/a",
        "https://sub.example.co.uk/a",
        "https://8.8.8.8/",
        "https://[2001:4860:4860::8888]/",
        "https://bücher.example/a",
        "https://münchen.de/",
        "https://例え.jp/",
        "https://myhome.com/",
        "https://localhost.example.com/",
    ],
)
def test_accepts_public_urls(url):
    assert validate_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # scheme
        "ftp://example.com/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "//example.com/",
        "example.com",
        "",
        # userinfo
        "http://user:pass@example.com/",
        "http://user@example.com/",
        "http://example.com@evil.example/",
        # whitespace, control chars, backslash
        "http://example.com/a b",
        " http://example.com/",
        "http://example.com/\n",
        "http://example.com/\x00",
        "http://example.com\\@evil.example/",
        "http://exa\tmple.com/",
        # malformed
        "http:///path",
        "http://",
        "http://example.com:99999/",
        "http://example.com:abc/",
        "http://[::1/",
        "http://[zzzz]/",
        "http://a..com/",
        "http://" + "a" * 64 + ".com/",
        "http://example.com../",
        "http://localhost../",
        # loopback and private IPv4
        "http://127.0.0.1/",
        "http://127.1.2.3:8080/",
        "http://0.0.0.0/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/",
        "http://169.254.1.1/",
        "http://100.64.0.1/",
        "http://100.127.255.255/",
        "http://224.0.0.1/",
        "http://240.0.0.1/",
        "http://255.255.255.255/",
        "http://192.0.2.1/",
        "http://127.0.0.1./",
        # numeric forms
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0x7f.1/",
        "http://017700000001/",
        "http://0177.0.0.1/",
        "http://127.1/",
        "http://127.0.1/",
        "http://0x7f.0x0.0x0.0x1/",
        "http://１２７.０.０.１/",
        "http://8.8.8/",
        # IPv6
        "http://[::1]/",
        "http://[::]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:7f00:1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://[::ffff:10.0.0.1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[fd12:3456::1]/",
        "http://[ff02::1]/",
        "http://[64:ff9b::7f00:1]/",
        "http://[2002:7f00:1::]/",
        "http://[2001:db8::1]/",
        "http://[::127.0.0.1]/",
        # hostnames
        "http://intranet/",
        "http://localhost/",
        "http://LOCALHOST/",
        "http://localhost./",
        "http://localhost.:8080/",
        "http://foo.localhost/",
        "http://foo.local/",
        "http://printer.lan/",
        "http://router.home/",
        "http://svc.internal/",
        "http://a.b.internal./",
        "http://x.test/",
        "http://x.invalid/",
        "http://abcdef.onion/",
        "http://nas.home.arpa/",
        "http://FOO.Local/",
    ],
)
def test_rejects_internal_or_malformed_urls(url):
    with pytest.raises(UrlError):
        validate_url(url)


def test_rejects_overlong_url():
    validate_url("https://example.com/" + "a" * (8192 - len("https://example.com/")))
    with pytest.raises(UrlError) as exc:
        validate_url("https://example.com/" + "a" * 8192)
    assert exc.value.code == "invalid_url"


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://127.0.0.1/", "ssrf_blocked"),
        ("http://user:pass@example.com/", "ssrf_blocked"),
        ("http://intranet/", "ssrf_blocked"),
        ("ftp://example.com/", "invalid_url"),
        ("http://example.com/a b", "invalid_url"),
    ],
)
def test_error_carries_code_and_message(url, code):
    with pytest.raises(UrlError) as exc:
        validate_url(url)
    assert exc.value.code == code
    assert exc.value.message
    assert str(exc.value) == exc.value.message


def test_url_etld1_extraction() -> None:
    from deepseek_search_everywhere_mcp.urls import _url_etld1

    assert _url_etld1("https://sub.example.com/page") == "example.com"
    assert _url_etld1("https://news.bbc.co.uk/articles/123") == "bbc.co.uk"
    assert _url_etld1("https://sub.portal.gov.pl/test") == "portal.gov.pl"
    assert _url_etld1("https://deep.sub.python.org.au/docs") == "python.org.au"
    assert _url_etld1("http://127.0.0.1:8080/foo") == "127.0.0.1"
    assert _url_etld1("http://localhost:5000") == "localhost"
    assert _url_etld1("https://example.com") == "example.com"
