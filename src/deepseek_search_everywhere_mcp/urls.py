"""Static URL validation for the fetch tools.

This is an abuse filter, not a security boundary. The page is fetched remotely by DeepSeek,
so resolving the hostname locally would say nothing about what DeepSeek connects to (split
horizon DNS, rebinding, different network vantage point). No DNS lookup is therefore
performed: only the syntax of the URL and literal IP addresses are inspected.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

MAX_URL_LENGTH = 8192

CODE_BLOCKED = "ssrf_blocked"
CODE_INVALID = "invalid_url"

_BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".test",
    ".invalid",
    ".onion",
    ".home.arpa",
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_METADATA = ipaddress.ip_address("169.254.169.254")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
# A last label that parsers such as WHATWG URL or inet_aton read as a number
# (decimal, octal or 0x hex), making the host an obfuscated IPv4 address.
_NUMERIC_LABEL = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)")


class UrlError(Exception):
    """A URL rejected by :func:`validate_url`, with a machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _blocked(message: str) -> UrlError:
    return UrlError(CODE_BLOCKED, message)


def _invalid(message: str) -> UrlError:
    return UrlError(CODE_INVALID, message)


def _check_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Raise :class:`UrlError` unless ``ip`` is a public, globally routable address."""
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            _check_ip(embedded)
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or ip == _METADATA
        or (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT)
        or not ip.is_global
    ):
        raise _blocked("URL points to a non-public IP address.")


def validate_url(url: str) -> str:
    """Return ``url`` unchanged if it may be sent to the remote fetcher, else raise.

    Accepts only ``http``/``https`` URLs of at most 8192 characters without whitespace,
    control characters, backslashes or userinfo. Hosts must be public: IP literals (including
    bracketed IPv6 and IPv4-mapped forms) must be globally routable, numeric/hex/octal IPv4
    spellings are rejected, and dotless or local-only names (``localhost``, ``*.local``,
    ``*.internal`` and similar) are refused. IDN hosts are accepted once IDNA-encodable.

    No DNS resolution happens here (see the module docstring).

    Raises:
        UrlError: with code ``invalid_url`` for malformed input or ``ssrf_blocked`` for
            internal targets.
    """
    if len(url) > MAX_URL_LENGTH:
        raise _invalid(f"URL is longer than {MAX_URL_LENGTH} characters.")
    if any(ch.isspace() or ch == "\\" or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise _invalid("URL contains whitespace, control characters or a backslash.")
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError:
        raise _invalid("URL is malformed.") from None
    if parts.scheme not in ("http", "https"):
        raise _invalid("Only http and https URLs are supported.")
    if "@" in parts.netloc:
        raise _blocked("URLs with userinfo are not allowed.")

    host = (parts.hostname or "").removesuffix(".")
    if not host or host.endswith("."):
        raise _invalid("URL has no valid host.")

    if ":" in host:  # IPv6 literal; urlsplit already removed the brackets
        try:
            literal6 = ipaddress.ip_address(host)
        except ValueError:
            raise _invalid("URL has an invalid IPv6 address.") from None
        _check_ip(literal6)
        return url

    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise _invalid("URL host is not a valid IDNA name.") from None

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None  # a name, not a dotted-quad literal
    if literal is not None:
        _check_ip(literal)
        return url
    if _NUMERIC_LABEL.fullmatch(host.rsplit(".", 1)[-1]):
        raise _blocked("Numeric IPv4 notation is not allowed.")
    if "." not in host:
        raise _blocked("Hosts without a dot are not allowed.")
    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        raise _blocked("Local and internal hostnames are not allowed.")
    return url
