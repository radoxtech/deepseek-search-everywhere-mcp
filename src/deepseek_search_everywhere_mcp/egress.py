"""Egress guard for outgoing search queries and fetch URLs (S2, S10).

Prevents prompt injection from exfiltrating credentials, keys, or secrets
through outgoing fetch requests or search queries.
"""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import Settings

# Secret patterns
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}")
_BEARER = re.compile(
    r"(?i)\bbearer(?:[\s/=]|%20)+(?:[A-Za-z0-9._~+/=-]*\d[A-Za-z0-9._~+/=-]*|[A-Za-z0-9._~+/=-]{20,})"
)
_KEY_VALUE = re.compile(r"(?i)\b((?:\w+[_-])?(?:api[_-]?key|token|secret|password))=[^&\s]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
_AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_USERINFO = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def _shannon_entropy(token: str) -> float:
    """Calculate Shannon entropy in bits per character."""
    if not token:
        return 0.0
    length = len(token)
    counts = Counter(token)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _has_high_entropy_token(text: str, threshold: float = 4.2) -> bool:
    """Return True if any token > 32 chars exceeds the Shannon entropy threshold."""
    # Split on common separators
    tokens = re.split(r"[\s/?&=_#\.-]+", text)
    for token in tokens:
        if len(token) > 32:
            if _shannon_entropy(token) >= threshold:
                return True
    return False


def _matches_secret_pattern(text: str) -> str | None:
    """Return description of matched secret pattern, or None if clean."""
    if _SK_KEY.search(text):
        return "sk- API key"
    if _BEARER.search(text):
        return "Bearer credential"
    if _KEY_VALUE.search(text):
        return "key-value credential pair"
    if _JWT.search(text):
        return "JWT token"
    if _AWS_KEY.search(text):
        return "AWS access key"
    if _USERINFO.search(text):
        return "embedded URL credentials"
    return None


def _is_domain_match(host: str, target_domain: str) -> bool:
    """Return True if host is target_domain or a subdomain of it."""
    host = host.lower().strip()
    target = target_domain.lower().strip()
    return host == target or host.endswith(f".{target}")


def check_search_egress(query: str, settings: Settings | None = None) -> str | None:
    """Check outgoing search query for secrets or egress violations.

    Returns error message if blocked, or None if permitted.
    """
    secret = _matches_secret_pattern(query)
    if secret:
        return f"Search query blocked: contains potential secret ({secret})."

    if settings is not None and settings.egress_entropy_check:
        if _has_high_entropy_token(query):
            return "Search query blocked: high-entropy token detected (>32 chars)."

    return None


def check_fetch_egress(url: str, settings: Settings | None = None) -> str | None:
    """Check outgoing fetch URL for secrets, allow/deny domains, and entropy.

    Returns error message if blocked, or None if permitted.
    """
    unquoted = urllib.parse.unquote(url)
    secret = _matches_secret_pattern(url) or _matches_secret_pattern(unquoted)
    if secret:
        return f"Fetch URL blocked: contains potential secret ({secret})."

    if settings is not None:
        try:
            parsed = urllib.parse.urlsplit(url)
            host = (parsed.hostname or "").lower()
        except ValueError:
            host = ""

        if host:
            if settings.fetch_deny_domains:
                for denied in settings.fetch_deny_domains:
                    if _is_domain_match(host, denied):
                        return f"Fetch URL blocked: domain {host} is in fetch_deny_domains."

            if settings.fetch_allow_domains:
                allowed = any(_is_domain_match(host, d) for d in settings.fetch_allow_domains)
                if not allowed:
                    return f"Fetch URL blocked: domain {host} is not in fetch_allow_domains."

        if settings.egress_entropy_check:
            # Check path and query string for high-entropy tokens
            target = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
            if _has_high_entropy_token(target):
                return "Fetch URL blocked: high-entropy token detected in URL (>32 chars)."

    return None
