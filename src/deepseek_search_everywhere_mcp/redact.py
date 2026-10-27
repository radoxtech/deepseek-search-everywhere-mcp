"""Secret redaction for text that leaves the process: error messages and notes.

Result content (``results[].content`` and summaries) is never passed through here, because
the user may deliberately be reading a page that contains a secret-like string.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

# Bearer tokens must look like credentials: they contain a digit or are long, so prose such
# as "bearer responsibility" is left alone.
_BEARER = re.compile(
    r"(?i)(\bbearer\s+)"
    r"(?:[A-Za-z0-9._~+/=-]*\d[A-Za-z0-9._~+/=-]*|[A-Za-z0-9._~+/=-]{20,})"
)
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}")
_USERINFO = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_KEY_VALUE = re.compile(r"(?i)\b((?:\w+[_-])?(?:api[_-]?key|token|secret))=[^&\s]+")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with credentials replaced by ``[REDACTED]``.

    Exact ``secrets`` are removed first, then bearer tokens, ``sk-`` style keys, URL
    userinfo and ``api_key``/``token``/``secret`` query-style pairs.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    text = _BEARER.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _SK_KEY.sub(REDACTED, text)
    text = _USERINFO.sub(f"://{REDACTED}@", text)
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
