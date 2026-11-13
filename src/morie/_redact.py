"""Keep a key or token out of error text.

A server may echo a credential back in an error body, as it was sent or in another encoding. What
this package then shows the user (or writes to a log) has every such form replaced: the value as
sent, percent-encoded, hex, base64 at each of its three alignments (standard and URL-safe), and
with separators taken out or put in. The same rule as rmoriebricklayer's.
"""

from __future__ import annotations

import base64
import re
from urllib.parse import quote

MARK = "[redacted]"
_SHORT = 8  # a shorter value would blank ordinary words


def key_forms(secret: str) -> list[str]:
    """Every form of ``secret`` that is replaced, longest first.

    Examples:
        >>> "c2stYWJjZGVmZ2hpams" in key_forms("sk-abcdefghijk")
        True
    """
    raw = secret.encode("utf-8")
    forms = {secret, quote(secret, safe=""), raw.hex(), raw.hex().upper(), re.sub(r"[^0-9A-Za-z]", "", secret)}
    for off in range(3):
        n = len(raw) + off
        for enc in (base64.b64encode, base64.urlsafe_b64encode):
            b = enc(b"\0" * off + raw).decode("ascii").rstrip("=")
            # the characters that depend on the bytes before and after the value are left out
            forms.add(b[(0, 2, 3)[off] : len(b) - (0 if n % 3 == 0 else 1)])
    return sorted((f for f in forms if len(f) >= _SHORT), key=len, reverse=True)


def redact(text, *secrets) -> str:
    """``text`` with every form of each secret replaced by ``[redacted]``.

    Examples:
        >>> redact("bad key sk-abcdefghijk", "sk-abcdefghijk")
        'bad key [redacted]'
    """
    text = str(text)
    for s in secrets:
        if not isinstance(s, str) or len(s) < _SHORT:
            continue
        for f in key_forms(s):
            text = text.replace(f, MARK)
        text = re.sub(re.escape(s.encode("utf-8").hex()), MARK, text, flags=re.IGNORECASE)
        bare = re.sub(r"[^0-9A-Za-z]", "", s)
        if len(bare) >= _SHORT:
            # the value with separators dropped, or put between its characters
            text = re.sub(r"[\W_]{0,4}".join(map(re.escape, bare)), MARK, text)
    return text
