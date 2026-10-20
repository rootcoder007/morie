# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON in the bytes rmoriebricklayer writes.

Two forms, both matching the R package's own codec (``tojs()``):

* :func:`canon` -- ``mnfcanon()``: one line, keys of every object sorted by
  their UTF-8 bytes, no spaces. Its SHA-256 is the manifest digest
  ``mnfdgst()`` and the attestation digest, so it has to be byte-exact.
* :func:`pretty` -- ``wrmnfjs()`` / ``cpsbndl()`` files: two-space indent,
  ``"key": value``, an array of scalars on one line (``[1, 2.5]``), an array
  holding objects or arrays one element per line.

Numbers: an integer below 2^31 is written in decimal, every other number
with 17 significant digits (``%.17g``: ``0.1`` is ``0.10000000000000001``,
``1e20`` is ``1e+20``, ``3.0`` is ``3``), so a double reads back exactly.
``None``, NaN and the infinities are ``null``. Strings are UTF-8, with
``"``, ``\\``, and the control characters below 0x20 escaped as JSON
requires; ``/`` and everything else is written as is.
"""

from __future__ import annotations

import json
import math

__all__ = ["canon", "pretty"]

_INT_MAX = 2**31 - 1


def _num(x) -> str:
    if isinstance(x, int) and -_INT_MAX <= x <= _INT_MAX:
        return str(x)
    x = float(x)
    if not math.isfinite(x):
        return "null"
    return f"{x:.17g}"


def _scalar(x) -> str | None:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return _num(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    return None


def _key(k) -> str:
    if not isinstance(k, str):
        raise TypeError(f"object keys must be strings, got {type(k).__name__}")
    return json.dumps(k, ensure_ascii=False)


def _items(d: dict, sort: bool):
    if not sort:
        return list(d.items())
    # R sorts names with method = "radix": byte order, which for UTF-8 is code point order
    return sorted(d.items(), key=lambda kv: kv[0].encode("utf-8"))


def canon(x) -> str:
    """One-line JSON with sorted keys, as ``mnfcanon()`` writes it."""
    s = _scalar(x)
    if s is not None:
        return s
    if isinstance(x, dict):
        return "{" + ",".join(_key(k) + ":" + canon(v) for k, v in _items(x, True)) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(canon(v) for v in x) + "]"
    raise TypeError(f"cannot write {type(x).__name__} as JSON")


def pretty(x, _ind: str = "") -> str:
    """Indented JSON in insertion order, as ``wrmnfjs()`` writes it (no final newline)."""
    s = _scalar(x)
    if s is not None:
        return s
    inner = _ind + "  "
    if isinstance(x, dict):
        if not x:
            return "{}"
        body = (",\n").join(inner + _key(k) + ": " + pretty(v, inner) for k, v in x.items())
        return "{\n" + body + "\n" + _ind + "}"
    if isinstance(x, (list, tuple)):
        if not x:
            return "[]"
        if all(_scalar(v) is not None for v in x):
            return "[" + ", ".join(_scalar(v) for v in x) + "]"
        return "[\n" + ",\n".join(inner + pretty(v, inner) for v in x) + "\n" + _ind + "]"
    raise TypeError(f"cannot write {type(x).__name__} as JSON")
