# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read a CSV the way ``utils::read.csv()`` sees it.

Only what the capsule checks need: the header, the data rows (blank lines
skipped, as ``read.csv`` does), and each column's type after R's
``type.convert()`` -- logical, integer, numeric or character -- with the
values converted and the missing ones (``NA``, and an empty field in a
non-character column) as ``None``.
"""

from __future__ import annotations

import csv
import io
import re

__all__ = ["read", "column", "frame"]

# what type.convert() takes as a number (decimal, exponent, hex, Inf, NaN)
_NUM = re.compile(r"^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|0[xX][0-9a-fA-F]+|Inf|inf|NaN)$")
_INT = re.compile(r"^[+-]?\d+$")
_LGL = {"TRUE": True, "FALSE": False, "T": True, "F": False, "true": True, "false": False, "True": True, "False": False}


def read(path) -> tuple[list[str], list[list[str]]]:
    """Header and data rows of a CSV. Errors (from ``open``/``csv``) when it cannot be read."""
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if not rows:
        raise ValueError(f"{path}: no lines available")
    return rows[0], rows[1:]


def _numval(s: str) -> float:
    if s.lower().startswith(("0x", "+0x", "-0x")):
        return float(int(s, 16))
    return float(s)


def column(values: list[str]) -> tuple[str, list]:
    """``(type, values)`` for one column's raw strings, as ``type.convert()`` decides."""
    present = [v for v in values if v not in ("NA", "")]
    if all(v in _LGL for v in present):
        return "logical", [None if v in ("NA", "") else _LGL[v] for v in values]
    if all(_NUM.match(v) for v in present):
        if all(_INT.match(v) and abs(int(v)) <= 2**31 - 1 for v in present):
            return "integer", [None if v in ("NA", "") else int(v) for v in values]
        return "numeric", [None if v in ("NA", "") else _numval(v) for v in values]
    return "character", [None if v == "NA" else v for v in values]


def frame(path) -> dict[str, tuple[str, list]]:
    """Every column of a CSV as ``{name: (type, values)}``, short rows padded with missing."""
    header, rows = read(path)
    out = {}
    for j, name in enumerate(header):
        out[name] = column([r[j] if j < len(r) else "NA" for r in rows])
    return out
