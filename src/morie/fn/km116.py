# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.4: BLEU brevity penalty."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_brevity_penalty"]


def kamath_ch8_brevity_penalty(c, r):
    r"""BP = 1 if c > r, else exp(1 - r/c).

    ``c`` is the candidate length and ``r`` the (effective) reference
    length, both in tokens. The penalty is applied exactly at the
    boundary c == r, where exp(1 - 1) = 1 anyway, so the piecewise
    definition is continuous.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.4, printed
    p. 323.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch8_brevity_penalty(3, 5)
    >>> abs(out["estimate"] - math.exp(1 - 5 / 3)) < 1e-12
    True
    >>> kamath_ch8_brevity_penalty(7, 5)["estimate"]
    1.0
    """
    c = float(c)
    r = float(r)
    if c <= 0:
        raise ValueError("the candidate length c must be positive; "
                         "1 - r/c is a pole at c = 0.")
    if r < 0:
        raise ValueError("the reference length r cannot be negative.")
    bp = 1.0 if c > r else float(np.exp(1.0 - r / c))
    return RichResult(payload={
        "estimate": bp, "c": c, "r": r, "penalized": bool(c <= r),
        "n": 1, "method": "BLEU brevity penalty (Kamath Eq 8.4)"})


def cheatsheet():
    return "km116: 1 when longer than the reference, exp(1-r/c) when not"
