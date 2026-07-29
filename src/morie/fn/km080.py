# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.4: the WEAT test statistic."""

import numpy as np

from ._richresult import RichResult
from .km081 import _s

__all__ = ["kamath_ch6_weat_function"]


def _sums(A_1, A_2, W_1, W_2):
    """Per-attribute s values for both protected sets (km082 reuses)."""
    a1 = np.atleast_2d(np.asarray(A_1, dtype=float))
    a2 = np.atleast_2d(np.asarray(A_2, dtype=float))
    if a1.shape[0] == 0 or a2.shape[0] == 0:
        raise ValueError("A_1 and A_2 must each contain at least one "
                         "attribute word.")
    if a1.shape[1] != a2.shape[1]:
        raise ValueError(
            f"A_1 has width {a1.shape[1]} but A_2 has {a2.shape[1]}.")
    s1 = [_s(v, W_1, W_2) for v in a1]
    s2 = [_s(v, W_1, W_2) for v in a2]
    return s1, s2


def kamath_ch6_weat_function(A_1, A_2, W_1, W_2):
    """f = sum_{a1 in A_1} s(a1,W_1,W_2) - sum_{a2 in A_2} s(a2,W_1,W_2).

    The raw (unstandardised) test statistic: how differently the two
    protected sets associate with the two neutral sets. Each s is
    km081's, delegated. Swapping A_1 and A_2 negates it, which the
    tests assert.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.4, printed
    p. 234.

    Examples
    --------
    >>> out = kamath_ch6_weat_function([[1.0, 0.0]], [[0.0, 1.0]],
    ...                                [[1.0, 0.0]], [[0.0, 1.0]])
    >>> out["estimate"], out["s_A1"], out["s_A2"]
    (2.0, [1.0], [-1.0])
    """
    s1, s2 = _sums(A_1, A_2, W_1, W_2)
    return RichResult(payload={
        "estimate": float(np.sum(s1) - np.sum(s2)),
        "s_A1": [float(v) for v in s1], "s_A2": [float(v) for v in s2],
        "n": len(s1) + len(s2),
        "method": "WEAT test statistic (Kamath Eq 6.4)"})


def cheatsheet():
    return "km080: sum s over A_1 minus sum s over A_2"
