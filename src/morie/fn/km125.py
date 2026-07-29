# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.13: MoverScore n-gram weight."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_ngram_weight"]


def kamath_ch8_ngram_weight(x, Z=None):
    r"""f_{x_i^n} = (1/Z) sum_{k=i}^{i+n-1} idf(x_k).

    ``x`` is either the idf values of ONE n-gram window (1-D) or one
    row per n-gram (2-D). ``Z`` is the normalization constant; left
    ``None`` it is set to the total idf mass over the rows supplied,
    which is what makes the weights sum to 1 as the book requires.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.13, printed
    p. 326.

    Examples
    --------
    >>> out = kamath_ch8_ngram_weight([[1.0, 2.0], [3.0, 4.0]])
    >>> [round(v, 12) for v in out["weights"]]      # 3/10 and 7/10
    [0.3, 0.7]
    >>> round(sum(out["weights"]), 12)
    1.0
    """
    A = np.atleast_2d(np.asarray(x, dtype=float))
    if A.size == 0:
        raise ValueError("no idf values given.")
    if np.any(A < 0):
        raise ValueError("idf values cannot be negative.")
    sums = A.sum(axis=1)
    if Z is None:
        Z = float(sums.sum())
        if Z <= 0:
            raise ValueError("every idf value is 0, so Z = 0 and the "
                             "weights are undefined.")
    else:
        Z = float(Z)
        if Z <= 0:
            raise ValueError(f"Z must be positive; got {Z}.")
    w = sums / Z
    est = float(w[0]) if w.size == 1 else [float(v) for v in w]
    return RichResult(payload={
        "estimate": est, "weights": [float(v) for v in w],
        "Z": Z, "n": int(w.size),
        "method": "MoverScore n-gram weight (Kamath Eq 8.13)"})


def cheatsheet():
    return "km125: window idf sum divided by the normalizer Z"
