# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.12: MoverScore n-gram embedding."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_ngram_embedding"]


def kamath_ch8_ngram_embedding(x, i, n):
    r"""E(x_i^n) = sum_{k=i}^{i+n-1} idf(x_k).

    ``x`` is the sequence of per-token idf values; ``i`` the 0-based
    start of the n-gram window and ``n`` its length. Eq 8.12 is
    printed exactly like this in the book -- a sum of idf WEIGHTS, not
    of idf-weighted vectors -- and it is implemented as printed; if
    ``x`` is 2-D (one idf-weighted vector per token) the same window
    sum is taken row-wise, which is the vector form implementations
    use.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.12, printed
    p. 326.

    Examples
    --------
    >>> out = kamath_ch8_ngram_embedding([1.0, 2.0, 3.0], 1, 2)
    >>> out["estimate"]        # 2 + 3
    5.0
    """
    A = np.asarray(x, dtype=float)
    if A.ndim > 2:
        raise ValueError("x must be a 1-D idf sequence or a 2-D matrix "
                         "of per-token vectors.")
    i = int(i)
    n = int(n)
    if n < 1:
        raise ValueError(f"the n-gram length must be >= 1; got {n}.")
    if i < 0 or i + n > A.shape[0]:
        raise ValueError(
            f"the window [{i}, {i + n}) runs off a sequence of length "
            f"{A.shape[0]}.")
    win = A[i:i + n]
    emb = win.sum(axis=0)
    est = float(emb) if np.ndim(emb) == 0 else [float(v) for v in emb]
    return RichResult(payload={
        "estimate": est,
        "embedding": est if isinstance(est, list) else [est],
        "window": [float(v) for v in np.ravel(win)],
        "i": i, "n": n,
        "method": "MoverScore n-gram embedding (Kamath Eq 8.12)"})


def cheatsheet():
    return "km124: sum of idf over the n tokens of one n-gram window"
