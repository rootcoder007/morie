# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.7: KronA's efficient (A (x) B) x without forming W."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_krona_efficient"]


def kamath_ch4_krona_efficient(A, B, x):
    """(A (x) B) x = gamma(B eta_{b2 x a2}(x) A^T).

    ``eta_{m x n}`` folds the vector x into an m by n matrix and
    ``gamma`` unfolds a matrix by STACKING ITS COLUMNS -- so both are
    column-major, which is what makes the identity hold. The big
    (a1 b1, a2 b2) matrix is never formed: cost drops from
    O(a1 a2 b1 b2) to two small products.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.7, printed
    p. 153.

    Examples
    --------
    >>> out = kamath_ch4_krona_efficient([[1.0, 2.0], [0.0, 1.0]],
    ...     [[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0, 3.0, 4.0])
    >>> out["y"]
    [7.0, 10.0, 3.0, 4.0]
    """
    Am = np.atleast_2d(np.asarray(A, dtype=float))
    Bm = np.atleast_2d(np.asarray(B, dtype=float))
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    if xv.ndim != 1:
        raise ValueError("x must be a vector.")
    a1, a2 = Am.shape
    b1, b2 = Bm.shape
    if xv.shape[0] != a2 * b2:
        raise ValueError(
            f"x has {xv.shape[0]} entries but A (x) B has {a2 * b2} "
            "columns.")
    X = np.reshape(xv, (b2, a2), order="F")
    Y = Bm @ X @ Am.T
    y = np.reshape(Y, (-1,), order="F")
    return RichResult(payload={
        "y": [float(v) for v in y],
        "folded_shape": (int(b2), int(a2)),
        "estimate": float(y[0]), "n": int(y.shape[0]),
        "products_avoided": int(a1 * a2 * b1 * b2),
        "method": "KronA matrix-free product (Kamath Eq 4.7)"})


def cheatsheet():
    return "km060: (A(x)B)x = gamma(B eta(x) A^T), column-major folds"
