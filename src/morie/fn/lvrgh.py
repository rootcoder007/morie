# morie.fn -- k02 batch (rootcoder007/morie)
"""Leverage: the diagonal of the hat matrix.

Source consulted: Belsley, D.A., Kuh, E. and Welsch, R.E. (1980), *Regression
Diagnostics: Identifying Influential Data and Sources of Collinearity*, Wiley,
chapter 2.  With H = X (X'X)^-1 X' the fitted values are H y, so

    h_ii = x_i' (X'X)^-1 x_i

is the weight point i puts on its own fit.  ``sum(h_ii) = p`` and
``1/n <= h_ii <= 1`` with an intercept; Belsley, Kuh and Welsch's rule of
thumb flags ``h_ii > 2p/n``.  Computed from the thin QR (h_ii is the squared
row norm of Q), which is stable for collinear designs.  Matches
``stats::hatvalues``.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hat_matrix_diagonal"]


def hat_matrix_diagonal(X, intercept=True):
    """Hat-matrix diagonal (leverages).

    Parameters
    ----------
    X : array-like
        Design matrix, n by q.
    intercept : bool, default True
        Prepend a column of ones.

    Returns
    -------
    RichResult
        estimate (largest leverage), leverage, rank, threshold, high,
        trace, n, method.
    """
    m = np.atleast_2d(np.asarray(X, dtype=float))
    if m.shape[0] == 1 and m.shape[1] > 1:
        m = m.T
    n = m.shape[0]
    d = np.column_stack([np.ones(n), m]) if intercept else m
    q, _r = np.linalg.qr(d)
    h = np.sum(q * q, axis=1)
    p = d.shape[1]
    thr = 2.0 * p / n
    hl = h.tolist()
    return RichResult(
        payload={
            "estimate": float(np.max(h)),
            "leverage": hl,
            "rank": int(p),
            "threshold": float(thr),
            "high": [bool(t > thr) for t in hl],
            "trace": float(np.sum(h)),
            "n": int(n),
            "method": "Hat-matrix diagonal / leverage (Belsley, Kuh & Welsch 1980, ch. 2)",
        }
    )


# CANONICAL TEST
# >>> X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7], [9, 10], [10, 9]]
# >>> r = hat_matrix_diagonal(X)
# >>> assert abs(r["leverage"][0] - 0.4) < 1e-12       # stats::hatvalues
# >>> assert abs(r["trace"] - 3.0) < 1e-12             # trace(H) = rank


def cheatsheet():
    return "lvrgh(X): hat-matrix diagonal (leverages)."


hatmatrixdiagonal = hat_matrix_diagonal
