# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the location shift from Mann-Whitney."""

import math

from ._richresult import RichResult

__all__ = ['mwuci', 'gibbons_mw_ci']


def mwuci(x, y, k):
    """Interval for theta built from the mn differences X_i - Y_j.

    Section 6.6.2 (book p. 267).  Inverting the Mann-Whitney test gives
    an interval whose endpoints are order statistics of the mn
    differences X_i - Y_j: with k the lower critical value of U,

    .. math:: (D_{(k+1)},\\; D_{(mn-k)}),

    where D_(1) <= ... <= D_(mn) are the ordered X_i - Y_j, matching
    the orientation of U in eq. (6.6.1).  The
    Hodges-Lehmann point estimate, the median of the same differences,
    is returned as ``estimate``.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    k : int
        Lower critical value of U, 0 <= k < mn/2.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``estimate``, ``k``, ``ndiff``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.6.2, p. 267.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    k = int(k)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    d = sorted(xi - yj for xi in xs for yj in ys)
    nd = len(d)
    if not 0 <= k < nd:
        raise ValueError("k must lie in 0..mn-1.")
    mid = nd // 2
    est = d[mid] if nd % 2 else (d[mid - 1] + d[mid]) / 2.0
    return RichResult(
        payload={
            "lower": float(d[k]),
            "upper": float(d[nd - 1 - k]),
            "estimate": float(est),
            "k": k,
            "ndiff": int(nd),
            "m": m,
            "n": n,
            "method": "Mann-Whitney shift CI from ordered differences",
        }
    )


gibbons_mw_ci = mwuci
