# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the shift from the rank-sum test."""

import math

from ._richresult import RichResult

__all__ = ['wrsci', 'gibbons_wrs_ci']


def wrsci(x, y, wcrit):
    """Shift interval from the rank-sum critical value.

    Section 8.2 (book p. 292).  The rank-sum and Mann-Whitney
    statistics differ only by the constant m(m+1)/2, so the acceptance
    region of W_N inverts to the same interval of ordered differences
    X_i - Y_j.  With w the lower critical value of W_N, the index into
    the ordered differences is k = w - m(m+1)/2, and the interval is
    (D_(k+1), D_(mn-k)).

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    wcrit : float
        Lower critical value of W_N from Table J.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``k``, ``estimate``, ``ndiff``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.2, p. 292.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    k = int(round(float(wcrit) - m * (m + 1) / 2.0))
    d = sorted(xi - yj for xi in xs for yj in ys)
    nd = len(d)
    if not 0 <= k < nd:
        raise ValueError("wcrit implies an index outside 0..mn-1.")
    mid = nd // 2
    est = d[mid] if nd % 2 else (d[mid - 1] + d[mid]) / 2.0
    return RichResult(
        payload={
            "lower": float(d[k]),
            "upper": float(d[nd - 1 - k]),
            "k": int(k),
            "estimate": float(est),
            "ndiff": int(nd),
            "m": m,
            "n": n,
            "method": "rank-sum shift CI from ordered differences",
        }
    )


gibbons_wrs_ci = wrsci
