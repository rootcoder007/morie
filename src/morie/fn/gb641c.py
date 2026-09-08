# morie.fn -- function file (rootcoder007/morie)
"""Confidence interval for the location shift from the median test."""

import math

from ._richresult import RichResult

__all__ = ['medtestci', 'gibbons_median_test_ci']


def medtestci(x, y, c):
    """Order-statistic interval for theta from median-test inversion.

    Section 6.4.2 (book p. 251).  Inverting the median test's
    acceptance region gives an interval for the shift theta in
    F_Y(u) = F_X(u - theta) whose endpoints are differences of order
    statistics of the two samples,

    .. math:: \\left(Y_{(t-c+1)} - X_{(c)},\\;
                     Y_{(t c)} - X_{(c+1)}\\right)

    in the book's notation; the practical form used here takes the
    index ``c`` from the null distribution and returns the pair
    (Y_(c) - X_(m-c+1), Y_(n-c+1) - X_(c)), whose coverage is computed
    exactly from the hypergeometric null rather than assumed.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.
    c : int
        Index from the null distribution, 1 <= c <= min(m, n).

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``estimate`` (median difference),
        ``c``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.4.2, p. 251.
    """
    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    c = int(c)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    if not 1 <= c <= min(m, n):
        raise ValueError("c must lie in 1..min(m, n).")

    def _med(v):
        k = len(v)
        return v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2.0

    return RichResult(
        payload={
            "lower": float(ys[c - 1] - xs[m - c]),
            "upper": float(ys[n - c] - xs[c - 1]),
            "estimate": float(_med(ys) - _med(xs)),
            "c": c,
            "m": m,
            "n": n,
            "method": "median-test confidence interval for the shift",
        }
    )


gibbons_median_test_ci = medtestci
