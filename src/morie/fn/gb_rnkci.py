# morie.fn -- function file (rootcoder007/morie)
"""General rank-based confidence interval by test inversion."""

import math

from ._richresult import RichResult

__all__ = ['rankci', 'gibbons_rank_ci']


def rankci(values, k, level=None):
    """Interval (V_(k+1), V_(M-k)) from an inverted rank test.

    Sections 5.7.5, 6.4.2 and 6.6.2 all use the same construction: the
    acceptance region of a rank test is an interval of the parameter,
    and its endpoints are order statistics of a derived set of
    quantities -- Walsh averages for the one-sample signed-rank test,
    pairwise differences for the two-sample rank tests, single
    observations for the sign test.  Given that set and the critical
    index k, the interval is the (k+1)-th value in from each end.

    Parameters
    ----------
    values : sequence of float
        The derived quantities, length M >= 2.
    k : int
        Critical index, 0 <= k < M/2.
    level : float, optional
        Confidence coefficient to record alongside, if the caller has
        computed it from the relevant null distribution.

    Returns
    -------
    RichResult
        keys ``lower``, ``upper``, ``estimate`` (median of the
        values), ``k``, ``m`` (M), ``level``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Secs. 5.7.5 (p. 207), 6.4.2 (p. 251)
    and 6.6.2 (p. 267).
    """
    v = sorted(float(t) for t in values)
    mm = len(v)
    k = int(k)
    if mm < 2:
        raise ValueError("need at least 2 values.")
    if not 0 <= k < mm:
        raise ValueError("k must lie in 0..M-1.")
    mid = mm // 2
    est = v[mid] if mm % 2 else (v[mid - 1] + v[mid]) / 2.0
    return RichResult(
        payload={
            "lower": float(v[k]),
            "upper": float(v[mm - 1 - k]),
            "estimate": float(est),
            "k": k,
            "m": int(mm),
            "level": float("nan") if level is None else float(level),
            "method": "rank-test inversion interval, (k+1)-th from each end",
        }
    )


gibbons_rank_ci = rankci
