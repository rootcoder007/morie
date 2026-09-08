# morie.fn -- function file (rootcoder007/morie)
"""Two-sample median test based on the count above the combined median."""

import math

from ._richresult import RichResult

__all__ = ['medtest', 'gibbons_median_test']


def medtest(x, y):
    """Median test statistic U and its hypergeometric null law.

    Section 6.4 (book p. 247).  Pool the samples, find the combined
    median, and let U be the number of X observations that exceed it.
    Under H0 the t = number of pooled values above the median are a
    random subset, so

    .. math:: P(U = u) = \\frac{\\binom{m}{u}\\binom{n}{t-u}}
        {\\binom{m+n}{t}},

    a hypergeometric law free of F.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, sizes m and n.

    Returns
    -------
    RichResult
        keys ``statistic`` (U), ``p_value`` (two-sided, doubled tail),
        ``p_lower``, ``p_upper``, ``median``, ``t`` (pooled count above
        the median), ``mean``, ``var``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.4, p. 247.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    pooled = sorted(xs + ys)
    nn = m + n
    med = (
        pooled[nn // 2]
        if nn % 2
        else (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    )
    t = sum(1 for v in pooled if v > med)
    u = sum(1 for v in xs if v > med)

    def _p(k):
        if k < 0 or k > m or t - k < 0 or t - k > n:
            return 0.0
        return math.comb(m, k) * math.comb(n, t - k) / math.comb(nn, t)

    lower = sum(_p(k) for k in range(0, u + 1))
    upper = sum(_p(k) for k in range(u, m + 1))
    mean = m * t / float(nn)
    var = (
        m * n * t * (nn - t) / (float(nn) ** 2 * (nn - 1.0))
        if nn > 1
        else float("nan")
    )
    return RichResult(
        payload={
            "statistic": int(u),
            "p_value": float(min(1.0, 2.0 * min(lower, upper))),
            "p_lower": float(lower),
            "p_upper": float(upper),
            "median": float(med),
            "t": int(t),
            "mean": float(mean),
            "var": float(var),
            "m": m,
            "n": n,
            "method": "two-sample median test, hypergeometric null",
        }
    )


gibbons_median_test = medtest
