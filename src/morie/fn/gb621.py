# morie.fn -- function file (rootcoder007/morie)
"""Wald-Wolfowitz two-sample runs test on the pooled ordered sequence."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wwruns', 'gibbons_ks2samp']


def wwruns(x, y):
    """Total runs in the pooled ordering of two samples.

    Section 6.2 (book p. 231).  Pool the m X's and n Y's, sort, and
    read off the sequence of sample labels; the number of runs R is
    small when the two samples separate, so the rejection region for
    H1: F_X != F_Y is the left tail of the null distribution of
    Theorem 3.2.2.  Both the exact left-tail probability and the
    normal approximation with the exact moments

    .. math:: E[R] = \\frac{2mn}{m+n} + 1, \\qquad
        Var[R] = \\frac{2mn(2mn - m - n)}{(m+n)^2(m+n-1)}

    are returned.

    Parameters
    ----------
    x, y : sequence of float
        The two samples, sizes m and n, both >= 1.

    Returns
    -------
    RichResult
        keys ``statistic`` (R), ``p_value`` (exact left tail), ``z``,
        ``p_normal``, ``mean``, ``var``, ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.2, p. 231; null distribution
    Theorem 3.2.2, p. 79.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    tagged = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    tagged.sort(key=lambda p: (p[0], p[1]))
    labels = [t for _, t in tagged]
    r = 1
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            r += 1
    nn = m + n
    den = math.comb(nn, m)
    tail = 0.0
    for rr in range(2, r + 1):
        if rr % 2 == 0:
            k = rr // 2
            p = 2.0 * math.comb(m - 1, k - 1) * math.comb(n - 1, k - 1)
        else:
            k = (rr - 1) // 2
            p = (
                math.comb(m - 1, k - 1) * math.comb(n - 1, k)
                + math.comb(m - 1, k) * math.comb(n - 1, k - 1)
            )
        tail += p / den
    mean = 2.0 * m * n / nn + 1.0
    var = 2.0 * m * n * (2.0 * m * n - nn) / (float(nn) ** 2 * (nn - 1.0))
    z = (r - mean) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": int(r),
            "p_value": float(min(1.0, tail)),
            "z": float(z),
            "p_normal": float(stats.norm.cdf(z)),
            "mean": float(mean),
            "var": float(var),
            "m": m,
            "n": n,
            "method": "Wald-Wolfowitz two-sample runs test (Sec. 6.2)",
        }
    )


gibbons_ks2samp = wwruns
