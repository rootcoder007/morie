# morie.fn -- function file (rootcoder007/morie)
"""k-sample extension of the median test."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['kmedtest', 'gibbons_k_median_test']


def kmedtest(samples):
    """Q for the 2 x k table of counts below the combined median.

    Section 10.2 (book pp. 344-346).  Let d be the combined-sample
    median, u_i the number of observations in sample i below d, and
    t = sum u_i.  Given t, the null law of (U_1, ..., U_k) is the
    multivariate hypergeometric

    .. math:: P = \\frac{\\prod_i \\binom{n_i}{u_i}}{\\binom{N}{t}},

    and the goodness-of-fit criterion of eq. (4.2.1) applied to the
    2k cells collapses to

    .. math:: Q = \\frac{N^2}{t(N-t)}
        \\sum_{i=1}^{k}\\frac{(u_i - n_i t/N)^2}{n_i},

    asymptotically chi-square with k - 1 degrees of freedom.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples, k >= 2, each non-empty.

    Returns
    -------
    RichResult
        keys ``statistic`` (Q), ``df``, ``p_value``, ``u`` (counts
        below d), ``t``, ``median``, ``prob`` (exact multivariate
        hypergeometric probability of the observed table), ``k``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.2, pp. 344-346, with
    eq. (4.2.1), p. 104.
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    if any(len(s) < 1 for s in ss):
        raise ValueError("every sample must be non-empty.")
    pooled = sorted(v for s in ss for v in s)
    nn = len(pooled)
    d = (
        pooled[nn // 2]
        if nn % 2
        else (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    )
    u = [sum(1 for v in s if v < d) for s in ss]
    t = sum(u)
    ns = [len(s) for s in ss]
    if t == 0 or t == nn:
        raise ValueError("no split at the combined median.")
    q = (float(nn) ** 2 / (t * (nn - t))) * sum(
        (u[i] - ns[i] * t / float(nn)) ** 2 / ns[i] for i in range(k)
    )
    prob = 1.0
    for i in range(k):
        prob *= math.comb(ns[i], u[i])
    prob /= math.comb(nn, t)
    return RichResult(
        payload={
            "statistic": float(q),
            "df": int(k - 1),
            "p_value": float(stats.chi2.sf(q, k - 1)),
            "u": u,
            "t": int(t),
            "median": float(d),
            "prob": float(prob),
            "k": int(k),
            "n": int(nn),
            "method": "k-sample median test (Sec. 10.2)",
        }
    )


gibbons_k_median_test = kmedtest
