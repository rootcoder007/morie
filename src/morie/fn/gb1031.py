# morie.fn -- function file (rootcoder007/morie)
"""Sen's k-sample extension of the control median test."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['kctrlmed', 'gibbons_k_ctrl_median']


def kctrlmed(samples, p=(0.5,)):
    """Block counts V_ij against the control-sample quantiles.

    Section 10.3 (book pp. 350-351), eq. (10.3.1) (Sen, 1962).  Sample
    1 is the control.  For fractions 0 < p_1 < ... < p_q < 1 take the
    control order statistics with r_i = [n_1 p_i] + 1; these cut the
    line into q + 1 contiguous blocks I_0, ..., I_q.  V_ij counts the
    observations of sample i falling in block I_j, and conditional on
    the control quantiles the rows are independent multinomials, which
    is eq. (10.3.1).

    Under H0 the expected cell probability is the Beta mean of the
    corresponding control order statistic,
    P(I_j) = (r_{j+1} - r_j)/(n_1 + 1) with r_0 = 0 and
    r_{q+1} = n_1 + 1, so a Pearson statistic on the (k-1) x (q+1)
    table of treatment counts follows, with (k-2)(q) + q degrees of
    freedom in the usual goodness-of-fit sense -- reported here as
    (k - 2) * q + q = (k - 1) q.

    Parameters
    ----------
    samples : sequence of sequence of float
        The k samples; ``samples[0]`` is the control.
    p : sequence of float, optional
        The quantile fractions (default the median alone).

    Returns
    -------
    RichResult
        keys ``counts`` (rows = treatments, cols = blocks),
        ``cuts`` (the control quantiles), ``r`` (their indices),
        ``pcell``, ``statistic``, ``df``, ``p_value``, ``k``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 10.3, eq. (10.3.1), pp. 350-351
    (Sen, 1962).
    """
    ss = [[float(v) for v in s] for s in samples]
    k = len(ss)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    ps = sorted(float(v) for v in p)
    if not ps or any(not 0.0 < v < 1.0 for v in ps):
        raise ValueError("p must lie strictly inside (0, 1).")
    ctrl = sorted(ss[0])
    n1 = len(ctrl)
    r = [int(math.floor(n1 * v)) + 1 for v in ps]
    if any(v > n1 for v in r):
        raise ValueError("a quantile index exceeds the control sample size.")
    cuts = [ctrl[v - 1] for v in r]
    q = len(cuts)
    counts = []
    for s in ss[1:]:
        row = [0] * (q + 1)
        for v in s:
            j = 0
            while j < q and v > cuts[j]:
                j += 1
            row[j] += 1
        counts.append(row)
    edges = [0] + r + [n1 + 1]
    pcell = [
        (edges[j + 1] - edges[j]) / (n1 + 1.0) for j in range(q + 1)
    ]
    stat = 0.0
    for i, row in enumerate(counts):
        ni = sum(row)
        for j in range(q + 1):
            e = ni * pcell[j]
            if e > 0.0:
                stat += (row[j] - e) ** 2 / e
    df = (k - 1) * q
    return RichResult(
        payload={
            "counts": counts,
            "cuts": cuts,
            "r": r,
            "pcell": pcell,
            "statistic": float(stat),
            "df": int(df),
            "p_value": float(stats.chi2.sf(stat, df)) if df > 0 else 1.0,
            "k": int(k),
            "method": "k-sample control median test, eq. (10.3.1)",
        }
    )


gibbons_k_ctrl_median = kctrlmed
