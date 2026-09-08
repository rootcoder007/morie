# morie.fn -- function file (rootcoder007/morie)
"""Wilcoxon rank-sum statistic W_N with its exact null distribution."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wrs', 'gibbons_wilcoxon_ranksum']


def wrs(x, y):
    """W_N = sum of the X ranks in the combined ordering.

    Section 8.2 (book p. 290).  With m X's and n Y's and N = m + n,

    .. math:: E[W_N] = \\frac{m(N+1)}{2}, \\qquad
              Var[W_N] = \\frac{mn(N+1)}{12},

    and the exact null distribution follows the recursion of book
    p. 291,

    .. math:: r_{m,n}(k) = r_{m-1,n}(k-N) + r_{m,n-1}(k),

    evaluated here by dynamic programming over all C(N, m)
    arrangements.  Ties get midranks.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic`` (W_N), ``p_value`` (exact two-sided),
        ``z``, ``p_normal``, ``mean``, ``var``, ``wmin``, ``wmax``,
        ``m``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 8.2, pp. 290-291.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    nn = m + n
    pooled = sorted(xs + ys)
    ranks = {}
    i = 0
    while i < nn:
        j = i
        while j + 1 < nn and pooled[j + 1] == pooled[i]:
            j += 1
        ranks[pooled[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w = sum(ranks[v] for v in xs)
    wmin = m * (m + 1) / 2.0
    wmax = m * (2.0 * nn - m + 1.0) / 2.0
    # exact counts over the shifted support 0..mn
    total = m * n
    counts = [0.0] * (total + 1)
    counts[0] = 1.0
    for _ in range(1, m + 1):
        new = [0.0] * (total + 1)
        run = 0.0
        for k in range(total + 1):
            run += counts[k]
            if k - n - 1 >= 0:
                run -= counts[k - n - 1]
            new[k] = run
        counts = new
    denom = math.comb(nn, m)
    pmf = [c / denom for c in counts]
    shifted = int(round(w - wmin))
    shifted = max(0, min(total, shifted))
    lower = sum(pmf[: shifted + 1])
    upper = sum(pmf[shifted:])
    mean = m * (nn + 1.0) / 2.0
    var = m * n * (nn + 1.0) / 12.0
    z = (w - mean) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": float(w),
            "p_value": float(min(1.0, 2.0 * min(lower, upper))),
            "z": float(z),
            "p_normal": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": float(mean),
            "var": float(var),
            "wmin": float(wmin),
            "wmax": float(wmax),
            "m": m,
            "n": n,
            "method": "Wilcoxon rank-sum W_N, exact recursion (Sec. 8.2)",
        }
    )


gibbons_wilcoxon_ranksum = wrs
