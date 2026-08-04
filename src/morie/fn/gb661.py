# morie.fn -- function file (rootcoder007/morie)
"""Mann-Whitney U counting the preferences between two samples."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['mwu', 'gibbons_mannwhitney']


def mwu(x, y):
    """U = #{(i, j) : Y_j < X_i}, with exact and normal-approximation tails.

    Section 6.6 (book p. 261), eq. (6.6.1): D_ij = 1 when Y_j < X_i,
    so U counts the times a Y precedes an X -- the book's orientation,
    which is the complement mn - U of the opposite convention.  Under H0 the null moments
    are E[U] = mn/2 and Var[U] = mn(m+n+1)/12, and the exact null
    distribution follows the recursion (6.6.14),

    .. math:: r_{m,n}(u) = r_{m-1,n}(u-n) + r_{m,n-1}(u),

    which is evaluated here by dynamic programming (a
    generating-function convolution), so the exact p-value is available
    for any m, n that fit in memory.  Ties contribute 1/2 each.

    Parameters
    ----------
    x, y : sequence of float
        The two samples.

    Returns
    -------
    RichResult
        keys ``statistic`` (U), ``p_value`` (exact, two-sided),
        ``z``, ``p_normal``, ``mean``, ``var``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 6.6, eqs. (6.6.1), (6.6.14),
    pp. 260-266.
    """
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    m = len(xs)
    n = len(ys)
    if m < 1 or n < 1:
        raise ValueError("both samples must be non-empty.")
    u = 0.0
    for xi in xs:
        for yj in ys:
            if yj < xi:
                u += 1.0
            elif yj == xi:
                u += 0.5
    # exact null counts of U over 0..mn by DP on the rank-sum recursion
    total = m * n
    counts = [0.0] * (total + 1)
    counts[0] = 1.0
    for i in range(1, m + 1):
        new = [0.0] * (total + 1)
        run = 0.0
        for k in range(total + 1):
            run += counts[k]
            if k - n - 1 >= 0:
                run -= counts[k - n - 1]
            new[k] = run
        counts = new
    denom = math.comb(m + n, m)
    pmf = [c / denom for c in counts]
    ui = int(round(u))
    lower = sum(pmf[: min(ui, total) + 1])
    upper = sum(pmf[min(ui, total):])
    mean = m * n / 2.0
    var = m * n * (m + n + 1.0) / 12.0
    z = (u - mean) / math.sqrt(var)
    return RichResult(
        payload={
            "statistic": float(u),
            "p_value": float(min(1.0, 2.0 * min(lower, upper))),
            "z": float(z),
            "p_normal": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": float(mean),
            "var": float(var),
            "m": m,
            "n": n,
            "method": "Mann-Whitney U, exact recursion (6.6.14)",
        }
    )


gibbons_mannwhitney = mwu
