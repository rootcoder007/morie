# morie.fn -- function file (rootcoder007/morie)
"""Multiple comparisons following Kruskal-Wallis -- eq. (10.4.8)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['kwmc', 'gibbons_kw_mult_comp']


def kwmc(rank_means, ns, alpha=0.20):
    """Pairwise rank-mean comparisons at an experimentwise level.

    Book p. 357, eq. (10.4.8): treatments i and j differ significantly
    when

    .. math:: |\\bar R_i - \\bar R_j| \\ge z^*
        \\sqrt{\\frac{N(N+1)}{12}
                \\left(\\frac{1}{n_i}+\\frac{1}{n_j}\\right)},

    with z* the upper alpha/[k(k-1)] normal quantile, i.e. a Bonferroni
    split over the k(k-1)/2 two-sided comparisons.  For equal
    n_i = N/k the bound reduces to z* sqrt(k(N+1)/6), which the book
    states on the same page.

    The book's Example 10.4.1 (k = 4, n_i = 10, N = 40, alpha = 0.20)
    gives a bound of 11.125.

    Parameters
    ----------
    rank_means : sequence of float
        The k average rank sums.
    ns : sequence of int
        The k sample sizes.
    alpha : float, optional
        Experimentwise level (default 0.20, the book's example).

    Returns
    -------
    RichResult
        keys ``bound`` (equal-n bound, else nan), ``bounds`` (pairwise
        matrix), ``diffs``, ``significant`` (pairs as [i, j]),
        ``zstar``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (10.4.8), p. 357.
    """
    rm = [float(v) for v in rank_means]
    nv = [int(v) for v in ns]
    k = len(rm)
    alpha = float(alpha)
    if k < 2 or len(nv) != k:
        raise ValueError("need at least 2 samples and matching sizes.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    nn = sum(nv)
    zstar = stats.norm.ppf(1.0 - alpha / (k * (k - 1.0)))
    bounds = []
    diffs = []
    sig = []
    for i in range(k):
        brow = []
        drow = []
        for j in range(k):
            b = zstar * math.sqrt(
                nn * (nn + 1.0) / 12.0 * (1.0 / nv[i] + 1.0 / nv[j])
            )
            d = abs(rm[i] - rm[j])
            brow.append(float(b))
            drow.append(float(d))
            if i < j and d >= b:
                sig.append([i, j])
        bounds.append(brow)
        diffs.append(drow)
    eq = float("nan")
    if len(set(nv)) == 1:
        eq = zstar * math.sqrt(k * (nn + 1.0) / 6.0)
    return RichResult(
        payload={
            "bound": eq,
            "bounds": bounds,
            "diffs": diffs,
            "significant": sig,
            "zstar": float(zstar),
            "k": int(k),
            "n": int(nn),
            "method": "Kruskal-Wallis multiple comparisons, eq. (10.4.8)",
        }
    )


gibbons_kw_mult_comp = kwmc
