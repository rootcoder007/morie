# morie.fn -- function file (rootcoder007/morie)
"""Multiple comparisons after the Friedman test -- eq. (12.2.13)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['friedmc', 'gibbons_friedman_mult']


def friedmc(rank_sums, k, alpha=0.20):
    """Pairwise treatment comparisons from the Friedman rank sums.

    Book p. 445, eq. (12.2.13): treatments i and j differ significantly
    when

    .. math:: |R_i - R_j| \\ge z^*\\sqrt{\\frac{kn(n+1)}{6}},

    with z* the upper alpha/[n(n-1)] normal quantile -- the same
    Bonferroni split over the n(n-1)/2 pairs used in Sec. 10.4 for the
    one-way case, and the book notes alpha is generally taken larger
    than in ordinary testing.

    Parameters
    ----------
    rank_sums : sequence of float
        The n treatment rank sums R_j.
    k : int
        Number of blocks.
    alpha : float, optional
        Experimentwise level (default 0.20).

    Returns
    -------
    RichResult
        keys ``bound``, ``zstar``, ``diffs`` (matrix), ``significant``
        (pairs as [i, j]), ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (12.2.13), p. 445.
    """
    rs = [float(v) for v in rank_sums]
    n = len(rs)
    k = int(k)
    alpha = float(alpha)
    if n < 2:
        raise ValueError("need at least 2 treatments.")
    if k < 2:
        raise ValueError("need at least 2 blocks.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1).")
    zstar = stats.norm.ppf(1.0 - alpha / (n * (n - 1.0)))
    bound = zstar * math.sqrt(k * n * (n + 1.0) / 6.0)
    diffs = []
    sig = []
    for i in range(n):
        row = []
        for j in range(n):
            d = abs(rs[i] - rs[j])
            row.append(float(d))
            if i < j and d >= bound:
                sig.append([i, j])
        diffs.append(row)
    return RichResult(
        payload={
            "bound": float(bound),
            "zstar": float(zstar),
            "diffs": diffs,
            "significant": sig,
            "k": k,
            "n": int(n),
            "method": "Friedman multiple comparisons, eq. (12.2.13)",
        }
    )


gibbons_friedman_mult = friedmc
