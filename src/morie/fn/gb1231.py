# morie.fn -- function file (rootcoder007/morie)
"""Page's test for ordered alternatives in a two-way layout."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['pagel', 'gibbons_page_test']


def pagel(data, weights=None):
    """L = sum of weighted treatment rank sums, eq. (12.3.1).

    Section 12.3 (book p. 448).  For the ordered alternative
    theta_1 <= ... <= theta_n,

    .. math:: L = \\sum_{j=1}^{n} Y_j R_j,

    with Y_j the hypothesised ranking of treatment j (1, 2, ..., n by
    default) and R_j its rank sum over the k blocks; H0 is rejected for
    large L.  The large-sample form with continuity correction is
    eq. (12.3.2),

    .. math:: Z = \\frac{12(L - 0.5) - 3kn(n+1)^2}
        {n(n+1)\\sqrt{k(n-1)}},

    and the average rank correlation implied by L is
    r_av = 12L/[k(n^3-n)] - 3(n+1)/(n-1).

    Parameters
    ----------
    data : sequence of sequence of float
        k blocks of n treatment observations, columns in the
        hypothesised order.
    weights : sequence of float, optional
        The Y_j (defaults to 1, 2, ..., n).

    Returns
    -------
    RichResult
        keys ``statistic`` (L), ``z``, ``p_value``, ``rav``,
        ``rank_sums``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.3, eqs. (12.3.1)-(12.3.2),
    pp. 448-449 (Page, 1963); Table Q, p. 591.
    """
    rows = [[float(v) for v in r] for r in data]
    k = len(rows)
    if k < 2:
        raise ValueError("need at least 2 blocks.")
    n = len(rows[0])
    if n < 2:
        raise ValueError("need at least 2 treatments.")
    w = (
        [float(i + 1) for i in range(n)]
        if weights is None
        else [float(v) for v in weights]
    )
    if len(w) != n:
        raise ValueError("weights must have length n.")
    rsum = [0.0] * n
    for r in rows:
        if len(r) != n:
            raise ValueError("every block must have n observations.")
        order = sorted(range(n), key=lambda i: r[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and r[order[j + 1]] == r[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = mid
            i = j + 1
        for j in range(n):
            rsum[j] += rk[j]
    ell = sum(w[j] * rsum[j] for j in range(n))
    z = (12.0 * (ell - 0.5) - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0))
    )
    rav = 12.0 * ell / (k * (float(n) ** 3 - n)) - 3.0 * (n + 1.0) / (n - 1.0)
    return RichResult(
        payload={
            "statistic": float(ell),
            "z": float(z),
            "p_value": float(1.0 - stats.norm.cdf(z)),
            "rav": float(rav),
            "rank_sums": rsum,
            "k": int(k),
            "n": int(n),
            "method": "Page's L test, eqs. (12.3.1)-(12.3.2)",
        }
    )


gibbons_page_test = pagel
