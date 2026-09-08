# morie.fn -- function file (rootcoder007/morie)
"""Rank von Neumann test of randomness -- eqs. (3.5.1) and (3.5.2)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['rvntest', 'gibbons_rvn_test']


def rvntest(x, alternative="two-sided"):
    """Bartels rank von Neumann ratio test for randomness.

    Book p. 95, eqs. (3.5.1) and (3.5.2):

    .. math:: NM = \\sum_{i=1}^{n-1}[R_i - R_{i+1}]^2, \\qquad
        RVN = \\frac{NM}{\\sum_{i=1}^{n}[R_i - (n+1)/2]^2},

    with R_i the rank of X_i in the time-ordered sequence.  Small RVN
    indicates trend, large RVN indicates alternation; the reference
    normal has mean 2 and the variance of Sec. 3.5.

    Parameters
    ----------
    x : sequence of float
        Time-ordered observations, n >= 3.
    alternative : str, optional
        ``"two-sided"``, ``"less"`` (trend) or ``"greater"``
        (alternation).

    Returns
    -------
    RichResult
        keys ``statistic`` (RVN), ``nm``, ``denom``, ``z``,
        ``p_value``, ``mean``, ``var``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (3.5.1)-(3.5.2), p. 95.
    """
    xs = [float(v) for v in x]
    n = len(xs)
    if n < 3:
        raise ValueError("need at least 3 observations.")
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    nm = sum((ranks[i] - ranks[i + 1]) ** 2 for i in range(n - 1))
    den = sum((ranks[i] - (n + 1.0) / 2.0) ** 2 for i in range(n))
    rvn = nm / den
    var = (
        4.0 * (n - 2.0) * (5.0 * n * n - 2.0 * n - 9.0)
        / (5.0 * n * (n + 1.0) * (n - 1.0) ** 2)
    )
    z = (rvn - 2.0) / math.sqrt(var)
    if alternative == "less":
        pv = stats.norm.cdf(z)
    elif alternative == "greater":
        pv = 1.0 - stats.norm.cdf(z)
    elif alternative == "two-sided":
        pv = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    else:
        raise ValueError("alternative must be two-sided, less or greater.")
    return RichResult(
        payload={
            "statistic": float(rvn),
            "nm": float(nm),
            "denom": float(den),
            "z": float(z),
            "p_value": float(min(1.0, pv)),
            "mean": 2.0,
            "var": float(var),
            "n": n,
            "method": "rank von Neumann ratio test, eqs. (3.5.1)-(3.5.2)",
        }
    )


gibbons_rvn_test = rvntest
