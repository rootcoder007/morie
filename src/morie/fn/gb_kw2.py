# morie.fn -- function file (rootcoder007/morie)
"""The defining form of the Kruskal-Wallis statistic -- eq. (10.4.2)."""

import math

from ._richresult import RichResult

__all__ = ['kwalt', 'gibbons_kw_alt_form']


def kwalt(rank_sums, ns):
    """H from the weighted sum of squared rank-sum deviations.

    Book p. 354, eq. (10.4.2):

    .. math:: H = \\frac{12}{N(N+1)}\\sum_{i=1}^{k}\\frac{1}{n_i}
        \\left[R_i - \\frac{n_i(N+1)}{2}\\right]^2,

    the form the statistic is defined by, with the reciprocal sample
    sizes as weights.  Eq. (10.4.7),
    H = 12/[N(N+1)] sum R_i^2/n_i - 3(N+1), is algebraically the same
    thing and is returned as ``h_computing`` so the identity can be
    checked rather than assumed.

    Parameters
    ----------
    rank_sums : sequence of float
        The k rank sums R_i.
    ns : sequence of int
        The k sample sizes.

    Returns
    -------
    RichResult
        keys ``statistic`` (eq. 10.4.2), ``h_computing`` (eq. 10.4.7),
        ``resid`` (their difference), ``df``, ``k``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (10.4.2) and (10.4.7),
    pp. 354, 357.
    """
    rs = [float(v) for v in rank_sums]
    nv = [int(v) for v in ns]
    k = len(rs)
    if k < 2 or len(nv) != k:
        raise ValueError("need at least 2 samples and matching sizes.")
    if any(v < 1 for v in nv):
        raise ValueError("sample sizes must be at least 1.")
    nn = sum(nv)
    h1 = 12.0 / (nn * (nn + 1.0)) * sum(
        (rs[i] - nv[i] * (nn + 1.0) / 2.0) ** 2 / nv[i] for i in range(k)
    )
    h2 = 12.0 / (nn * (nn + 1.0)) * sum(
        rs[i] ** 2 / nv[i] for i in range(k)
    ) - 3.0 * (nn + 1.0)
    return RichResult(
        payload={
            "statistic": float(h1),
            "h_computing": float(h2),
            "resid": float(h1 - h2),
            "df": int(k - 1),
            "k": int(k),
            "n": int(nn),
            "method": "Kruskal-Wallis H, defining form eq. (10.4.2)",
        }
    )


gibbons_kw_alt_form = kwalt
