# morie.fn -- function file (rootcoder007/morie)
"""Chi-square approximation for Friedman's statistic."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['friedchi', 'gibbons_friedman_chi2_approp']


def friedchi(q, k, n):
    """Reference chi-square tail for Q, with its exact first two moments.

    Book p. 442, following eq. (12.2.8): Q has

    .. math:: E[Q] = n - 1, \\qquad
        Var[Q] = \\frac{2(n-1)(k-1)}{k},

    "which are the first two moments of a chi-square distribution with
    n - 1 degrees of freedom" only in the limit -- the exact variance
    is smaller by the factor (k-1)/k, and the book notes the higher
    moments are likewise only closely approximated.  Both the
    chi-square variance and the exact one are returned so the gap at
    small k is visible.

    Parameters
    ----------
    q : float
        Observed Q.
    k : int
        Number of blocks.
    n : int
        Number of treatments, n >= 2.

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``mean``,
        ``var_exact``, ``var_chi2``, ``ratio``, ``k``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.2, p. 442, following
    eq. (12.2.8).
    """
    q = float(q)
    k = int(k)
    n = int(n)
    if k < 2 or n < 2:
        raise ValueError("need k >= 2 blocks and n >= 2 treatments.")
    df = n - 1
    ve = 2.0 * (n - 1.0) * (k - 1.0) / k
    vc = 2.0 * (n - 1.0)
    return RichResult(
        payload={
            "statistic": q,
            "df": int(df),
            "p_value": float(stats.chi2.sf(q, df)),
            "mean": float(n - 1.0),
            "var_exact": float(ve),
            "var_chi2": float(vc),
            "ratio": float(ve / vc),
            "k": k,
            "n": n,
            "method": "chi-square approximation to Friedman Q (Sec. 12.2)",
        }
    )


gibbons_friedman_chi2_approp = friedchi
