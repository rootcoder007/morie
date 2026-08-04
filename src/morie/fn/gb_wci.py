# morie.fn -- function file (rootcoder007/morie)
"""Significance test for the coefficient of concordance W."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['wsignif', 'gibbons_concordance_signif']


def wsignif(w, k, n):
    """Chi-square test of independence based on W, Sec. 12.4.2.

    Book p. 455.  With k sets of rankings of n objects, the null
    distribution of S (hence of W) is exactly that of Sec. 12.2, so
    for large k

    .. math:: Q = \\frac{12S}{kn(n+1)} = k(n-1)W

    is approximately chi-square with n - 1 degrees of freedom, with
    the rejection region in the upper tail (W = 0 under independence,
    W = 1 under perfect agreement).  For small k Table N gives the
    exact distribution.

    Parameters
    ----------
    w : float
        Observed coefficient of concordance, 0 <= w <= 1.
    k : int
        Number of rankings (judges), k >= 2.
    n : int
        Number of objects ranked, n >= 2.

    Returns
    -------
    RichResult
        keys ``statistic`` (Q), ``w``, ``df``, ``p_value``,
        ``s`` (the implied sum of squares), ``table_n`` (1 when the
        exact table should be preferred), ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 12.4.2, p. 455; Table N, p. 588.
    """
    w = float(w)
    k = int(k)
    n = int(n)
    if not 0.0 <= w <= 1.0:
        raise ValueError("w must lie in [0, 1].")
    if k < 2 or n < 2:
        raise ValueError("need k >= 2 rankings of n >= 2 objects.")
    q = k * (n - 1.0) * w
    s = q * k * n * (n + 1.0) / 12.0
    return RichResult(
        payload={
            "statistic": float(q),
            "w": w,
            "df": int(n - 1),
            "p_value": float(stats.chi2.sf(q, n - 1)),
            "s": float(s),
            "table_n": int(k <= 5 or n <= 5),
            "k": k,
            "n": n,
            "method": "concordance W significance test (Sec. 12.4.2)",
        }
    )


gibbons_concordance_signif = wsignif
