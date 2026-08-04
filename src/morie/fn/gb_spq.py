# morie.fn -- function file (rootcoder007/morie)
"""Test of the null hypothesis of zero Spearman rank correlation."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['rhotest', 'gibbons_spearman_test']


def rhotest(r, n, alternative="two-sided"):
    """Both large-sample tests of H0: rho = 0 from Sec. 11.3.3.

    Book p. 413.  Under independence E[R] = 0 and Var[R] = 1/(n-1)
    (Sec. 11.3.2, p. 412), giving the normal statistic

    .. math:: Z = R\\sqrt{n-1},

    while the equivalent Student form, obtained by treating R as an
    ordinary product-moment correlation of the ranks, is

    .. math:: t = R\\sqrt{\\frac{n-2}{1-R^2}}

    on n - 2 degrees of freedom.  Both are returned; for small n the
    exact null distribution of Sec. 11.3.1 (Table M) should be used
    instead of either.

    Parameters
    ----------
    r : float
        Observed Spearman coefficient, |r| <= 1.
    n : int
        Number of pairs, n >= 3.
    alternative : str, optional
        ``"two-sided"``, ``"greater"`` or ``"less"``.

    Returns
    -------
    RichResult
        keys ``z``, ``p_normal``, ``t``, ``df``, ``p_value``
        (the t-based value), ``var``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Secs. 11.3.2-11.3.3, pp. 412-413.
    """
    r = float(r)
    n = int(n)
    if n < 3:
        raise ValueError("n must be at least 3.")
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must lie in [-1, 1].")
    z = r * math.sqrt(n - 1.0)
    if abs(r) >= 1.0:
        t = math.inf if r > 0 else -math.inf
    else:
        t = r * math.sqrt((n - 2.0) / (1.0 - r * r))
    if alternative == "greater":
        pn = 1.0 - stats.norm.cdf(z)
        pt = stats.t.sf(t, n - 2)
    elif alternative == "less":
        pn = stats.norm.cdf(z)
        pt = 1.0 - stats.t.sf(t, n - 2)
    elif alternative == "two-sided":
        pn = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        pt = 2.0 * stats.t.sf(abs(t), n - 2)
    else:
        raise ValueError("alternative must be two-sided, greater or less.")
    return RichResult(
        payload={
            "z": float(z),
            "p_normal": float(min(1.0, pn)),
            "t": float(t),
            "df": int(n - 2),
            "p_value": float(min(1.0, pt)),
            "var": 1.0 / (n - 1.0),
            "n": n,
            "method": "test of zero Spearman correlation (Sec. 11.3.3)",
        }
    )


gibbons_spearman_test = rhotest
