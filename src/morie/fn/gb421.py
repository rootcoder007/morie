# morie.fn -- function file (rootcoder007/morie)
"""Chi-square goodness-of-fit statistic for grouped data."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['chigof', 'gibbons_chisq_gof']


def chigof(observed, expected, ddof=0):
    """Pearson goodness-of-fit statistic Q for k categories.

    Section 4.2 (book p. 104), eq. (4.2.1):

    .. math:: Q = \\sum_{i=1}^{k}\\frac{(f_i - e_i)^2}{e_i},

    asymptotically chi-square with k - 1 degrees of freedom, reduced by
    one further degree for each parameter estimated from the data
    (``ddof``).

    Parameters
    ----------
    observed : sequence of float
        Observed cell frequencies, k >= 2.
    expected : sequence of float
        Expected cell frequencies under H0, all strictly positive.
    ddof : int, optional
        Number of parameters estimated from the sample (default 0).

    Returns
    -------
    RichResult
        keys ``statistic``, ``df``, ``p_value``, ``k``, ``n``,
        ``contrib`` (per-cell terms), ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 4.2, eq. (4.2.1), p. 104.
    """
    o = [float(v) for v in observed]
    e = [float(v) for v in expected]
    k = len(o)
    if k < 2:
        raise ValueError("need at least 2 categories.")
    if len(e) != k:
        raise ValueError("observed and expected must have equal length.")
    if any(v <= 0.0 for v in e):
        raise ValueError("expected frequencies must be strictly positive.")
    contrib = [(o[i] - e[i]) ** 2 / e[i] for i in range(k)]
    q = sum(contrib)
    df = k - 1 - int(ddof)
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1.")
    return RichResult(
        payload={
            "statistic": float(q),
            "df": int(df),
            "p_value": float(stats.chi2.sf(q, df)),
            "k": int(k),
            "n": float(sum(o)),
            "contrib": contrib,
            "method": "chi-square goodness of fit, eq. (4.2.1)",
        }
    )


gibbons_chisq_gof = chigof
