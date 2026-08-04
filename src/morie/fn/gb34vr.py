# morie.fn -- function file (rootcoder007/morie)
"""Null moments of the total runs up-and-down statistic."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['runsudvar', 'gibbons_runs_ud_var']


def runsudvar(n, r=None, alpha=0.05):
    """Mean and variance of R, the number of runs up and down.

    Section 3.4 (book p. 93): Levene (1952) showed that R standardised
    by

    .. math:: E[R] = \\frac{2n-1}{3}, \\qquad
              Var[R] = \\frac{16n-29}{90}

    is asymptotically standard normal.  With ``r`` supplied the
    continuity-corrected tail statistics printed on the same page,
    (R +- 0.5 - E[R])/sqrt(Var[R]), are returned as ``z_left`` and
    ``z_right``.

    Parameters
    ----------
    n : int
        Number of observations (the sign sequence has length n - 1).
    r : int, optional
        Observed number of runs up and down.
    alpha : float, optional
        Level used to report the two-sided normal critical value.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``sd``, ``z_left``, ``z_right``,
        ``p_value``, ``zcrit``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 3.4, p. 93 (Levene, 1952).
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2.")
    mean = (2.0 * n - 1.0) / 3.0
    var = (16.0 * n - 29.0) / 90.0
    sd = math.sqrt(var)
    zl = zr = pv = float("nan")
    if r is not None:
        r = float(r)
        zl = (r + 0.5 - mean) / sd
        zr = (r - 0.5 - mean) / sd
        pv = 2.0 * min(stats.norm.cdf(zl), 1.0 - stats.norm.cdf(zr))
        pv = min(1.0, pv)
    return RichResult(
        payload={
            "mean": float(mean),
            "var": float(var),
            "sd": float(sd),
            "z_left": float(zl),
            "z_right": float(zr),
            "p_value": float(pv),
            "zcrit": float(stats.norm.ppf(1.0 - float(alpha) / 2.0)),
            "n": n,
            "method": "runs up and down: E[R]=(2n-1)/3, Var[R]=(16n-29)/90",
        }
    )


gibbons_runs_ud_var = runsudvar
