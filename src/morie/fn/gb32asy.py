# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic normality of the total number of runs -- eq. (3.2.9)."""

import math

from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ['runsz', 'gibbons_runs_asymp_normal']


def runsz(r, n1, n2, correct=False):
    """Standardised total-runs statistic for the Wald-Wolfowitz test.

    Book p. 82, eq. (3.2.9): with lambda = n1/n and n = n1 + n2,

    .. math:: Z = \\frac{R - 2n\\lambda(1-\\lambda)}
        {2\\sqrt{n}\\,\\lambda(1-\\lambda)},

    which tends to the standard normal.  The exact null moments,
    E[R] = 2 n1 n2 / n + 1 and Var[R] = 2 n1 n2 (2 n1 n2 - n) /
    (n^2 (n-1)), are returned alongside for comparison; ``correct``
    applies a 0.5 continuity correction toward the mean.

    Parameters
    ----------
    r : int
        Observed total number of runs.
    n1, n2 : int
        Counts of the two element types.
    correct : bool, optional
        Apply the continuity correction (default False).

    Returns
    -------
    RichResult
        keys ``z`` (eq. 3.2.9), ``z_exact`` (using the exact moments),
        ``p_value``, ``mean``, ``var``, ``mean_exact``, ``var_exact``,
        ``lam``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (3.2.9), p. 82.
    """
    n1 = int(n1)
    n2 = int(n2)
    r = float(r)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    n = n1 + n2
    lam = n1 / float(n)
    mean = 2.0 * n * lam * (1.0 - lam)
    sd = 2.0 * math.sqrt(n) * lam * (1.0 - lam)
    me = 2.0 * n1 * n2 / float(n) + 1.0
    ve = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (float(n) ** 2 * (n - 1.0))
    d = r - mean
    de = r - me
    if correct:
        d = d - 0.5 if d > 0 else (d + 0.5 if d < 0 else d)
        de = de - 0.5 if de > 0 else (de + 0.5 if de < 0 else de)
    z = d / sd
    return RichResult(
        payload={
            "z": float(z),
            "z_exact": float(de / math.sqrt(ve)),
            "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
            "mean": float(mean),
            "var": float(sd * sd),
            "mean_exact": float(me),
            "var_exact": float(ve),
            "lam": float(lam),
            "n": int(n),
            "method": "total runs asymptotic normality, eq. (3.2.9)",
        }
    )


gibbons_runs_asymp_normal = runsz
