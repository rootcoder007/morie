# morie.fn -- function file (rootcoder007/morie)
"""Null moments of the rank von Neumann statistic."""

import math

from ._richresult import RichResult

__all__ = ['rvnmom', 'gibbons_rvn_moments']


def rvnmom(n):
    """Asymptotic null mean and variance of RVN.

    Book p. 95: RVN is asymptotically normal with mean 2 and

    .. math:: Var[RVN] = \\frac{4(n-2)(5n^2-2n-9)}{5n(n+1)(n-1)^2}
        \\;\\approx\\; \\frac{20}{5n+7}.

    With no ties the denominator of RVN is exactly n(n^2-1)/12, which
    is returned as ``denom``.

    Parameters
    ----------
    n : int
        Number of observations, n >= 3.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``var_approx``, ``sd``, ``denom``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 3.5, p. 95 (Bartels, 1982).
    """
    n = int(n)
    if n < 3:
        raise ValueError("n must be at least 3.")
    var = (
        4.0 * (n - 2.0) * (5.0 * n * n - 2.0 * n - 9.0)
        / (5.0 * n * (n + 1.0) * (n - 1.0) ** 2)
    )
    return RichResult(
        payload={
            "mean": 2.0,
            "var": float(var),
            "var_approx": float(20.0 / (5.0 * n + 7.0)),
            "sd": float(math.sqrt(var)),
            "denom": float(n * (n * n - 1.0) / 12.0),
            "n": n,
            "method": "RVN null moments (Gibbons Sec. 3.5, Bartels 1982)",
        }
    )


gibbons_rvn_moments = rvnmom
