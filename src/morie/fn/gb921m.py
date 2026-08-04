# morie.fn -- function file (rootcoder007/morie)
"""Null moments of the Mood scale statistic."""

import math

from ._richresult import RichResult

__all__ = ['moodmom', 'gibbons_mood_moments']


def moodmom(m, n):
    """E[M_N] and Var[M_N] under H0 -- eqs. (9.2.2) and (9.2.3).

    Book pp. 315-316.  Derived from Theorem 7.3.2 with the Mood scores
    a_i = (i - (N+1)/2)^2 and the sum identities for i^3 and i^4 that
    the book proves by induction:

    .. math:: E[M_N] = \\frac{m(N^2-1)}{12}, \\qquad
        Var[M_N] = \\frac{mn(N+1)(N^2-4)}{180}.

    Parameters
    ----------
    m, n : int
        The two sample sizes.

    Returns
    -------
    RichResult
        keys ``mean``, ``var``, ``sd``, ``N``, ``m``, ``n``,
        ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (9.2.2)-(9.2.3), pp. 315-316.
    """
    m = int(m)
    n = int(n)
    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")
    nn = m + n
    mean = m * (nn * nn - 1.0) / 12.0
    var = m * n * (nn + 1.0) * (nn * nn - 4.0) / 180.0
    return RichResult(
        payload={
            "mean": float(mean),
            "var": float(var),
            "sd": float(math.sqrt(var)),
            "N": int(nn),
            "m": m,
            "n": n,
            "method": "Mood null moments, eqs. (9.2.2)-(9.2.3)",
        }
    )


gibbons_mood_moments = moodmom
