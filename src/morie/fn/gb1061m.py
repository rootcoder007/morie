# morie.fn -- function file (rootcoder007/morie)
"""Null moments of the Jonckheere-Terpstra statistic."""

import math

from ._richresult import RichResult

__all__ = ['jtmom', 'gibbons_jt_moments']


def jtmom(ns):
    """E_0[B] and Var_0[B] -- eqs. (10.6.2) and (10.6.3).

    Book pp. 365-366.  Since E[U_ij] = n_i n_j / 2 under H0,

    .. math:: E_0[B] = \\sum_{i<j}\\frac{n_i n_j}{2}
        = \\frac{N^2 - \\sum_i n_i^2}{4},

    and the variance the book leaves as an exercise is

    .. math:: Var_0[B] = \\frac{N^2(2N+3)
        - \\sum_i n_i^2 (2n_i+3)}{72}.

    The pairwise sum is computed as well as the closed form, so the
    first identity is verified rather than assumed.

    Parameters
    ----------
    ns : sequence of int
        The k sample sizes.

    Returns
    -------
    RichResult
        keys ``mean``, ``mean_pairwise``, ``var``, ``sd``, ``k``,
        ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (10.6.2)-(10.6.3), pp. 365-366.
    """
    nv = [int(v) for v in ns]
    k = len(nv)
    if k < 2:
        raise ValueError("need at least 2 samples.")
    if any(v < 1 for v in nv):
        raise ValueError("sample sizes must be at least 1.")
    nn = sum(nv)
    mean = (float(nn) ** 2 - sum(float(v) ** 2 for v in nv)) / 4.0
    pair = sum(
        nv[i] * nv[j] / 2.0 for i in range(k) for j in range(i + 1, k)
    )
    var = (
        float(nn) ** 2 * (2.0 * nn + 3.0)
        - sum(float(v) ** 2 * (2.0 * v + 3.0) for v in nv)
    ) / 72.0
    return RichResult(
        payload={
            "mean": float(mean),
            "mean_pairwise": float(pair),
            "var": float(var),
            "sd": float(math.sqrt(var)),
            "k": int(k),
            "n": int(nn),
            "method": "JT null moments, eqs. (10.6.2)-(10.6.3)",
        }
    )


gibbons_jt_moments = jtmom
