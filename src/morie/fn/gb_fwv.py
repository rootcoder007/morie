# morie.fn -- function file (rootcoder007/morie)
"""Exact moments of S and Q for a small number of blocks."""

import math

from ._richresult import RichResult

__all__ = ['friedvar', 'gibbons_friedman_variance']


def friedvar(k, n):
    """E[S], Var[S] and the moments of Q -- eqs. (12.2.7) and after.

    Book p. 442, eq. (12.2.7):

    .. math:: E[S] = \\frac{kn(n^2-1)}{12}, \\qquad
        Var[S] = \\frac{n^2 k(k-1)(n+1)^2}{72},

    with S the sum of squared deviations of the treatment rank sums
    from their mean.  Q = 12S / [kn(n+1)] therefore has E[Q] = n - 1
    and Var[Q] = 2(n-1)(k-1)/k, which falls short of the chi-square
    variance 2(n-1) by exactly (k-1)/k -- the small-k discrepancy this
    routine quantifies.

    Parameters
    ----------
    k : int
        Number of blocks, k >= 2.
    n : int
        Number of treatments, n >= 2.

    Returns
    -------
    RichResult
        keys ``mean_s``, ``var_s``, ``mean_q``, ``var_q``,
        ``var_chi2``, ``deficit``, ``k``, ``n``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eq. (12.2.7), p. 442.
    """
    k = int(k)
    n = int(n)
    if k < 2 or n < 2:
        raise ValueError("need k >= 2 blocks and n >= 2 treatments.")
    ms = k * n * (float(n) ** 2 - 1.0) / 12.0
    vs = float(n) ** 2 * k * (k - 1.0) * (n + 1.0) ** 2 / 72.0
    vq = 2.0 * (n - 1.0) * (k - 1.0) / k
    vc = 2.0 * (n - 1.0)
    return RichResult(
        payload={
            "mean_s": float(ms),
            "var_s": float(vs),
            "mean_q": float(n - 1.0),
            "var_q": float(vq),
            "var_chi2": float(vc),
            "deficit": float(vc - vq),
            "k": k,
            "n": n,
            "method": "Friedman S and Q moments, eq. (12.2.7)",
        }
    )


gibbons_friedman_variance = friedvar
