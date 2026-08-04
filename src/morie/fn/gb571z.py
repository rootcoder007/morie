# morie.fn -- function file (rootcoder007/morie)
"""Tie correction for the signed-rank variance -- eq. (5.7.11)."""

import math

from ._richresult import RichResult

__all__ = ['wsrties', 'gibbons_wsrt_ties_zeros']


def wsrties(d, m0=0.0):
    """Variance of T+ corrected for tied |d| and dropped zeros.

    Book p. 203, eqs. (5.7.10) and (5.7.11).  A group of t tied
    absolute differences reduces the sum of squares of the ranks by
    t(t^2 - 1)/12, so

    .. math:: Var[T^+ | H_0] = \\frac{N(N+1)(2N+1)}{24}
        - \\frac{\\sum t(t^2-1)}{48}.

    Parameters
    ----------
    d : sequence of float
        Differences (or observations, with ``m0`` subtracted).
    m0 : float, optional
        Hypothesised median (default 0).

    Returns
    -------
    RichResult
        keys ``var``, ``var_uncorrected``, ``correction``, ``n``,
        ``nzero``, ``ties``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), eqs. (5.7.10)-(5.7.11), p. 203.
    """
    ds = [float(v) - float(m0) for v in d]
    nzero = sum(1 for v in ds if v == 0.0)
    a = sorted(abs(v) for v in ds if v != 0.0)
    n = len(a)
    if n < 1:
        raise ValueError("no non-zero differences.")
    ties = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    corr = sum(t * (t * t - 1.0) for t in ties) / 48.0
    v0 = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    return RichResult(
        payload={
            "var": float(v0 - corr),
            "var_uncorrected": float(v0),
            "correction": float(corr),
            "n": int(n),
            "nzero": int(nzero),
            "ties": ties,
            "method": "tie-corrected Var[T+], eq. (5.7.11)",
        }
    )


gibbons_wsrt_ties_zeros = wsrties
