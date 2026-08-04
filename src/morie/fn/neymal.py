# morie.fn -- slice s03 (rootcoder007/morie)
"""Neyman optimal allocation, stated as n_h proportional to N_h S_h.

Same source and same equations as :mod:`morie.fn.neyman` -- Neyman, J.
(1934), *Journal of the Royal Statistical Society* 97(4), 558-625, JSTOR
2342192, p. 580 eq. (39) and eq. (41), read as a rendered page image.
This entry point differs only in taking the population size N = M_0 as
its first argument instead of the stratum means, and in checking it
against Sum_h N_h, which eq. (39) requires to be the same number.

The optimum allocation, the vanishing of the term B of eq. (39) at it,
and the erratum in the printed eq. (40) are all documented in
:mod:`morie.fn.neyman`; that module is the primary write-up.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from .neyman import _abc, _alloc_check, _alloc_variance, _largest_remainder

from ._richresult import RichResult

__all__ = ["neyman_allocation"]


def neyman_allocation(N, Nh, Sh, n):
    """Optimum allocation n_h proportional to N_h S_h.

    Parameters
    ----------
    N : float or None
        Population size M_0.  When given it must equal Sum_h N_h to
        within 1e-8 relative, since eq. (39) uses the one number in both
        roles; None skips the check and uses Sum_h N_h.
    Nh : array-like
        Stratum sizes M_i, all positive.
    Sh : array-like
        Within-stratum standard deviations S_i, divisor M_i - 1.
    n : float
        Total number of units to allocate.

    Returns
    -------
    estimate     : the variance sigma^2 the optimum allocation achieves
    allocation   : the real-valued n_h = n N_h S_h / Sum_k N_k S_k
    allocation_int : the same rounded to integers summing to round(n)
    variance     : same as estimate, eq. (37)
    A, B, C      : the three pieces of eq. (39) at this allocation
    N            : the population size actually used
    """
    M, S, m0, H = _alloc_check(Nh, Sh, n)
    A, C, M0, T = _abc(M, S, m0)
    if N is not None:
        Ngiven = float(N)
        if Ngiven != Ngiven or not (Ngiven > 0):
            raise ValueError("neyman_allocation: the population size N must be positive")
        if abs(Ngiven - M0) > 1e-8 * max(1.0, abs(M0)):
            raise ValueError("neyman_allocation: N does not equal sum(Nh)")
    if not (T > 0):
        raise ValueError("neyman_allocation: sum(Nh Sh) must be positive; every stratum has Sh = 0")
    alloc = [m0 * M[i] * S[i] / T for i in range(H)]
    var = _alloc_variance(M, S, alloc)
    B = 0.0
    for i in range(H):
        if alloc[i] > 0:
            d = M[i] * S[i] / alloc[i] - T / m0
            B += alloc[i] * d * d
    return RichResult(
        payload={
            "estimate": var,
            "allocation": alloc,
            "allocation_int": _largest_remainder(alloc, m0),
            "variance": var,
            "A": A,
            "B": B,
            "C": C,
            "N": M0,
            "n": H,
            "method": "Neyman (1934) optimum allocation n_h ~ N_h S_h, eq. (39) p. 580",
        }
    )


def cheatsheet():
    return "neymal: Neyman optimal allocation"
