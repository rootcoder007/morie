# morie.fn -- slice s03 (rootcoder007/morie)
"""Proportional stratum allocation, n_h = n N_h / N.

Neyman, J. (1934), *Journal of the Royal Statistical Society* 97(4),
558-625, JSTOR 2342192.  Page 567, read as a rendered page image, credits
this rule to Bowley: "Professor Bowley considered only the case when the
sizes, say m'_i, of the partial samples are proportional to the sizes of
corresponding strata" (the sentence runs on to p. 568).  Page 580,
also read as an image, gives what it costs: writing the variance of
eq. (37) as A + B - C of eq. (39), proportional allocation makes B = C
identically, so

    sigma^2 = A = ((M_0 - m_0) / m_0) Sum_i M_i S_i^2,

which is the page's eq. (41)... printed there as eq. (40) with a
misprint, "(M_0 - M_0)/m_0" for "(M_0 - m_0)/m_0"; see
:mod:`morie.fn.neyman` for the full note on that erratum.  The optimum
allocation reaches A - C instead, so proportional allocation is worse by
exactly C >= 0, and C = 0 precisely when every S_i is equal -- in which
case the two allocations coincide.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from .neyman import _abc, _alloc_check, _alloc_variance, _largest_remainder

from ._richresult import RichResult

__all__ = ["proportional_allocation"]


def proportional_allocation(N, Nh, n):
    """Allocate n units to strata in proportion to stratum size.

    Parameters
    ----------
    N : float or None
        Population size M_0; must equal Sum_h N_h when given.
    Nh : array-like
        Stratum sizes M_i, all positive.
    n : float
        Total number of units to allocate.

    Returns
    -------
    estimate     : the real-valued allocation is in allocation; this
                   is A = ((M_0 - n)/n) Sum_i N_i, the variance of
                   eq. (40) evaluated with every S_i = 1, which is the
                   only variance available without S_h
    allocation   : n_h = n N_h / N
    allocation_int : the same rounded to integers summing to round(n)
    fraction     : the sampling fraction n / N, common to every stratum
    N            : the population size actually used
    """
    M, S, m0, H = _alloc_check(Nh, None, n)
    A, C, M0, T = _abc(M, S, m0)
    if N is not None:
        Ngiven = float(N)
        if Ngiven != Ngiven or not (Ngiven > 0):
            raise ValueError("proportional_allocation: the population size N must be positive")
        if abs(Ngiven - M0) > 1e-8 * max(1.0, abs(M0)):
            raise ValueError("proportional_allocation: N does not equal sum(Nh)")
    alloc = [m0 * M[i] / M0 for i in range(H)]
    var = _alloc_variance(M, S, alloc)
    return RichResult(
        payload={
            "estimate": var,
            "allocation": alloc,
            "allocation_int": _largest_remainder(alloc, m0),
            "variance": var,
            "A": A,
            "fraction": m0 / M0,
            "N": M0,
            "n": H,
            "method": "Bowley proportional allocation n_h = n N_h / N; Neyman (1934) eq. (40) p. 580",
        }
    )


def cheatsheet():
    return "propal: Proportional stratum allocation"
