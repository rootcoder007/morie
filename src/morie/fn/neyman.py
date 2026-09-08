# morie.fn -- slice s03 (rootcoder007/morie)
"""Neyman optimal allocation across strata.

Neyman, J. (1934), "On the Two Different Aspects of the Representative
Method: The Method of Stratified Sampling and the Method of Purposive
Selection", *Journal of the Royal Statistical Society* 97(4), 558-625;
JSTOR 2342192.  Pages 579 and 580 were rendered as images with pdftoppm
and read visually, because the equations are load-bearing.

p. 579, eq. (37), the variance of the estimate Sum(M_i ubar_i) of the
population total under stratified random sampling::

    sigma^2 = Sum_i { (M_i^2 / m_i) ((M_i - m_i) / (M_i - 1)) sigma_i^2 }

p. 580 puts S_i^2 = M_i sigma_i^2 / (M_i - 1) and rewrites (37) as
eq. (39), whose three terms the page names A, B and -C::

    A =  ((M_0 - m_0) / m_0) Sum_i M_i S_i^2
    B =  Sum_i m_i ( M_i S_i / m_i - Sum_j M_j S_j / m_0 )^2
    C =  (M_0 / m_0) Sum_i M_i ( S_i - Sum_j M_j S_j / M_0 )^2

with M_0 = Sum_i M_i, m_0 = Sum_i m_i and sigma^2 = A + B - C.  Only B
depends on the individual m_i.  B is a sum of squares, so it is >= 0,
and it vanishes exactly when

    m_i = m_0 M_i S_i / Sum_j M_j S_j,

which is Neyman's optimum allocation; there sigma^2 = A - C, the page's
eq. (41).  Under Bowley's proportional allocation m_i = m_0 M_i / M_0
instead, B = C identically and sigma^2 = A, the page's eq. (40).

ERRATUM in the original.  Equation (40) on p. 580 is printed as
"sigma^2 = ((M_0 - M_0) / m_0) Sum(M_i S_i^2) = A".  The second M_0 must
be m_0: A is defined three lines above as the first term of (39), and as
printed the leading factor is identically zero, which would make the
variance under proportional allocation vanish for every population.
This was read off the rendered page image, so it is a misprint in the
1934 typesetting and not a pdftotext artifact.

Argument convention.  In eq. (37) sigma_i^2 is the within-stratum
variance taken with divisor M_i; S_i^2 = M_i sigma_i^2 / (M_i - 1) is the
same variance with divisor M_i - 1, which is the form survey practice
reports.  S_h below is S_i, so eq. (37) with sigma_i^2 eliminated is

    sigma^2 = Sum_i M_i (M_i - m_i) S_i^2 / m_i,

and that is what variance returns.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["neyman_allocation"]


def _alloc_check(N_h, S_h, n):
    """Coerce and validate the common stratum arguments.

    Returns (M, S, m0, H).  S is all-ones when S_h is None, which turns
    Neyman allocation into proportional allocation.
    """
    M = core.vec(N_h)
    H = len(M)
    if H == 0:
        raise ValueError("neyman_allocation: no strata")
    for v in M:
        if not (v > 0) or v != v:
            raise ValueError("neyman_allocation: every stratum size N_h must be positive")
    if S_h is None:
        S = [1.0] * H
    else:
        S = core.vec(S_h)
        if len(S) != H:
            raise ValueError("neyman_allocation: N_h and S_h have different lengths")
        for v in S:
            if v != v or v < 0:
                raise ValueError("neyman_allocation: every stratum standard deviation S_h must be >= 0")
    m0 = float(n)
    if m0 != m0 or not (m0 > 0):
        raise ValueError("neyman_allocation: the total sample size n must be positive")
    return M, S, m0, H


def _largest_remainder(alloc, m0):
    """Round a real allocation to integers summing to round(m0).

    Deterministic: the units left over after flooring go to the largest
    fractional parts, ties broken by the lower stratum index, so both
    language arms land on the same integers.
    """
    H = len(alloc)
    base = [int(math.floor(a)) for a in alloc]
    rem = int(round(m0)) - sum(base)
    if rem <= 0:
        return base
    order = sorted(range(H), key=lambda i: (-(alloc[i] - base[i]), i))
    for k in range(min(rem, H)):
        base[order[k]] += 1
    return base


def _alloc_variance(M, S, m):
    """Neyman (1934) eq. (37), p. 579, with sigma_i^2 eliminated.

    sigma^2 = Sum_i M_i (M_i - m_i) S_i^2 / m_i.
    """
    tot = 0.0
    for i in range(len(M)):
        if m[i] <= 0:
            return float("inf")
        tot += M[i] * (M[i] - m[i]) * S[i] * S[i] / m[i]
    return tot


def _abc(M, S, m0):
    """The A, B-free and C pieces of Neyman (1934) eq. (39), p. 580.

    B is omitted because it depends on the allocation, not on the
    population; the callers add it themselves when they need it.
    """
    M0 = 0.0
    for v in M:
        M0 += v
    T = 0.0
    for i in range(len(M)):
        T += M[i] * S[i]
    a = 0.0
    for i in range(len(M)):
        a += M[i] * S[i] * S[i]
    A = (M0 - m0) / m0 * a
    c = 0.0
    for i in range(len(M)):
        d = S[i] - T / M0
        c += M[i] * d * d
    C = M0 / m0 * c
    return A, C, M0, T


def neyman_allocation(y, N_h, S_h, n):
    """Optimum allocation of n sampling units over strata, Neyman (1934).

    Parameters
    ----------
    y : array-like or None
        Per-stratum sample means ybar_h, one per stratum, or None.  When
        supplied, the stratified estimate of the population total
        Sum_h N_h ybar_h is returned as total.  It plays no part in
        the allocation, which by eq. (39) depends only on N_h and S_h.
    N_h : array-like
        Stratum sizes M_i, all positive.
    S_h : array-like
        Within-stratum standard deviations S_i, divisor M_i - 1.
    n : float
        Total number of units to allocate, m_0 of eq. (38).

    Returns
    -------
    estimate    : the variance sigma^2 the optimum allocation achieves
    allocation  : the real-valued m_i = m_0 M_i S_i / Sum_j M_j S_j
    allocation_int : the same rounded to integers summing to round(n)
    variance    : same as estimate, by eq. (37)
    variance_prop : the variance proportional allocation would achieve
    A, B, C     : the three pieces of eq. (39) at this allocation
    total       : Sum_h N_h ybar_h when y is supplied, else None
    """
    M, S, m0, H = _alloc_check(N_h, S_h, n)
    A, C, M0, T = _abc(M, S, m0)
    if not (T > 0):
        raise ValueError("neyman_allocation: sum(N_h S_h) must be positive; every stratum has S_h = 0")
    alloc = [m0 * M[i] * S[i] / T for i in range(H)]
    var = _alloc_variance(M, S, alloc)
    B = 0.0
    for i in range(H):
        if alloc[i] > 0:
            d = M[i] * S[i] / alloc[i] - T / m0
            B += alloc[i] * d * d
    prop = [m0 * M[i] / M0 for i in range(H)]
    var_prop = _alloc_variance(M, S, prop)
    total = None
    if y is not None:
        yv = core.vec(y)
        if len(yv) != H:
            raise ValueError("neyman_allocation: y and N_h have different lengths")
        total = 0.0
        for i in range(H):
            total += M[i] * yv[i]
    return RichResult(
        payload={
            "estimate": var,
            "allocation": alloc,
            "allocation_int": _largest_remainder(alloc, m0),
            "variance": var,
            "variance_prop": var_prop,
            "A": A,
            "B": B,
            "C": C,
            "total": total,
            "n": H,
            "method": "Neyman (1934) optimum allocation m_i = m_0 M_i S_i / sum(M_j S_j), eq. (39) p. 580",
        }
    )


def cheatsheet():
    return "neyman: Neyman optimal allocation across strata"
