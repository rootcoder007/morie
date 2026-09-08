# morie.fn -- function file (rootcoder007/morie)
"""Bound under a treatment-variation (exclusion) assumption."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_treatment_variation"]


def bound_treatment_variation(y, D, X):
    """Intersection bounds on the ATE under mean independence of ``X``.

    If the excluded variable ``X`` shifts who gets treated but not the
    counterfactual means, then each value of ``X`` yields its own
    worst-case interval for the same ``E[y(t)]``, and the parameter must
    lie in all of them at once.  Intersecting is what makes the assumption
    bite; it is also what makes it refutable, since an empty intersection
    contradicts mean independence.

    Formula (Molinari 2021 eq. (2.15)):
    ``E[y(t)] in [max_x lower(x), min_x upper(x)]`` with each ``(x)``
    interval the worst-case bound computed within that cell, then the ATE
    interval formed from the two intersected arms.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    X : array-like
        Discrete excluded variable, one value per unit.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``n_cells``,
        ``refuted``, ``n``.

    References
    ----------
    Manski, C. F. (1995).  Identification Problems in the Social Sciences.
    Harvard University Press.  Intersection-bound form as equation (2.15)
    of Molinari, F. (2021), Microeconometrics with partial identification,
    Handbook of Econometrics 7A (arXiv:2004.11751 p. 19).
    """
    yv, dv = B.yd(y, D, "bound_treatment_variation")
    xv = C.vec(X)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_treatment_variation: X must have one value per unit")
    y0, y1 = B.support(yv)
    grp = B.cells(xv)
    lo1 = None
    hi1 = None
    lo0 = None
    hi0 = None
    for g in grp:
        gy = [yv[i] for i in range(n) if xv[i] == g]
        gd = [dv[i] for i in range(n) if xv[i] == g]
        p1, m1, p0, m0 = B.cellmeans(gy, gd)
        a1 = B.wc_arm(m1, p1, y0, y1)
        a0 = B.wc_arm(m0, p0, y0, y1)
        if lo1 is None or a1[0] > lo1:
            lo1 = a1[0]
        if hi1 is None or a1[1] < hi1:
            hi1 = a1[1]
        if lo0 is None or a0[0] > lo0:
            lo0 = a0[0]
        if hi0 is None or a0[1] < hi0:
            hi0 = a0[1]
    refuted = 1.0 if (lo1 > hi1 or lo0 > hi0) else 0.0
    lo = lo1 - hi0
    hi = hi1 - lo0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "n_cells": len(grp),
        "refuted": refuted, "n": n,
        "method": "Bound with treatment-variation assumption"})


def cheatsheet():
    return "bnstvr: Bound with treatment-variation assumption"
