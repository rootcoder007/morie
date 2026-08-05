# morie.fn -- function file (rootcoder007/morie)
"""Bound without an unobserved-heterogeneity invariance assumption."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_no_unobserved_inv"]


def bound_no_unobserved_inv(y, D, X, X_inv):
    """Conditional intersection bound, dropping the invariance assumption.

    Point identification here rests on assuming the unobserved
    heterogeneity is invariant to the excluded variable.  Dropping it
    leaves only the exclusion restriction itself, which is refutable
    rather than free: within each stratum of ``X`` the counterfactual mean
    must lie in every ``X_inv`` cell's worst-case interval at once, and an
    empty intersection in any stratum contradicts the restriction.  The
    stratum bounds are then averaged, so conditioning tightens and never
    widens.

    Formula: within each ``X`` stratum, the arm intersection of Molinari
    (2021) eq. (2.15) over the cells of ``X_inv``; the ATE interval is
    formed per stratum and averaged with the stratum shares.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment, coded 0/1.
    X : array-like
        Discrete conditioning covariate.
    X_inv : array-like
        Discrete variable excluded from the counterfactual means.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``n_strata``,
        ``n_cells``, ``refuted``, ``n``.

    References
    ----------
    Tchetgen Tchetgen, E. J. (2014) is the stub's attribution.  That
    construction could not be obtained, so what is implemented is the
    conditional form of Manski's intersection bound, equation (2.15) of
    Molinari, F. (2021), Handbook of Econometrics 7A (arXiv:2004.11751
    p. 19), applied within stratum -- stated here rather than attributed
    to a source that was not read.
    """
    yv, dv = B.yd(y, D, "bound_no_unobserved_inv")
    xv = C.vec(X)
    wv = C.vec(X_inv)
    n = len(yv)
    if len(xv) != n or len(wv) != n:
        raise ValueError("bound_no_unobserved_inv: X and X_inv must have one value per unit")
    y0, y1 = B.support(yv)
    grp = B.cells(xv)
    lo = 0.0
    hi = 0.0
    refuted = 0.0
    ncells = 0
    for g in grp:
        idx = [i for i in range(n) if xv[i] == g]
        gy = [yv[i] for i in idx]
        gd = [dv[i] for i in idx]
        gw = [wv[i] for i in idx]
        ncells += len(B.cells(gw))
        lo1, hi1, lo0, hi0 = B.wc_intersect(gy, gd, gw, y0, y1)
        if lo1 > hi1 or lo0 > hi0:
            refuted = 1.0
        wgt = len(idx) / float(n)
        lo += wgt * (lo1 - hi0)
        hi += wgt * (hi1 - lo0)
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "n_strata": len(grp),
        "n_cells": ncells, "refuted": refuted, "n": n,
        "method": "Bound under no unobserved invariance"})


def cheatsheet():
    return "bnsnig: conditional intersection bound without invariance"
