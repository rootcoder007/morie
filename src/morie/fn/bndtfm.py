# morie.fn -- function file (rootcoder007/morie)
"""Bound under a monotone outcome transformation."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_transform"]


def bound_transform(y, D, X, transform):
    """Worst-case bounds on the ATE for a monotonically transformed outcome.

    Mean bounds are not invariant under a monotone transform -- the
    counterfactual mass is placed at the transformed support, which moves.
    Quantile bounds are invariant, because both ends are quantiles of the
    observed distribution and a monotone map commutes with the type-1
    quantile.  Both facts are reported: the mean bound on ``t(y)`` and the
    equivariance gap of the median bound, which is exactly zero for any
    increasing ``t``.

    Formula: worst-case bound of Molinari (2021) eq. (2.11) applied to
    ``t(y)``; gap ``= |t(r_y(1/2)) - r_{t(y)}(1/2)|
                    + |t(s_y(1/2)) - s_{t(y)}(1/2)|``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    X : array-like
        Discrete stratum label, one per unit; the mean bound is computed
        within stratum and averaged.
    transform : array-like
        ``t(y_i)`` already evaluated, one value per unit.  Must be
        non-decreasing in ``y``; passing the values rather than a callable
        keeps the Python and R arms evaluating the same map.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``gap``,
        ``n_strata``, ``n``.

    References
    ----------
    Chernozhukov, V., Lee, S. & Rosen, A. M. (2013).  Intersection bounds:
    estimation and inference.  Econometrica 81(2), 667-737.
    doi:10.3982/ECTA8718 -- the source of the stub's attribution.  The
    invariance evaluated here is the quantile bound of Manski (2003)
    Section 1.3 as printed on pp. 12-13 of Molinari, F. (2021), Handbook of
    Econometrics 7A (arXiv:2004.11751), and the mean bound is eq. (2.11)
    of the same, which is the copy used.
    """
    yv, dv = B.yd(y, D, "bound_transform")
    xv = C.vec(X)
    tv = C.vec(transform)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_transform: X must have one value per unit")
    if len(tv) != n:
        raise ValueError("bound_transform: transform must have one value per unit")
    order = sorted(range(n), key=lambda i: yv[i])
    for a, b in zip(order[:-1], order[1:]):
        if tv[b] < tv[a]:
            raise ValueError("bound_transform: transform is not monotone in y")
    t0, t1 = B.support(tv)
    grp = B.cells(xv)
    lo = 0.0
    hi = 0.0
    for g in grp:
        idx = [i for i in range(n) if xv[i] == g]
        gt = [tv[i] for i in idx]
        gd = [dv[i] for i in idx]
        a = B.wc_ate(gt, gd, t0, t1)
        w = len(idx) / float(n)
        lo += w * a[0]
        hi += w * a[1]
    obs_y = [yv[i] for i in range(n) if dv[i] == 1.0]
    obs_t = [tv[i] for i in range(n) if dv[i] == 1.0]
    gap = 0.0
    if obs_y:
        p1 = len(obs_y) / float(n)
        y0, y1 = B.support(yv)
        for lev, endpoint in ((1.0 - 0.5 / p1, 0), (0.5 / p1, 1)):
            if 0.0 < lev <= 1.0:
                gap += abs(_apply(yv, tv, B.q1(obs_y, lev)) - B.q1(obs_t, lev))
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "gap": gap,
        "n_strata": len(grp), "n": n,
        "method": "Bound under outcome transformation"})


def _apply(yv, tv, value):
    """``t(value)`` read off the tabulated pairs; ``value`` is some ``y_i``."""
    for i in range(len(yv)):
        if yv[i] == value:
            return tv[i]
    raise ValueError("bound_transform: transform undefined at a sample quantile")


def cheatsheet():
    return "bndtfm: Bound under outcome transformation"
