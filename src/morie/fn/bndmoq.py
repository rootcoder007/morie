# morie.fn -- function file (rootcoder007/morie)
"""Quantile bound for a selectively observed outcome."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_moment_qed"]


def bound_moment_qed(y, D, X, quantile):
    """Sharp bounds on the ``alpha``-quantile of a selectively observed ``y``.

    Quantiles behave better than means under selection.  The mean bound is
    informative only when the outcome support is bounded on the relevant
    side; the quantile bound is informative whenever the observed fraction
    exceeds ``1 - alpha`` (lower) or ``alpha`` (upper), regardless of the
    range of ``y``.  Because both ends are quantiles of the observed
    distribution, the bound is equivariant under any increasing transform
    of ``y``.

    Formula (Molinari 2021 pp. 12-13, from Manski 2003 Section 1.3), with
    ``q(p)`` the type-1 quantile of ``y`` among the observed and
    ``p_1 = P(D = 1)``::

        lower = q(1 - (1 - alpha) / p_1)   if p_1 > 1 - alpha,  else y_0
        upper = q(alpha / p_1)             if p_1 >= alpha,     else y_1

    Parameters
    ----------
    y : array-like
        Outcome; entries with ``D = 0`` are unobserved.
    D : array-like
        Observation indicator, coded 0/1.
    X : array-like
        Discrete stratum label, one per unit; used to report the widest
        within-stratum interval alongside the pooled one.
    quantile : float
        Quantile level ``alpha`` in (0, 1).

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``max_width``,
        ``n_strata``, ``p_observed``, ``n``.

    Notes
    -----
    The stub this replaced attributed the construction to Chernozhukov and
    Hansen (2005).  That paper is the instrumental-variable quantile
    regression model, a different object; the bound evaluated here is
    Manski's, and is cited as such.

    References
    ----------
    Manski, C. F. (2003).  Partial Identification of Probability
    Distributions.  Springer, Section 1.3.  The two displayed expressions
    are ``r(alpha, x)`` and ``s(alpha, x)`` on pp. 12-13 of Molinari, F.
    (2021), Handbook of Econometrics 7A (arXiv:2004.11751), the copy used.
    """
    yv, dv = B.yd(y, D, "bound_moment_qed")
    xv = C.vec(X)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_moment_qed: X must have one value per unit")
    a = float(quantile)
    if not (0.0 < a < 1.0):
        raise ValueError("bound_moment_qed: quantile must lie in (0, 1)")
    y0, y1 = B.support(yv)

    def band(ys, ds):
        m = len(ys)
        obs = [ys[i] for i in range(m) if ds[i] == 1.0]
        p1 = len(obs) / float(m)
        if not obs:
            return (y0, y1, p1)
        lo = B.q1(obs, 1.0 - (1.0 - a) / p1) if p1 > 1.0 - a else y0
        hi = B.q1(obs, a / p1) if p1 >= a else y1
        return (lo, hi, p1)

    lo, hi, p1 = band(yv, dv)
    grp = B.cells(xv)
    mw = 0.0
    for g in grp:
        gy = [yv[i] for i in range(n) if xv[i] == g]
        gd = [dv[i] for i in range(n) if xv[i] == g]
        b = band(gy, gd)
        if b[1] - b[0] > mw:
            mw = b[1] - b[0]
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "max_width": mw,
        "n_strata": len(grp), "p_observed": p1, "n": n,
        "method": "Quantile-equivariant bound"})


def cheatsheet():
    return "bndmoq: Quantile-equivariant bound"

# public names resolved by fn/_lazy_map.json
boundmomentqed = bound_moment_qed
