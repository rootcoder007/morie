# morie.fn -- function file (rootcoder007/morie)
"""Worst-case bound on the causal odds ratio."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_logistic"]


def bound_logistic(y, D, X):
    """Bounds on the causal odds ratio for a binary outcome.

    The odds ratio is a strictly increasing function of the treated risk
    and a strictly decreasing function of the control risk, so the extreme
    odds ratios are attained at the corners of the two risk intervals --
    no search is needed and the bound is exact rather than conservative.
    The risks themselves are the worst-case bounds on ``E[y(t)]``,
    computed within stratum and averaged.

    Formula: ``OR = [p_1 / (1 - p_1)] / [p_0 / (1 - p_0)]``, so
    ``OR_low = odds(p_1^L) / odds(p_0^U)`` and
    ``OR_high = odds(p_1^U) / odds(p_0^L)``.

    Parameters
    ----------
    y : array-like
        Binary outcome, coded 0/1.
    D : array-like
        Binary treatment indicator, coded 0/1.
    X : array-like
        Discrete stratum label, one per unit.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate`` (geometric
        midpoint), ``p1_lower``, ``p1_upper``, ``p0_lower``, ``p0_upper``,
        ``n_strata``, ``n``.

    References
    ----------
    Robins, J. M. (2002) is the stub's attribution.  The risk bounds used
    are Manski's worst case, equation (2.11) of Molinari, F. (2021),
    Handbook of Econometrics 7A (arXiv:2004.11751 p. 17); the corner
    argument above is written out here rather than copied, because the
    attributed source could not be obtained.
    """
    yv, dv = B.yd(y, D, "bound_logistic")
    for v in yv:
        if v != 0.0 and v != 1.0:
            raise ValueError("bound_logistic: y must be coded 0/1")
    xv = C.vec(X)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_logistic: X must have one value per unit")
    p1lo = p1hi = p0lo = p0hi = 0.0
    grp = B.cells(xv)
    for g in grp:
        idx = [i for i in range(n) if xv[i] == g]
        gy = [yv[i] for i in idx]
        gd = [dv[i] for i in idx]
        q1, m1, q0, m0 = B.cellmeans(gy, gd)
        a1 = B.wc_arm(m1, q1, 0.0, 1.0)
        a0 = B.wc_arm(m0, q0, 0.0, 1.0)
        w = len(idx) / float(n)
        p1lo += w * a1[0]
        p1hi += w * a1[1]
        p0lo += w * a0[0]
        p0hi += w * a0[1]

    def odds(p):
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return float("inf")
        return p / (1.0 - p)

    lo = odds(p1lo) / odds(p0hi) if odds(p0hi) > 0.0 else 0.0
    hi = odds(p1hi) / odds(p0lo) if odds(p0lo) > 0.0 else float("inf")
    est = (lo * hi) ** 0.5 if (lo > 0.0 and hi < float("inf")) else float("nan")
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo, "estimate": est,
        "p1_lower": p1lo, "p1_upper": p1hi,
        "p0_lower": p0lo, "p0_upper": p0hi,
        "n_strata": len(grp), "n": n,
        "method": "Logistic odds-ratio bound"})


def cheatsheet():
    return "bndlgt: Logistic odds-ratio bound"
