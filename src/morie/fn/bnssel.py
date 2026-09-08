# morie.fn -- function file (rootcoder007/morie)
"""Sample-selection bound for a selectively observed outcome."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_selection"]


def bound_selection(y, D, X):
    """Bounds on ``E[y]`` when ``y`` is observed only for the selected.

    Heckman's selection model buys point identification with an exclusion
    restriction and a joint normality assumption.  Dropping both leaves the
    identified set of Manski's worst-case bound: the observed part of the
    mean is known, and the unobserved part can sit anywhere in the support.
    Stratifying on a discrete ``X`` and averaging the within-stratum
    intervals sharpens the bound whenever selection rates differ by
    stratum, and reproduces the pooled bound when they do not.

    Formula (per stratum, Molinari 2021 eq. (2.2)):
    ``[E(y | D = 1, x) P(D = 1 | x) + y_0 P(D = 0 | x),
       E(y | D = 1, x) P(D = 1 | x) + y_1 P(D = 0 | x)]``,
    averaged with weights ``P(x)``.

    Parameters
    ----------
    y : array-like
        Outcome; entries with ``D = 0`` are treated as unobserved and their
        recorded value is never used.
    D : array-like
        Selection indicator, 1 when ``y`` is observed.
    X : array-like
        Discrete stratum label, one per unit.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``p_observed``,
        ``n_strata``, ``n``.

    References
    ----------
    Heckman, J. J. (1979).  Sample selection bias as a specification error.
    Econometrica 47(1), 153-161.  doi:10.2307/1912352.
    Manski, C. F. (2003).  Partial Identification of Probability
    Distributions.  Springer.  Worst-case region as Theorem SIR-2.1,
    equation (2.2), of Molinari, F. (2021), Handbook of Econometrics 7A
    (arXiv:2004.11751 p. 12).
    """
    yv, dv = B.yd(y, D, "bound_selection")
    xv = C.vec(X)
    n = len(yv)
    if len(xv) != n:
        raise ValueError("bound_selection: X must have one value per unit")
    obs = [yv[i] for i in range(n) if dv[i] == 1.0]
    if not obs:
        raise ValueError("bound_selection: no observed outcome")
    y0, y1 = B.support(obs)
    lo = 0.0
    hi = 0.0
    grp = B.cells(xv)
    for g in grp:
        idx = [i for i in range(n) if xv[i] == g]
        ng = len(idx)
        n1 = 0
        s1 = 0.0
        for i in idx:
            if dv[i] == 1.0:
                n1 += 1
                s1 += yv[i]
        p1 = n1 / float(ng)
        m1 = s1 / n1 if n1 else 0.0
        a = B.wc_arm(m1, p1, y0, y1)
        wgt = ng / float(n)
        lo += wgt * a[0]
        hi += wgt * a[1]
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "p_observed": len(obs) / float(n),
        "n_strata": len(grp), "n": n,
        "method": "Sample-selection bound"})


def cheatsheet():
    return "bnssel: Sample-selection bound (Heckman model)"

# public names resolved by fn/_lazy_map.json
boundselection = bound_selection
