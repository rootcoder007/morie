# morie.fn -- function file (rootcoder007/morie)
"""Identification by intersection of moment inequalities."""

from . import _bndmi as MI
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_iii"]


def bound_iii(y, X, moments):
    """The identified set as the zero-level set of the criterion.

    The set estimate and the confidence set are different objects and the
    difference is visible here: this module reports the level set at
    criterion zero, with no critical value inflating it.  Every candidate
    inside the interval satisfies both inequalities in sample and scores
    exactly zero; every candidate outside scores strictly positive.

    Formula: ``H = {theta : q(theta) = 0}`` with ``q`` the sum of squared
    positive parts, Molinari (2021) equations (4.2) and (4.4).

    Parameters
    ----------
    y : array-like
        Lower end of each observation's interval.
    X : array-like
        Upper end of each observation's interval, same length as ``y``.
    moments : array-like
        Candidate parameter values at which the criterion is evaluated.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width`` (the identified set),
        ``n_in_set``, ``q_min``, ``q_max_stat``, ``n``.

    References
    ----------
    Chernozhukov, V., Hong, H. & Tamer, E. (2007).  Estimation and
    confidence regions for parameter sets in econometric models.
    Econometrica 75(5), 1243-1284.  doi:10.1111/j.1468-0262.2007.00794.x.
    Criterion and level set as equations (4.2), (4.3) and (4.4) of
    Molinari, F. (2021), Handbook of Econometrics 7A (arXiv:2004.11751
    pp. 89, 88), the copy used.
    """
    yl = C.vec(y)
    yu = C.vec(X)
    if len(yl) != len(yu):
        raise ValueError("bound_iii: y and X must have the same length")
    pairs = [[yl[i], yu[i]] for i in range(len(yl))]
    yl, yu = MI.interval_data(pairs, "bound_iii")
    n, mL, sL, mU, sU = MI.stats(yl, yu)
    grid = C.vec(moments)
    if len(grid) == 0:
        raise ValueError("bound_iii: moments grid is empty")
    nin = 0
    qmin = None
    qmax = 0.0
    for t in grid:
        q = MI.crit(t, n, mL, sL, mU, sU)
        if qmin is None or q < qmin:
            qmin = q
        qm = MI.critmax(t, n, mL, sL, mU, sU)
        if qm > qmax:
            qmax = qm
        if q <= 0.0:
            nin += 1
    return RichResult(payload={
        "lower": mL, "upper": mU, "width": mU - mL,
        "n_in_set": nin, "q_min": qmin, "q_max_stat": qmax, "n": n,
        "method": "Identification by intersection of inequalities"})


def cheatsheet():
    return "bnsiii: identified set as the zero-level set of the CHT criterion"

# public names resolved by fn/_lazy_map.json
boundiii = bound_iii
