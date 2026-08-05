# morie.fn -- function file (rootcoder007/morie)
"""Positive-only treatment bound (monotone treatment response, increasing)."""

from . import _bndcore as B

from ._richresult import RichResult

__all__ = ["bound_pos_treatment"]


def bound_pos_treatment(y, D, y_max):
    """Bound on the ATE when treatment is assumed never to hurt.

    Monotone treatment response in the increasing direction,
    ``y(1) >= y(0)`` for every unit, pins the lower bound at exactly zero
    and leaves only the upper bound to be estimated.  Unlike the worst-case
    bound, the arm bounds here use every ``(y, D)`` pair rather than only
    the pairs in the arm of interest, so they stay informative even when
    one arm is nearly empty.

    Formula (Manski 1997b Propositions M1 and M2, binary treatment):
    ``upper = [E(y | D = 1) P(D = 1) + y_max P(D = 0)]
              - [E(y | D = 0) P(D = 0) + y_min P(D = 1)]``, ``lower = 0``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    y_max : float
        Upper end of the logically possible outcome support; must be at
        least ``max(y)``.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``p_treated``, ``n``.

    References
    ----------
    Manski, C. F. (1997).  Monotone treatment response.  Econometrica
    65(6), 1311-1334.  doi:10.2307/2171738.  The binary-treatment form used
    here is equation (2.13) and the paragraph beneath it in Molinari, F.
    (2021), Microeconometrics with partial identification, Handbook of
    Econometrics 7A (arXiv:2004.11751 pp. 18-19), which is the copy used.
    """
    yv, dv = B.yd(y, D, "bound_pos_treatment")
    ymax = float(y_max)
    y0, y1 = B.support(yv)
    if ymax < y1:
        raise ValueError("bound_pos_treatment: y_max is below max(y)")
    p1, m1, p0, m0 = B.cellmeans(yv, dv)
    hi1 = m1 * p1 + ymax * p0
    lo0 = m0 * p0 + y0 * p1
    lo = 0.0
    hi = hi1 - lo0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "p_treated": p1, "n": len(yv),
        "method": "Positive-only treatment bound"})


def cheatsheet():
    return "bndpos: Positive-only treatment bound"
