# morie.fn -- function file (rootcoder007/morie)
"""Negative-only treatment bound (monotone treatment response, decreasing)."""

from . import _bndcore as B

from ._richresult import RichResult

__all__ = ["bound_neg_treatment"]


def bound_neg_treatment(y, D, y_min):
    """Bound on the ATE when treatment is assumed never to help.

    The mirror image of :func:`~morie.fn.bndpos.bound_pos_treatment`.
    Decreasing monotone treatment response, ``y(1) <= y(0)`` for every
    unit, pins the upper bound at exactly zero; only the lower bound is
    estimated, and it uses every ``(y, D)`` pair.

    Formula: ``lower = [E(y | D = 1) P(D = 1) + y_min P(D = 0)]
                       - [E(y | D = 0) P(D = 0) + y_max P(D = 1)]``,
    ``upper = 0``, with ``y_max = max(y)``.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment indicator, coded 0/1.
    y_min : float
        Lower end of the logically possible outcome support; must be at
        most ``min(y)``.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``p_treated``, ``n``.

    References
    ----------
    Manski, C. F. (1997).  Monotone treatment response.  Econometrica
    65(6), 1311-1334.  doi:10.2307/2171738.  Equation (2.13) of Molinari,
    F. (2021), Handbook of Econometrics 7A (arXiv:2004.11751 pp. 18-19),
    read with the order on the treatment set reversed.
    """
    yv, dv = B.yd(y, D, "bound_neg_treatment")
    ymin = float(y_min)
    y0, y1 = B.support(yv)
    if ymin > y0:
        raise ValueError("bound_neg_treatment: y_min is above min(y)")
    p1, m1, p0, m0 = B.cellmeans(yv, dv)
    lo1 = m1 * p1 + ymin * p0
    hi0 = m0 * p0 + y1 * p1
    lo = lo1 - hi0
    hi = 0.0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "p_treated": p1, "n": len(yv),
        "method": "Negative-only treatment bound"})


def cheatsheet():
    return "bndngt: Negative-only treatment bound"
