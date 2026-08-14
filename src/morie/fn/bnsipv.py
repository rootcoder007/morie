# morie.fn -- function file (rootcoder007/morie)
"""Partial IV bound intersected with monotone treatment response."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_iv_partial"]


def bound_iv_partial(y, D, Z):
    """Intersection of the instrument bound and the shape bound.

    The two assumptions restrict different things -- the instrument
    restricts selection, monotone response restricts the outcome -- so
    their identified sets intersect rather than nest, and the joint bound
    can be strictly tighter than either.  Reporting the two components
    alongside the intersection is the point: it shows which assumption is
    doing the work, and an empty intersection means the two are jointly
    refuted by the data.

    Formula: instrument bound from Molinari (2021) eq. (2.15), shape bound
    ``[0, upper]`` from eq. (2.13), intersected coordinate-wise.

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment, coded 0/1.
    Z : array-like
        Discrete instrument, one value per unit.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``iv_lower``,
        ``iv_upper``, ``mtr_lower``, ``mtr_upper``, ``refuted``, ``n``.

    References
    ----------
    Mogstad, M., Santos, A. & Torgovitsky, A. (2018).  Using instrumental
    variables for inference about policy relevant treatment parameters.
    Econometrica 86(5), 1589-1619.  doi:10.3982/ECTA15463 -- the stub's
    attribution; their general linear-programming machinery already ships
    as :func:`~morie.fn.bndcvx.bound_convex_estimator` and is not
    duplicated here.  The two component bounds are equations (2.15) and
    (2.13) of Molinari, F. (2021), Handbook of Econometrics 7A
    (arXiv:2004.11751 pp. 18-19).
    """
    yv, dv = B.yd(y, D, "bound_iv_partial")
    zv = C.vec(Z)
    n = len(yv)
    if len(zv) != n:
        raise ValueError("bound_iv_partial: Z must have one value per unit")
    y0, y1 = B.support(yv)
    lo1, hi1, lo0, hi0 = B.wc_intersect(yv, dv, zv, y0, y1)
    iv_lo = lo1 - hi0
    iv_hi = hi1 - lo0
    p1, m1, p0, m0 = B.cellmeans(yv, dv)
    mtr_lo = 0.0
    mtr_hi = (m1 * p1 + y1 * p0) - (m0 * p0 + y0 * p1)
    lo = iv_lo if iv_lo > mtr_lo else mtr_lo
    hi = iv_hi if iv_hi < mtr_hi else mtr_hi
    refuted = 1.0 if (lo > hi or lo1 > hi1 or lo0 > hi0) else 0.0
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "iv_lower": iv_lo, "iv_upper": iv_hi,
        "mtr_lower": mtr_lo, "mtr_upper": mtr_hi, "refuted": refuted,
        "n": n, "method": "Partial IV bound under one-sided compliance"})


def cheatsheet():
    return "bnsipv: IV bound intersected with monotone treatment response"

# public names resolved by fn/_lazy_map.json
boundivpartial = bound_iv_partial
