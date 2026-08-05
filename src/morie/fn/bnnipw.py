# morie.fn -- function file (rootcoder007/morie)
"""Bound from a proxy used in place of an instrument."""

from . import _bndcore as B
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_no_iv_proxy"]


def bound_no_iv_proxy(y, D, Z_proxy):
    """Intersection bound treating a proxy as if it were an instrument.

    A proxy is only worth using if it moves selection without moving the
    counterfactual means, and both halves of that show up here.  How much
    it moves selection is the width reduction against the assumption-free
    bound, reported directly.  Whether the exclusion half survives is the
    emptiness of the intersection: an empty one is a refutation, so a
    proxy that is too good to be true announces itself instead of quietly
    producing a narrow interval.

    Formula: arm intersection over the proxy's cells, Molinari (2021)
    eq. (2.15), against the unconditional worst case of eq. (2.11).

    Parameters
    ----------
    y : array-like
        Observed outcome.
    D : array-like
        Binary treatment, coded 0/1.
    Z_proxy : array-like
        Discrete proxy variable, one value per unit.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``estimate``, ``wc_width``,
        ``informativeness``, ``valid``, ``n_cells``, ``n``.

    References
    ----------
    Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X. & Miao, W.
    (2020).  An introduction to proximal causal learning.
    arXiv:2009.10982 -- the stub's attribution.  Their bridge-function
    identification is a different construction and is not implemented
    here; what is implemented is the intersection bound of Manski, stated
    as equation (2.15) of Molinari, F. (2021), Handbook of Econometrics
    7A (arXiv:2004.11751 p. 19), applied to the proxy.
    """
    yv, dv = B.yd(y, D, "bound_no_iv_proxy")
    zv = C.vec(Z_proxy)
    n = len(yv)
    if len(zv) != n:
        raise ValueError("bound_no_iv_proxy: Z_proxy must have one value per unit")
    y0, y1 = B.support(yv)
    wlo, whi = B.wc_ate(yv, dv, y0, y1)
    lo1, hi1, lo0, hi0 = B.wc_intersect(yv, dv, zv, y0, y1)
    empty = lo1 > hi1 or lo0 > hi0
    lo = lo1 - hi0
    hi = hi1 - lo0
    wcw = whi - wlo
    info = 0.0 if wcw <= 0.0 else 1.0 - (hi - lo) / wcw
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "estimate": 0.5 * (lo + hi), "wc_width": wcw,
        "informativeness": info, "valid": 0.0 if empty else 1.0,
        "n_cells": len(B.cells(zv)), "n": n,
        "method": "Bound without IV using proxy"})


def cheatsheet():
    return "bnnipw: proxy intersection bound with a validity check"
