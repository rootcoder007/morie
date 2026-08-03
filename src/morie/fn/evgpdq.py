# morie.fn -- function file (rootcoder007/morie)
"""GPD quantile function.

Implements eq. (4.2) inverted of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_quantile"]


def evt_gpd_quantile(p, sigma, xi):
    """GPD quantile y_p = (sigma/xi)[(1-p)^(-xi) - 1] (inverse of
    Coles 2001 eq. 4.2)."""
    ps = _ev._flat(p)
    q = [_ev.gpd_quantile(v, float(sigma), float(xi)) for v in ps]
    out = q[0] if len(q) == 1 else q
    res = RichResult(payload={"y_p": out,
                              "method": "GPD quantile (Coles 2001 eq. 4.2 inverse)"})
    return with_describe_pointer(res, "evgpdq")


def cheatsheet():
    return "evgpdq: GPD quantile function"
