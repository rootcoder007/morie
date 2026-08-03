# morie.fn -- function file (rootcoder007/morie)
"""GEV quantile (return-level) function.

Implements eq. (3.4) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_quantile"]


def evt_gev_quantile(p, mu, sigma, xi):
    """GEV quantile x_p = mu - (sigma/xi)[1 - (-log p)^(-xi)]
    (Coles 2001 eq. 3.4 with p the non-exceedance probability)."""
    ps = _ev._flat(p)
    q = [_ev.gev_quantile(v, float(mu), float(sigma), float(xi))
         for v in ps]
    out = q[0] if len(q) == 1 else q
    res = RichResult(payload={"x_p": out,
                              "method": "GEV quantile (Coles 2001 eq. 3.4)"})
    return with_describe_pointer(res, "evgevq")


def cheatsheet():
    return "evgevq: GEV quantile (return-level) function"
