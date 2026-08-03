# morie.fn -- function file (rootcoder007/morie)
"""GPD CDF.

Implements eq. (4.2) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_cdf"]


def evt_gpd_cdf(y, sigma, xi):
    """GPD distribution function H(y) = 1 - (1+xi y/sigma)^(-1/xi)
    for exceedances y >= 0 (Coles 2001 eq. 4.2; exponential limit
    eq. 4.3)."""
    ys = _ev._flat(y)
    F = [_ev.gpd_cdf(v, float(sigma), float(xi)) for v in ys]
    out = F[0] if len(F) == 1 else F
    res = RichResult(payload={"F": out,
                              "method": "GPD CDF (Coles 2001 eq. 4.2)"})
    return with_describe_pointer(res, "evgpdc")


def cheatsheet():
    return "evgpdc: GPD CDF"
