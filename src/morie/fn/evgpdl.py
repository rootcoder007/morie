# morie.fn -- function file (rootcoder007/morie)
"""Log-likelihood of a GPD sample.

Implements eq. (4.10) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_loglik"]


def evt_gpd_loglik(y, sigma, xi):
    """GPD log-likelihood over threshold excesses (Coles 2001
    eq. 4.10); -inf outside the support."""
    ll = _ev.gpd_loglik(y, float(sigma), float(xi))
    n = len(_ev._flat(y))
    res = RichResult(payload={"ll": float(ll), "n": n,
                              "method": "GPD log-likelihood (Coles 2001 eq. 4.10)"})
    return with_describe_pointer(res, "evgpdl")


def cheatsheet():
    return "evgpdl: Log-likelihood of a GPD sample"
