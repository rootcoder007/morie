# morie.fn -- function file (rootcoder007/morie)
"""Log-likelihood of a GEV sample.

Implements eq. (3.7)-(3.9) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_loglik"]


def evt_gev_loglik(x, mu, sigma, xi):
    """GEV log-likelihood (Coles 2001 eq. 3.7; Gumbel form eq. 3.9).
    Returns -inf when any observation violates the support constraint
    1 + xi (x - mu)/sigma > 0 (the sec. 3.3.2 warning)."""
    ll = _ev.gev_loglik(x, float(mu), float(sigma), float(xi))
    n = len(_ev._flat(x))
    res = RichResult(payload={"ll": float(ll), "n": n,
                              "method": "GEV log-likelihood (Coles 2001 eq. 3.7)"})
    return with_describe_pointer(res, "evgevl")


def cheatsheet():
    return "evgevl: Log-likelihood of a GEV sample"
