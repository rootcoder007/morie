# morie.fn -- function file (rootcoder007/morie)
"""MLE of GEV parameters from block maxima.

Implements sec. 3.3.2 (likelihood 3.7) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_mle"]


def evt_gev_mle(x):
    """GEV maximum-likelihood fit (Coles 2001 sec. 3.3.2): maximize
    eq. (3.7) by Nelder-Mead; covariance from the observed information
    (sec. 2.6.4)."""
    f = _ev.gev_mle(x)
    res = RichResult(payload={"mu": f["mu"], "sigma": f["sigma"],
                              "xi": f["xi"], "ll": f["loglik"],
                              "cov": f["cov"], "n": f["n"],
                              "converged": f["converged"],
                              "method": "GEV MLE (Coles 2001 sec. 3.3.2)"})
    return with_describe_pointer(res, "evgevm")


def cheatsheet():
    return "evgevm: MLE of GEV parameters from block maxima"
