# morie.fn -- function file (rootcoder007/morie)
"""MLE of GPD parameters above threshold.

Implements sec. 4.3.2 (likelihood 4.10) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_mle"]


def evt_gpd_mle(y):
    """GPD maximum-likelihood fit over excesses (Coles 2001
    sec. 4.3.2), Nelder-Mead on eq. (4.10), observed-information
    covariance."""
    f = _ev.gpd_mle(y)
    res = RichResult(payload={"sigma": f["sigma"], "xi": f["xi"],
                              "ll": f["loglik"], "cov": f["cov"],
                              "n": f["n"], "converged": f["converged"],
                              "method": "GPD MLE (Coles 2001 sec. 4.3.2)"})
    return with_describe_pointer(res, "evgpdm")


def cheatsheet():
    return "evgpdm: MLE of GPD parameters above threshold"
