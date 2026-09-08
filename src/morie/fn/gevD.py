# morie.fn -- function file (rootcoder007/morie)
"""Generalized extreme value distribution.

Implements eq. (3.2) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["gev_distribution"]


def gev_distribution(x, mu, sigma, xi):
    """GEV distribution bundle: CDF, density and log-density at ``x``
    (Coles 2001 eq. 3.2 and sec. 3.3.2). ``estimate`` is the CDF."""
    import math
    xs = _ev._flat(x)
    F = [_ev.gev_cdf(v, float(mu), float(sigma), float(xi))
         for v in xs]
    lp = [_ev.gev_logpdf(v, float(mu), float(sigma), float(xi))
          for v in xs]
    f = [math.exp(v) if v > -700 else 0.0 for v in lp]
    one = len(xs) == 1
    res = RichResult(payload={"estimate": F[0] if one else F,
                              "F": F[0] if one else F,
                              "pdf": f[0] if one else f,
                              "logpdf": lp[0] if one else lp,
                              "method": "GEV distribution (Coles 2001 eq. 3.2)"})
    return with_describe_pointer(res, "gevD")


def cheatsheet():
    return "gevD: Generalized extreme value distribution"
