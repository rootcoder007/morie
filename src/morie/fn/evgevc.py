# morie.fn -- function file (rootcoder007/morie)
"""GEV distribution CDF.

Implements eq. (3.2) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_cdf"]


def evt_gev_cdf(x, mu, sigma, xi):
    """GEV distribution function G(x) = exp{-[1+xi(x-mu)/sigma]^(-1/xi)}
    (Coles 2001 eq. 3.2; Gumbel limit as xi -> 0)."""
    xs = _ev._flat(x)
    F = [_ev.gev_cdf(v, float(mu), float(sigma), float(xi)) for v in xs]
    out = F[0] if len(F) == 1 else F
    res = RichResult(payload={"F": out, "mu": float(mu),
                              "sigma": float(sigma), "xi": float(xi),
                              "method": "GEV CDF (Coles 2001 eq. 3.2)"})
    return with_describe_pointer(res, "evgevc")


def cheatsheet():
    return "evgevc: GEV distribution CDF"


# compact alias per ledger/NAMING.md
evtgevcdf = evt_gev_cdf
