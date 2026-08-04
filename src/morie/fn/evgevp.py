# morie.fn -- function file (rootcoder007/morie)
"""GEV distribution density.

Implements eq. (3.2) differentiated (sec. 3.3.2) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gev_pdf"]


def evt_gev_pdf(x, mu, sigma, xi):
    """GEV density f = (1/sigma) t(x)^(xi+1) exp(-t(x)) with
    t(x) = [1+xi(x-mu)/sigma]^(-1/xi) (Coles 2001 sec. 3.3.2)."""
    import math
    xs = _ev._flat(x)
    f = [math.exp(_ev.gev_logpdf(v, float(mu), float(sigma),
                                 float(xi))) for v in xs]
    out = f[0] if len(f) == 1 else f
    res = RichResult(payload={"f": out, "mu": float(mu),
                              "sigma": float(sigma), "xi": float(xi),
                              "method": "GEV density (Coles 2001 sec. 3.3.2)"})
    return with_describe_pointer(res, "evgevp")


def cheatsheet():
    return "evgevp: GEV distribution density"


# compact alias per ledger/NAMING.md
evtgevpdf = evt_gev_pdf
