# morie.fn -- function file (rootcoder007/morie)
"""GPD density above threshold.

Implements eq. (4.2) differentiated of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_pdf"]


def evt_gpd_pdf(y, sigma, xi):
    """GPD density h(y) = (1/sigma)(1+xi y/sigma)^(-1-1/xi)
    (derivative of Coles 2001 eq. 4.2)."""
    import math
    ys = _ev._flat(y)
    f = [math.exp(_ev.gpd_logpdf(v, float(sigma), float(xi)))
         for v in ys]
    out = f[0] if len(f) == 1 else f
    res = RichResult(payload={"f": out,
                              "method": "GPD density (Coles 2001 eq. 4.2)"})
    return with_describe_pointer(res, "evgpdp")


def cheatsheet():
    return "evgpdp: GPD density above threshold"


# compact alias per ledger/NAMING.md
evtgpdpdf = evt_gpd_pdf
