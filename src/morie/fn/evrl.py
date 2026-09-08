# morie.fn -- function file (rootcoder007/morie)
"""T-year return level under GEV.

Implements eq. (3.4) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_return_level"]


def evt_return_level(mu, sigma, xi, T):
    """T-period GEV return level z_T = mu - (sigma/xi)[1 - y_T^(-xi)],
    y_T = -log(1 - 1/T) (Coles 2001 eq. 3.4; Gumbel form eq. 3.5).
    z_T is exceeded once per T periods on average."""
    z = _ev.gev_return_level(float(T), float(mu), float(sigma),
                             float(xi))
    res = RichResult(payload={"z_T": float(z), "T": float(T),
                              "method": "GEV return level (Coles 2001 eq. 3.4)"})
    return with_describe_pointer(res, "evrl")


def cheatsheet():
    return "evrl: T-year return level under GEV"


# compact alias per ledger/NAMING.md
evtreturnlevel = evt_return_level
