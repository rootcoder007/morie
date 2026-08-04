# morie.fn -- function file (rootcoder007/morie)
"""Return level z_p.

Implements eq. (3.4) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["return_level"]


def return_level(mu, sigma, xi, T):
    """Return level (Coles 2001 eq. 3.4) -- front-end returning the
    shelf's scalar contract."""
    z = _ev.gev_return_level(float(T), float(mu), float(sigma),
                             float(xi))
    res = RichResult(payload={"estimate": float(z), "T": float(T),
                              "method": "GEV return level (Coles 2001 eq. 3.4)"})
    return with_describe_pointer(res, "retLvl")


def cheatsheet():
    return "retLvl: Return level z_p"


# compact alias per ledger/NAMING.md
returnlevel = return_level
