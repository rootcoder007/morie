# morie.fn -- function file (rootcoder007/morie)
"""T-period return level under POT/GPD.

Implements eq. (4.12)-(4.13) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_return_level_pot"]


def evt_return_level_pot(u, sigma, xi, zeta_u, m):
    """m-observation POT return level
    x_m = u + (sigma/xi)[(m zeta_u)^xi - 1] where zeta_u = P(X > u)
    (Coles 2001 eq. 4.12-4.13; log form as xi -> 0)."""
    z = _ev.pot_return_level(float(m), float(u), float(sigma),
                             float(xi), float(zeta_u))
    res = RichResult(payload={"z_T": float(z), "m": float(m),
                              "u": float(u), "zeta_u": float(zeta_u),
                              "method": "POT return level (Coles 2001 eq. 4.13)"})
    return with_describe_pointer(res, "evrlpot")


def cheatsheet():
    return "evrlpot: T-period return level under POT/GPD"
