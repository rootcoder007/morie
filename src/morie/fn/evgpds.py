# morie.fn -- function file (rootcoder007/morie)
"""Sample from a GPD.

Implements eq. (4.2) inverted (inverse-CDF sampling) of Coles (2001), *An Introduction to Statistical
Modeling of Extreme Values*, Springer. The mathematics live in
``morie.fn._evt_core``; this module is the named entry point with the
shelf's result contract.
"""

from . import _array_core as np
from . import _evt_core as _ev
from ._richresult import RichResult, with_describe_pointer

__all__ = ["evt_gpd_sample"]


def evt_gpd_sample(n, sigma, xi, seed=42):
    """Inverse-CDF GPD sampling of threshold excesses."""
    rng = np.random.default_rng(seed)
    y = _ev.gpd_sample(int(n), float(sigma), float(xi), rng)
    res = RichResult(payload={"y": y, "n": int(n),
                              "method": "GPD inverse-CDF sampler (Coles 2001 eq. 4.2)"})
    return with_describe_pointer(res, "evgpds")


def cheatsheet():
    return "evgpds: Sample from a GPD"


# compact alias per ledger/NAMING.md
evtgpdsample = evt_gpd_sample
