# morie.fn -- function file (rootcoder007/morie)
"""Ripley's K function for a planar point pattern.

Includes the isotropic edge correction, without which K is biased
downwards near the window boundary.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ripley_k"]


def ripley_k(coords, r_grid, area=None, edge_correction=True):
    """K(r) = |A| n^-2 sum_i sum_{j != i} w_ij 1(d_ij <= r).

    Under complete spatial randomness K(r) = pi r^2, so K above that
    indicates clustering and below it regularity.  ``L`` in the result
    is the variance-stabilising sqrt(K/pi), which is flat under CSR and
    easier to read. Keys: estimate."""
    r = _rc.ripley_k(coords, r_grid, area=area,
                     edge_correction=edge_correction)
    res = RichResult(payload={"estimate": r["K"], "r": r["r"],
                              "K": r["K"], "L": r["L"],
                              "csr_K": r["csr_K"],
                              "intensity": r["intensity"],
                              "method": r["method"]})
    return with_describe_pointer(res, "rkfunc")


def cheatsheet():
    return "rkfunc: Ripley's K function for a planar point pattern"
