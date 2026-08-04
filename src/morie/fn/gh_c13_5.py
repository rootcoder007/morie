# morie.fn -- function file (rootcoder007/morie)
"""Beta-process Lévy measure.

Implements sec. 13.3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bp_cont"]


def ghosal_bp_cont(c=2.0, t_max=1.0, n_grid=2000):
    """BP(c, H0) Levy measure nu(du, dt) = c u^{-1} (1-u)^{c-1} du
    dH0(t) (sec. 13.3.2): the expected total jump mass
    int_0^1 u nu(du) dH0 = int_0^1 c (1-u)^{c-1} du H0(t_max)
    = H0(t_max). Quadrature check. Keys: estimate."""
    tot = 0.0
    for i in range(n_grid):
        u = (i + 0.5) / n_grid
        tot += u * c / u * (1.0 - u) ** (c - 1.0) / n_grid
    expected_mass = tot * t_max            # H0(t) = t
    res = RichResult(payload={"estimate": expected_mass,
                              "H0_t_max": t_max,
                              "gap": abs(expected_mass - t_max),
                              "method": "BP Levy measure (GvdV 2017 sec. 13.3.2)"})
    return with_describe_pointer(res, "gh_c13_5")


def cheatsheet():
    return "gh_c13_5: Beta-process Lévy measure"


# compact alias per ledger/NAMING.md
ghosalbpcont = ghosal_bp_cont
