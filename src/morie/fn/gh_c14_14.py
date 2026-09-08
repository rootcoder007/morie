# morie.fn -- function file (rootcoder007/morie)
"""Normalized inverse-Gaussian process.

Implements sec. 14.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_nig_proc"]


def ghosal_nig_proc(alpha_par=1.0, u_max=10.0, n_grid=6000):
    """NIG Levy density rho(u) = (2 pi)^{-1/2} u^{-3/2}
    e^{-alpha^2 u / 2} (sec. 14.6): total jump mass
    int u rho(u) du = (2 pi)^{-1/2} int u^{-1/2} e^{-alpha^2 u/2} du
    = 1/alpha (Gaussian integral). Quadrature check. Keys: estimate."""
    tot = 0.0
    for i in range(n_grid):
        u = (i + 0.5) * u_max / n_grid
        tot += u * u ** (-1.5) * math.exp(-alpha_par ** 2 * u / 2.0) \
            / math.sqrt(2.0 * math.pi) * u_max / n_grid
    res = RichResult(payload={"estimate": tot,
                              "theory": 1.0 / alpha_par,
                              "gap": abs(tot - 1.0 / alpha_par),
                              "method": "NIG Levy mass (GvdV 2017 sec. 14.6)"})
    return with_describe_pointer(res, "gh_c14_14")


def cheatsheet():
    return "gh_c14_14: Normalized inverse-Gaussian process"


# compact alias per ledger/NAMING.md
ghosalnigproc = ghosal_nig_proc
