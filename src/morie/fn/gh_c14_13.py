# morie.fn -- function file (rootcoder007/morie)
"""Poisson-Kingman Lévy intensity.

Implements sec. 14.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_pk_levy"]


def ghosal_pk_levy(u_grid_max=8.0, n_grid=4000):
    """rho(du) on (0, infty) with int min(u, 1) rho(du) < infty: for
    the gamma Levy density rho(u) = u^{-1} e^{-u} the expected total
    mass int_0^infty u rho(u) du = 1 (sec. 14.5). Quadrature.
    Keys: estimate."""
    tot = 0.0
    for i in range(n_grid):
        u = (i + 0.5) * u_grid_max / n_grid
        tot += u * (1.0 / u) * math.exp(-u) * u_grid_max / n_grid
    res = RichResult(payload={"estimate": tot,
                              "gap_to_one": abs(tot - 1.0),
                              "method": "PK Levy intensity mass (GvdV 2017 sec. 14.5)"})
    return with_describe_pointer(res, "gh_c14_13")


def cheatsheet():
    return "gh_c14_13: Poisson-Kingman Lévy intensity"


# compact alias per ledger/NAMING.md
ghosalpklevy = ghosal_pk_levy
