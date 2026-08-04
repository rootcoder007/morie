# morie.fn -- function file (rootcoder007/morie)
"""NCRM Laplace functional.

Implements sec. 14.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ncrm_levy"]


def ghosal_ncrm_levy(f_vals, nu_masses, u_atoms):
    """E exp(-int f dM) = exp(-int (1 - e^{-f(x) u}) nu(du, dx))
    (sec. 14.7): exact for a discrete Levy measure with atoms
    (u_j, mass m_j) paired with f values. Keys: estimate."""
    fs = _bnp._flat(f_vals)
    ms = _bnp._flat(nu_masses)
    us = _bnp._flat(u_atoms)
    expo = sum(m * (1.0 - math.exp(-f * u))
               for f, m, u in zip(fs, ms, us))
    res = RichResult(payload={"estimate": math.exp(-expo),
                              "exponent": expo,
                              "method": "NCRM Laplace functional (GvdV 2017 sec. 14.7)"})
    return with_describe_pointer(res, "gh_c14_16")


def cheatsheet():
    return "gh_c14_16: NCRM Laplace functional"


# compact alias per ledger/NAMING.md
ghosalncrmlevy = ghosal_ncrm_levy
