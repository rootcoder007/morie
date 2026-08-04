# morie.fn -- function file (rootcoder007/morie)
"""Species sampling process.

Implements sec. 14.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ssp_def"]


def ghosal_ssp_def(weights, atoms):
    """G = sum_k p_k delta_{theta_k}, theta_k iid G0, sum p_k = 1
    a.s. (sec. 14.2): validates the species-sampling structure.
    Keys: estimate."""
    p = _bnp.normalize_weights(weights)
    th = _bnp._flat(atoms)
    mean = sum(pi * t for pi, t in zip(p, th))
    res = RichResult(payload={"estimate": mean,
                              "total_mass": sum(p),
                              "n_species": len(p),
                              "method": "species sampling process (GvdV 2017 sec. 14.2)"})
    return with_describe_pointer(res, "gh_c14_5")


def cheatsheet():
    return "gh_c14_5: Species sampling process"


# compact alias per ledger/NAMING.md
ghosalsspdef = ghosal_ssp_def
