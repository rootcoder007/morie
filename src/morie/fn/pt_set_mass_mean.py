# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree first two moments.

Implements eq. (3.21), p.49 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["pt_set_mass_mean"]


def pt_set_mass_mean(alpha_epsilon, epsilon=None,
                                            m=None):
    """E P(A_e) = prod_j alpha_taken/(alpha_0 + alpha_1) and
    E P(A_e)^2 = prod_j alpha_taken (alpha_taken + 1) /
    ((alpha_0+alpha_1)(alpha_0+alpha_1+1)) (eq. 3.21).
    ``alpha_epsilon`` is a list of (alpha_taken, alpha_other) pairs
    down the branch. Keys: value."""
    pairs = [(float(a), float(b)) for a, b in alpha_epsilon]
    mean = _bnp.pt_set_mass_mean(pairs)
    m2 = 1.0
    for a_take, a_other in pairs:
        s = a_take + a_other
        m2 *= a_take * (a_take + 1.0) / (s * (s + 1.0))
    var = m2 - mean * mean
    res = RichResult(payload={"estimate": mean,
                              "value": [mean, m2],
                              "second_moment": m2, "variance": var,
                              "method": "PT set-mass moments (GvdV 2017 eq. 3.21)"})
    return with_describe_pointer(res, "ghs028")


def cheatsheet():
    return "ghs028: Pólya tree first two moments"


# compact alias per ledger/NAMING.md
ptsetmassmean = pt_set_mass_mean
