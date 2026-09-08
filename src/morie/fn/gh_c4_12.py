# morie.fn -- function file (rootcoder007/morie)
"""Discreteness of DP realizations.

Implements Theorem 4.14 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dp_discrete"]


def ghosal_dp_discrete(n_terms, alpha, seed=42):
    """Almost every realization of DP(alpha) is discrete
    (Theorem 4.14): the stick-breaking representation exhibits it as
    a countable sum of atoms; the largest atom has positive mass.
    Keys: estimate."""
    M = float(alpha)
    rng = np.random.default_rng(seed)
    V = [float(rng.beta(1.0, M)) for _ in range(int(n_terms))]
    W = _bnp.stick_breaking(V)
    biggest = max(W)
    res = RichResult(payload={"estimate": biggest,
                              "largest_atom": biggest,
                              "atoms_carry_all_mass": sum(W) > 1.0
                              - (M / (M + 1.0)) ** int(n_terms) - 1e-9,
                              "method": "DP discreteness via atoms (GvdV 2017 Thm 4.14)"})
    return with_describe_pointer(res, "gh_c4_12")


def cheatsheet():
    return "gh_c4_12: Discreteness of DP realizations"
